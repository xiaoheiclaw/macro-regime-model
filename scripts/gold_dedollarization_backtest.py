"""Gold de-dollarization leading-indicator *size-modulation* backtest (PR #8).

Tests the user's last de-dollarization cut end-to-end: does a **forward / high-
frequency de-dollarization proxy** (foreign official UST custody share, falling =
central banks retreating from USD) ADD value over PR #5's **S1 pure-trend blend**
— used as a SIZE / persistence modulator, not an entry signal? The bet: when de-
dollarization is strong, a gold up-trend persists longer → hold a bigger slice of
the S1 position; when weak, trim it. Prior: three earlier cuts (PR #5-S2 real-rate
gate, PR #6 regime dominance, PR #7 fair-value dispersion) all LOST to S1, because
any external-relationship signal lags a price that already discounts the flow.

Four comparison lines, all on the SAME panel + SAME engine (same-track):
  S0       buy-and-hold gold
  S1       pure trend (3/6/12 blend, vol-targeted) — the PR #5 standard to beat
  S5_soft  S1 × linear de-dollarization factor (f_min..f_max increasing in rank)
  S5_hard  S1 × tercile factor (weak/neutral/strong → f_min/1.0/f_max)

The de-dollarization strength is the negated trailing-12m change of the foreign
official custody share, turned into a leak-free rolling percentile rank ∈[0,1].
Strong (rank→1) amplifies S1 (capped at 100%); weak (rank→0) trims it; no signal
(warm-up / missing proxy) → neutral → S5 = S1.

Outputs:
  * markdown report (analysis/): four-strategy metric table full-sample +
    sub-segments (esp. the 2022+ sanctions/de-dollarization era), the de-
    dollarization timeline at known episodes with an explicit 2022+ check, an
    honest kill-condition verdict (S5 vs S1 on Sharpe AND Calmar net of cost), a
    rank-window sensitivity band {36, 60}, and a cost band {0,10,20}.
  * equity-curve CSV (data/): S0 / S1 / S5_soft / S5_hard net growth of $1.
  * signal CSV (data/): monthly custody, debt, share, strength, rank, factor.

Usage:
    uv run python scripts/gold_dedollarization_backtest.py
    uv run python scripts/gold_dedollarization_backtest.py --cost-bps 10 --rank-window 48
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_trend_timing import (  # noqa: E402
    DEFAULT_COST_BPS,
    DEFAULT_LOOKBACKS,
    equity_curve,
    run_backtest,
    s0_buy_hold,
    s1_trend,
)
from lib.gold_dedollarization_leading import (  # noqa: E402
    DEFAULT_CHANGE_WINDOW,
    DEFAULT_RANK_WINDOW,
    build_dedollar_panel,
    custody_share,
    dedollar_factor,
    dedollar_rank,
    dedollar_strength,
    s5_dedollar,
    signal_available,
)
# Reuse PR #5's reporting helpers verbatim so S5 is scored on the SAME common
# investable window and rendered identically to S0/S1 (no parallel scoring path).
from scripts.gold_trend_backtest import (  # noqa: E402
    common_span,
    metrics_table,
    _fmt,
    _fmt_simple,
    _md_table,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402

# De-dollarization episodes — descriptive anchors to read the timeline against
# (NOT traded signals). The 2022-02 Russia reserve freeze is the canonical
# de-dollarization catalyst; the strength should be elevated 2015+ and 2022+.
LANDMARKS: List[tuple[str, str]] = [
    ("2011-08 debt-ceiling/downgrade", "2011-08"),
    ("2014-08 foreign-official peak", "2014-08"),
    ("2018-06 trade-war escalation", "2018-06"),
    ("2022-02 Russia reserve freeze", "2022-02"),
    ("2023-12 BRICS de-dollarization push", "2023-12"),
]

# Segments tuned to the custody-data era (post-2002), not the 1968 trend window.
DEDOLLAR_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("2008-2014 accumulation", "2008-01-01", "2014-12-31"),
    ("2015-2021 plateau/decline", "2015-01-01", "2021-12-31"),
    ("2022-2026 sanctions era", "2022-01-01", "2026-12-31"),
)

RANK_WINDOW_BAND = (36, 60)

# The "2022+ de-dollarization should be strong" check reads months from here on.
CHECK_2022 = pd.Timestamp("2022-01-01")


def _month_start(s: str) -> pd.Timestamp:
    """Snap a date string (YYYY-MM or YYYY-MM-DD) to the first day of its month."""
    return pd.Period(s, freq="M").to_timestamp()


def _month_end(s: str) -> pd.Timestamp:
    """Snap a date string to the LAST day (month-end) of its month — the panel is
    month-end indexed, so an end/landmark bound must be a month-end or `<= bound`
    drops the final month."""
    return pd.Period(s, freq="M").to_timestamp("M")


def build_positions(panel: pd.DataFrame, rank: pd.Series) -> Dict[str, pd.Series]:
    """S0 / S1 / S5_soft / S5_hard. S1 and S5 share the 3/6/12 blend + vol target,
    so the only difference is the de-dollarization size factor on S5."""
    lbs = tuple(DEFAULT_LOOKBACKS)
    return {
        "S0_buyhold": s0_buy_hold(panel.index),
        "S1_blend": s1_trend(panel, lookbacks=lbs),
        "S5_soft": s5_dedollar(panel, rank, mode="soft", lookbacks=lbs),
        "S5_hard": s5_dedollar(panel, rank, mode="hard", lookbacks=lbs),
    }


def run_all(panel: pd.DataFrame, rank: pd.Series, cost_bps: float) -> Dict[str, pd.DataFrame]:
    positions = build_positions(panel, rank)
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


def segment_investable_months(m: pd.DataFrame) -> int:
    """Min investable months across strategies in a segment metrics table."""
    n = m["n_months"].dropna()
    return int(n.min()) if len(n) else 0


def timeline_table(
    share: pd.Series,
    strength: pd.Series,
    rank: pd.Series,
    panel_index: pd.Index,
) -> pd.DataFrame:
    """De-dollarization signal snapped to the nearest valid panel month at each
    landmark. `rank` may be n/a early (windows not yet filled) even where the share
    exists — shown honestly."""
    rows = []
    for name, ds in LANDMARKS:
        ts = _month_end(ds)
        valid = panel_index[panel_index <= ts]
        idx = valid[-1] if len(valid) else panel_index[0]
        rows.append({
            "landmark": name,
            "month": f"{idx:%Y-%m}",
            "custody_share": share.get(idx, np.nan),
            "strength": strength.get(idx, np.nan),
            "rank": rank.get(idx, np.nan),
        })
    return pd.DataFrame(rows).set_index("landmark")


def check_2022(rank: pd.Series) -> str:
    """Descriptive validity: is de-dollarization PERSISTENTLY strong post-2022? The
    thesis needs the rank to sit high (≥2/3, S5_hard's amplify tier) through the
    sanctions era. Reports the share of post-2022 months in each rank tier."""
    r = rank.loc[rank.index >= CHECK_2022].dropna()
    if len(r) == 0:
        return ("- **2022+ check**: no ranked months post-2022 (signal still in "
                "warm-up or unavailable) — cannot assess persistence.")
    hi = float((r >= 2.0 / 3.0).mean())
    mid = float(((r >= 1.0 / 3.0) & (r < 2.0 / 3.0)).mean())
    lo = float((r < 1.0 / 3.0).mean())
    mean_rank = float(r.mean())
    verdict_word = ("PERSISTENTLY STRONG" if hi >= 0.5 else
                    "MIXED" if hi + mid >= 0.5 else "WEAK/ABSENT")
    return (f"- **2022+ check** ({len(r)} ranked months, mean rank {mean_rank:.2f}): "
            f"{hi:.0%} strong (≥2/3) · {mid:.0%} neutral · {lo:.0%} weak → "
            f"de-dollarization signal is **{verdict_word}** post-2022.")


def verdict(full: pd.DataFrame, rank: pd.Series) -> str:
    """Honest kill-condition adjudication. Decisive test: S5 (either variant) vs S1
    (the PR #5 standard). If modulating trend SIZE by a de-dollarization proxy does
    not beat pure trend on BOTH Sharpe and Calmar net of cost, the proxy lags the
    price that already discounts the flow — the fourth independent confirmation that
    S1 suffices and the de-dollarization intuition is closed out for trading."""
    s0 = full.loc["S0_buyhold"]
    s1 = full.loc["S1_blend"]
    s5s = full.loc["S5_soft"]
    s5h = full.loc["S5_hard"]

    for name, row in (("S0", s0), ("S1_blend", s1), ("S5_soft", s5s), ("S5_hard", s5h)):
        if not (pd.notna(row["sharpe"]) and pd.notna(row["calmar"])):
            return ("## Verdict\n\n**Insufficient sample / invalid metrics "
                    f"({name} has NaN Sharpe or Calmar) — cannot adjudicate. "
                    "Widen the sample window.**")

    def better(a, b):  # risk-adjusted = Sharpe AND Calmar
        return (a["sharpe"] > b["sharpe"]) and (a["calmar"] > b["calmar"])

    s1_beats_s0 = better(s1, s0)
    s5s_beats_s1 = better(s5s, s1)
    s5h_beats_s1 = better(s5h, s1)
    s5_beats_s1 = s5s_beats_s1 or s5h_beats_s1
    winners = [lbl for lbl, won in (("S5_soft", s5s_beats_s1), ("S5_hard", s5h_beats_s1)) if won]

    lines = ["## Verdict (honest kill conditions, net of cost)\n"]
    lines.append(f"- S1_blend vs S0: Sharpe {s1['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {s1['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'S1 beats buy-and-hold (PR#5 replicates)' if s1_beats_s0 else 'S1 does NOT beat buy-and-hold here'}")
    lines.append(f"- **S5_soft vs S1** (decisive): Sharpe {s5s['sharpe']:.2f} vs "
                 f"{s1['sharpe']:.2f}, Calmar {s5s['calmar']:.2f} vs {s1['calmar']:.2f} → "
                 f"{'soft size modulation ADDS edge' if s5s_beats_s1 else 'no edge'}")
    lines.append(f"- **S5_hard vs S1** (decisive): Sharpe {s5h['sharpe']:.2f} vs "
                 f"{s1['sharpe']:.2f}, Calmar {s5h['calmar']:.2f} vs {s1['calmar']:.2f} → "
                 f"{'hard size modulation ADDS edge' if s5h_beats_s1 else 'no edge'}")
    lines.append("")
    lines.append(check_2022(rank))
    lines.append("")

    if s5_beats_s1:
        lines.append(f"**① DE-DOLLARIZATION HAS EDGE: {' and '.join(winners)} beat S1 pure "
                     "trend on a risk-adjusted, cost-net basis — the leading proxy genuinely "
                     "led price and sized the trend correctly. (Must also survive the "
                     "rank-window band below to count as OOS structure, not a parameter "
                     "artifact.)**")
    else:
        lines.append("**② KILLED (4th independent confirmation): modulating trend SIZE by a "
                     "de-dollarization proxy adds nothing over S1 pure trend on both decisive "
                     "metrics (Sharpe AND Calmar). The proxy is a slow institutional flow that "
                     "LAGS a gold price already discounting it — exactly the prior. After the "
                     "real-rate gate (PR#5-S2), regime dominance (PR#6) and fair-value "
                     "dispersion (PR#7), this closes the de-dollarization intuition for "
                     "trading: the signal may be REAL (see the 2022+ check) but it is "
                     "pre-priced. Sit on S1.**")
        lines.append("")
        lines.append("_Mechanism: S5 = clip(S1 × f(dedollar_rank), 0, 1). Sizing UP a trend "
                     "when de-dollarization is strong only helps if the proxy turns strong "
                     "BEFORE the price trend it should prolong — but foreign-official flows are "
                     "reported with a lag and the trend has usually already moved, so the "
                     "up-size either arrives late (no gain) or amplifies a trend about to mean-"
                     "revert (a drag); the down-size symmetrically trims trends that S1's own "
                     "momentum would have ridden. The signal's VALIDITY (is de-dollarization "
                     "real post-2022?) is separate from its incremental trading EDGE._")
    return "\n".join(lines)


def _nonneg_float(x: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return v


def _pos_int(x: str) -> int:
    v = int(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _int_ge2(x: str) -> int:
    # the rank window must be >= 2 (dedollar_rank rejects 1); fail at the argparse
    # stage so the user gets the error immediately, not mid-run.
    v = int(x)
    if v < 2:
        raise argparse.ArgumentTypeError("must be an integer >= 2")
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=_nonneg_float, default=DEFAULT_COST_BPS,
                    help="per-rebalance trading cost in bps (non-negative)")
    ap.add_argument("--change-window", type=_pos_int, default=DEFAULT_CHANGE_WINDOW,
                    help="months for the trailing custody-share change (default 12)")
    ap.add_argument("--rank-window", type=_int_ge2, default=DEFAULT_RANK_WINDOW,
                    help="months for the de-dollarization rank window (>=2, default 48)")
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    print("Building de-dollarization panel (gold + custody + debt)...")
    dp = build_dedollar_panel(start=args.start, end=args.end)
    # the signal reads gold/custody/debt; the shared engine also needs gold_ret +
    # tbill_ret, so attach the timing-panel cash/return legs (same as PR #7).
    from lib.gold_trend_timing import build_timing_panel
    tp = build_timing_panel(start=args.start, end=args.end)
    panel = dp.data.join(tp.data[["gold_ret", "tbill_ret"]], how="left")
    assert panel["gold_nominal"].notna().any()
    print(f"  panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}, "
          f"{len(panel)} months")

    share = custody_share(panel["foreign_official_custody"], panel["total_public_debt"])
    strength = dedollar_strength(share, args.change_window)
    rank = dedollar_rank(strength, args.rank_window)

    available = signal_available(rank)
    if not available:
        print("  WARNING: de-dollarization proxy unavailable (no ranked months) — "
              "S5 degrades to S1. Report will note the fallback.", file=sys.stderr)

    backtests = run_all(panel, rank, args.cost_bps)
    cstart, cend = common_span(backtests)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for warm-up). Widen --start/--end.", file=sys.stderr)
        raise SystemExit(2)
    full = metrics_table(backtests, cstart, cend)

    # ── rank-window sensitivity band {36, 60} (anti-overfit), one shared window ──
    sens_bts: Dict[int, Dict[str, pd.DataFrame]] = {}
    sens_spans: Dict[int, tuple] = {}
    for w in RANK_WINDOW_BAND:
        rank_w = dedollar_rank(strength, w)
        bt_w = run_all(panel, rank_w, args.cost_bps)
        cs_w, ce_w = common_span(bt_w)
        if cs_w is None or ce_w is None:
            continue
        sens_bts[w] = bt_w
        sens_spans[w] = (cs_w, ce_w)
    sens_windows: Dict[int, pd.DataFrame] = {}
    if len(sens_spans) == len(RANK_WINDOW_BAND):
        sens_start = max(s for s, _ in sens_spans.values())
        sens_end = min(e for _, e in sens_spans.values())
        for w, bt_w in sens_bts.items():
            m = metrics_table(bt_w, sens_start, sens_end).loc[
                ["S1_blend", "S5_soft", "S5_hard"]]
            sens_windows[w] = m[["sharpe", "calmar", "cagr", "max_dd", "n_months"]]

    # ── cost sensitivity {0, 10, 20} bps on the SAME common window ──
    sens_cost: Dict[float, pd.DataFrame] = {}
    for c in (0.0, 10.0, 20.0):
        bt_c = run_all(panel, rank, c)
        m = metrics_table(bt_c, cstart, cend).loc[["S0_buyhold", "S1_blend", "S5_soft", "S5_hard"]]
        sens_cost[c] = m[["sharpe", "calmar", "max_dd", "cagr"]]

    tl = timeline_table(share, strength, rank, panel.index)

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")
    report_path = os.path.join(args.out_dir, f"gold_dedollarization_{file_stamp}.md")

    parts: List[str] = []
    parts.append(f"# Gold de-dollarization leading-indicator size-modulation backtest — {stamp}\n")
    parts.append(
        "**Question.** Does a *forward / high-frequency de-dollarization proxy* "
        "(foreign official UST custody share — falling = central banks retreating "
        "from USD) ADD value over PR #5's **S1 pure-trend blend**, used as a SIZE / "
        "persistence modulator (not an entry signal)? Strong de-dollarization → size "
        "UP (trend persists longer); weak → size DOWN. Prior: PR #5-S2 (real-rate "
        "gate), PR #6 (regime dominance) and PR #7 (fair-value dispersion) all LOST "
        "to S1 — any external-relationship signal lags a price that already discounts "
        "it. This is the cleanest, fastest de-dollarization proxy — the fairest single "
        "shot to falsify (or flip) that intuition.\n"
    )
    if not available:
        parts.append("> **DATA FALLBACK ACTIVE**: the de-dollarization proxy returned no "
                     "ranked months (series unavailable). S5 degrades to S1 (neutral factor) "
                     "and the head-to-head below is therefore S5 ≡ S1 — read as 'no proxy, no "
                     "modulation', not as evidence either way.\n")
    parts.append(f"Panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m} "
                 f"({len(panel)} months). Cost: {args.cost_bps:g} bps/rebalance. "
                 f"Long-only 0↔100%, cash leg = 3m T-bill. Custody-share change "
                 f"window = {args.change_window}m; rank window = {args.rank_window}m.\n")
    parts.append(f"All metrics on the **common investable window** {cstart:%Y-%m}–"
                 f"{cend:%Y-%m} (after warm-up) so S0/S1/S5 share the same months. "
                 "Because the factor is neutral (not NaN) where the rank is undefined, "
                 "S5 and S1 share an identical investable window — the modulation simply "
                 "switches on once the proxy's rank exists.\n")
    parts.append("Strategies: **S0** buy-and-hold · **S1** pure trend (3/6/12 blend, "
                 "vol-targeted, PR #5 standard) · **S5_soft** S1 × linear de-dollarization "
                 "factor (f∈[0.5,1.5] increasing in rank) · **S5_hard** S1 × tercile factor "
                 "(weak/neutral/strong → 0.5/1.0/1.5).\n")

    parts.append("## De-dollarization timeline at known episodes (validity check)\n")
    parts.append("Descriptive test of the thesis: the custody share should FALL (strength "
                 "rises, rank → 1) through the de-dollarization era — esp. post-2022 (Russia "
                 "reserve freeze). Read-only anchors, not traded signals. `rank` may be n/a "
                 "early where the change/rank windows have not yet filled.\n")
    tl_show = tl.copy()
    tl_show["custody_share"] = tl_show["custody_share"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    tl_show["strength"] = tl_show["strength"].map(
        lambda v: f"{v:+.5f}" if pd.notna(v) else "n/a")
    tl_show["rank"] = tl_show["rank"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    parts.append(_md_table(tl_show.reset_index()))
    parts.append("")
    parts.append(check_2022(rank))
    parts.append("")

    parts.append("## Full sample (net of cost)\n")
    parts.append("_Note: the custody proxy only exists from ~2002-12, so pre-2003 the "
                 "factor is neutral and S5 ≡ S1 by construction. The full-sample table "
                 "therefore DILUTES the modulation (it acts on only ~1/3 of these months); "
                 "the decisive evidence is the post-2008 active segments below, esp. the "
                 "2022+ sanctions era where de-dollarization is strongest._\n")
    parts.append(_fmt(full))
    parts.append("")
    parts.append(verdict(full, rank))
    parts.append("")

    # ── sub-segments ──
    parts.append("## Sub-segments (net of cost)\n")
    pmin, pmax = panel.index.min(), panel.index.max()
    min_seg_months = 12
    for name, s, e in DEDOLLAR_SEGMENTS:
        seg_lo, seg_hi = _month_start(s), _month_end(e)
        score_lo = max(cstart, seg_lo)
        score_hi = min(cend, seg_hi)
        if seg_lo > pmax or seg_hi < pmin:
            parts.append(f"### {name}\n_(skipped: no overlap with sample)_\n")
            continue
        m = metrics_table(backtests, score_lo, score_hi)
        n_tradeable = segment_investable_months(m)
        if score_lo > score_hi or n_tradeable < min_seg_months:
            parts.append(f"### {name}\n_(skipped: only {n_tradeable} investable "
                         f"months in {score_lo:%Y-%m}–{score_hi:%Y-%m}, <{min_seg_months})_\n")
            continue
        parts.append(f"### {name} ({score_lo:%Y-%m}–{score_hi:%Y-%m})\n")
        parts.append(_fmt(m))
        parts.append("")

    # ── sensitivity bands ──
    parts.append("## rank-window sensitivity (S1 vs S5, shared window)\n")
    parts.append("If S5's verdict vs S1 flips across {36, 60}m the edge is a window "
                 "artifact, not structure. All windows scored on ONE shared investable "
                 "window so the comparison isolates the rank-window parameter.\n")
    if sens_windows:
        for w, m in sens_windows.items():
            parts.append(f"### rank window = {w}m\n")
            parts.append(_fmt_simple(m))
            parts.append("")
    else:
        parts.append("_(not enough common investable window to score the band)_\n")

    parts.append("## Cost sensitivity (S0 / S1 / S5_soft / S5_hard)\n")
    for c, m in sens_cost.items():
        parts.append(f"### {c:.0f} bps\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Data & method notes\n")
    parts.append("- **Leading proxy**: foreign official custody of marketable UST (Fed "
                 "H.4.1 `WMTSECL1`, weekly) — the fastest clean public fingerprint of "
                 "foreign central-bank USD Treasury exposure. De-dollarization strength = "
                 "−Δ(custody / total public debt) over the trailing change window (a "
                 "falling foreign-official SHARE of the UST market), ranked over a trailing "
                 "window into [0,1]. Standard windows (change 12m, rank 48m) — NOT tuned; "
                 "the {36,60} rank band is reported.\n")
    parts.append("- **Why the share, not the level**: raw custody grows with the debt "
                 "stock; only the SHARE captures foreign officials holding relatively LESS "
                 "USD. The change is negated so falling share → positive strength.\n")
    parts.append("- **Size modulation, not entry**: position = clip(S1 × f(rank), 0, 1), f "
                 "increasing. Strong de-dollarization amplifies S1 (f>1) but the 0–100% cap "
                 "means never leverage (amplification only bites where vol-targeting left "
                 "headroom); weak trims it (f<1); no view (warm-up / missing proxy) → f=1 → "
                 "S5 = S1. The factor never CREATES a position (S1=0 → S5=0).\n")
    parts.append("- **No look-ahead**: the share change is trailing; the rank is a trailing "
                 "percentile; the position decided at t is held t+1 via the shared engine's "
                 "`.shift(1)`. S5 and S1 share an identical investable window.\n")
    parts.append("- **Availability constraint**: custody starts ~2002-12 → a post-2002 "
                 "backtest by construction. TIC 'Major Foreign Holders' monthly and auction "
                 "indirect-bid aggregates are not clean single FRED series; IMF COFER USD "
                 "reserve share is the slow quarterly fallback (not on FRED as a simple "
                 "series). Custody is used for speed — the task's 'fastest available' rule.\n")
    parts.append("## Provenance\n")
    for k, v in dp.notes.items():
        parts.append(f"- **{k}**: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  report → {report_path}")

    # ── write equity-curve CSV (common window, rebased to $1) ──
    os.makedirs(DATA_DIR, exist_ok=True)
    curves = {label: equity_curve(bt.loc[cstart:cend]) for label, bt in backtests.items()}
    eq = pd.DataFrame(curves)
    eq_path = os.path.join(DATA_DIR, f"gold_dedollarization_curves_{file_stamp}.csv")
    eq.to_csv(eq_path)
    print(f"  equity curves → {eq_path}")

    # ── write signal CSV ──
    out = pd.DataFrame({
        "foreign_official_custody": panel["foreign_official_custody"],
        "total_public_debt": panel["total_public_debt"],
        "custody_share": share,
        "dedollar_strength": strength,
        "dedollar_rank": rank,
        "factor_soft": dedollar_factor(rank, mode="soft"),
        "factor_hard": dedollar_factor(rank, mode="hard"),
        "gold_nominal": panel["gold_nominal"],
    })
    sig_path = os.path.join(DATA_DIR, f"gold_dedollarization_series_{file_stamp}.csv")
    out.to_csv(sig_path)
    print(f"  signal series → {sig_path}")

    print("\n" + _fmt(full))
    print("\n" + verdict(full, rank))


if __name__ == "__main__":
    main()
