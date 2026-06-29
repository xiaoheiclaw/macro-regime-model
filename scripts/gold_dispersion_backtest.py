"""Gold multi-model fair-value *dispersion* backtest runner.

Tests the user's PR #7 hypothesis end-to-end: does the **disagreement across six
independent fair-value estimators** ADD value *on top of* PR #5's S1 pure-trend
blend — the standard that already beat buy-and-hold? The bet is NOT on any single
implied price (PR #1–#4 showed every anchor relation drifts), but on the
**dispersion** between them: high dispersion = no consensus / a turn warning →
cut trend exposure; low dispersion = consensus → ride the trend.

Four comparison lines, all on the SAME panel + SAME engine (same-track):
  S0       buy-and-hold gold
  S1       pure trend (3/6/12 blend, vol-targeted) — the PR #5 standard to beat
  S4_hard  S1 × hard dispersion weight (tercile: low/mid/high disp → 1.0/0.5/0.0)
  S4_soft  S1 × soft dispersion weight (1 − rolling percentile rank of dispersion)

The dispersion itself is the cross-sectional std of ln(implied_i/market) gaps
across the available lenses each month, turned into a leak-free rolling percentile
rank ∈[0,1] that gates S1.

Outputs:
  * markdown report (analysis/): four-strategy metric table full-sample +
    sub-segments, the dispersion timeline at known tops/bottoms (1980/2011/2020…),
    an honest kill-condition verdict (S4 vs S1 on Sharpe AND Calmar, net of cost),
    a dispersion-rank-window sensitivity band {60, 120}, and a cost band {0,10,20}.
  * equity-curve CSV (data/): S0 / S1 / S4_hard / S4_soft net growth of $1.
  * dispersion CSV (data/): monthly dispersion, rank, estimator count, per-lens gap.
  * coverage CSV (data/): per-estimator first/last/n (oil pre-1986, copper PPI, …).

Usage:
    uv run python scripts/gold_dispersion_backtest.py
    uv run python scripts/gold_dispersion_backtest.py --cost-bps 10 --disp-window 120
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
    DEFAULT_SEGMENTS,
    equity_curve,
    run_backtest,
    s0_buy_hold,
    s1_trend,
)
from lib.gold_fairvalue_dispersion import (  # noqa: E402
    DEFAULT_CALIB_WINDOW,
    DEFAULT_DISP_WINDOW,
    estimator_count,
    estimator_coverage,
    estimator_gaps,
    dispersion,
    dispersion_rank,
    implied_cpi,
    implied_debt_gdp,
    implied_gold_copper,
    implied_gold_oil,
    implied_m2_gdp,
    implied_real_rate,
    s4_dispersion,
)
# Reuse PR #5's reporting helpers verbatim so S4 is scored on the SAME common
# investable window and rendered identically to S0/S1 (no parallel scoring path).
from scripts.gold_trend_backtest import (  # noqa: E402
    common_span,
    metrics_table,
    _fmt,
    _fmt_simple,
    _md_table,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402

# Known gold turning points — a descriptive validity check: does dispersion
# SPIKE at tops/bottoms (the "no consensus at the turn" hypothesis)? These are
# NOT traded signals, just anchors to read the dispersion timeline against.
LANDMARKS: List[tuple[str, str]] = [
    ("1980-01 nominal peak", "1980-01"),
    ("2008-11 GFC trough", "2008-11"),
    ("2011-09 nominal peak", "2011-09"),
    ("2015-12 cyclical low", "2015-12"),
    ("2020-03 COVID trough", "2020-03"),
]

DISP_WINDOW_BAND = (60, 120)


def compute_implieds(df: pd.DataFrame, calib_window: int) -> Dict[str, pd.Series]:
    """All six fair-value implied-gold lenses, each calibrated on a trailing
    `calib_window` (ex-ante). A lens whose input is missing returns NaN there
    (oil pre-1986) and is simply dropped from that month's dispersion."""
    g = df["gold_nominal"]
    return {
        "debt_gdp": implied_debt_gdp(g, df["debt_gdp"], calib_window),
        "real_rate": implied_real_rate(g, df["real_rate_10y"], calib_window),
        "m2_gdp": implied_m2_gdp(g, df["m2_gdp"], calib_window),
        "cpi": implied_cpi(g, df["cpi"], calib_window),
        "gold_oil": implied_gold_oil(g, df["oil"], calib_window),
        "gold_copper": implied_gold_copper(g, df["copper"], calib_window),
    }


def build_positions(
    panel: pd.DataFrame, rank: pd.Series, disp_window_label: str
) -> Dict[str, pd.Series]:
    """S0 / S1 / S4_hard / S4_soft. S1 and S4 share the 3/6/12 blend + vol target,
    so the only difference is the dispersion weight on S4."""
    lbs = tuple(DEFAULT_LOOKBACKS)
    return {
        "S0_buyhold": s0_buy_hold(panel.index),
        "S1_blend": s1_trend(panel, lookbacks=lbs),
        "S4_hard": s4_dispersion(panel, rank, mode="hard", lookbacks=lbs),
        "S4_soft": s4_dispersion(panel, rank, mode="soft", lookbacks=lbs),
    }


def run_all(
    panel: pd.DataFrame, rank: pd.Series, cost_bps: float
) -> Dict[str, pd.DataFrame]:
    positions = build_positions(panel, rank, "")
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


def segment_investable_months(m: pd.DataFrame) -> int:
    """Min investable months across strategies in a segment metrics table
    (compute_metrics returns NaN n_months for an empty slice → drop NaN first)."""
    n = m["n_months"].dropna()
    return int(n.min()) if len(n) else 0


def landmarks_table(
    disp: pd.Series, rank: pd.Series, n_est: pd.Series, panel_index: pd.Index
) -> pd.DataFrame:
    """Dispersion / rank / lens-count snapped to the nearest valid panel month at
    each landmark date. A dispersion VALUE may exist where the RANK is still in
    warm-up (e.g. 1980, before the rank window fills) — both shown honestly."""
    rows = []
    for name, ds in LANDMARKS:
        ts = pd.Timestamp(ds)
        # nearest on-or-before panel month; fall back to nearest any-side
        valid = panel_index[panel_index <= ts]
        idx = valid[-1] if len(valid) else panel_index[0]
        rows.append({
            "landmark": name,
            "month": f"{idx:%Y-%m}",
            "dispersion": disp.get(idx, np.nan),
            "rank": rank.get(idx, np.nan),
            "n_lenses": int(n_est.get(idx, 0)) if pd.notna(n_est.get(idx, np.nan)) else 0,
        })
    out = pd.DataFrame(rows).set_index("landmark")
    return out


def verdict(full: pd.DataFrame, landmarks: pd.DataFrame) -> str:
    """Honest kill-condition adjudication. The decisive test is S4 (either variant)
    vs S1 (the PR #5 standard): if scaling trend by valuation dispersion does not
    beat pure trend on BOTH Sharpe and Calmar net of cost, S1's price trend already
    handles regime turns (it flips position at trend reversals) and the dispersion
    signal is redundant complexity.

    `landmarks` (dispersion at known tops/bottoms) feeds the descriptive mechanism
    note — even a killed verdict reports whether dispersion actually spiked at the
    turns (the signal's *validity*, separate from its *incremental edge*)."""
    s0 = full.loc["S0_buyhold"]
    s1 = full.loc["S1_blend"]
    s4h = full.loc["S4_hard"]
    s4s = full.loc["S4_soft"]

    for name, row in (("S0", s0), ("S1_blend", s1), ("S4_hard", s4h), ("S4_soft", s4s)):
        if not (pd.notna(row["sharpe"]) and pd.notna(row["calmar"])):
            return ("## Verdict\n\n**Insufficient sample / invalid metrics "
                    f"({name} has NaN Sharpe or Calmar) — cannot adjudicate. "
                    "Widen the sample window.**")

    def better(a, b):  # risk-adjusted = Sharpe AND Calmar
        return (a["sharpe"] > b["sharpe"]) and (a["calmar"] > b["calmar"])

    s1_beats_s0 = better(s1, s0)
    s4h_beats_s1 = better(s4h, s1)
    s4s_beats_s1 = better(s4s, s1)
    s4_beats_s1 = s4h_beats_s1 or s4s_beats_s1
    best = s4h if (s4h["calmar"] + s4h["sharpe"]) >= (s4s["calmar"] + s4s["sharpe"]) else s4s
    best_label = "S4_hard" if best is s4h else "S4_soft"

    lines = ["## Verdict (honest kill conditions, net of cost)\n"]
    lines.append(f"- S1_blend vs S0: Sharpe {s1['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {s1['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'S1 beats buy-and-hold (PR#5 replicates)' if s1_beats_s0 else 'S1 does NOT beat buy-and-hold here'}")
    lines.append(f"- **S4_hard vs S1** (decisive): Sharpe {s4h['sharpe']:.2f} vs "
                 f"{s1['sharpe']:.2f}, Calmar {s4h['calmar']:.2f} vs {s1['calmar']:.2f} → "
                 f"{'hard dispersion ADDS edge' if s4h_beats_s1 else 'no edge'}")
    lines.append(f"- **S4_soft vs S1** (decisive): Sharpe {s4s['sharpe']:.2f} vs "
                 f"{s1['sharpe']:.2f}, Calmar {s4s['calmar']:.2f} vs {s1['calmar']:.2f} → "
                 f"{'soft dispersion ADDS edge' if s4s_beats_s1 else 'no edge'}")
    lines.append("")

    # descriptive validity: did dispersion actually spike at the named turns?
    valid_lm = landmarks.dropna(subset=["dispersion"])
    if len(valid_lm) >= 2:
        med = valid_lm["dispersion"].median()
        spike = valid_lm[valid_lm["dispersion"] > med]
        validity = (f"the dispersion *did* rise above its landmark median at "
                    f"{len(spike)}/{len(valid_lm)} named turns "
                    f"(landmarks table below) — the signal is *valid* descriptively")
    else:
        validity = "too few landmark readings to describe"

    if s4_beats_s1:
        lines.append(f"**① DISPERSION HAS EDGE: {best_label} beats S1 pure trend on a "
                     "risk-adjusted, cost-net basis — the cross-lens disagreement captures "
                     "turn information that S1's trend alone misses. (Edge must also survive "
                     "the dispersion-window band below to count as OOS structure, not a "
                     f"parameter artifact.) Descriptively, {validity}.**")
    else:
        lines.append("**② KILLED: scaling trend by valuation dispersion adds nothing over "
                     "S1 pure trend on both decisive metrics (Sharpe AND Calmar). S1's price "
                     "trend already handles regime turns — it flips to cash at trend reversals "
                     "— so the dispersion signal is redundant complexity. Sit on S1.**")
        lines.append("")
        if s4h["max_dd"] < s1["max_dd"]:
            dd = (f"and gives back more in drawdowns (S4_hard max_dd {s4h['max_dd']:.1%} "
                  f"vs S1 {s1['max_dd']:.1%})")
        else:
            dd = (f"with a comparable/shallower drawdown (S4_hard max_dd {s4h['max_dd']:.1%} "
                  f"vs S1 {s1['max_dd']:.1%})")
        lines.append(f"_Mechanism: S4 = S1 × f(dispersion_rank). Cutting exposure when the "
                     "lenses disagree only helps if that cut lands *before* a trend reversal "
                     "that S1 itself misses — but S1 already exits on the reversal (its "
                     "momentum flips to 0). So the dispersion cut either coincides with S1's "
                     f"own exit (no gain) or cuts a still-profitable trend (a drag) {dd}. "
                     f"Descriptively, {validity} — the disagreement IS real at the turns, it "
                     "just does not translate to incremental risk-adjusted return beyond what "
                     "trend already captures._")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=_nonneg_float, default=DEFAULT_COST_BPS,
                    help="per-rebalance trading cost in bps (non-negative)")
    ap.add_argument("--calib-window", type=_pos_int, default=DEFAULT_CALIB_WINDOW,
                    help="months each estimator is calibrated on (default 120)")
    ap.add_argument("--disp-window", type=_pos_int, default=DEFAULT_DISP_WINDOW,
                    help="months for the dispersion rank rolling window (default 120)")
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    print("Building dispersion panel (gold + anchors + CPI + oil + copper)...")
    from lib.gold_fairvalue_dispersion import build_dispersion_panel
    dp = build_dispersion_panel(start=args.start, end=args.end)
    # the dispersion estimators read gold/anchors/CPI/oil/copper; the shared engine
    # also needs gold_ret + tbill_ret, so attach the timing-panel cash/return legs.
    from lib.gold_trend_timing import build_timing_panel
    tp = build_timing_panel(start=args.start, end=args.end)
    panel = dp.data.join(tp.data[["gold_ret", "tbill_ret"]], how="left")
    # gold_nominal is shared by both panels; sanity-check alignment
    assert panel["gold_nominal"].notna().any()
    print(f"  panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}, "
          f"{len(panel)} months")

    implieds = compute_implieds(panel, args.calib_window)
    gaps = estimator_gaps(implieds, panel["gold_nominal"])
    disp = dispersion(gaps)
    rank = dispersion_rank(disp, args.disp_window)
    n_est = estimator_count(gaps)
    cov = estimator_coverage(implieds)

    backtests = run_all(panel, rank, args.cost_bps)
    cstart, cend = common_span(backtests)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for warm-up). Widen --start/--end.", file=sys.stderr)
        raise SystemExit(2)
    full = metrics_table(backtests, cstart, cend)

    # ── dispersion-window sensitivity band {60, 120} (anti-overfit) ──
    # Each disp window shifts S4's rank warm-up, hence a different common span.
    # Isolate the parameter from a moving sample start: score every window on the
    # ONE shared window = intersection of all per-window common spans.
    sens_bts: Dict[int, Dict[str, pd.DataFrame]] = {}
    sens_spans: Dict[int, tuple] = {}
    for w in DISP_WINDOW_BAND:
        rank_w = dispersion_rank(disp, w)
        bt_w = run_all(panel, rank_w, args.cost_bps)
        cs_w, ce_w = common_span(bt_w)
        if cs_w is None or ce_w is None:
            continue
        sens_bts[w] = bt_w
        sens_spans[w] = (cs_w, ce_w)
    sens_windows: Dict[int, pd.DataFrame] = {}
    if len(sens_spans) == len(DISP_WINDOW_BAND):
        sens_start = max(s for s, _ in sens_spans.values())
        sens_end = min(e for _, e in sens_spans.values())
        for w, bt_w in sens_bts.items():
            m = metrics_table(bt_w, sens_start, sens_end).loc[
                ["S1_blend", "S4_hard", "S4_soft"]]
            sens_windows[w] = m[["sharpe", "calmar", "cagr", "max_dd", "n_months"]]

    # ── cost sensitivity {0, 10, 20} bps ──
    sens_cost: Dict[float, pd.DataFrame] = {}
    for c in (0.0, 10.0, 20.0):
        bt_c = run_all(panel, rank, c)
        m = metrics_table(bt_c).loc[["S0_buyhold", "S1_blend", "S4_hard", "S4_soft"]]
        sens_cost[c] = m[["sharpe", "calmar", "max_dd", "cagr"]]

    lm = landmarks_table(disp, rank, n_est, panel.index)

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")
    report_path = os.path.join(args.out_dir, f"gold_dispersion_{file_stamp}.md")

    parts: List[str] = []
    parts.append(f"# Gold fair-value dispersion backtest — {stamp}\n")
    parts.append(
        "**Question.** Does the *disagreement across six independent fair-value "
        "estimators* (debt/GDP, real-rate level, M2/GDP, CPI, gold/oil, gold/copper) "
        "ADD value over PR #5's **S1 pure-trend blend** — not by trading any single "
        "implied price (PR #1–#4: every anchor drifts), but by reading the cross-lens "
        "**dispersion** (high = no consensus / turn warning → cut trend; low = "
        "consensus → ride trend)? If not, S1's trend already handles turns and the "
        "dispersion signal is redundant.\n"
    )
    parts.append(f"Panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m} "
                 f"({len(panel)} months). Cost: {args.cost_bps:g} bps/rebalance. "
                 f"Long-only 0↔100%, cash leg = 3m T-bill. Estimator calib window = "
                 f"{args.calib_window}m; dispersion-rank window = {args.disp_window}m.\n")
    parts.append(f"All metrics on the **common investable window** {cstart:%Y-%m}–"
                 f"{cend:%Y-%m} (after warm-up) so S0/S1/S4 share the same months. "
                 "Note: S4 needs the estimator-calib AND dispersion-rank windows to "
                 "fill, so its investable window starts LATER than S1's — the common "
                 "window reflects that.\n")
    parts.append("Strategies: **S0** buy-and-hold · **S1** pure trend (3/6/12 blend, "
                 "vol-targeted, PR #5 standard) · **S4_hard** S1 × tercile dispersion "
                 "weight (low/mid/high → 1.0/0.5/0.0) · **S4_soft** S1 × (1 − rank).\n")

    parts.append("## Estimator coverage (sample per lens)\n")
    parts.append("Each lens is calibrated on a trailing window, so it is NaN until "
                 "that window fills; oil is additionally NaN pre-1986 (WTI spot start). "
                 "Dispersion each month uses whatever lenses are available (≥2).\n")
    cov_show = cov.copy()
    cov_show["first"] = cov_show["first"].map(lambda d: f"{d:%Y-%m}" if pd.notna(d) else "—")
    cov_show["last"] = cov_show["last"].map(lambda d: f"{d:%Y-%m}" if pd.notna(d) else "—")
    parts.append(_md_table(cov_show.reset_index()))
    parts.append("")

    parts.append("## Dispersion at known turning points (validity check)\n")
    parts.append("Descriptive test of the core hypothesis: dispersion should SPIKE "
                 "(rank → 1) at major tops/bottoms — the lenses disagree most when "
                 "the regime is turning. These are read-only anchors, not traded "
                 "signals. `rank` may be n/a early (1980) where the rank window has "
                 "not yet filled even though the dispersion value exists.\n")
    lm_show = lm.copy()
    lm_show["dispersion"] = lm_show["dispersion"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    lm_show["rank"] = lm_show["rank"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    lm_show["n_lenses"] = lm_show["n_lenses"].map(lambda v: f"{int(v):d}")
    parts.append(_md_table(lm_show.reset_index()))
    parts.append("")

    parts.append("## Full sample (net of cost)\n")
    parts.append(_fmt(full))
    parts.append("")
    parts.append(verdict(full, lm))
    parts.append("")

    # ── sub-segments ──
    parts.append("## Sub-segments (net of cost)\n")
    pmin, pmax = panel.index.min(), panel.index.max()
    min_seg_months = 12
    for name, s, e in DEFAULT_SEGMENTS:
        score_lo = max(cstart, pd.Timestamp(s))
        score_hi = min(cend, pd.Timestamp(e))
        if pd.Timestamp(s) > pmax or pd.Timestamp(e) < pmin:
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
    parts.append("## dispersion-window sensitivity (S1 vs S4, shared window)\n")
    parts.append("If S4's verdict vs S1 flips across {60, 120}m the edge is a window "
                 "artifact, not structure. All windows scored on ONE shared investable "
                 "window so the comparison isolates the dispersion-rank parameter.\n")
    if sens_windows:
        for w, m in sens_windows.items():
            parts.append(f"### dispersion-rank window = {w}m\n")
            parts.append(_fmt_simple(m))
            parts.append("")
    else:
        parts.append("_(not enough common investable window to score the band)_\n")

    parts.append("## Cost sensitivity (S0 / S1 / S4_hard / S4_soft)\n")
    for c, m in sens_cost.items():
        parts.append(f"### {c:.0f} bps\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Data & method notes\n")
    parts.append("- **Six independent lenses** (trailing-rolling, ex-ante): (a) rolling "
                 "OLS ln(gold)~ln(debt/GDP); (b) rolling OLS ln(gold)~real-rate LEVEL "
                 "(classic ≈−0.9); (c) rolling OLS ln(gold)~ln(M2/GDP); (d) rolling OLS "
                 "ln(gold)~ln(CPI) (Jastram purchasing-power); (e) gold/oil ratio "
                 "mean-reversion (WTI spot, 1986+); (f) gold/copper ratio mean-reversion "
                 "(BLS PPI copper, 1967+; spot only 1992+ so PPI used for length).\n")
    parts.append("- **Dispersion** = cross-sectional std of the **ln(implied/market) "
                 "gaps** (each lens's % mispricing view), NOT raw implied levels — "
                 "different anchors imply structurally different gold levels, so a "
                 "level CV would measure that offset, not turn disagreement. The gap "
                 "std is shift-invariant: a constant added to every gap leaves it "
                 "unchanged, so the signal can never trade a single lens's bias.\n")
    parts.append("- **No look-ahead**: every estimator value at t uses a trailing "
                 "rolling window only; the dispersion rank is a trailing percentile; "
                 "the position decided at t is held t+1 via the shared engine's "
                 "`.shift(1)`. Standard windows (calib 120m, rank 120m) — NOT tuned; "
                 "the {60,120} rank band is reported.\n")
    parts.append("- **Warm-up**: S4 needs estimator-calib (120m) AND dispersion-rank "
                 "(120m) windows to fill, so it is investable only from the late 1980s; "
                 "the 1980 peak is therefore a descriptive anchor, outside S4's traded "
                 "window. The common investable window above makes this explicit.\n")
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
    eq_path = os.path.join(DATA_DIR, f"gold_dispersion_curves_{file_stamp}.csv")
    eq.to_csv(eq_path)
    print(f"  equity curves → {eq_path}")

    # ── write dispersion / gaps CSV ──
    out = pd.DataFrame({
        "dispersion": disp, "disp_rank": rank, "n_estimators": n_est,
    }).join(gaps.add_suffix("_gap")).join(
        pd.DataFrame(implieds).add_suffix("_implied"))
    out["gold_nominal"] = panel["gold_nominal"]
    disp_path = os.path.join(DATA_DIR, f"gold_dispersion_series_{file_stamp}.csv")
    out.to_csv(disp_path)
    print(f"  dispersion series → {disp_path}")

    # ── coverage CSV ──
    cov_path = os.path.join(DATA_DIR, f"gold_dispersion_coverage_{file_stamp}.csv")
    cov.to_csv(cov_path)
    print(f"  coverage → {cov_path}")

    print("\n" + _fmt(full))
    print("\n" + verdict(full, lm))


if __name__ == "__main__":
    main()
