"""Gold regime-dominance backtest runner.

Tests the user's hypothesis end-to-end: does *explicitly classifying* gold's
dominant factor (real-rate regime vs de-dollarization regime) and trading the
dominant factor beat PR #5's S1 pure-trend blend — the standard that already
beat buy-and-hold (PR #5)?

Three comparison lines, all on the SAME panel + SAME engine (so the head-to-
head is same-track, not a re-implementation):
  S0  buy-and-hold gold
  S1  pure trend (3/6/12 blend, vol-targeted) — the PR #5 standard to beat
  SD  regime-dominance conditional: real-rate-dominant months → real-rate
      signal; de-dollarization months → trend signal (smooth prob blend)

Outputs:
  * markdown report (analysis/): three-strategy metric table full-sample +
    sub-segments, the regime timeline (does it flip to de-dollarization around
    2022?), an honest kill-condition verdict (SD vs S1 on Sharpe AND Calmar),
    a corr-window sensitivity band {24, 36, 48}, and a cost band {0,10,20}bps.
  * equity-curve CSV (data/): S0 / S1_blend / SD_blend net growth of $1.
  * regime CSV (data/): monthly de-dollarization probability + hard label.

Usage:
    uv run python scripts/gold_dominance_backtest.py
    uv run python scripts/gold_dominance_backtest.py --cost-bps 10 --corr-window 36
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_trend_timing import (  # noqa: E402
    DEFAULT_COST_BPS,
    DEFAULT_LOOKBACKS,
    DEFAULT_SEGMENTS,
    build_timing_panel,
    equity_curve,
    run_backtest,
    s0_buy_hold,
    s1_trend,
)
from lib.gold_regime_dominance import (  # noqa: E402
    DEFAULT_CORR_WINDOW,
    divergence_share,
    dominance_probability,
    level_divergence,
    regime_label,
    regime_timeline,
    rolling_gold_realrate_corr,
    s3_dominance,
)
# Reuse PR #5's reporting helpers verbatim so SD is scored on the SAME common
# investable window and rendered identically to S0/S1 (no parallel scoring path).
from scripts.gold_trend_backtest import (  # noqa: E402
    common_span,
    metrics_table,
    _fmt,
    _fmt_simple,
    _md_table,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402


def build_positions(panel: pd.DataFrame, prob: pd.Series) -> Dict[str, pd.Series]:
    """S0 / S1_blend / SD_blend position series, keyed by label. All three use
    the 3/6/12 blend so the only difference is the *signal*, not the lookback."""
    return {
        "S0_buyhold": s0_buy_hold(panel.index),
        "S1_blend": s1_trend(panel, lookbacks=tuple(DEFAULT_LOOKBACKS)),
        "SD_blend": s3_dominance(panel, prob, lookbacks=tuple(DEFAULT_LOOKBACKS)),
    }


def run_all(panel: pd.DataFrame, prob: pd.Series, cost_bps: float) -> Dict[str, pd.DataFrame]:
    positions = build_positions(panel, prob)
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


def segment_investable_months(m: pd.DataFrame) -> int:
    """Min investable months across strategies in a segment metrics table.
    compute_metrics returns NaN n_months for an empty slice, so a plain
    int(m['n_months'].min()) would crash on a segment that overlaps the panel
    but has no tradeable months after warm-up — drop NaN first, treat as 0."""
    n = m["n_months"].dropna()
    return int(n.min()) if len(n) else 0


def verdict(full: pd.DataFrame, mean_prob: float = float("nan")) -> str:
    """Honest kill-condition adjudication. The decisive test is SD vs S1
    (the PR #5 standard): if explicit regime classification does not beat pure
    trend on BOTH Sharpe and Calmar net of cost, S1 already implicitly follows
    the dominant factor through price and the classifier is redundant.

    ``mean_prob`` is the mean regime probability over the traded window — used
    only to describe the actual smooth blend in the mechanism note (a hard
    switch claim would not match the (1-prob)/prob implementation)."""
    s0 = full.loc["S0_buyhold"]
    s1 = full.loc["S1_blend"]
    sd = full.loc["SD_blend"]
    mean_p = mean_prob

    for name, row in (("S0", s0), ("S1_blend", s1), ("SD_blend", sd)):
        if not (pd.notna(row["sharpe"]) and pd.notna(row["calmar"])):
            return ("## Verdict\n\n**Insufficient sample / invalid metrics "
                    f"({name} has NaN Sharpe or Calmar) — cannot adjudicate. "
                    "Widen the sample window.**")

    def better(a, b):  # risk-adjusted = Sharpe AND Calmar
        return (a["sharpe"] > b["sharpe"]) and (a["calmar"] > b["calmar"])

    sd_beats_s1 = better(sd, s1)
    sd_beats_s0 = better(sd, s0)
    s1_beats_s0 = better(s1, s0)

    lines = ["## Verdict (honest kill conditions, blend variant, net of cost)\n"]
    lines.append(f"- S1_blend vs S0: Sharpe {s1['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {s1['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'S1 beats buy-and-hold (PR#5 replicates)' if s1_beats_s0 else 'S1 does NOT beat buy-and-hold here'}")
    lines.append(f"- **SD_blend vs S1_blend** (the decisive test): Sharpe "
                 f"{sd['sharpe']:.2f} vs {s1['sharpe']:.2f}, Calmar {sd['calmar']:.2f} "
                 f"vs {s1['calmar']:.2f} → "
                 f"{'regime classification ADDS value over pure trend' if sd_beats_s1 else 'regime classification adds NOTHING over pure trend'}")
    lines.append(f"- SD_blend vs S0: Sharpe {sd['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {sd['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'SD beats buy-and-hold' if sd_beats_s0 else 'SD does NOT beat buy-and-hold'}")
    lines.append("")
    if sd_beats_s1:
        lines.append("**① 'JUDGE THE DOMINANT FACTOR' HAS EDGE: SD beats S1 pure trend "
                     "on a risk-adjusted, cost-net basis — explicitly detecting the "
                     "dominant factor leads price or sizes better than trend alone.**")
    else:
        lines.append("**② KILLED: explicitly judging the dominant factor adds nothing "
                     "over S1 pure trend. S1 already implicitly follows whatever factor "
                     "is in charge (price trend tracks the dominant driver), so the "
                     "explicit classifier is redundant complexity. Sit on S1.**")
        lines.append("")
        # the drawdown clause is generated from the actual figures, never asserted:
        # SD only "gives back more in drawdowns" if its max_dd is genuinely deeper.
        if sd["max_dd"] < s1["max_dd"]:
            dd_clause = (f"and gives back more in drawdowns (SD max_dd {sd['max_dd']:.1%} "
                         f"vs S1 {s1['max_dd']:.1%})")
        else:
            dd_clause = (f"with a comparable/shallower drawdown (SD max_dd {sd['max_dd']:.1%} "
                         f"vs S1 {s1['max_dd']:.1%}), so the shortfall is on risk-adjusted "
                         f"return, not tail risk")
        # branch the mechanism on the actual blend weight so the note never
        # contradicts the figures: at mean_prob≈1 SD is just S1 (no real-rate
        # anchor to blame); only below ~1 does the real-rate leg drag.
        if pd.notna(mean_p) and mean_p >= 0.999:
            lines.append("_Mechanism: the mean regime probability is ~1.0, so SD is "
                         "EFFECTIVELY pure trend — it duplicates S1 rather than lagging "
                         "it, and the strict `>` kill simply records no INCREMENTAL edge. "
                         "The classifier can correctly flip to de-dollarization post-2022 "
                         "(see the timeline) yet still not add anything, precisely because "
                         "'follow price when de-dollarization dominates' is what S1 already "
                         "does everywhere._")
        else:
            lines.append("_Mechanism: SD blends the real-rate signal (weight "
                         f"{1 - mean_p:.0%}) and the trend signal (weight {mean_p:.0%}) by "
                         "the regime probability — a SMOOTH handoff, not a hard switch. "
                         "Because the blend is never fully trend (mean prob < 1), SD stays "
                         "partly anchored to the real-rate 'not rising' signal, which lags "
                         f"pure price trend {dd_clause}. SD does not beat S1 on both "
                         "decisive metrics (Sharpe AND Calmar). The classifier can correctly "
                         "flip to de-dollarization post-2022 (see the timeline) yet still "
                         "not help, precisely because 'follow price when de-dollarization "
                         "dominates' is what S1 already does everywhere._")
    return "\n".join(lines)


def _nonneg_float(x: str) -> float:
    v = float(x)
    # `nan < 0` and `inf < 0` are both False, so a bare `v < 0` would silently
    # accept "nan"/"inf" as a cost — reject any non-finite value explicitly.
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
    ap.add_argument("--corr-window", type=_pos_int, default=DEFAULT_CORR_WINDOW,
                    help="months for the rolling gold-realrate relation (default 36)")
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    print("Building timing panel (gold + real rate + USD + T-bill)...")
    tp = build_timing_panel(start=args.start, end=args.end)
    panel = tp.data
    print(f"  panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}, "
          f"{len(panel)} months")

    # ── regime classifier (cb_demand=None: gold-realrate relation is the sole
    #    fingerprint — the documented full-sample fallback) ──
    prob = dominance_probability(
        panel["gold_nominal"], panel["real_rate_10y"], window=args.corr_window
    )
    label = regime_label(prob)
    corr = rolling_gold_realrate_corr(
        panel["gold_nominal"], panel["real_rate_10y"], window=args.corr_window
    )
    # expose the other two sub-signals for the auditable regime CSV
    div_share = divergence_share(
        panel["gold_nominal"], panel["real_rate_10y"], window=args.corr_window
    )
    lvl_div = level_divergence(
        panel["gold_nominal"], panel["real_rate_10y"], window=args.corr_window
    )

    backtests = run_all(panel, prob, args.cost_bps)
    cstart, cend = common_span(backtests)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for warm-up). Widen --start/--end.", file=sys.stderr)
        raise SystemExit(2)
    # full-sample metrics on the common investable window. metrics_table() also
    # derives this window internally from common_span(), so passing cstart/cend
    # is the same result made explicit (and self-documents the contract that
    # S0/S1/SD are scored on identical months — verified by equal n_months).
    full = metrics_table(backtests, cstart, cend)

    # ── corr-window sensitivity band {24, 36, 48} (anti-overfit) ──
    # A longer corr window pushes SD's warm-up later, so each window has a
    # different common investable span. To isolate the *parameter* effect from
    # a moving sample start, score every window on ONE shared window =
    # intersection of all per-window common spans (verified by equal n_months).
    sens_bts: Dict[int, Dict[str, pd.DataFrame]] = {}
    sens_spans: Dict[int, tuple] = {}
    sens_skipped: List[str] = []
    for w in (24, 36, 48):
        prob_w = dominance_probability(
            panel["gold_nominal"], panel["real_rate_10y"], window=w
        )
        bt_w = run_all(panel, prob_w, args.cost_bps)
        cs_w, ce_w = common_span(bt_w)
        if cs_w is None or ce_w is None:
            sens_skipped.append(f"{w}m: no common investable span (warm-up eats the sample)")
            continue
        sens_bts[w] = bt_w
        sens_spans[w] = (cs_w, ce_w)
    sens_windows: Dict[int, pd.DataFrame] = {}
    if sens_spans:
        # ONE shared window = intersection of every per-window span, so a longer
        # window's later warm-up does not mix a moving sample start into the band.
        sens_start = max(s for s, _ in sens_spans.values())
        sens_end = min(e for _, e in sens_spans.values())
        if sens_start > sens_end:
            # the per-window spans don't share a common month (short sample /
            # large --corr-window) — there is NO shared investable window, so the
            # whole band is skipped rather than emitting empty/all-NaN tables.
            sens_skipped.append(f"no shared investable span across corr windows "
                                f"(latest start {sens_start:%Y-%m} after earliest "
                                f"end {sens_end:%Y-%m})")
        else:
            for w, bt_w in sens_bts.items():
                lo, hi = sens_spans[w]
                # a window whose own span doesn't cover [sens_start, sens_end] can't
                # be scored on the shared window — flag it skipped rather than emit a
                # misleading all-NaN table (matters under a short sample / large window).
                if lo > sens_start or hi < sens_end:
                    sens_skipped.append(f"{w}m: span {lo:%Y-%m}–{hi:%Y-%m} doesn't "
                                        f"cover the shared band {sens_start:%Y-%m}–{sens_end:%Y-%m}")
                    continue
                m = metrics_table(bt_w, sens_start, sens_end).loc[["S1_blend", "SD_blend"]]
                sens_windows[w] = m[["sharpe", "calmar", "cagr", "max_dd", "n_months"]]

    # ── cost sensitivity {0, 10, 20} bps ──
    sens_cost: Dict[float, pd.DataFrame] = {}
    for c in (0.0, 10.0, 20.0):
        bt_c = run_all(panel, prob, c)
        m = metrics_table(bt_c).loc[["S0_buyhold", "S1_blend", "SD_blend"]]
        sens_cost[c] = m[["sharpe", "calmar", "max_dd", "cagr"]]

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")
    report_path = os.path.join(args.out_dir, f"gold_regime_dominance_{file_stamp}.md")

    parts: List[str] = []
    parts.append(f"# Gold regime-dominance backtest — {stamp}\n")
    parts.append(
        "**Question.** Can explicitly *classifying* gold's dominant factor "
        "(real-rate regime ↔ de-dollarization regime) and trading the dominant "
        "factor beat PR #5's **S1 pure-trend blend** (already shown to beat "
        "buy-and-hold)? If not, S1's price trend already implicitly follows "
        "whatever factor is in charge and the explicit classifier is redundant.\n"
    )
    parts.append(f"Panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m} "
                 f"({len(panel)} months). Trading cost: {args.cost_bps:g} bps/rebalance. "
                 f"Long-only 0↔100%, cash leg = 3m T-bill. corr window = {args.corr_window}m.\n")
    parts.append(f"All metrics on the **common investable window** {cstart:%Y-%m}–"
                 f"{cend:%Y-%m} (after warm-up) so S0/S1/SD share the same months.\n")
    parts.append("Strategies: **S0** buy-and-hold · **S1** pure trend (3/6/12 blend, "
                 "vol-targeted, the PR #5 standard) · **SD** regime-dominance "
                 "conditional (real-rate-dominant→real-rate signal; "
                 "de-dollarization→trend; smooth probability blend).\n")

    parts.append("## Full sample (net of cost)\n")
    parts.append(_fmt(full))
    parts.append("")
    # mean regime probability over the common investable window — describes the
    # actual smooth blend in the mechanism note. shift(1) because month-m's HELD
    # position was decided at m-1 (run_backtest applies .shift(1)), so the blend
    # in force over the traded window is prob.shift(1), not the un-shifted prob.
    mean_prob = float(prob.shift(1).loc[cstart:cend].dropna().mean())
    parts.append(verdict(full, mean_prob))
    parts.append("")

    # ── regime timeline ──
    parts.append("## Regime timeline (de-dollarization share per year)\n")
    parts.append("Share of months each year the classifier calls "
                 "**de-dollarization-dominant** (1.0 = wholly de-dollarization, "
                 "0.0 = wholly real-rate). The hypothesis predicts a flip toward "
                 "1.0 around 2022.\n")
    tl = regime_timeline(label, freq="YE")
    tl_show = tl.copy()
    tl_show.index = [d.year for d in tl_show.index]
    tl_show["de-dollarization_share"] = tl_show["de-dollarization_share"].map(
        lambda v: f"{v:.2f}")
    tl_show["n_months"] = tl_show["n_months"].map(lambda v: f"{int(v):d}")
    parts.append(_md_table(tl_show))
    parts.append("")
    # explicit 2019→2026 monthly close-up so the 2022 handover is auditable
    parts.append("### 2019–2026 monthly close-up (prob, label, trailing corr)\n")
    recent = pd.DataFrame({
        "de_doll_prob": prob, "label": label, "gold_realrate_corr": corr,
    })
    recent = recent[recent.index >= pd.Timestamp("2019-01-01")]
    rshow = recent.copy()
    rshow.index = [f"{d:%Y-%m}" for d in rshow.index]
    rshow["de_doll_prob"] = rshow["de_doll_prob"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    rshow["label"] = rshow["label"].map(
        lambda v: ("de-doll" if v == 1.0 else "real-rate") if pd.notna(v) else "n/a")
    rshow["gold_realrate_corr"] = rshow["gold_realrate_corr"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    parts.append(_md_table(rshow))
    parts.append("")

    # ── sub-segments ──
    parts.append("## Sub-segments (net of cost)\n")
    pmin, pmax = panel.index.min(), panel.index.max()
    min_seg_months = 12
    for name, s, e in DEFAULT_SEGMENTS:
        # score window = segment ∩ common investable window (cstart..cend). Score
        # on THIS window and show THIS window in the title, so the dates a reader
        # sees match the months actually scored (metrics_table otherwise trims
        # silently to the common window, making e.g. 1968-2000 misleading).
        score_lo = max(cstart, pd.Timestamp(s))
        score_hi = min(cend, pd.Timestamp(e))
        if pd.Timestamp(s) > pmax or pd.Timestamp(e) < pmin:
            parts.append(f"### {name}\n_(skipped: no overlap with sample)_\n")
            continue
        m = metrics_table(backtests, score_lo, score_hi)
        # gate on *investable* months (after the common-window warm-up), not the
        # raw panel count: a long --corr-window or a tight --start/--end can leave
        # a segment with enough panel months but too few tradeable ones, which
        # would otherwise print a near-empty / all-NaN table (and int(NaN) crash).
        n_tradeable = segment_investable_months(m)
        if score_lo > score_hi or n_tradeable < min_seg_months:
            parts.append(f"### {name}\n_(skipped: only {n_tradeable} investable "
                         f"months in {score_lo:%Y-%m}–{score_hi:%Y-%m}, <{min_seg_months})_\n")
            continue
        parts.append(f"### {name} ({score_lo:%Y-%m}–{score_hi:%Y-%m})\n")
        parts.append(_fmt(m))
        parts.append("")

    # ── sensitivity bands ──
    parts.append("## corr-window sensitivity (S1 vs SD, blend)\n")
    parts.append("If SD's verdict vs S1 flips across {24,36,48}m the edge is a "
                 "window artifact, not structure. All rendered windows are scored "
                 "on ONE shared investable window so the comparison isolates the "
                 "corr-window parameter, not a moving sample start.\n")
    for w, m in sens_windows.items():
        parts.append(f"### corr window = {w}m\n")
        parts.append(_fmt_simple(m))
        parts.append("")
    for note in sens_skipped:
        parts.append(f"- _(skipped: {note})_")

    parts.append("## Cost sensitivity (S0 / S1_blend / SD_blend)\n")
    for c, m in sens_cost.items():
        parts.append(f"### {c:.0f} bps\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Data & method notes\n")
    parts.append("- **De-dollarization fingerprint (3-way OR, combined by max)**: "
                 "(a) rolling Δ(gold, real-rate) correlation → 0 / positive, "
                 "(b) per-month divergence share (Δgold>0 ∧ Δreal_rate>0), and "
                 "(c) **trailing-window level divergence** (gold higher AND real "
                 "rate higher over the window) — (c) is the signal that actually "
                 "fires in the post-2022 break and can on its own drive the "
                 "probability to 1. An optional central-bank-buying proxy is a "
                 "co-equal 4th disjunct when supplied.\n")
    parts.append("- **Central-bank gold demand**: WGC quarterly net-purchase data "
                 "is 2010+ and lagged, with no clean FRED series; `cb_demand` is an "
                 "optional injectable proxy (publication-lagged by `cb_lag_months`). "
                 "This run uses **cb_demand=None**, so the three gold–real-rate "
                 "fingerprints above are the **sole signals** across the full sample "
                 "— the documented fallback.\n")
    parts.append("- **No look-ahead**: every classifier value at t uses a trailing "
                 "rolling window / forward shift only; the position decided at t is "
                 "held t+1 via the shared engine's `.shift(1)`. Standard parameter "
                 "values (corr window 36m, real-rate signal 12m, thresholds at "
                 "conventional levels) — not tuned; the sensitivity band is reported.\n")
    parts.append("## Provenance\n")
    for k, v in tp.notes.items():
        parts.append(f"- **{k}**: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  report → {report_path}")

    # ── write equity-curve CSV ──
    os.makedirs(DATA_DIR, exist_ok=True)
    # clip every curve to the common investable window [cstart, cend] before
    # rebasing to $1, so S0 (invested from month 1) doesn't compound over extra
    # early months that S1/SD spend in warm-up — the CSV curves are then
    # same-track comparable, matching the metrics table.
    curves = {label_: equity_curve(bt.loc[cstart:cend]) for label_, bt in backtests.items()}
    eq = pd.DataFrame(curves)
    eq_path = os.path.join(DATA_DIR, f"gold_regime_dominance_curves_{file_stamp}.csv")
    eq.to_csv(eq_path)
    print(f"  equity curves → {eq_path}")

    # ── write regime CSV ──
    reg = pd.DataFrame({
        "de_dollarization_prob": prob, "regime_label": label,
        "gold_realrate_corr": corr,
        "divergence_share": div_share, "level_divergence": lvl_div,
    })
    reg_path = os.path.join(DATA_DIR, f"gold_regime_dominance_regime_{file_stamp}.csv")
    reg.to_csv(reg_path)
    print(f"  regime series → {reg_path}")

    print("\n" + _fmt(full))
    print("\n" + verdict(full, mean_prob))


if __name__ == "__main__":
    main()
