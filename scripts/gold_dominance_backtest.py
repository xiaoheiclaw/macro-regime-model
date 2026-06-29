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
    dominance_probability,
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


def verdict(full: pd.DataFrame) -> str:
    """Honest kill-condition adjudication. The decisive test is SD vs S1
    (the PR #5 standard): if explicit regime classification does not beat pure
    trend on BOTH Sharpe and Calmar net of cost, S1 already implicitly follows
    the dominant factor through price and the classifier is redundant."""
    s0 = full.loc["S0_buyhold"]
    s1 = full.loc["S1_blend"]
    sd = full.loc["SD_blend"]

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
        lines.append("_Mechanism: in de-dollarization months SD switches to the trend "
                     "signal, so it can at best tie S1 there; in real-rate-dominant "
                     f"months SD trades the real-rate 'not rising' signal, which lags trend "
                     f"{dd_clause}. Net SD ≤ S1. The classifier can correctly flip to "
                     "de-dollarization post-2022 (see the timeline) yet still not help, "
                     "precisely because 'follow price when de-dollarization dominates' is "
                     "what S1 already does everywhere._")
    return "\n".join(lines)


def _nonneg_float(x: str) -> float:
    v = float(x)
    if v < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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
    sens_windows: Dict[int, pd.DataFrame] = {}
    for w in (24, 36, 48):
        prob_w = dominance_probability(
            panel["gold_nominal"], panel["real_rate_10y"], window=w
        )
        bt_w = run_all(panel, prob_w, args.cost_bps)
        m = metrics_table(bt_w).loc[["S1_blend", "SD_blend"]]
        sens_windows[w] = m[["sharpe", "calmar", "cagr", "max_dd"]]

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
                 f"({len(panel)} months). Trading cost: {args.cost_bps:.0f} bps/rebalance. "
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
    parts.append(verdict(full))
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
        lo, hi = max(pd.Timestamp(s), pmin), min(pd.Timestamp(e), pmax)
        if lo > hi:
            parts.append(f"### {name}\n_(skipped: no overlap with sample)_\n")
            continue
        n_in = int(((panel.index >= lo) & (panel.index <= hi)).sum())
        if n_in < min_seg_months:
            parts.append(f"### {name}\n_(skipped: only {n_in} months, <{min_seg_months})_\n")
            continue
        parts.append(f"### {name} ({lo:%Y-%m}–{hi:%Y-%m})\n")
        parts.append(_fmt(metrics_table(backtests, s, e)))
        parts.append("")

    # ── sensitivity bands ──
    parts.append("## corr-window sensitivity (S1 vs SD, blend)\n")
    parts.append("If SD's verdict vs S1 flips across {24,36,48}m the edge is a "
                 "window artifact, not structure.\n")
    for w, m in sens_windows.items():
        parts.append(f"### corr window = {w}m\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Cost sensitivity (S0 / S1_blend / SD_blend)\n")
    for c, m in sens_cost.items():
        parts.append(f"### {c:.0f} bps\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Data & method notes\n")
    parts.append("- **Central-bank gold demand**: WGC quarterly net-purchase data "
                 "is 2010+ and lagged, with no clean FRED series; `cb_demand` is an "
                 "optional injectable proxy. This run uses **cb_demand=None**, so the "
                 "gold–real-rate relationship (rolling corr + divergence share) is the "
                 "**sole fingerprint** across the full sample — the documented fallback.\n")
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
    curves = {label_: equity_curve(bt) for label_, bt in backtests.items()}
    eq = pd.DataFrame(curves)
    eq_path = os.path.join(DATA_DIR, f"gold_regime_dominance_curves_{file_stamp}.csv")
    eq.to_csv(eq_path)
    print(f"  equity curves → {eq_path}")

    # ── write regime CSV ──
    reg = pd.DataFrame({
        "de_dollarization_prob": prob, "regime_label": label,
        "gold_realrate_corr": corr,
    })
    reg_path = os.path.join(DATA_DIR, f"gold_regime_dominance_regime_{file_stamp}.csv")
    reg.to_csv(reg_path)
    print(f"  regime series → {reg_path}")

    print("\n" + _fmt(full))
    print("\n" + verdict(full))


if __name__ == "__main__":
    main()
