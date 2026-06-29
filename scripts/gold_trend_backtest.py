"""Gold long-only trend-timing backtest runner.

Builds the monthly panel (reusing `build_anchor_panel` for gold + real rate,
adding a trade-weighted USD and a T-bill cash leg), runs three strategy
families — S0 buy-and-hold, S1 pure trend (lookbacks {3,6,12,blend}), S2 trend +
regime gate — scores them net of trading cost across the full sample and
sub-segments, writes an equity-curve CSV plus a markdown report with an honest
kill-condition verdict.

Usage:
    uv run python scripts/gold_trend_backtest.py
    uv run python scripts/gold_trend_backtest.py --cost-bps 10 --start 1968-01-01
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
    compute_metrics,
    equity_curve,
    run_backtest,
    s0_buy_hold,
    s1_trend,
    s2_trend_regime,
    slice_segment,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402

# Strategy spec: label → (kind, lookbacks). "blend" uses all DEFAULT_LOOKBACKS.
TREND_VARIANTS: List[tuple[str, tuple[int, ...]]] = [
    ("L3", (3,)),
    ("L6", (6,)),
    ("L12", (12,)),
    ("blend", tuple(DEFAULT_LOOKBACKS)),
]

METRIC_COLS = ["sharpe", "calmar", "cagr", "ann_vol", "max_dd", "hit_rate",
               "ann_turnover", "n_months"]


def build_positions(panel: pd.DataFrame) -> Dict[str, pd.Series]:
    """All strategy position series keyed by label."""
    pos: Dict[str, pd.Series] = {"S0_buyhold": s0_buy_hold(panel.index)}
    for label, lbs in TREND_VARIANTS:
        pos[f"S1_{label}"] = s1_trend(panel, lookbacks=lbs)
        pos[f"S2_{label}"] = s2_trend_regime(panel, lookbacks=lbs)
    return pos


def run_all(panel: pd.DataFrame, cost_bps: float) -> Dict[str, pd.DataFrame]:
    positions = build_positions(panel)
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


def common_span(backtests: Dict[str, pd.DataFrame]):
    """The window over which *every* strategy is investable — the latest start
    and earliest end across all backtests. S1/S2 start after their trend/vol
    warm-up while S0 is invested from month 1, so a fair head-to-head must
    evaluate them all on this shared window (else S0 gets free extra months).

    Returns ``(None, None)`` if there is no shared window: any empty strategy
    (the 'every strategy investable' contract fails) or a start that lands after
    the earliest end. Callers treat that as unadjudicable rather than comparing
    on a window some strategy is absent from."""
    spans = []
    for bt in backtests.values():
        if len(bt) == 0:
            return None, None
        spans.append((bt.index.min(), bt.index.max()))
    if not spans:
        return None, None
    cstart, cend = max(s for s, _ in spans), min(e for _, e in spans)
    if cstart > cend:
        return None, None
    return cstart, cend


def metrics_table(backtests: Dict[str, pd.DataFrame], start=None, end=None) -> pd.DataFrame:
    cstart, cend = common_span(backtests)
    if cstart is None or cend is None:
        # No shared investable window → all-NaN table; the verdict guard / caller
        # turns this into an explicit "cannot adjudicate" rather than a crash.
        empty = {label: compute_metrics(bt.iloc[0:0]) for label, bt in backtests.items()}
        return pd.DataFrame(empty).T[METRIC_COLS]
    lo = cstart if start is None else max(cstart, pd.Timestamp(start))
    hi = cend if end is None else min(cend, pd.Timestamp(end))
    rows = {label: compute_metrics(slice_segment(bt, lo, hi)) for label, bt in backtests.items()}
    return pd.DataFrame(rows).T[METRIC_COLS]


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate
    dependency). The index becomes the first column ('strategy')."""
    cols = list(df.columns)
    header = "| strategy | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for idx, row in df.iterrows():
        cells = " | ".join(str(row[c]) for c in cols)
        rows.append(f"| {idx} | {cells} |")
    return "\n".join(rows)


def _fmt(df: pd.DataFrame) -> str:
    show = df.copy()
    for c in ["cagr", "ann_vol", "max_dd", "hit_rate"]:
        show[c] = (show[c] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    for c in ["sharpe", "calmar"]:
        show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    # ann_turnover is an annualised *turnover multiple* (not a percentage)
    show["ann_turnover"] = show["ann_turnover"].map(
        lambda v: f"{v:.2f}x" if pd.notna(v) else "n/a")
    show["n_months"] = show["n_months"].map(lambda v: f"{int(v):d}" if pd.notna(v) else "0")
    return _md_table(show)


def verdict(full: pd.DataFrame) -> str:
    """Honest kill-condition adjudication using the *blend* variants (no
    cherry-picking the best single lookback)."""
    s0 = full.loc["S0_buyhold"]
    s1 = full.loc["S1_blend"]
    s2 = full.loc["S2_blend"]

    # Guard: short/constant/empty samples produce NaN Sharpe/Calmar; a NaN
    # comparison is silently False and would masquerade as a "killed" verdict.
    for name, row in (("S0", s0), ("S1_blend", s1), ("S2_blend", s2)):
        if not (pd.notna(row["sharpe"]) and pd.notna(row["calmar"])):
            return ("## Verdict\n\n**Insufficient sample / invalid metrics "
                    f"({name} has NaN Sharpe or Calmar) — cannot adjudicate the "
                    "kill conditions. Widen the sample window.**")

    def better(a, b):  # risk-adjusted = Sharpe AND Calmar
        return (a["sharpe"] > b["sharpe"]) and (a["calmar"] > b["calmar"])

    lines = ["## Verdict (honest kill conditions, blend variant, net of cost)\n"]
    s1_beats_s0 = better(s1, s0)
    s2_beats_s1 = better(s2, s1)
    s2_beats_s0 = better(s2, s0)

    lines.append(f"- S1_blend vs S0: Sharpe {s1['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {s1['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'S1 wins' if s1_beats_s0 else 'S1 does NOT beat buy-and-hold'}")
    lines.append(f"- S2_blend vs S1_blend: Sharpe {s2['sharpe']:.2f} vs {s1['sharpe']:.2f}, "
                 f"Calmar {s2['calmar']:.2f} vs {s1['calmar']:.2f} → "
                 f"{'regime gate adds value' if s2_beats_s1 else 'regime gate adds NOTHING over trend'}")
    lines.append(f"- S2_blend vs S0: Sharpe {s2['sharpe']:.2f} vs {s0['sharpe']:.2f}, "
                 f"Calmar {s2['calmar']:.2f} vs {s0['calmar']:.2f} → "
                 f"{'S2 beats buy-and-hold' if s2_beats_s0 else 'S2 does NOT beat buy-and-hold'}")

    lines.append("")
    if not s1_beats_s0:
        lines.append("**③ KILLED: timing as a whole is useless — pure trend cannot beat "
                     "holding gold. Sit on the metal.**")
    elif not s2_beats_s1:
        lines.append("**② regime gate KILLED: it does not beat pure trend. Trend alone "
                     "suffices; the macro gate is redundant.**")
    elif s2_beats_s1 and s2_beats_s0:
        lines.append("**① regime gate has edge: S2 beats BOTH S0 and S1 on a "
                     "risk-adjusted, cost-net basis.**")
    else:
        lines.append("**Ambiguous: S2 beats S1 but not S0 (or vice versa) — gate helps "
                     "vs trend but the combo still lags buy-and-hold. Inconclusive edge.**")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    print("Building timing panel (gold + real rate + USD + T-bill)...")
    tp = build_timing_panel(start=args.start, end=args.end)
    panel = tp.data
    print(f"  panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}, "
          f"{len(panel)} months")

    # Headline run at the default cost.
    backtests = run_all(panel, args.cost_bps)
    cstart, cend = common_span(backtests)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for the trend/vol warm-up). Widen --start/--end.",
              file=sys.stderr)
        raise SystemExit(2)
    full = metrics_table(backtests)

    # Cost sensitivity (0 / 20 bps) for the blend variants + S0.
    sens = {}
    for c in (0.0, 20.0):
        bt_c = run_all(panel, c)
        m = metrics_table(bt_c).loc[["S0_buyhold", "S1_blend", "S2_blend"]]
        sens[c] = m[["sharpe", "calmar", "max_dd", "cagr"]]

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    # filename carries the time so repeated same-day runs don't silently overwrite
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S")
    report_path = os.path.join(args.out_dir, f"gold_trend_timing_{file_stamp}.md")

    parts: List[str] = []
    parts.append(f"# Gold long-only trend-timing backtest — {stamp}\n")
    parts.append(f"Panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m} "
                 f"({len(panel)} months). Trading cost: {args.cost_bps:.0f} bps/rebalance. "
                 "Long-only 0↔100%, cash leg = 3m T-bill.\n")
    parts.append(f"All metrics are computed on the **common investable window** "
                 f"{cstart:%Y-%m}–{cend:%Y-%m} (after S1/S2 trend+vol warm-up) so "
                 "S0/S1/S2 are compared on the same months.\n")
    parts.append("Strategies: **S0** buy-and-hold · **S1** pure trend (vol-targeted, "
                 "lookbacks {3,6,12,blend}) · **S2** trend + regime gate "
                 "(real-rate-not-rising ∧ dollar-not-strengthening fast exit).\n")

    parts.append("## Full sample (net of cost)\n")
    parts.append(_fmt(full))
    parts.append("")

    parts.append(verdict(full))
    parts.append("")

    parts.append("## Sub-segments (net of cost)\n")
    pmin, pmax = panel.index.min(), panel.index.max()
    min_seg_months = 12
    for name, s, e in DEFAULT_SEGMENTS:
        lo, hi = max(pd.Timestamp(s), pmin), min(pd.Timestamp(e), pmax)
        if lo > hi:
            parts.append(f"### {name}\n_(skipped: no overlap with sample "
                         f"{pmin:%Y-%m}–{pmax:%Y-%m})_\n")
            continue
        n_in = int(((panel.index >= lo) & (panel.index <= hi)).sum())
        if n_in < min_seg_months:
            parts.append(f"### {name}\n_(skipped: only {n_in} months in sample, "
                         f"<{min_seg_months})_\n")
            continue
        parts.append(f"### {name} ({lo:%Y-%m}–{hi:%Y-%m})\n")
        parts.append(_fmt(metrics_table(backtests, s, e)))
        parts.append("")

    parts.append("## Cost sensitivity (S0 / S1_blend / S2_blend)\n")
    for c, m in sens.items():
        parts.append(f"### {c:.0f} bps\n")
        parts.append(_fmt_simple(m))
        parts.append("")

    parts.append("## Attribution note\n")
    parts.append("Compare the **1968-2000** segment (contains the 1980–2000 gold bear) "
                 "against later segments: if S1/S2's edge concentrates there and "
                 "evaporates in 2001-2011 (the bull), the edge is *avoiding the bear*, "
                 "not generic alpha. See per-segment max_dd and CAGR above.\n")
    parts.append("## Provenance\n")
    for k, v in tp.notes.items():
        parts.append(f"- **{k}**: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  report → {report_path}")

    # ── write equity-curve CSV ──
    os.makedirs(DATA_DIR, exist_ok=True)
    curves = {label: equity_curve(bt) for label, bt in backtests.items()}
    eq = pd.DataFrame(curves)
    csv_path = os.path.join(DATA_DIR, f"gold_trend_timing_curves_{file_stamp}.csv")
    eq.to_csv(csv_path)
    print(f"  equity curves → {csv_path}")

    print("\n" + _fmt(full))
    print("\n" + verdict(full))


def _fmt_simple(df: pd.DataFrame) -> str:
    show = df.copy()
    for c in ["cagr", "max_dd"]:
        if c in show:
            show[c] = (show[c] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    for c in ["sharpe", "calmar"]:
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    return _md_table(show)


if __name__ == "__main__":
    main()
