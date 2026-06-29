"""S1 trend-timing post-2000 sub-period verdict runner.

Reuses PR#5's leak-free panel + backtest engine (`lib.gold_trend_timing`) and
the sub-period analytics in `lib.gold_s1_subperiod` to answer one question:

    Does S1 (vol-targeted pure trend) still beat buy-and-hold gold (S0) AFTER
    2000 — net of cost — or is its full-sample Sharpe just historical
    bear-avoidance that has decayed for today's trader?

Writes a markdown report (per-segment S0-vs-S1 tables incl. lived-experience
drawdown caliber, cost-sensitivity sweep, parameter-robustness panel, paired
post-2000 significance, and an honest verdict) plus a tidy metrics CSV.

Usage:
    uv run python scripts/gold_s1_subperiod.py
    uv run python scripts/gold_s1_subperiod.py --start 1968-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_s1_subperiod import (  # noqa: E402
    COST_GRID,
    PAIRED_DISPLAY_COST,
    POST2000_SEGMENT,
    PRIMARY_LABEL,
    S0_LABEL,
    S1_VARIANTS,
    SUBPERIOD_SEGMENTS,
    build_positions,
    common_window,
    paired_net_diff_stats,
    segment_metrics,
    segment_window,
    verdict,
)
from lib.gold_trend_timing import (  # noqa: E402
    DEFAULT_COST_BPS,
    build_timing_panel,
    run_backtest,
    slice_segment,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402


def run_all(panel: pd.DataFrame, cost_bps: float) -> Dict[str, pd.DataFrame]:
    """Backtest every strategy (S0 + S1 variants) at a single cost level."""
    positions = build_positions(panel)
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


# ── markdown helpers ───────────────────────────────────────────────────────
def _cell(x) -> str:
    """Escape a value for a GitHub-flavoured markdown table cell: a literal `|`
    would start a new column and a newline would break the row."""
    return str(x).replace("|", "\\|").replace("\n", "<br>")


def _md_table(df: pd.DataFrame, index_name: str = "strategy") -> str:
    cols = list(df.columns)
    header = f"| {_cell(index_name)} | " + " | ".join(_cell(c) for c in cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for idx, row in df.iterrows():
        rows.append(f"| {_cell(idx)} | " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def _fmt_segment(df: pd.DataFrame) -> str:
    """Format a segment_metrics frame for the report."""
    show = df.copy()
    for c in ("cagr", "max_dd", "hit_rate"):
        if c in show:
            show[c] = (show[c] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    for c in ("sharpe", "calmar"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    if "ann_turnover" in show:
        show["ann_turnover"] = show["ann_turnover"].map(
            lambda v: f"{v:.2f}x" if pd.notna(v) else "n/a")
    # Integer count columns: NaN means "no sample / not computable" → show
    # "n/a", never "0" (which would conflate an empty segment with a real zero).
    for c in ("longest_underwater_m", "max_consec_loss_m", "n_trades", "n_months"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{int(v):d}" if pd.notna(v) else "n/a")
    return _md_table(show)


def _excess_sharpe(seg_tbl: pd.DataFrame, variant_label: str) -> float:
    if variant_label not in seg_tbl.index or S0_LABEL not in seg_tbl.index:
        return float("nan")
    return seg_tbl.loc[variant_label, "sharpe"] - seg_tbl.loc[S0_LABEL, "sharpe"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    print("Building timing panel (gold + real rate + USD + T-bill)...")
    tp = build_timing_panel(start=args.start, end=args.end)
    panel = tp.data
    print(f"  panel: {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}, "
          f"{len(panel)} months")

    # Backtest the whole strategy set at every cost level. The headline cost is
    # the verdict/segment-display cost, so it MUST be one of the grid points —
    # guard the cross-module constant contract rather than KeyError downstream.
    if DEFAULT_COST_BPS not in COST_GRID:
        print(f"ERROR: DEFAULT_COST_BPS={DEFAULT_COST_BPS} (from lib.gold_trend_timing) "
              f"is not in COST_GRID={COST_GRID}; the headline/verdict cost must be a "
              "grid point. Add it to COST_GRID.", file=sys.stderr)
        raise SystemExit(2)
    bt_by_cost: Dict[float, Dict[str, pd.DataFrame]] = {
        c: run_all(panel, c) for c in COST_GRID
    }
    headline = bt_by_cost[DEFAULT_COST_BPS]
    cstart, cend = common_window(headline)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for the trend/vol warm-up).", file=sys.stderr)
        raise SystemExit(2)
    print(f"  fair common window @{DEFAULT_COST_BPS:.0f}bps: {cstart:%Y-%m} → {cend:%Y-%m}")

    # Per-segment metrics at every cost (fair common window applied inside).
    seg_by_cost: Dict[float, Dict[str, pd.DataFrame]] = {}
    for c, bts in bt_by_cost.items():
        common = common_window(bts)
        seg_by_cost[c] = {
            name: segment_metrics(bts, s, e, common=common)
            for name, s, e in SUBPERIOD_SEGMENTS
        }

    # Paired post-2000 significance (S1 primary vs S0) at every cost. Window
    # bounds come from SUBPERIOD_SEGMENTS via POST2000_SEGMENT — the SAME source
    # the segment tables and verdict use, so they can never silently desync.
    # Each cost re-derives its OWN common window (cost can in principle change
    # the investable rows), so paired stats stay aligned with that cost's table.
    post_start, post_end = segment_window(POST2000_SEGMENT)
    paired_by_cost: Dict[float, Dict[str, float]] = {}
    for c, bts in bt_by_cost.items():
        ccstart, ccend = common_window(bts)
        lo = None if ccstart is None else max(ccstart, pd.Timestamp(post_start))
        hi = None if ccend is None else min(ccend, pd.Timestamp(post_end))
        # explicit empty-window handling (don't rely on slice_segment's behaviour
        # for a reversed lo>hi range, which is fragile on a DatetimeIndex)
        if lo is None or hi is None or lo > hi:
            a = bts[PRIMARY_LABEL].iloc[0:0]
            b = bts[S0_LABEL].iloc[0:0]
        else:
            a = slice_segment(bts[PRIMARY_LABEL], lo, hi)
            b = slice_segment(bts[S0_LABEL], lo, hi)
        paired_by_cost[c] = paired_net_diff_stats(a, b)

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")
    report_path = os.path.join(args.out_dir, f"gold_s1_subperiod_{file_stamp}.md")

    parts: List[str] = []
    parts.append(f"# S1 trend-timing — post-2000 sub-period verdict — {stamp}\n")
    parts.append(
        f"Panel {panel.index.min():%Y-%m}→{panel.index.max():%Y-%m} ({len(panel)} months). "
        f"Fair common window (after S1 warm-up) **{cstart:%Y-%m}→{cend:%Y-%m}** — every "
        "metric below is computed on this shared month set so S0 and all S1 variants "
        "see identical months.\n")
    parts.append(
        "**Question.** S1 (vol-targeted pure trend, long-only 0↔100%) has a full-sample "
        "Sharpe ~0.63 widely attributed to dodging the 1968–2000 bear. Does it *still* "
        "beat buy-and-hold gold (S0) **after 2000**, net of cost — or has the edge "
        "decayed to historical bear-avoidance?\n")
    parts.append(
        "> **Caveat (read first).** Everything here is **in-sample and ex-post** — no "
        "walk-forward, no out-of-sample holdout, no parameter search discipline. A "
        "favourable verdict means *worth hardening with a walk-forward test*, not "
        "*proven out-of-sample*. The blend/vote windows are fixed a priori (PR#5), not "
        "tuned here.\n")

    # Per-segment tables at the headline cost.
    parts.append(f"## Per-segment metrics @ {DEFAULT_COST_BPS:.0f}bps (headline cost)\n")
    parts.append(
        "Drawdown caliber is reported three ways: `max_dd` (month-end snapshot, "
        "understates pain), `longest_underwater_m` (months below the prior peak — the "
        "wait-to-even), `max_consec_loss_m` (longest losing streak — whipsaw burn). "
        "`n_trades` counts discrete entries/exits that occur *inside* the window: "
        "buy-and-hold S0 shows 1 only in the window-opening segment (its single "
        "entry) and **0 in later segments** (the position is carried in, not "
        "re-traded); `ann_turnover` is the continuous vol-target rebalancing.\n")
    for name, s, e in SUBPERIOD_SEGMENTS:
        tag = " — **COMBINED POST-2000 (verdict window)**" if name == POST2000_SEGMENT else ""
        parts.append(f"### {name}{tag}\n")
        parts.append(_fmt_segment(seg_by_cost[DEFAULT_COST_BPS][name]))
        parts.append("")

    # Core post-2000 focus: S1 primary vs S0 across the post-2000 sub-windows.
    parts.append("## Core question — S1 (primary blend) vs S0, post-2000 excess\n")
    focus_segs = ["2000-2011", "2011-2015", "2016-2026", POST2000_SEGMENT]
    foc_rows = {}
    for name in focus_segs:
        tbl = seg_by_cost[DEFAULT_COST_BPS][name]
        if PRIMARY_LABEL in tbl.index and S0_LABEL in tbl.index:
            s1r, s0r = tbl.loc[PRIMARY_LABEL], tbl.loc[S0_LABEL]
            foc_rows[name] = {
                "S1_sharpe": round(s1r["sharpe"], 2),
                "S0_sharpe": round(s0r["sharpe"], 2),
                "Δsharpe": round(s1r["sharpe"] - s0r["sharpe"], 2),
                "S1_cagr%": round(s1r["cagr"] * 100, 1),
                "S0_cagr%": round(s0r["cagr"] * 100, 1),
                "Δcagr%": round((s1r["cagr"] - s0r["cagr"]) * 100, 1),
                "S1_maxdd%": round(s1r["max_dd"] * 100, 1),
                "S0_maxdd%": round(s0r["max_dd"] * 100, 1),
            }
    parts.append(_md_table(pd.DataFrame(foc_rows).T, index_name="segment"))
    parts.append("")
    parts.append("_Δ = S1 − S0. Positive Δsharpe/Δcagr = S1 ahead; less-negative maxdd = "
                 "S1 cushioned the give-back. The 2011-2015 and 2016-2026 rows are the "
                 "modern whipsaw / recent-decade acid test._\n")

    # Cost sensitivity: S0 vs S1 primary Sharpe per segment across the cost grid.
    parts.append("## Cost sensitivity — Sharpe by segment across the cost grid\n")
    parts.append("S1 primary (blend 3/6/12) Sharpe minus S0 Sharpe — the post-2000 "
                 "rows show whether S1's edge survives higher costs.\n")
    cost_rows = {}
    for name, s, e in SUBPERIOD_SEGMENTS:
        cost_rows[name] = {
            f"{int(c)}bps": round(_excess_sharpe(seg_by_cost[c][name], PRIMARY_LABEL), 2)
            for c in COST_GRID
        }
    parts.append(_md_table(pd.DataFrame(cost_rows).T, index_name="segment (ΔSharpe S1−S0)"))
    parts.append("")

    # Parameter robustness: all variants vs S0 on the post-2000 window @10bps.
    parts.append(f"## Parameter robustness — all S1 variants on {POST2000_SEGMENT} @ "
                 f"{DEFAULT_COST_BPS:.0f}bps\n")
    parts.append("If the post-2000 verdict only holds for the one 3/6/12 blend, it is "
                 "fragile. Single windows (L2/L6/L12) and the majority vote test that.\n")
    parts.append(_fmt_segment(seg_by_cost[DEFAULT_COST_BPS][POST2000_SEGMENT]))
    parts.append("")

    # Paired significance.
    parts.append("## Paired in-sample significance — S1 primary − S0, post-2000\n")
    _pj0 = paired_by_cost[PAIRED_DISPLAY_COST]
    parts.append(
        f"Method params shown at the {PAIRED_DISPLAY_COST:.0f}bps display cost. "
        "Monthly net-return difference treated as the time series it is (NOT IID): "
        f"**Newey-West HAC** t-stat (Bartlett kernel, lag≈n^⅓={_pj0.get('hac_lag', 0)}) and a "
        f"**moving-block bootstrap** 95% CI (block≈√n={_pj0.get('block_len', 0)}, 2000 paths, "
        "seed 0) so autocorrelation / volatility clustering do not understate the "
        "uncertainty. CI excluding zero ⇒ the gap is reliable on this sample (NOT an "
        "out-of-sample claim).\n")
    psig = {}
    for c in COST_GRID:
        pj = paired_by_cost[c]
        psig[f"{int(c)}bps"] = {
            "ann_diff%": round(pj["ann_mean"] * 100, 2) if pd.notna(pj["ann_mean"]) else "n/a",
            "t_stat": round(pj["t_stat"], 2) if pd.notna(pj["t_stat"]) else "n/a",
            "ci_lo%": round(pj["ci_lo"] * 100, 2) if pd.notna(pj["ci_lo"]) else "n/a",
            "ci_hi%": round(pj["ci_hi"] * 100, 2) if pd.notna(pj["ci_hi"]) else "n/a",
            "excl_zero": pj["ci_excludes_zero"],
        }
    parts.append(_md_table(pd.DataFrame(psig).T, index_name="cost"))
    parts.append("")

    parts.append(verdict(seg_by_cost, paired_by_cost))
    parts.append("")

    parts.append("## Provenance\n")
    for k, v in tp.notes.items():
        parts.append(f"- **{k}**: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  report → {report_path}")

    # ── write tidy metrics CSV (cost × segment × strategy × metrics) ──
    os.makedirs(DATA_DIR, exist_ok=True)
    tidy: List[pd.DataFrame] = []
    for c in COST_GRID:
        for name, s, e in SUBPERIOD_SEGMENTS:
            t = seg_by_cost[c][name].copy()
            t.insert(0, "segment", name)
            t.insert(0, "cost_bps", c)
            t.index.name = "strategy"
            tidy.append(t.reset_index())
    csv_path = os.path.join(DATA_DIR, f"gold_s1_subperiod_{file_stamp}.csv")
    pd.concat(tidy, ignore_index=True).to_csv(csv_path, index=False)
    print(f"  metrics CSV → {csv_path}")

    print("\n" + _fmt_segment(seg_by_cost[DEFAULT_COST_BPS][POST2000_SEGMENT]))
    print("\n" + verdict(seg_by_cost, paired_by_cost))


if __name__ == "__main__":
    main()
