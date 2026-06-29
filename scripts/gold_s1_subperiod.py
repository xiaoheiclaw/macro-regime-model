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
    METRIC_COLS,
    POST2000_SEGMENT,
    PRIMARY_VARIANT,
    S0_LABEL,
    S1_VARIANTS,
    SUBPERIOD_SEGMENTS,
    build_positions,
    common_window,
    paired_net_diff_stats,
    segment_metrics,
)
from lib.gold_trend_timing import (  # noqa: E402
    DEFAULT_COST_BPS,
    build_timing_panel,
    run_backtest,
    slice_segment,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402

PRIMARY_LABEL = f"S1_{PRIMARY_VARIANT}"


def run_all(panel: pd.DataFrame, cost_bps: float) -> Dict[str, pd.DataFrame]:
    """Backtest every strategy (S0 + S1 variants) at a single cost level."""
    positions = build_positions(panel)
    return {
        label: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost_bps)
        for label, p in positions.items()
    }


# ── markdown helpers ───────────────────────────────────────────────────────
def _md_table(df: pd.DataFrame, index_name: str = "strategy") -> str:
    cols = list(df.columns)
    header = f"| {index_name} | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for idx, row in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |")
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
    for c in ("longest_underwater_m", "max_consec_loss_m", "n_trades", "n_months"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{int(v):d}" if pd.notna(v) else "0")
    return _md_table(show)


def _excess_sharpe(seg_tbl: pd.DataFrame, variant_label: str) -> float:
    if variant_label not in seg_tbl.index or S0_LABEL not in seg_tbl.index:
        return float("nan")
    return seg_tbl.loc[variant_label, "sharpe"] - seg_tbl.loc[S0_LABEL, "sharpe"]


def verdict(
    seg_by_cost: Dict[float, Dict[str, pd.DataFrame]],
    paired_by_cost: Dict[float, Dict[str, float]],
) -> str:
    """Adjudicate the post-2000 kill condition on the PRIMARY (3/6/12 blend)
    variant — along BOTH axes, because they disagree and the honest answer is
    two-dimensional:

      • risk-adjusted (Sharpe AND Calmar, the same caliber PR#5 used): does S1
        beat S0 net of cost at the realistic 10bps and punitive 25bps?
      • raw return (CAGR + the paired monthly net-return excess and its CI): is
        the return excess positive and distinguishable from zero?

    A timing overlay can win the first while losing the second — that *is* the
    'stress insurance, not uniform alpha' pattern. Reporting only one axis would
    be the cherry-pick. Robustness (how many variants beat S0 on Sharpe
    post-2000 @10bps) is reported but does not flip the headline."""
    lines = ["## Verdict — is S1 still alive after 2000? (in-sample, ex-post)\n"]

    post = POST2000_SEGMENT

    def row(cost, label):
        return seg_by_cost[cost][post].loc[label]

    # Guard: need valid metrics on the post-2000 window.
    try:
        s1_10 = row(10.0, PRIMARY_LABEL)
        s0_10 = row(10.0, S0_LABEL)
        s1_25 = row(25.0, PRIMARY_LABEL)
        s0_25 = row(25.0, S0_LABEL)
    except KeyError:
        return "## Verdict\n\n**Cannot adjudicate: post-2000 window missing from results.**"
    if not (pd.notna(s1_10["sharpe"]) and pd.notna(s0_10["sharpe"])
            and pd.notna(s1_10["calmar"]) and pd.notna(s0_10["calmar"])):
        return ("## Verdict\n\n**Insufficient sample on the post-2000 window "
                "(NaN Sharpe/Calmar) — cannot adjudicate. Widen the data.**")

    def risk_adj_beats(s1, s0):  # PR#5 caliber: Sharpe AND Calmar
        return (s1["sharpe"] > s0["sharpe"]) and (s1["calmar"] > s0["calmar"])

    ra_10 = risk_adj_beats(s1_10, s0_10)
    ra_25 = risk_adj_beats(s1_25, s0_25)
    ret_10 = s1_10["cagr"] > s0_10["cagr"]   # raw-return win @10bps
    ret_25 = s1_25["cagr"] > s0_25["cagr"]

    pj = paired_by_cost[10.0]
    ci_positive = bool(pj["ci_excludes_zero"]) and pj["ann_mean"] > 0
    ci_negative = bool(pj["ci_excludes_zero"]) and pj["ann_mean"] < 0

    # robustness across variants @10bps (risk-adjusted)
    tbl10 = seg_by_cost[10.0][post]
    variant_labels = [f"S1_{lbl}" for lbl, _, _ in S1_VARIANTS]
    n_beat = sum(
        1 for v in variant_labels
        if v in tbl10.index and pd.notna(tbl10.loc[v, "sharpe"])
        and tbl10.loc[v, "sharpe"] > tbl10.loc[S0_LABEL, "sharpe"]
    )
    n_var = len(variant_labels)

    lines.append(f"Post-2000 window **{post}**, primary variant **{PRIMARY_VARIANT}** "
                 "(PR#5's 3/6/12 blend), net of cost. **Two axes, read both:**\n")
    lines.append(f"- Risk-adjusted @10bps: S1 Sharpe {s1_10['sharpe']:.2f} vs S0 "
                 f"{s0_10['sharpe']:.2f}, Calmar {s1_10['calmar']:.2f} vs {s0_10['calmar']:.2f}, "
                 f"MaxDD {s1_10['max_dd']*100:.1f}% vs {s0_10['max_dd']*100:.1f}% → "
                 f"{'S1 WINS' if ra_10 else 'S1 does not win'}")
    lines.append(f"- Risk-adjusted @25bps: S1 Sharpe {s1_25['sharpe']:.2f} vs S0 "
                 f"{s0_25['sharpe']:.2f}, Calmar {s1_25['calmar']:.2f} vs {s0_25['calmar']:.2f} → "
                 f"{'S1 WINS' if ra_25 else 'S1 does not win'}")
    lines.append(f"- Raw return @10bps: S1 CAGR {s1_10['cagr']*100:.1f}% vs S0 "
                 f"{s0_10['cagr']*100:.1f}% → "
                 f"{'S1 higher' if ret_10 else 'S1 GIVES UP return'}")
    lines.append(f"- Paired monthly net excess (S1−S0) @10bps: ann {pj['ann_mean']*100:.1f}%, "
                 f"t={pj['t_stat']:.2f}, 95% CI [{pj['ci_lo']*100:.1f}%, {pj['ci_hi']*100:.1f}%] "
                 f"→ {'excludes zero' if pj['ci_excludes_zero'] else 'includes zero (NOT distinguishable from luck)'}")
    lines.append(f"- Robustness: {n_beat}/{n_var} S1 variants beat S0 on Sharpe over {post} @10bps")
    lines.append("")

    ra = ra_10 and ra_25         # risk-adjusted win at BOTH realistic & punitive cost
    ret = ret_10 and ret_25       # raw-return win at BOTH costs
    sig_phrase = ("distinguishable from zero" if ci_positive
                  else "significantly NEGATIVE" if ci_negative
                  else "statistically indistinguishable from zero")

    if not ra:
        lines.append(
            "**② S1 edge has DECAYED.** After 2000 it does not even win risk-adjusted "
            "net of cost — its full-sample Sharpe is largely the 1968–2000 bear it "
            "sidestepped. For a trader operating today, 'use S1 to trade gold' is "
            "close to void: just hold GLD/physical, or don't single-bet gold. "
            "Honest kill.")
    elif ra and ret and ci_positive:
        lines.append(
            "**① S1 STILL HAS EDGE post-2000 — on both axes.** It beats buy-and-hold "
            "risk-adjusted *and* on raw return at realistic and punitive costs, with "
            "the paired excess distinguishable from zero in-sample. The S1 story is "
            "not *only* 1968–2000 bear-avoidance. Worth hardening with a proper "
            "walk-forward / out-of-sample test before trading.")
    elif ra and ret:  # both axes nominally, but the return excess is not significant
        lines.append(
            "**①a S1 LEANS POSITIVE on both axes post-2000, but the return edge is "
            f"not significant.** It wins risk-adjusted (higher Sharpe & Calmar) net of "
            "cost at 10 and 25bps and is nominally ahead on raw CAGR too — yet the "
            f"paired monthly excess is {sig_phrase}, so the return advantage is not "
            "reliably separable from luck on this single path. Promising, not proven: "
            "the risk-adjusted edge is the solid part; a walk-forward test is needed "
            "before leaning on the return premium.")
    else:  # ra and not ret → risk-reducer, gives up raw return
        lines.append(
            "**①′ MIXED — S1 is a risk-reducer post-2000, NOT a return-enhancer.** "
            "It still wins risk-adjusted (higher Sharpe & Calmar, roughly half the "
            "drawdown) net of cost at both 10 and 25bps, but it GIVES UP raw CAGR to "
            f"buy-and-hold and the paired return excess is {sig_phrase}. This is the "
            "same 'stress insurance, not uniform alpha' character the v2 SP-CVaR layer "
            "shows. Read-through: the full-sample Sharpe 0.63 *return* edge was indeed "
            "largely 1968–2000 bear-avoidance and has decayed — but the "
            "*drawdown-control* edge persists. **For a drawdown-averse holder S1 is "
            "still worth a walk-forward test; for a total-return maximiser, just "
            "hold the metal.**")
    return "\n".join(lines)


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

    # Backtest the whole strategy set at every cost level.
    bt_by_cost: Dict[float, Dict[str, pd.DataFrame]] = {
        c: run_all(panel, c) for c in COST_GRID
    }
    headline = bt_by_cost[DEFAULT_COST_BPS]
    cstart, cend = common_window(headline)
    if cstart is None or cend is None:
        print("ERROR: no common investable window across strategies "
              "(sample too short for the trend/vol warm-up).", file=sys.stderr)
        raise SystemExit(2)
    print(f"  fair common window: {cstart:%Y-%m} → {cend:%Y-%m}")

    # Per-segment metrics at every cost (fair common window applied inside).
    seg_by_cost: Dict[float, Dict[str, pd.DataFrame]] = {}
    for c, bts in bt_by_cost.items():
        common = common_window(bts)
        seg_by_cost[c] = {
            name: segment_metrics(bts, s, e, common=common)
            for name, s, e in SUBPERIOD_SEGMENTS
        }

    # Paired post-2000 significance (S1 primary vs S0) at every cost.
    paired_by_cost: Dict[float, Dict[str, float]] = {}
    for c, bts in bt_by_cost.items():
        lo = max(cstart, pd.Timestamp("2000-01-01"))
        hi = min(cend, pd.Timestamp("2026-12-31"))
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
    _pj0 = paired_by_cost[10.0]
    parts.append(
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
