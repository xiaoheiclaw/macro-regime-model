"""Gold vs de-dollarization fundamentals — deviation / valuation monitor (PR #13).

Answers the user's worry: **has gold run far ahead of the de-dollarization
fundamentals it is supposed to track, and is the 2025-2026 hedge being put on at
the most-expensive point of the narrative?** This is a quantitative anchor for a
hedging decision — NOT a forecast and NOT a trading backtest.

Pipeline (all in ``lib.gold_dedollar_gap``):
  1. build_gap_panel  — gold + CPI + cb_cum_excess (WGC, PR #10) + custody_share
                        (WMTSECL1/GFDEBTN, PR #8) [+ optional DXY].
  2. build_di         — de-dollarization index = signed z-scored composite.
  3. compute_deviation— rolling-OLS residual of ln(gold) on DI (+ ln(gold/CPI)),
                        z-scored vs history → "how far is gold above fundamentals".
  4. conditional_forward_table — historical extreme-high deviation → forward
                        1/2/3y return distribution (DESCRIPTIVE, not a forecast).
  5. current_reading + adjudicate — where 2025-2026 sits + the headline verdict.
  6. robustness — weight perturbation, rolling-window band {36,60,120}, and a
                  first-difference stationarity read.

Outputs:
  * markdown report (analysis/gold_dedollar_gap_<stamp>.md)
  * series CSV (data/gold_dedollar_gap_series_<stamp>.csv): panel + DI + deviation
  * conditional-table CSV (data/gold_dedollar_gap_conditional_<stamp>.csv)

Usage:
    uv run python scripts/gold_dedollar_gap.py
    uv run python scripts/gold_dedollar_gap.py --include-dxy --top-q 0.9
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_dedollar_gap import (  # noqa: E402
    DEFAULT_COMPONENTS,
    DEFAULT_HORIZONS,
    DEFAULT_REG_WINDOW,
    DEFAULT_TOP_Q,
    DXY_COMPONENT,
    WINDOW_BAND,
    adjudicate,
    build_di,
    build_gap_panel,
    compute_deviation,
    conditional_forward_table,
    current_reading,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x * 100:.0f}%"


def _fmt(x: float, d: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _cond_table_md(tbl: pd.DataFrame) -> str:
    if tbl.empty:
        return "_(no conditioning observations)_"
    rows = []
    for _, r in tbl.iterrows():
        rows.append([
            f"{int(r['horizon'])}m", str(r["regime"]), str(int(r["n"])),
            _fmt(r["mean"]), _fmt(r["median"]), _fmt(r["p25"]),
            _fmt(r["p75"]), _fmt_pct(r["hit"]),
        ])
    return _md_table(
        ["horizon", "regime", "n", "mean(lnret)", "median", "p25", "p75", "hit>0"],
        rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2002-12")
    ap.add_argument("--end", default=None)
    ap.add_argument("--reg-window", type=int, default=DEFAULT_REG_WINDOW)
    ap.add_argument("--top-q", type=float, default=DEFAULT_TOP_Q)
    ap.add_argument("--include-dxy", action="store_true",
                    help="add the broad USD index (sign −) as a 3rd DI component")
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    # local-timezone wall clock for the human-facing report date, so a run near
    # the UTC day boundary is not stamped "yesterday" (codex PR#14 P3). The "截至
    # <asof>" data date below is always the panel's own date, independent of this.
    now = datetime.now().astimezone()
    tzname = now.strftime("%Z") or "local"
    stamp = f"{now.strftime('%Y-%m-%d')} {tzname}"
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")

    print("building gap panel (gold + CPI + cb_cum_excess + custody_share"
          + (" + dxy" if args.include_dxy else "") + ") ...")
    panel = build_gap_panel(args.start, args.end, include_dxy=args.include_dxy)
    df = panel.data

    components = list(DEFAULT_COMPONENTS) + (
        [DXY_COMPONENT] if args.include_dxy else [])

    # ── DI (headline, equal weight) ──
    di_res = build_di(df, components=components)
    di = di_res.di

    # ── deviation: two口径 (nominal + real purchasing power) ──
    dev_nom = compute_deviation(df["ln_gold"], di, window=args.reg_window)
    dev_real = compute_deviation(df["ln_gold_real"], di, window=args.reg_window)

    # ── historical extreme → forward returns (descriptive) ──
    cond_nom = conditional_forward_table(
        dev_nom.gap_z_full, df["gold_nominal"],
        horizons=DEFAULT_HORIZONS, top_q=args.top_q)

    # ── current positioning + verdict ──
    cr_nom = current_reading(dev_nom, di, roll_window=args.reg_window)
    cr_real = current_reading(dev_real, di, roll_window=args.reg_window)
    label, verdict_msg = adjudicate(cr_nom, min_n=max(args.reg_window, 36))

    # ── robustness: weight perturbation (only if >=2 components present) ──
    weight_rows: List[List[str]] = []
    present = list(di_res.weights.keys())
    if len(present) >= 2:
        c0 = present[0]
        for tilt in (0.3, 0.4, 0.5, 0.6, 0.7):
            w = {c0: tilt}
            rest = (1.0 - tilt) / (len(present) - 1)
            for c in present[1:]:
                w[c] = rest
            di_w = build_di(df, components=components, weights=w).di
            dev_w = compute_deviation(df["ln_gold"], di_w, window=args.reg_window)
            cr_w = current_reading(dev_w, di_w, roll_window=args.reg_window)
            weight_rows.append([
                ", ".join(f"{k}={v:.2f}" for k, v in w.items()),
                _fmt(cr_w.gap_z_full), _fmt_pct(cr_w.gap_pct_full)])

    # ── robustness: rolling-window band ──
    window_rows: List[List[str]] = []
    for w in WINDOW_BAND:
        dev_w = compute_deviation(df["ln_gold"], di, window=w)
        cr_w = current_reading(dev_w, di, roll_window=w)
        window_rows.append([
            f"{w}m", str(int(dev_w.resid.notna().sum())),
            _fmt(cr_w.gap_z_full), _fmt_pct(cr_w.gap_pct_full)])

    # ── robustness: first-difference stationarity read ──
    # Δln(gold) vs ΔDI rolling residual → is the deviation conclusion an artifact
    # of two trending levels? (PR #11 placebo lesson). Same machinery on diffs.
    dln_gold = df["ln_gold"].diff()
    ddi = di.diff()
    dev_diff = compute_deviation(dln_gold, ddi, window=args.reg_window)
    cr_diff = current_reading(dev_diff, ddi, roll_window=args.reg_window)

    # ── guard degenerate states before rendering (codex PR#14 P1) ──
    di_cov = di.dropna()
    if di_cov.empty:
        raise RuntimeError(
            "DI has no valid months — the de-dollarization components have no "
            "common overlap on this window; cannot compute a deviation reading. "
            "Widen --start/--end or check the component coverage in the panel.")
    asof_str = cr_nom.asof.strftime("%Y-%m") if cr_nom.asof is not None else "n/a"
    di_latest_str = _fmt(di_cov.iloc[-1])

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"gold_dedollar_gap_{file_stamp}.md")
    P: List[str] = []
    P.append(f"# 金价 vs 去美元化基本面偏离度监控 ({stamp})\n")
    P.append("> **这是对冲决策的量化锚 — 非预测、非交易回测。** 回答:金价是否已远远"
             "走在去美元化基本面之前?现在相对基本面贵到什么程度、是不是买在叙事最贵处?\n")

    P.append("## 裁决 (Verdict)\n")
    P.append(f"**{label}** — {verdict_msg}\n")
    P.append("- 名义口径 ln(gold) vs DI:当前偏离 "
             f"z={_fmt(cr_nom.gap_z_full)},历史分位 {_fmt_pct(cr_nom.gap_pct_full)}"
             f"(截至 {asof_str},残差样本 n={cr_nom.n_resid},裁决最小样本要求 "
             f"{max(args.reg_window, 36)}); 滚动分位 {_fmt_pct(cr_nom.gap_pct_roll)}。")
    P.append("- 实际购买力口径 ln(gold/CPI) vs DI:当前偏离 "
             f"z={_fmt(cr_real.gap_z_full)},历史分位 {_fmt_pct(cr_real.gap_pct_full)}。")
    P.append(f"- DI 自身历史分位 {_fmt_pct(cr_nom.di_pct_full)}(去美元化基本面本身处于"
             "历史什么位置)。\n")

    P.append("## 1. 去美元化基本面指数 (DI) 构造\n")
    P.append(f"- {di_res.notes['definition']}")
    P.append(f"- 权重: {di_res.notes['weights']}")
    if "dropped" in di_res.notes:
        P.append(f"- ⚠️ {di_res.notes['dropped']}")
    P.append("- 成分与符号(符号统一为「越大=去美元化越强」):")
    sign_map = {c: s for c, s in components}
    for c in present:
        cov = df[c].dropna()
        sgn = "+" if sign_map.get(c, 1) > 0 else "−"
        if len(cov):
            P.append(f"  - `{c}` (sign {sgn}) 覆盖 {cov.index.min():%Y-%m}.."
                     f"{cov.index.max():%Y-%m} (n={len(cov)})")
        else:
            P.append(f"  - `{c}` (sign {sgn}) 无数据")
    P.append(f"- DI z-score 基准: {di_res.notes.get('z_base', 'n/a')}。")
    P.append(f"- DI 时序覆盖: {di_cov.index.min():%Y-%m}..{di_cov.index.max():%Y-%m} "
             f"(n={len(di_cov)});DI 最新值 {di_latest_str}。\n")

    P.append("## 2. 金价相对 DI 的偏离度 (两口径)\n")
    P.append("偏离度 = ln(gold) 对 DI 的**滚动 OLS** 残差,再标准化。滚动(非全样本"
             "单回归)是对「两条共同趋势线伪回归」(repo PR #11 placebo 教训)的刻意防范"
             "——衡量金价偏离*当前局部*关系的程度,而非协整断言。正值=金价高于 DI 隐含水平"
             "(走在基本面前)。\n")
    P.append(_md_table(
        ["口径", "残差 n", "当前 z (全样本)", "当前历史分位", "滚动分位"],
        [["名义 ln(gold)", str(int(dev_nom.resid.notna().sum())),
          _fmt(cr_nom.gap_z_full), _fmt_pct(cr_nom.gap_pct_full),
          _fmt_pct(cr_nom.gap_pct_roll)],
         ["实际 ln(gold/CPI)", str(int(dev_real.resid.notna().sum())),
          _fmt(cr_real.gap_z_full), _fmt_pct(cr_real.gap_pct_full),
          _fmt_pct(cr_real.gap_pct_roll)]]))
    P.append("")

    P.append("## 3. 历史极端高位 → 之后金价表现 (条件描述,非预测)\n")
    P.append(f"当名义偏离度处于历史 top {_fmt_pct(1 - args.top_q)} 时,之后 1/2/3 年的"
             "金价前瞻对数收益分布,对照非极端月份。**⚠️ 这是对过去共同走势的条件描述,"
             "不是预测也不是因果声明:偏离度高 ≠ 必然回调。** 阈值用全样本分位(in-sample);"
             "前瞻收益本身是真前瞻(尾部 NaN 不可观测)。N 较小(后 2010 基本面)。\n")
    P.append(_cond_table_md(cond_nom))
    P.append("")

    P.append("## 4. 稳健性\n")
    P.append("### 4a. DI 权重扰动 (名义口径当前偏离)\n")
    if weight_rows:
        P.append(_md_table(["权重", "当前 z", "历史分位"], weight_rows))
    else:
        P.append("_(单一成分,无权重可扰动)_")
    P.append("")
    P.append("### 4b. 滚动窗口 {36,60,120} 月\n")
    P.append(_md_table(["窗口", "残差 n", "当前 z", "历史分位"], window_rows))
    P.append("")
    P.append("### 4c. 平稳化:一阶差分口径 (Δln gold vs ΔDI)\n")
    P.append("对趋势伪回归的稳健性检查:若 levels 口径的「偏离」只是两条趋势线的产物,"
             "差分口径应显著弱化。\n")
    P.append(f"- Δ 口径当前偏离 z={_fmt(cr_diff.gap_z_full)},"
             f"历史分位 {_fmt_pct(cr_diff.gap_pct_full)}"
             f"(残差 n={int(dev_diff.resid.notna().sum())})。\n")

    P.append("## 5. 诚实与局限\n")
    P.append("- **DI 是合成代理,权重主观**;其 level/scale 是标准化与成分选择的产物。")
    P.append("- **成分频率粗糙**:央行购金为 WGC 年度估计→月度均摊(形态是插值产物,见 "
             "`lib.gold_cb_flow`);托管周→月;CPI 月度。基本面是低频的,DI 移动缓慢,"
             "次年内偏离形态主要由金价驱动。")
    P.append("- **全样本 z/分位是 in-sample/描述性**(本就是把今天对照整段历史排序)。")
    P.append("- **共同覆盖窗口**:央行超额购金起于约 2010、托管起于约 2002-12,故双成分 DI "
             "本质是**后 2010** 对象;前瞻收益条件分桶的 N 因此较小——已如实标注,未隐藏。")
    P.append("- 偏离度高仅描述「相对历史的估值偏贵」,**不预测**必然回调。\n")

    P.append("## Provenance\n")
    for k, v in {**panel.notes, **{f"DI.{k}": v for k, v in di_res.notes.items()},
                 **{f"dev.{k}": v for k, v in dev_nom.notes.items()}}.items():
        P.append(f"- **{k}**: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print(f"  report → {report_path}")

    # ── write series CSV ──
    os.makedirs(DATA_DIR, exist_ok=True)
    series = pd.DataFrame({
        "gold_nominal": df["gold_nominal"],
        "cpi": df["cpi"],
        "cb_cum_excess": df["cb_cum_excess"],
        "custody_share": df["custody_share"],
        "ln_gold": df["ln_gold"],
        "ln_gold_real": df["ln_gold_real"],
        "DI": di,
        "resid_nominal": dev_nom.resid,
        "gap_z_nominal": dev_nom.gap_z_full,
        "resid_real": dev_real.resid,
        "gap_z_real": dev_real.gap_z_full,
    })
    if args.include_dxy and "dxy" in df.columns:
        series["dxy"] = df["dxy"]
    series_path = os.path.join(
        DATA_DIR, f"gold_dedollar_gap_series_{file_stamp}.csv")
    series.to_csv(series_path)
    print(f"  series → {series_path}")

    cond_path = os.path.join(
        DATA_DIR, f"gold_dedollar_gap_conditional_{file_stamp}.csv")
    cond_nom.to_csv(cond_path, index=False)
    print(f"  conditional table → {cond_path}")

    print(f"\nVERDICT: {label} — {verdict_msg}")


if __name__ == "__main__":
    main()
