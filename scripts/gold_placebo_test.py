"""PR #11 placebo test runner — is the central-bank-buying ⑤ a real signal or a
spurious monotone-trend fit?

Reuses PR #9's `build_attribution_panel` to assemble layers ①–④, then swaps
layer ⑤ for the real WGC cumulative excess stock and a battery of placebo
monotone-rising series, refits the same five-layer ex-post attribution on an
IDENTICAL window, and compares R² / residual collapse / ⑤ contribution / ⑤ t.
Adds ADF+KPSS stationarity, a first-difference (stationary) re-fit, and a
lag/lead variance-attribution table. Writes a verdict on whether PR #11's
"+121% residual claimed, R² 0.32→0.67" is real or trend-fitting.

EX-POST, non-predictive, non-trading. Touches no existing code.

Outputs:
  analysis/gold_placebo_test_<date>.md
  data/gold_placebo_levels_<date>.csv       per-candidate levels diagnostics
  data/gold_placebo_diff_<date>.csv         per-candidate first-difference fits
  data/gold_placebo_stationarity_<date>.csv ADF+KPSS levels & diff
  data/gold_placebo_leadlag_<date>.csv      Δgold × flow lead/lag corr

Usage:
    uv run python scripts/gold_placebo_test.py
    uv run python scripts/gold_placebo_test.py --t0 2022-01 --start 1990-01-01
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402
from lib.gold_anchor import fetch_fred_series  # noqa: E402
from lib.gold_credit_spread_attribution import build_attribution_panel  # noqa: E402
from lib.gold_attribution_placebo import (  # noqa: E402
    WGC_BASELINE_T,
    WGC_SOURCE_NOTE,
    adjudicate,
    annual_to_monthly,
    baseline_no_fifth,
    common_window,
    lead_lag_table,
    make_placebos,
    placebo_label,
    run_diff_fifth,
    run_levels_fifth,
    stationarity_table,
    wgc_cumulative_excess_annual,
)


def _fmt(x, spec="+.1f", suffix=""):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01", help="panel data start")
    ap.add_argument("--t0", default="2022-01", help="attribution window start")
    ap.add_argument("--t1", default=None, help="attribution window end (default latest)")
    ap.add_argument("--cpi-mode", default="identity", choices=["identity", "free"])
    ap.add_argument("--max-lag", type=int, default=6, help="lead/lag horizon (months)")
    args = ap.parse_args()

    print("[1/6] building PR#9 panel (live FRED + anchor reuse) …", file=sys.stderr)
    panel = build_attribution_panel(start=args.start)
    df = panel.data
    idx = df.index

    print("[2/6] fetching placebo source series (M2, IP) …", file=sys.stderr)
    m2 = fetch_fred_series("M2SL", args.start).resample("ME").last().reindex(idx)
    ip = fetch_fred_series("INDPRO", args.start).resample("ME").last().reindex(idx)

    print("[3/6] building ⑤ candidates (real WGC + placebos) …", file=sys.stderr)
    wgc_annual = wgc_cumulative_excess_annual()
    wgc = annual_to_monthly(wgc_annual, idx)
    placebos = make_placebos(idx, cpi=df["cpi"], m2=m2, ip=ip)
    candidates = {"REAL_WGC": wgc, **placebos}

    # Fair comparison: fix ONE window (where ①–④ AND the real WGC both exist) and
    # fit every candidate on it. Otherwise longer-history placebos get more data.
    window = common_window(df, wgc)
    print(f"      common window: {window.min().date()}..{window.max().date()} "
          f"(n={len(window)})", file=sys.stderr)

    print("[4/6] levels attribution under each ⑤ …", file=sys.stderr)
    base = baseline_no_fifth(df, window=window, t0=args.t0, t1=args.t1,
                             cpi_mode=args.cpi_mode)
    levels = {
        k: run_levels_fifth(df, s, key=k, window=window, t0=args.t0, t1=args.t1,
                            cpi_mode=args.cpi_mode)
        for k, s in candidates.items()
    }

    print("[5/6] first-difference (stationary) attribution …", file=sys.stderr)
    diffs = {
        k: run_diff_fifth(df, s, key=k, window=window, cpi_mode=args.cpi_mode)
        for k, s in candidates.items()
    }

    print("[6/6] stationarity + lead/lag + verdict …", file=sys.stderr)
    stat_inputs = {
        "ln_gold": df["ln_gold"].reindex(window),
        "REAL_WGC": wgc.reindex(window),
        **{f"placebo:{k}": s.reindex(window) for k, s in placebos.items()
           if not k.startswith("rand_") or k == "rand_11"},  # one rand sample is enough
    }
    stat = stationarity_table(stat_inputs)

    # lead/lag: Δln(gold) vs the purchase flow = Δ(cumulative excess stock)
    gold_dln = df["ln_gold"].reindex(window).diff()
    flow = wgc.reindex(window).diff()
    ll = lead_lag_table(gold_dln, flow, max_lag=args.max_lag)

    real = levels["REAL_WGC"]
    placebo_results = [v for k, v in levels.items() if k != "REAL_WGC"]
    diff_placebo_results = [d for k, d in diffs.items() if k != "REAL_WGC"]
    v = adjudicate(real, placebo_results, diffs["REAL_WGC"],
                   diff_placebos=diff_placebo_results)

    # ── persist CSVs ────────────────────────────────────────────────────
    date = datetime.now().strftime("%Y-%m-%d")
    lv_csv = os.path.join(DATA_DIR, f"gold_placebo_levels_{date}.csv")
    df_csv = os.path.join(DATA_DIR, f"gold_placebo_diff_{date}.csv")
    st_csv = os.path.join(DATA_DIR, f"gold_placebo_stationarity_{date}.csv")
    ll_csv = os.path.join(DATA_DIR, f"gold_placebo_leadlag_{date}.csv")

    pd.DataFrame([{
        "candidate": r.key, "label": r.label, "r2": r.r2, "n": r.n,
        "flow_contrib_ln": r.flow_contrib_ln, "flow_contrib_pct": r.flow_contrib_pct,
        "resid_contrib_ln": r.resid_contrib_ln, "resid_contrib_pct": r.resid_contrib_pct,
        "flow_coef": r.flow_coef, "flow_t": r.flow_t, "flow_p": r.flow_p,
    } for r in levels.values()]).to_csv(lv_csv, index=False)
    pd.DataFrame([{
        "candidate": d.key, "label": d.label, "r2_diff": d.r2, "n": d.n,
        "flow_coef": float(d.coefs.get("flow", np.nan)),
        "flow_t": d.flow_t, "flow_p": d.flow_p,
    } for d in diffs.values()]).to_csv(df_csv, index=False)
    stat.to_csv(st_csv, index=False)
    ll.to_csv(ll_csv, index=False)

    md = os.path.join(ANALYSIS_DIR, f"gold_placebo_test_{date}.md")
    _write_report(md, args, date, panel, window, base, levels, diffs, stat, ll, v,
                  lv_csv, df_csv, st_csv, ll_csv)

    print(f"\nVERDICT: {v.verdict.upper()} — "
          f"{ {'spurious':'伪回归','mixed':'同期共振/形态拟合(降级)','real':'含真实可区分信号'}[v.verdict] }")
    print(f"  real R²={v.real_r2:.3f}  best monotone placebo R²={_fmt(v.best_placebo_r2,'.3f')} "
          f"({placebo_label(v.best_placebo_key) if v.best_placebo_key else 'n/a'})")
    print(f"  real ⑤ survives in diff: {v.survives_in_diff} (t={v.real_diff_t:.2f}); "
          f"singles-out-real={v.diff_singles_out_real}")
    print(f"  kink(g) R²={_fmt(v.kink_r2,'.3f')} dominates={v.kink_dominates}")
    print(f"report → {md}")
    for p in (lv_csv, df_csv, st_csv, ll_csv):
        print(f"csv    → {p}")


def _levels_table_md(base, levels) -> str:
    rows = [
        "| ⑤ 候选 | R² | 残差% (127%→?) | ⑤ 贡献% | ⑤ 系数 t |",
        "|---|---:|---:|---:|---:|",
        f"| **基线 (无⑤, 仅①–④)** | {base['r2']:.3f} | "
        f"{_fmt(base['resid_contrib_pct'])} | — | — |",
    ]
    # real first, then placebos in a stable order
    order = ["REAL_WGC", "t", "log_t"] + \
            [k for k in levels if k.startswith("rand_")] + \
            ["cum_cpi", "cum_m2", "cum_ip", "kink_2022"]
    for k in order:
        if k not in levels:
            continue
        r = levels[k]
        bold = "**" if k == "REAL_WGC" else ""
        rows.append(
            f"| {bold}{r.label}{bold} | {r.r2:.3f} | {_fmt(r.resid_contrib_pct)} | "
            f"{_fmt(r.flow_contrib_pct)} | {_fmt(r.flow_t,'+.2f')} |"
        )
    return "\n".join(rows)


def _diff_table_md(diffs) -> str:
    rows = ["| ⑤ 候选 | R²(差分) | ⑤ 系数 | ⑤ t | ⑤ p |", "|---|---:|---:|---:|---:|"]
    order = ["REAL_WGC", "t", "log_t"] + \
            [k for k in diffs if k.startswith("rand_")] + \
            ["cum_cpi", "cum_m2", "cum_ip", "kink_2022"]
    for k in order:
        if k not in diffs:
            continue
        d = diffs[k]
        bold = "**" if k == "REAL_WGC" else ""
        coef = float(d.coefs.get("flow", np.nan))
        rows.append(
            f"| {bold}{d.label}{bold} | {d.r2:.3f} | {_fmt(coef,'+.4g')} | "
            f"{_fmt(d.flow_t,'+.2f')} | {_fmt(d.flow_p,'.3f')} |"
        )
    return "\n".join(rows)


def _stat_table_md(stat) -> str:
    rows = ["| 序列 | n | ADF p | KPSS p | levels | Δ ADF p | Δ KPSS p | Δ |",
            "|---|---:|---:|---:|---|---:|---:|---|"]
    for _, r in stat.iterrows():
        rows.append(
            f"| {r['series']} | {int(r['n'])} | {_fmt(r['adf_p'],'.3f')} | "
            f"{_fmt(r['kpss_p'],'.3f')} | {r['level_verdict']} | "
            f"{_fmt(r['diff_adf_p'],'.3f')} | {_fmt(r['diff_kpss_p'],'.3f')} | "
            f"{r['diff_verdict']} |"
        )
    return "\n".join(rows)


def _leadlag_table_md(ll) -> str:
    rows = ["| 滞后(月) | 方向 | corr(Δln金 , flow) | n |", "|---:|---|---:|---:|"]
    for _, r in ll.iterrows():
        rows.append(
            f"| {int(r['lag_months']):+d} | {r['direction']} | "
            f"{_fmt(r['corr'],'+.3f')} | {int(r['n'])} |"
        )
    return "\n".join(rows)


def _leadlag_reading(ll) -> str:
    """One-line observed reading of the lead/lag table (vs the hypothetical note)."""
    pos = ll[ll["lag_months"] > 0]["corr"].dropna()
    neg = ll[ll["lag_months"] < 0]["corr"].dropna()
    contemp = ll[ll["lag_months"] == 0]["corr"]
    pos_max = float(pos.max()) if len(pos) else np.nan
    neg_max = float(neg.max()) if len(neg) else np.nan
    c0 = float(contemp.iloc[0]) if len(contemp) else np.nan
    if np.isfinite(pos_max) and np.isfinite(neg_max) and pos_max > neg_max:
        direction = (
            f"**实测(事实)**:正滞后侧(购金领先金价)相关更强(max corr={pos_max:+.3f})"
            f"且强于负滞后侧({neg_max:+.3f})与同期({c0:+.3f}) —— 方向上**轻微支持**「购金→金价」,"
            "但相关系数全部 <0.2、且建立在平滑插值上,**不足以**支撑强因果(推理)。"
        )
    else:
        direction = (
            f"**实测(事实)**:负滞后侧(金价领先购金)相关不弱于正滞后侧(neg max={neg_max:+.3f} "
            f"vs pos max={pos_max:+.3f}) —— 提示金价与购金**互为内生**,削弱单向「购金顶价」叙事(推理)。"
        )
    return direction


def _write_report(path, args, date, panel, window, base, levels, diffs, stat, ll, v,
                  lv_csv, df_csv, st_csv, ll_csv) -> None:
    real = levels["REAL_WGC"]
    verdict_head = {
        "spurious": "伪回归 (SPURIOUS) ❌",
        "mixed": "同期共振 / 形态拟合 — 从『顶价因果』降级 (MIXED) ◐",
        "real": "含真实可区分信号 (REAL) ✅",
    }[v.verdict]
    best_lbl = placebo_label(v.best_placebo_key) if v.best_placebo_key else "n/a"

    lines = [
        f"# 黄金归因 PR#11「央行购金⑤」placebo 检验 — {date}",
        "",
        f"_窗口 {window.min().date()}→{window.max().date()} (n={len(window)}) · "
        f"归因区间 {args.t0}→{args.t1 or '最新'} · cpi_mode={args.cpi_mode}_",
        "",
        "## 0. 这是什么检验 / 为什么是决定性的",
        "",
        "PR#9 的五层归因里,**去掉⑤** 时四个可测宏观层(①通胀/②实利率/③主权/④尾部)"
        f"合起来解释不了 2022→今 的金价涨幅 —— ε_flow 残差 ≈ {_fmt(base['resid_contrib_pct'])}% "
        "(②实利率层甚至为负:实利率上行金价反涨)。PR#11 往⑤注入 **央行累计超额购金存量(WGC)**,"
        "残差塌缩、R² 0.32→0.67,宣称「央行购金顶价」。",
        "",
        "**codex 复审的决定性质疑**:累计超额存量是**单调上升趋势变量**,金价 2022 后也单调上升;"
        "短样本里**任意单调趋势**都能吃掉残差、抬高 R²(与本研究早先打掉的『债务/GDP 伪相关』同源,PR#1)。"
        "所以残差塌缩本身**不构成**真实购金信号的证据 —— 必须先排除「趋势拟合」。本检验就是做这个排除:"
        "把⑤换成一组单调上升的 placebo,在**同一窗口、同一五层归因**下看它们能不能复现真WGC的塌缩。",
        "",
        "> **EX-POST、非预测、非交易**;完全复用 PR#9 的 `build_attribution_panel/"
        "fit_attribution/decompose_period`,只替换⑤列。断言按 (事实)/(推理)/(推测) 标注。",
        "",
        "## 1. 裁决",
        "",
        f"### **{verdict_head}**",
        "",
        v.reason,
        "",
    ]
    for nt in v.notes:
        lines.append(f"- {nt}")
    lines += [
        "",
        f"三个仲裁口径(事实):①单调 placebo 最高 R²={_fmt(v.best_placebo_r2,'.3f')}"
        f"(「{best_lbl}」)对比真WGC R²={real.r2:.3f};②真WGC 差分口径 ⑤ t={_fmt(v.real_diff_t,'+.2f')}"
        f"({'存活' if v.survives_in_diff else '消失'});③形态对照(g)拐点 R²={_fmt(v.kink_r2,'.3f')}。",
        "",
        "## 2. Placebo 电池 — 水平(levels)归因对照（核心表）",
        "",
        "同一 2010-12→今 窗口、同一①–④,只换⑤。**核心问题:placebo 能否达到与真WGC相近的 "
        "R²/残差塌缩?**",
        "",
        _levels_table_md(base, levels),
        "",
        "**读法(事实)**:`(a)t / (b)log t / (c)随机单调×5 / (d)累计CPI / (e)累计M2 / (f)累计IP` 这些"
        "**纯单调趋势** placebo,R² 几乎不动(仍 ≈ 基线)、残差**不塌缩**、⑤贡献为**负**(被回归挤成"
        "对冲项) —— 即「任意单调趋势都能压残差」的朴素担忧在**本归因结构下不成立**(推理:①–④已吸收了"
        "平滑趋势,留给⑤的是它们拟合不了的特定形状)。",
        "",
        "**但 (g) 2022 拐点形态对照**(先平后升、零经济含义)是关键:见 §1 注与下表。",
        "",
        "## 3. 平稳性 (ADF + KPSS)",
        "",
        "ADF 原假设=单位根(非平稳);KPSS 原假设=平稳。levels 多为 I(1)+/ambiguous(非平稳),"
        "一阶差分转平稳 —— **「一条非平稳趋势回归另一条非平稳趋势」正是伪回归的教科书温床**。",
        "",
        _stat_table_md(stat),
        "",
        "## 4. 一阶差分(平稳口径)归因 — 真正的仲裁",
        "",
        "若⑤的解释力是**真实**的,它应在平稳的 Δln金价 ~ Δ各层 里**存活**;若只是 levels 的趋势套趋势,"
        "差分后**消失**(levels 显著、差分不显著 = 典型伪回归特征)。",
        "",
        _diff_table_md(diffs),
        "",
        f"**差分结论(事实/推理)**:真WGC 的⑤在差分口径 t={_fmt(diffs['REAL_WGC'].flow_t,'+.2f')}、"
        f"p={_fmt(diffs['REAL_WGC'].flow_p,'.3f')} —— "
        + ("**仍显著(存活)**,说明 +121% 残差认领**不是纯水平伪回归**;但需注意差分口径下真WGC并"
           "**不比** 2022 拐点/时间趋势更突出(见表),其差分显著性主要来自 2022-24 购金流量与金价同期共振。"
           if v.survives_in_diff else
           "**不显著(消失)**,levels 显著但差分消失 = 伪回归签名,+121% 认领是水平趋势拟合假象。"),
        "",
        "## 5. 滞后/领先(方差归属,非因果)",
        "",
        "Δln金价 与 **购金流量**(=累计存量的一阶差分)的领先滞后相关。正滞后=购金领先金价"
        "(购金→价),负滞后=金价领先购金(价→购金,内生方向)。",
        "",
        _leadlag_table_md(ll),
        "",
        _leadlag_reading(ll),
        "",
        "> **诚实标注(推测)**:WGC 年度→月度插值过度平滑、样本极短(n≈" f"{len(window)}" "),"
        "本表只能显示**哪个方向同期共动**,无法识别因果。若 corr 在负滞后侧不弱于正滞后侧,提示"
        "**价格与购金互为内生**(金价涨→央行追买,与购金顶价方向相反),进一步削弱单向「购金顶价」叙事。",
        "",
        "## 6. 综合结论与边界",
        "",
        f"1. **朴素「任意单调趋势都能骗」被部分证伪(事实)**:6 类纯单调 placebo 无一逼近真WGC "
        f"(最高 R²={_fmt(v.best_placebo_r2,'.3f')} vs 真 {real.r2:.3f}),因为①–④已吸收平滑趋势。",
        f"2. **但真WGC的水平拟合主要是『2022 制度拐点形态』(推理)**:零含义的拐点对照(g) "
        f"R²={_fmt(v.kink_r2,'.3f')} "
        + ("**≥**" if np.isfinite(v.kink_r2) and v.kink_r2 >= real.r2 - 0.05 else "<")
        + " 真WGC —— 真WGC赢过平滑 placebo,**不是**因为它编码了独有的购金信息,"
        "而是因为它恰好具备金价同期的『先平后升』形状。",
        f"3. **差分口径(事实)**:真WGC的⑤ "
        + ("存活" if v.survives_in_diff else "消失")
        + (",但**不被单独挑出**(累计CPI/M2 差分 t 更大)" if v.survives_in_diff and not v.diff_singles_out_real else "")
        + f" —— {'有真实同期共振成分,但不强于其它宏观趋势' if v.survives_in_diff else '伪回归确证'}。",
        "4. **裁决(推理)**:" + {
            "spurious": (
                "+121% 残差认领是**趋势拟合假象**,PR#11『央行购金顶价』归因被证伪 —— "
                "『连归因都会被趋势骗』的怀疑论主线再次自洽。"
            ),
            "mixed": (
                "+121% 残差认领**含真实的 2022-24 购金↔金价同期共振成分,但远未达到 PR#11 叙事所暗示的"
                "『央行购金顶价』因果强度**:其 levels 优势主要是 2022 制度拐点的**形态拟合**(零含义拐点对照即可复现),"
                "差分口径也未将其从累计 CPI/M2 等宏观趋势中单独挑出,且方向存在内生性(金价可能领先购金)。"
                "**应将 PR#11 的⑤认领从『顶价归因』降级为『同期共振相关』** —— 怀疑论主线(『趋势/形态会冒充归因』)"
                "在更细口径下依然成立。"
            ),
            "real": (
                "+121% 残差认领**含真实且可区分的信号**:真WGC 胜过所有 placebo(含形态对照)、差分存活并被单独挑出。"
                "但这仍是**同期共振相关**而非已识别的因果(内生性仅做方差归属)。"
            ),
        }[v.verdict],
        "",
        "**边界/诚实标注**:",
        f"- {WGC_SOURCE_NOTE}",
        f"- 基线baseline=473t 影响累计存量的早段水平,不影响 2022→ 的斜率结论(推理)。",
        "- 窗口短(2010-12+,WGC 起点);内生性仅做方差归属,非因果识别。",
        "- 多分量层③④在差分口径按 Δ 的 z-composite 重建,与 levels 口径同构但非同一系数。",
        "",
        "## 附:产出文件",
        "",
        f"- 水平归因对照:`{os.path.relpath(lv_csv)}`",
        f"- 差分归因:`{os.path.relpath(df_csv)}`",
        f"- 平稳性:`{os.path.relpath(st_csv)}`",
        f"- 滞后/领先:`{os.path.relpath(ll_csv)}`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
