#!/usr/bin/env python3
"""Gold "anchor + deviation" — steps 0–1 runner.

Builds the monthly FRED+gold panel, runs unit-root tests (ADF/PP/KPSS) to
classify integration order, then runs Johansen cointegration on
[ln gold, ln(anchor/GDP)] for the three anchor candidates (debt/GDP main;
Fed/GDP and M2/GDP as controls). Writes the panel to data/ and an analysis
report to analysis/.

Core question answered: does an anchor *hold* (cointegration rank>=1) or is it
a spurious common trend (rank=0)? And is the long-run elasticity beta ~ 1?

Usage:
    uv run python scripts/gold_anchor_analysis.py [--start 1968-01-01] [--end YYYY-MM]
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.gold_anchor import (
    build_anchor_panel,
    combined_verdict,
    integration_table,
    johansen_robustness,
    johansen_test,
    select_var_order,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR

ROBUST_LAGS = (1, 2, 3, 4)
ROBUST_DET_ORDERS = (-1, 0, 1)

LEVEL_COLS = ["ln_gold_nominal", "ln_debt_gdp", "ln_m2_gdp", "ln_fed_gdp", "real_rate_10y"]
ANCHORS = [
    ("debt_gdp", "ln_debt_gdp", "Debt/GDP (主锚)"),
    ("fed_gdp", "ln_fed_gdp", "Fed assets/GDP (对照)"),
    ("m2_gdp", "ln_m2_gdp", "M2/GDP (对照)"),
]


def should_run_johansen(gold_verdict: str, anchor_verdict: str) -> bool:
    """Johansen cointegration assumes all series are I(1). Only run when BOTH
    gold and the anchor are classified I(1); I(0)/ambiguous/mixed-order inputs
    would yield a meaningless 'anchor holds' verdict."""
    return gold_verdict == "I(1)" and anchor_verdict == "I(1)"


def _fmt_johansen(j: dict) -> str:
    lines = []
    n = len(j["columns"])
    lines.append(f"- n_obs = {j['n_obs']}, α = {j['alpha_level']}, "
                 f"det_order = {j['det_order']}, k_ar_diff = {j['k_ar_diff']}")
    lines.append("- Trace test (H0: rank ≤ r):")
    for r in range(n):
        mark = "✓ reject" if j["trace_stat"][r] > j["trace_cv"][r] else "✗ fail"
        lines.append(f"    r≤{r}: stat={j['trace_stat'][r]:.2f} vs cv={j['trace_cv'][r]:.2f}  {mark}")
    lines.append("- Max-eigen test (H0: rank = r):")
    for r in range(n):
        mark = "✓ reject" if j["maxeig_stat"][r] > j["maxeig_cv"][r] else "✗ fail"
        lines.append(f"    r={r}: stat={j['maxeig_stat'][r]:.2f} vs cv={j['maxeig_cv'][r]:.2f}  {mark}")
    lines.append(f"- **coint_rank = {j['coint_rank']}** (valid_coint={j['valid_coint']}; "
                 f"trace={j['trace_rank']}, max-eigen={j['maxeig_rank']}; "
                 f"raw trace/max-eigen={j['raw_trace_rank']}/{j['raw_maxeig_rank']})")
    if j["full_rank_stationary"]:
        lines.append("- ⚠️ **full-rank** (raw rank = n): series look stationary / model assumptions "
                     "may not hold — this is NOT evidence the anchor holds.")
    lines.append(f"- cointegrating vector (normalized on gold): {[round(x,4) for x in j['coint_vector_normalized']]}")
    if j["beta"] is None:
        lines.append("- **long-run elasticity β = n/a** (no valid cointegrating relation)")
    else:
        lines.append(f"- **long-run elasticity β = {j['beta']:.4f}** (gold vs anchor)")
    return "\n".join(lines)


def _robust_summary(entry: dict) -> str:
    """One-line robustness verdict over the lag×det_order grid."""
    ranks = entry.get("rank_set") or []
    if not ranks:
        return "robustness grid produced no valid cells"
    br = entry.get("beta_range")
    beta_s = "" if not br else f", β∈[{br[0]:.2f}, {br[1]:.2f}]"
    if len(ranks) == 1:
        return f"coint_rank = {ranks[0]} 在所有 lag∈{list(ROBUST_LAGS)}×det∈{list(ROBUST_DET_ORDERS)} 下**稳定**{beta_s}"
    return f"coint_rank ∈ {ranks} **随设定变化(不稳定)**{beta_s}"


def _build_report(df, notes, itab, jres, args) -> str:
    L = []
    L.append("# 黄金「锚 + 偏离」— 第 0–1 步:单整 + 协整生死门\n")
    L.append(f"> 生成时间区间: {df.index.min().date()} .. {df.index.max().date()} "
             f"({len(df)} 个月)。对应 `docs/gold-anchor-vecm-spec.md` 第 0–1 步。\n")
    L.append("> 本报告只做单整检验与协整秩判决,**不含 VECM / 2022 分解**(留作下一 PR)。\n")

    L.append("\n## 0. 数据与口径\n")
    for k, v in notes.items():
        L.append(f"- **{k}**: {v}")

    L.append("\n## ⚠️ Limitations(本 PR 已知假设,留待 VECM PR 处理)\n")
    L.append("- **季度 GDP/debt → 月末 ffill 含前视偏差** (事实):季度值被前填到季度内各月末,"
             "**这不是实时(发布日)历史协整**——某月用到的 GDP/debt 在当时可能尚未发布。本 PR 为全样本"
             "结构检验,可接受;**发布日对齐 + 对该假设的敏感性**留待 VECM PR (推测)。")
    L.append("- 实利率 1997 前(实为 DFII10 起点 2003 前)为 GS10−CPI 拼接代理,见 §0 注记 (事实)。")

    L.append("\n## 1a. 单整阶数 (ADF + PP + KPSS, regression ∈ {c, ct})\n")
    L.append("ADF/PP 原假设 = 单位根 (I(1));KPSS 原假设 = 平稳 (I(0))。"
             "`verdict` = c 与 ct 两套判定的合并(一致取该判定,分歧→ambiguous)。"
             "`*_ct` 为含趋势项的 p 值。\n")
    L.append("```")
    L.append(itab.to_string())
    L.append("```")
    L.append("\n判读(检验统计量/p 值为 (事实);I(d) 判定为 (推理)):")
    for s, row in itab.iterrows():
        L.append(f"- `{s}`: **{row['verdict']}** (c={row['verdict_c']}, ct={row['verdict_ct']}) (推理)")
    rr = itab.loc["real_rate_10y", "verdict"] if "real_rate_10y" in itab.index else "n/a"
    L.append(f"\n- 实利率 `real_rate_10y` 合并判定 **{rr}** (推理)。若 I(0)→只能进短期 ECM(下一 PR),"
             "符合「实利率是偏离驱动、不是锚」的直觉 (推测);若 I(1)→可进长期向量,需另行解释 (spec §1)。")

    L.append("\n## 1b. 协整 (Johansen) — pairwise 子样本 + lag/det_order 稳健性\n")
    L.append("系统 = [ln(名义金价), ln(锚/GDP)]。**前置 I(1) 检验在 Johansen 实际使用的同一 "
             "pairwise complete-case 子样本上、用 c+ct 双设定重跑**(不用整列判定)。lag 由 VAR "
             "select_order(AIC) 在该子样本上选,k_ar_diff = p−1;并报告 rank 对 lag∈{1..4}×"
             "det_order∈{−1,0,1} 的稳健性。\n")
    for key, e in jres.items():
        L.append(f"\n### {e['label']}\n")
        pw = e["pairwise"]
        L.append(f"- pairwise 子样本 n={pw['n']};I(1) 检验(同子样本,c+ct):"
                 f"gold→**{pw['gold']['combined']}** (c={pw['gold']['c']},ct={pw['gold']['ct']}), "
                 f"anchor→**{pw['anchor']['combined']}** (c={pw['anchor']['c']},ct={pw['anchor']['ct']}) (推理)")
        if e["skip"]:
            L.append(f"- **skipped: {e['skip']}** — 未跑 Johansen (推理)。")
            continue
        lag = e["lag"]
        L.append(f"- VAR select_order(AIC) p={lag['var_order']} → k_ar_diff={lag['k_ar_diff']} (事实)")
        L.append("\n点估计(选定 lag, det_order=0):")
        L.append(_fmt_johansen(e["point"]))
        L.append("\n稳健性网格 (coint_rank / valid_coint / β):")
        L.append("```")
        L.append(e["robust"].to_string(index=False))
        L.append("```")
        L.append(f"- **稳健性 (推理): {_robust_summary(e)}**")

    L.append("\n## 核心结论(稳健性视角)\n")
    main = jres.get("debt_gdp")
    if main:
        e = main
        if e["skip"]:
            L.append(f"- **主锚 Debt/GDP: 未跑 Johansen({e['skip']})** — 前置单整条件不满足,不作锚判定 (推理)。")
        else:
            ranks = e.get("rank_set") or []
            pt = e["point"]
            br = e.get("beta_range")
            beta_pt = "n/a" if pt["beta"] is None else f"{pt['beta']:.3f}"
            if ranks == [1]:
                L.append(f"- **主锚 Debt/GDP: coint_rank=1 在 lag×det 全设定下稳定 (事实) → 锚成立(非伪趋势)(推理)。** "
                         f"点估计 β={beta_pt} (事实);稳健性区间 β∈[{br[0]:.2f},{br[1]:.2f}] (事实),"
                         f"{'接近 1(纯贬值假说获支持)' if (br and abs((br[0]+br[1])/2-1)<0.25) else '中枢偏离 1(贬值故事需进一步解释,见 spec §2)'} (推理)。")
            elif ranks == [0]:
                L.append(f"- **主锚 Debt/GDP: coint_rank=0 在所有设定下稳定 (事实) → 没有协整,基准线是伪共同趋势,"
                         f"假说被证伪(spec 杀死条件 1)(推理)。**")
            else:
                L.append(f"- **主锚 Debt/GDP: coint_rank ∈ {ranks} 随 lag/det 变化(不稳定)(事实) → 协整证据脆弱,"
                         f"不能稳健地宣称锚成立 (推理)**;点估计(det=0,选定 lag)coint_rank={pt['coint_rank']}, β={beta_pt}。")
    for key in ("fed_gdp", "m2_gdp"):
        if key in jres:
            e = jres[key]
            if e["skip"]:
                L.append(f"- 对照 {e['label']}: skipped({e['skip']})(推理)。")
            else:
                ranks = e.get("rank_set") or []
                stab = "稳定" if len(ranks) == 1 else "不稳定"
                L.append(f"- 对照 {e['label']}: coint_rank∈{ranks}({stab}) (事实)。")
    L.append("\n- 下一步 (下一 PR) (推测): 对协整成立的锚估 VECM,看误差修正项 λ 是否显著、"
             "实利率 δ 是否在偏离项里现形,并做发布日对齐 + 2022 分解 (spec §2–§3)。")
    L.append("\n> Claim types: 检验统计量与样本 β 估计值为 (事实);I(d) 判定、协整成立与否、"
             "「β≈1 支持纯贬值假说」等模型判读为 (推理);未来路径与机制故事为 (推测)。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    # data/ and analysis/ are gitignored → may not exist on a fresh checkout
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

    print("[1/4] Building monthly gold-anchor panel...")
    panel = build_anchor_panel(start=args.start, end=args.end)
    df = panel.data
    if df.empty:
        raise ValueError(
            f"empty panel for start={args.start}, end={args.end} — "
            "check the date window and upstream data availability"
        )
    print(f"  panel: {df.shape[0]} months × {df.shape[1]} cols, "
          f"{df.index.min().date()}..{df.index.max().date()}")

    panel_path = os.path.join(DATA_DIR, "gold_anchor_panel.csv")
    df.to_csv(panel_path)
    with open(os.path.join(DATA_DIR, "gold_anchor_panel_notes.json"), "w", encoding="utf-8") as f:
        json.dump(panel.notes, f, indent=2, ensure_ascii=False)
    print(f"  saved → {panel_path}")

    print("[2/4] Unit-root tests (ADF + PP + KPSS)...")
    itab = integration_table(df, [c for c in LEVEL_COLS if c in df.columns])
    print(itab.to_string())

    print("[3/4] Johansen cointegration (debt/GDP main; Fed, M2 controls)...")
    # Johansen assumes all series are I(1); the I(1) check is done on the SAME
    # pairwise complete-case subsample Johansen actually uses (not the full
    # column), under both 'c' and 'ct' regressions. Lag is selected per pair.
    jres = {}
    for key, lncol, label in ANCHORS:
        cols = ["ln_gold_nominal", lncol]
        sub = df[cols].dropna()
        gold_cv = combined_verdict(sub["ln_gold_nominal"])
        anc_cv = combined_verdict(sub[lncol])
        pairwise = {"gold": gold_cv, "anchor": anc_cv, "n": int(len(sub))}
        entry = {"label": label, "cols": cols, "pairwise": pairwise,
                 "skip": None, "lag": None, "point": None, "robust": None}

        if not should_run_johansen(gold_cv["combined"], anc_cv["combined"]):
            entry["skip"] = (f"non-I(1) on pairwise subsample "
                             f"(gold={gold_cv['combined']}, anchor={anc_cv['combined']})")
            jres[key] = entry
            print(f"  {label}: skipped — {entry['skip']}")
            continue
        if len(sub) < 30:
            entry["skip"] = f"too few obs ({len(sub)})"
            jres[key] = entry
            print(f"  {label}: skipped — {entry['skip']}")
            continue

        lag = select_var_order(df, cols, max_lags=4)
        entry["lag"] = lag
        point = johansen_test(df, cols, det_order=0, k_ar_diff=lag["k_ar_diff"])
        entry["point"] = point
        robust = johansen_robustness(df, cols, lags=ROBUST_LAGS, det_orders=ROBUST_DET_ORDERS)
        entry["robust"] = robust
        ranks = sorted(set(robust["coint_rank"].dropna().astype(int)))
        betas = [b for b in robust["beta"].tolist() if b is not None]
        entry["rank_set"] = ranks
        entry["beta_range"] = (min(betas), max(betas)) if betas else None
        jres[key] = entry

        beta_s = "n/a" if point["beta"] is None else f"{point['beta']:.3f}"
        print(f"  {label}: VAR(p)={lag['var_order']} k_ar_diff={lag['k_ar_diff']} → "
              f"coint_rank={point['coint_rank']} valid={point['valid_coint']} β={beta_s}; "
              f"robust ranks across lag×det={ranks}")

    print("[4/4] Writing analysis report...")
    today = (args.end or datetime.now().strftime("%Y-%m-%d"))
    report = _build_report(df, panel.notes, itab, jres, args)
    out = os.path.join(ANALYSIS_DIR, f"gold_anchor_cointegration_{today}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved → {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
