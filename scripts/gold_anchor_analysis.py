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
from lib.gold_anchor import build_anchor_panel, integration_table, johansen_test
from lib.paths import ANALYSIS_DIR, DATA_DIR

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
    lines.append(f"- n_obs = {j['n_obs']}, α = {j['alpha_level']}")
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
    # Johansen assumes all series are I(1); only run when both gold and the
    # anchor are classified I(1), else the cointegration verdict is meaningless.
    gold_v = itab.loc["ln_gold_nominal", "verdict"] if "ln_gold_nominal" in itab.index else "missing"
    jres = {}  # key -> (label, j_or_None, skip_reason_or_None)
    for key, lncol, label in ANCHORS:
        anc_v = itab.loc[lncol, "verdict"] if lncol in itab.index else "missing"
        sub = df[["ln_gold_nominal", lncol]].dropna()
        if not should_run_johansen(gold_v, anc_v):
            reason = f"non-I(1) (gold={gold_v}, anchor={anc_v})"
            jres[key] = (label, None, reason)
            print(f"  {label}: skipped — {reason}")
            continue
        if len(sub) < 30:
            reason = f"too few obs ({len(sub)})"
            jres[key] = (label, None, reason)
            print(f"  {label}: skipped — {reason}")
            continue
        j = johansen_test(df, ["ln_gold_nominal", lncol])
        jres[key] = (label, j, None)
        beta_s = "n/a" if j["beta"] is None else f"{j['beta']:.3f}"
        print(f"  {label}: coint_rank={j['coint_rank']}, valid={j['valid_coint']}, β={beta_s}, n={j['n_obs']}")

    print("[4/4] Writing analysis report...")
    today = (args.end or datetime.now().strftime("%Y-%m-%d"))
    report = _build_report(df, panel.notes, itab, jres, args)
    out = os.path.join(ANALYSIS_DIR, f"gold_anchor_cointegration_{today}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved → {out}")
    print("\nDone.")


def _build_report(df, notes, itab, jres, args) -> str:
    L = []
    L.append("# 黄金「锚 + 偏离」— 第 0–1 步:单整 + 协整生死门\n")
    L.append(f"> 生成时间区间: {df.index.min().date()} .. {df.index.max().date()} "
             f"({len(df)} 个月)。对应 `docs/gold-anchor-vecm-spec.md` 第 0–1 步。\n")
    L.append("> 本报告只做单整检验与协整秩判决,**不含 VECM / 2022 分解**(留作下一 PR)。\n")

    L.append("\n## 0. 数据与口径\n")
    for k, v in notes.items():
        L.append(f"- **{k}**: {v}")

    L.append("\n## 1a. 单整阶数 (ADF + PP + KPSS)\n")
    L.append("ADF/PP 原假设 = 单位根 (I(1));KPSS 原假设 = 平稳 (I(0))。`diff_stationary` = 一阶差分是否平稳。\n")
    L.append("```")
    L.append(itab.to_string())
    L.append("```")
    L.append("\n判读 (推理):")
    for s, row in itab.iterrows():
        L.append(f"- `{s}`: **{row['verdict']}**")
    rr = itab.loc["real_rate_10y", "verdict"] if "real_rate_10y" in itab.index else "n/a"
    L.append(f"\n- 实利率 `real_rate_10y` 判定 **{rr}**。若 I(0)→只能进短期 ECM(下一 PR),"
             "符合「实利率是偏离驱动、不是锚」的直觉;若 I(1)→可进长期向量,需另行解释 (spec §1)。")

    L.append("\n## 1b. 协整 (Johansen, trace + max-eigen)\n")
    L.append("系统 = [ln(名义金价), ln(锚/GDP)],仅当两者都判 I(1) 才跑。"
             "valid_coint=False → 锚是伪共同趋势/满秩,假说被证伪;"
             "valid_coint=True → 锚成立,报告长期弹性 β。\n")
    for key, (label, j, skip) in jres.items():
        L.append(f"\n### {label}\n")
        if j is None:
            L.append(f"- **skipped: {skip}** — 未跑 Johansen。")
        else:
            L.append(_fmt_johansen(j))

    L.append("\n## 核心结论\n")
    main = jres.get("debt_gdp")
    if main:
        label, j, skip = main
        if j is None:
            L.append(f"- **主锚 Debt/GDP: 未跑 Johansen({skip})** — 前置单整条件不满足,不作锚判定。")
        elif j["full_rank_stationary"]:
            L.append(f"- **主锚 Debt/GDP: 满秩(raw rank=n)→ 序列疑似平稳/模型假设不成立,"
                     f"协整解读无效,不作锚成立判定(需复查单整阶数)。**")
        elif j["valid_coint"]:
            beta_note = ("接近 1(纯贬值假说获支持)" if abs(j["beta"] - 1.0) < 0.25
                         else "显著偏离 1(贬值故事需进一步解释,见 spec §2)")
            L.append(f"- **主锚 Debt/GDP: coint_rank = {j['coint_rank']} ≥ 1 → 锚成立(非伪趋势)。** "
                     f"长期弹性 β = {j['beta']:.3f},{beta_note}。")
        else:
            L.append(f"- **主锚 Debt/GDP: coint_rank = 0 → 没有协整,基准线是伪共同趋势,"
                     f"用户「锚」假说在此被证伪(spec 杀死条件 1)。**")
    for key in ("fed_gdp", "m2_gdp"):
        if key in jres:
            label, j, skip = jres[key]
            if j is None:
                L.append(f"- 对照 {label}: skipped({skip})。")
            else:
                beta_s = "n/a" if j["beta"] is None else f"{j['beta']:.3f}"
                L.append(f"- 对照 {label}: coint_rank={j['coint_rank']}, valid={j['valid_coint']}, β={beta_s}。")
    L.append("\n- 下一步 (下一 PR): 对协整成立的锚估 VECM,看误差修正项 λ 是否显著、"
             "实利率 δ 是否在偏离项里现形,并做 2022 分解 (spec §2–§3)。")
    L.append("\n> Claim types: 检验统计量与样本 β 估计值为 (事实);I(d) 判定、协整成立与否、"
             "「β≈1 支持纯贬值假说」等模型判读为 (推理);未来路径与机制故事为 (推测)。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
