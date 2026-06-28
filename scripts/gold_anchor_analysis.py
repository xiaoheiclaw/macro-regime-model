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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.gold_anchor import (
    build_anchor_panel,
    combined_verdict,
    estimate_vecm,
    integration_segments,
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

# PR #2: the trivariate anchor — joint long-run equilibrium surface
# [ln gold, ln(debt/GDP), 10y real rate]. β1 lifts the anchor (debasement),
# β2 presses it down (opportunity cost). β2 significant ⇒ the long end IS part
# of the anchor; short-run Δreal_rate significant ⇒ it drives the deviation.
TRIVARIATE_COLS = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]


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
        if j["coint_rank"] > 1:
            lines.append(f"- **long-run β = n/a** (coint_rank={j['coint_rank']}>1 → 单一协整向量"
                         "不唯一,不报告 β)")
        else:
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


def _run_real_rate_segments(df, notes) -> dict:
    """Step 1 (PR #2): the long-end real rate's I(d) read on TWO regimes — the
    full spliced series (GS10−CPI proxy + DFII10) vs the clean post-TIPS DFII10
    subsample. PR #1 judged the (full) real rate I(0) and dropped it from the
    cointegrating vector; the split makes that auditable per segment."""
    cutoff = notes.get("real_rate_tips_start", "n/a")
    segs = {"full (spliced)": (None, None)}
    if cutoff and cutoff != "n/a":
        segs[f"clean TIPS (≥{cutoff})"] = (cutoff, None)
    out = integration_segments(df["real_rate_10y"], segs)
    out["_cutoff"] = cutoff
    return out


def _gate_check(sub) -> tuple:
    """All-I(1) gate for Johansen: every column must be combined-verdict I(1) on
    the SAME complete-case subsample (c+ct). Mixing an I(0)/ambiguous leg into
    Johansen invalidates the rank and the long-run β — so we gate, not paper over."""
    verdicts = {c: combined_verdict(sub[c]) for c in sub.columns}
    passed = all(v["combined"] == "I(1)" for v in verdicts.values())
    return passed, verdicts


def _run_trivariate(df, notes) -> dict:
    """Step 1–2 (PR #2): three-variable Johansen on the joint anchor surface,
    plus VECM iff the cointegration rank is robustly and *uniquely* 1.

    Johansen/VECM are run only on a window where ALL THREE legs are I(1) (the
    full 1968+ spliced sample if it passes; else the clean post-TIPS window
    where the real rate is I(1)). If neither window is all-I(1), the trivariate
    test is skipped and the real rate is confined to the short-run ECM (P1)."""
    cols = TRIVARIATE_COLS
    entry = {"cols": cols, "window": None, "id_window": None, "pairwise_n": None,
             "id_check": {}, "gate_full": None, "gate_full_n": None,
             "gate_clean": None, "gate_clean_n": None, "skip": None, "lag": None,
             "point": None, "robust": None, "rank_set": None,
             "robust_unique_rank1": False, "vecm": None, "vecm_note": None}

    full = df[cols].dropna()
    passed_full, v_full = _gate_check(full)
    entry["gate_full"], entry["gate_full_n"] = v_full, int(len(full))

    use, window, used_gate = None, None, None
    if passed_full and len(full) >= 40:
        use, window, used_gate = full, "full (1968+ spliced)", v_full
    else:
        cutoff = notes.get("real_rate_tips_start", "n/a")
        if cutoff and cutoff != "n/a":
            clean = df.loc[df.index >= pd.Timestamp(cutoff), cols].dropna()
            passed_clean, v_clean = _gate_check(clean)
            entry["gate_clean"], entry["gate_clean_n"] = v_clean, int(len(clean))
            if passed_clean and len(clean) >= 40:
                use, window, used_gate = clean, f"clean TIPS (≥{cutoff})", v_clean

    if use is None:
        # report id_check + n from the LAST evaluated gate window (clean if it was
        # tried, else full) so id verdicts and the sample size never mismatch (P2).
        if entry["gate_clean"] is not None:
            entry["id_check"], entry["pairwise_n"], entry["id_window"] = (
                entry["gate_clean"], entry["gate_clean_n"], "clean TIPS")
        else:
            entry["id_check"], entry["pairwise_n"], entry["id_window"] = (
                entry["gate_full"], entry["gate_full_n"], "full (1968+ spliced)")
        entry["skip"] = ("not all three legs I(1) on any gated window — Johansen "
                         "requires all-I(1); real rate remains eligible only for a "
                         "future short-run ECM specification (PR #3)")
        return entry

    # id_check / n / window all come from the SAME (used) subsample
    entry["window"] = entry["id_window"] = window
    entry["id_check"] = used_gate
    entry["pairwise_n"] = int(len(use))
    lag = select_var_order(use, cols, max_lags=4)
    entry["lag"] = lag
    try:
        entry["point"] = johansen_test(use, cols, det_order=0, k_ar_diff=lag["k_ar_diff"])
    except (ValueError, np.linalg.LinAlgError) as e:
        entry["skip"] = f"point Johansen failed: {type(e).__name__}: {str(e)[:60]}"
        return entry
    robust = johansen_robustness(use, cols, lags=ROBUST_LAGS, det_orders=ROBUST_DET_ORDERS)
    entry["robust"] = robust
    ranks = sorted(set(robust["coint_rank"].dropna().astype(int)))
    entry["rank_set"] = ranks

    # VECM only when the cointegrating vector is well-identified: EVERY grid cell
    # succeeded (no failures), every cell is valid_coint, AND rank is uniquely 1.
    # rank>1 ⇒ the single eigenvector is not unique → β1/β2 arbitrary; any failed
    # cell ⇒ the grid isn't robust. (P2)
    n_cells = len(robust)
    all_cells_ok = bool(robust["coint_rank"].notna().all()) and n_cells > 0
    all_valid = bool(robust["valid_coint"].fillna(False).all()) if n_cells else False
    robust_unique_rank1 = all_cells_ok and all_valid and ranks == [1]
    entry["robust_unique_rank1"] = robust_unique_rank1
    if robust_unique_rank1:
        k = max(1, lag["k_ar_diff"])
        if k != lag["k_ar_diff"]:
            entry["vecm_note"] = (f"AIC lag→k_ar_diff={lag['k_ar_diff']} (VAR(1)); "
                                  f"bumped to k_ar_diff={k} so a short-run Δ block exists.")
        try:
            entry["vecm"] = estimate_vecm(use, cols, k_ar_diff=k, coint_rank=1, det_order=0)
        except (ValueError, np.linalg.LinAlgError) as e:
            entry["vecm_note"] = f"VECM estimation failed: {type(e).__name__}: {e}"
    return entry


def _fmt_trivariate(tri) -> list:
    L = []
    L.append("\n## 2. 三变量锚 — Johansen [ln金价, ln(债务/GDP), 10y实利率]\n")
    L.append("锚升级为**联合长期均衡面** `ln金价* = α + β₁·ln(债务/GDP) + β₂·实利率`:"
             "债务上抬(贬值)、实利率下压(机会成本),两者共同定均衡。"
             "「长端是不是锚」= β₂ 是否在**长期向量**里显著(对 spec §修订)。\n")
    L.append("**全-I(1) gate (推理)**: Johansen 要求三列同子样本都 I(1);混入 I(0)/ambiguous "
             "腿会让 rank 与 β₂ 失效,故先 gate 再跑。")

    def _gate_lines(label, n, verdicts):
        out = [f"- gate 窗口 **{label}** (n={n}),同子样本 c+ct 单整预检:"]
        for c, cv in verdicts.items():
            out.append(f"    - `{c}`: **{cv['combined']}** (c={cv['c']}, ct={cv['ct']})")
        return out

    if tri["skip"]:
        # print BOTH gate windows that were evaluated, each with its OWN n (P2)
        L.extend(_gate_lines("full (1968+ spliced)", tri["gate_full_n"], tri["gate_full"]))
        if tri.get("gate_clean"):
            L.extend(_gate_lines(f"clean TIPS", tri["gate_clean_n"], tri["gate_clean"]))
        L.append(f"- **skipped (gate 未通过): {tri['skip']}** (推理)")
        return L

    L.extend(_gate_lines(tri["id_window"], tri["pairwise_n"], tri["id_check"]))
    L.append(f"- **gate 通过窗口: {tri['window']}** — Johansen/VECM 在此窗口估计 (事实)")
    lag = tri["lag"]
    L.append(f"- VAR select_order(AIC) p={lag['var_order']} → k_ar_diff={lag['k_ar_diff']} (事实)")
    L.append("\n点估计 (选定 lag, det_order=0):")
    L.append(_fmt_johansen(tri["point"]))
    pt = tri["point"]
    if pt["betas"] is not None:
        L.append(f"- 长期向量系数: **β₁(债务)={pt['betas'][0]:.3f}, "
                 f"β₂(实利率)={pt['betas'][1]:.3f}** (事实)")
    L.append("\n稳健性网格 (coint_rank / valid_coint / β₁,β₂=betas):")
    L.append("```")
    L.append(tri["robust"].to_string(index=False))
    L.append("```")
    ranks = tri["rank_set"] or []
    if ranks == [1]:
        rank_verdict = f"coint_rank=1 在 lag∈{list(ROBUST_LAGS)}×det∈{list(ROBUST_DET_ORDERS)} 全设定下**稳定**"
    elif ranks == [0]:
        rank_verdict = "coint_rank=0 在所有设定下稳定 → **无协整**"
    else:
        rank_verdict = f"coint_rank ∈ {ranks} **随设定变化(不稳定)**"
    L.append(f"- **稳健性 (推理): {rank_verdict}**")
    return L


def _fmt_vecm(tri) -> list:
    L = []
    L.append("\n## 3. VECM — 实利率在锚里、在偏离里、还是两者皆有?\n")
    if tri.get("skip") or tri["rank_set"] is None:
        L.append("- 三变量 Johansen 未运行(gate 未通过),VECM 跳过 (推理)。")
        return L
    if tri["vecm"] is None:
        if tri.get("robust_unique_rank1") and tri.get("vecm_note"):
            L.append(f"- coint_rank 稳健且唯一=1,但 VECM 未产出: {tri['vecm_note']} (推理)。")
        else:
            L.append(f"- **coint_rank∈{tri['rank_set']} 非「稳健且唯一=1」→ 不估 VECM** "
                     "(rank>1 时单一协整向量不唯一,β₁/β₂ 不可识别;含失败单元则网格不稳健) (推理)。")
        return L
    v = tri["vecm"]
    if tri.get("vecm_note"):
        L.append(f"> {tri['vecm_note']}\n")
    L.append(f"- 设定: coint_rank={v['coint_rank']}, k_ar_diff={v['k_ar_diff']}, "
             f"deterministic={v['deterministic']}, n_obs={v['n_obs']} (事实)\n")
    L.append("**长期协整向量 (gold 归一化, 系数移到 RHS):**")
    for var, b in v["betas"].items():
        sig = "显著" if b["significant"] else "不显著"
        L.append(f"- {var}: β={b['beta']:.4f} (t={b['t']:.2f}, p={b['p']:.3f}, **{sig}**) (事实)")
    ec = v["ec_speed"]
    ec_sig = "显著" if ec["significant"] else "不显著"
    hl = ""
    if ec["lambda"] < 0 and ec["significant"]:
        import math
        hl = f"; 半衰期≈{math.log(0.5)/math.log(1+ec['lambda']):.1f} 月" if -1 < ec["lambda"] < 0 else ""
    L.append(f"\n**误差修正速度 λ (金价方程): {ec['lambda']:.4f}** "
             f"(t={ec['t']:.2f}, p={ec['p']:.3f}, {ec_sig}; λ<0 且显著={ec['corrects']}{hl}) (事实)")
    L.append("\n**短期 Δ 系数 (金价方程):**")
    L.append("```")
    L.append(f"{'var':<16}{'lag':>4}{'coef':>10}{'t':>8}{'p':>8}  sig")
    for t in v["short_run"]:
        L.append(f"{t['var']:<16}{t['lag']:>4}{t['coef']:>10.4f}{t['t']:>8.2f}"
                 f"{t['p']:>8.3f}  {'*' if t['significant'] else ''}")
    L.append("```")

    # the verdict the spec asks for
    b2 = v["betas"].get("real_rate_10y", {})
    long_sig = bool(b2.get("significant"))
    rr_short = [t for t in v["short_run"] if t["var"] == "real_rate_10y"]
    short_sig = any(t["significant"] for t in rr_short)
    if long_sig and short_sig:
        verdict = "**两者皆有**: 实利率既在长期锚 (β₂ 显著) 又驱动短期偏离"
    elif long_sig:
        verdict = "**在锚里**: 实利率长期 β₂ 显著、短期 Δ 不显著 → 长端是锚的一部分"
    elif short_sig:
        verdict = "**在偏离里**: 长期 β₂ 不显著、短期 Δ实利率显著 → 实利率驱动金价对锚的偏离(印证 Baur 漏了 ECT)"
    else:
        verdict = "**两者皆不显著**: 实利率长期与短期都不显著 → '长端是锚' 在此线性设定下未获支持"
    L.append(f"\n- **裁决 (推理): {verdict}。** "
             f"(λ {'修正成立' if ec['corrects'] else '不显著/非负→无回归力,存疑'})")
    return L


def _build_report(df, notes, itab, jres, args, rr_segs=None, tri=None) -> str:
    L = []
    L.append("# 黄金「锚 + 偏离」— 单整 + 协整 + 三变量 VECM\n")
    L.append(f"> 生成时间区间: {df.index.min().date()} .. {df.index.max().date()} "
             f"({len(df)} 个月)。对应 `docs/gold-anchor-vecm-spec.md` 第 0–2 步。\n")
    L.append("> PR #1: 单整检验 + 双变量协整生死门。PR #2: 长端实利率两段 I(d) 复核 + "
             "**三变量 Johansen [金价,债务/GDP,实利率]** + (若稳健协整) **VECM** 拆长期锚/短期偏离。"
             "**2022 分解 / Gregory-Hansen 断点**留作 PR #3。\n")

    L.append("\n## 0. 数据与口径\n")
    for k, v in notes.items():
        L.append(f"- **{k}**: {v}")

    L.append("\n## ⚠️ Limitations(本 PR 已知假设,留待 PR #3 处理)\n")
    L.append("- **季度 GDP/debt → 月末 ffill 含前视偏差** (事实):季度值被前填到季度内各月末,"
             "**这不是实时(发布日)历史协整**——某月用到的 GDP/debt 在当时可能尚未发布。本 PR 为全样本"
             "结构检验,可接受;**发布日对齐 + 对该假设的敏感性**留待 PR #3 (推测)。")
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
    L.append(f"\n- 实利率 `real_rate_10y` 合并判定 **{rr}** (推理)。若 I(0)→只能进短期 ECM,"
             "符合「实利率是偏离驱动、不是锚」的直觉 (推测);若 I(1)→可进长期向量,需另行解释 (spec §1)。")

    if rr_segs:
        cutoff = rr_segs.get("_cutoff", "n/a")
        L.append("\n### 1a-bis. 长端实利率的两段 I(d) 复核 (PR #2 重点)\n")
        L.append("PR #1 把(全样本拼接)实利率判 I(0) 是它被踢出长期协整向量的**直接原因**;"
                 "因拼接断点(代理↔TIPS)可能污染单一判定,这里分两段各报 c+ct (推理):\n")
        L.append(f"- 注:干净 TIPS 子样本实为 **DFII10 起点 {cutoff}**(10y 不变期限 TIPS 自 2003 起),"
                 "而非 spec 假设的 1997——如实标注 (事实)。\n")
        for name, seg in rr_segs.items():
            if name == "_cutoff":
                continue
            L.append(f"- **{name}** (n={seg['n']}, {seg['start']}..{seg['end']}): "
                     f"合并 **{seg['combined']}** (c={seg['c']}, ct={seg['ct']}) (推理)")
        L.append("\n- 判读 (推理): 若全样本与干净 TIPS 段都非 I(1) → 实利率作为**长期锚的一条腿**"
                 "证据弱,三变量协整大概率仍由 [金价,债务] 主导;若任一段 I(1) → 它有资格进长期向量,"
                 "由下面三变量 Johansen 与 VECM β₂ 给出最终裁决。")

    L.append("\n## 1b. 双变量协整 (Johansen) — PR #1 基线: [金价, 单锚] pairwise + 稳健性\n")
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

    # PR #2: trivariate anchor + VECM
    if tri is not None:
        L.extend(_fmt_trivariate(tri))
        L.extend(_fmt_vecm(tri))

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

    # ── trivariate anchor verdict (PR #2 centerpiece) ──
    if tri is not None:
        L.append("\n### 三变量锚 [金价, 债务/GDP, 实利率] 裁决\n")
        ranks = tri.get("rank_set")
        v = tri.get("vecm")
        if tri.get("skip"):
            rr = (tri.get("id_check") or {}).get("real_rate_10y", {})
            L.append(f"- **全-I(1) gate 未通过(实利率合并判定 {rr.get('combined', '?')})→ "
                     "三列未同为 I(1),无法构成有效三变量协整 (事实/推理)。** "
                     "实利率只能进短期 ECM,不报告长期 β₂。")
            L.append("- ⚠️ **杀死条件触发 (spec 修订版扩展)**: 全样本线性常参数三变量锚因实利率"
                     "非稳健 I(1) 无法估计 → 留待 regime-switching / 断点协整 (PR #3) (推理)。")
        elif tri.get("robust_unique_rank1") and v is not None:
            b2 = v["betas"].get("real_rate_10y", {})
            ec = v["ec_speed"]
            long_sig = bool(b2.get("significant"))
            rr_short = [t for t in v["short_run"] if t["var"] == "real_rate_10y"]
            short_sig = any(t["significant"] for t in rr_short)
            where = ("锚里+偏离里" if (long_sig and short_sig) else
                     "锚里" if long_sig else "偏离里" if short_sig else "都不显著")
            L.append(f"- **三变量锚稳健成立 (coint_rank 全网格唯一=1, 窗口={tri['window']}) (事实/推理)。** "
                     f"长期 β₁(债务)={v['betas']['ln_debt_gdp']['beta']:.3f}, "
                     f"β₂(实利率)={b2.get('beta', float('nan')):.3f} "
                     f"(p={b2.get('p', float('nan')):.3f}); 误差修正 λ={ec['lambda']:.3f} "
                     f"(p={ec['p']:.3f}, 修正={ec['corrects']}) (事实)。")
            L.append(f"- **实利率落点: {where}** (推理) — 回答了 spec §修订的核心问句。")
        elif ranks == [0]:
            L.append("- **三变量仍 coint_rank=0 全网格稳定 (事实) → 加入实利率后仍无协整。**")
            L.append("- ⚠️ **杀死条件触发 (spec 修订版条件 1+扩展)**: 线性常参数锚(含实利率)"
                     "也被证伪 (推理) → 留待 regime-switching / 断点协整 (Gregory-Hansen, PR #3)。")
        else:
            L.append(f"- **三变量 coint_rank∈{ranks} 非「稳健且唯一=1」(事实) → 协整证据脆弱"
                     "(rank>1 时向量不唯一/含失败单元),不能稳健宣称含实利率的三变量锚成立 (推理)。**")
            L.append("- ⚠️ **杀死条件触发 (扩展)**: 线性常参数三变量锚不稳健 → "
                     "留待 regime-switching / 断点协整 (PR #3)。")

    L.append("\n- 下一步 (下一 PR) (推测): 若三变量锚成立但 λ/β₂ 边际,或不稳健 → "
             "Gregory-Hansen 断点协整 + regime-switching VECM;并做发布日对齐 + 2022 分解 (spec §3–§4)。")
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

    print("[4/4] Trivariate anchor (Johansen + VECM) & real-rate I(d) segments...")
    rr_segs = _run_real_rate_segments(df, panel.notes)
    for name, seg in rr_segs.items():
        if name != "_cutoff":
            print(f"  real_rate {name}: {seg['combined']} (c={seg['c']}, ct={seg['ct']}, n={seg['n']})")
    tri = _run_trivariate(df, panel.notes)
    if tri.get("skip"):
        print(f"  trivariate: skipped — {tri['skip']}")
    else:
        print(f"  trivariate Johansen: robust ranks={tri['rank_set']}; "
              f"VECM={'estimated' if tri.get('vecm') else 'not estimated'}")

    print("[5/5] Writing analysis report...")
    today = (args.end or datetime.now().strftime("%Y-%m-%d"))
    report = _build_report(df, panel.notes, itab, jres, args, rr_segs=rr_segs, tri=tri)
    out = os.path.join(ANALYSIS_DIR, f"gold_anchor_cointegration_{today}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved → {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
