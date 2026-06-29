#!/usr/bin/env python3
"""Gold "anchor + deviation" — step 3 / PR #4: Markov-switching regression.

PR #1–#3 killed every linear / constant-parameter / single-break anchor
(bivariate + trivariate Johansen, Gregory-Hansen endogenous-break cointegration
all fail). The surviving hypothesis is a **discrete time-varying anchor**: the
world jumps between 2–3 regimes, each with its own driver coefficients. This
runner tests it with a Markov-switching regression on the STATIONARY left-hand
side Δln(gold) (not I(1) levels — avoids spurious regression), regressing on
Δln(debt/GDP) and Δ(real rate) with regime-dependent coefficients AND residual
variance.

Two samples are reported (matching the spec's full-vs-clean discipline):
  * full 1968+ — debt-only MS (real rate splice is dirty pre-2003, so the long
    sample uses Δln(debt/GDP) alone);
  * post-2003 — debt + clean TIPS real rate (the regime-conditional test of
    "is the long end an anchor": in which regimes is Δreal_rate significant?).

For each sample we fit K=2 then K=3 and select by BIC + economic readability.
The report gives per-regime coefficient tables (which driver is significant in
which regime), the transition matrix, expected durations, and the smoothed
regime-probability path checked against known macro episodes (1970s inflation,
Volcker, GFC, 2022). Kill condition: no interpretable, significantly-distinct
regime structure → discrete time-varying anchor also fails → TVP (PR #5).

Usage:
    uv run python scripts/gold_anchor_markov.py [--start 1968-01-01] [--end YYYY-MM]
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
    build_ms_frame,
    regime_coeff_spread,
    select_markov_k,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR

K_VALUES = (2, 3)
N_RESTARTS = 25          # MS likelihood is multimodal → random restarts
SEED = 0

# Known macro episodes — the smoothed regime path should (if regimes are real)
# light up distinct regimes across these. Descriptive only; NOT fitted to them.
EPISODES = {
    "1970s 高通胀 (1973–1980)": ("1973-01-01", "1980-12-31"),
    "Volcker 紧缩 (1980–1985)": ("1980-01-01", "1985-12-31"),
    "GFC 危机 (2008–2009)": ("2008-01-01", "2009-12-31"),
    "2022+ 地缘/紧缩 (2022–)": ("2022-01-01", None),
}

# the two samples (label → (exog level cols, start override or None))
SAMPLES = [
    ("full 1968+ (debt-only)", ["ln_debt_gdp"], None),
    ("post-2003 (debt + clean TIPS real rate)", ["ln_debt_gdp", "real_rate_10y"], "tips"),
]

# pretty names for the diffed drivers
DRIVER_LABEL = {
    "const": "截距 (常数漂移)",
    "d_ln_debt_gdp": "Δln(债务/GDP)",
    "d_real_rate_10y": "Δ实利率(10y)",
}


def _fit_sample(df, notes, exog_cols, start_override):
    """Build the diffed frame for one sample window and select K by BIC."""
    sub = df
    if start_override == "tips":
        cutoff = notes.get("real_rate_tips_start", "n/a")
        if not cutoff or cutoff == "n/a":
            return {"skip": "no clean-TIPS start in panel notes", "frame_n": 0}
        sub = df.loc[df.index >= pd.Timestamp(cutoff)]
    cols = ["ln_gold_nominal"] + exog_cols
    avail = [c for c in cols if c in sub.columns]
    if len(avail) < 2:
        return {"skip": f"missing columns {set(cols) - set(avail)}", "frame_n": 0}
    frame = build_ms_frame(sub[avail].dropna(), exog_cols)
    if len(frame) < 60:
        return {"skip": f"too few obs after diff (n={len(frame)})", "frame_n": len(frame)}
    sel = select_markov_k(frame, k_values=K_VALUES, n_restarts=N_RESTARTS, seed=SEED)
    return {"skip": None, "frame": frame, "frame_n": len(frame), "sel": sel,
            "window": (frame.index.min().date().isoformat(),
                       frame.index.max().date().isoformat())}


def _regime_label(regime, exog):
    """Heuristic economic label for a regime by its significant drivers + vol."""
    sig = [DRIVER_LABEL.get(nm, nm) for nm in exog
           if regime["coeffs"].get(nm, {}).get("significant")]
    vol = regime["sigma2"]
    drivers = ("、".join(sig) + " 驱动") if sig else "无显著驱动"
    return f"{drivers}; σ²={vol:.4f}"


def _episode_alignment(fit):
    """For the selected fit, the mean smoothed prob of each regime within each
    known macro episode + the dominant regime there (descriptive cross-check)."""
    smp = fit["smoothed_probabilities"]
    rows = []
    for name, (s, e) in EPISODES.items():
        seg = smp
        if s is not None:
            seg = seg[seg.index >= pd.Timestamp(s)]
        if e is not None:
            seg = seg[seg.index <= pd.Timestamp(e)]
        if len(seg) == 0:
            rows.append((name, None, None, 0))
            continue
        means = seg.mean(axis=0)
        dom = int(means.values.argmax())
        rows.append((name, dom, float(means.iloc[dom]), len(seg)))
    return rows


def _annual_dominant(fit):
    """Year → dominant regime (mode of monthly argmax) — a compact timeline."""
    smp = fit["smoothed_probabilities"]
    dom = pd.Series(smp.values.argmax(axis=1), index=smp.index)
    out = dom.groupby(dom.index.year).agg(lambda s: int(s.mode().iloc[0]))
    return out


def _fmt_fit(fit) -> list:
    L = []
    exog = fit["exog"]
    deg = fit.get("degenerate")
    trust = fit.get("trustworthy")
    L.append(f"- 展示 **K={fit['k_regimes']}** (BIC={fit['bic']:.1f}, AIC={fit['aic']:.1f}, "
             f"llf={fit['llf']:.1f}, n_obs={fit['n_obs']}, "
             f"switching_variance={fit.get('switching_variance')}, "
             f"n_restarts={fit.get('n_restarts')}, **converged={fit['converged']}**, "
             f"non-degenerate regimes={fit.get('n_nondegenerate')}/{fit['k_regimes']}, "
             f"**trustworthy={trust}**) (事实)\n")
    if deg or fit.get("converged") is False:
        L.append(f"- ⚠️ **该解退化/未收敛**(σ² floor={fit.get('var_floor', float('nan')):.2e}, "
                 f"min duration={fit.get('min_duration')}m, |t| 识别上限={fit.get('t_ident_max'):g}):"
                 "含退化 regime 或 MLE 未收敛 → 下列系数仅作诊断,**不作 regime 结构证据** (推理)。\n")
    L.append("**各 regime 系数表** (Δln金价 ~ 各驱动的 Δ;系数/方差均 regime 依赖):\n")
    for r in fit["regimes"]:
        tag = ""
        if r.get("degenerate"):
            tag = f" — ⚠️**退化**[{', '.join(r.get('degenerate_reasons', []))}]"
        L.append(f"- **Regime {r['regime']}** — {_regime_label(r, exog)} "
                 f"(期望持续 {r['expected_duration']:.1f} 月){tag} (事实):")
        for nm, cell in r["coeffs"].items():
            sig = "**显著**" if cell["significant"] else ("不显著(未识别 |t|>1e6)"
                  if not cell.get("identified", True) else "不显著")
            tval = f"{cell['t']:.2e}" if abs(cell["t"]) >= 1e4 or not np.isfinite(cell["t"]) else f"{cell['t']:.2f}"
            L.append(f"    - {DRIVER_LABEL.get(nm, nm)}: coef={cell['coef']:+.4f} "
                     f"(t={tval}, p={cell['p']:.3f}, {sig})")
    tm = np.asarray(fit["transition_matrix"])
    L.append("\n**转移矩阵** P[i,j]=P(s_t=i | s_{t-1}=j),列和=1 (事实):")
    L.append("```")
    hdr = "from→  " + "  ".join(f"r{j}" for j in range(fit["k_regimes"]))
    L.append(hdr)
    for i in range(fit["k_regimes"]):
        L.append(f"to r{i}: " + "  ".join(f"{tm[i, j]:.3f}" for j in range(fit["k_regimes"])))
    L.append("```")
    L.append(f"- 期望持续期 (月): {[round(d, 1) for d in fit['expected_durations']]} (事实)")

    # regime-conditional anchor read — degenerate regimes EXCLUDED
    spread = regime_coeff_spread(fit)
    nd = fit.get("n_nondegenerate", fit["k_regimes"])
    L.append(f"\n**驱动的 regime 区分度** (仅在 {nd} 个 non-degenerate regime 间比较;退化 regime 已剔除):")
    for nm in exog:
        s = spread[nm]
        L.append(f"    - {DRIVER_LABEL.get(nm, nm)}: 跨 non-deg regime 系数={[round(c, 3) for c in s['coefs']]}, "
                 f"可比 regime 数={s['n_compared']}, 符号翻转={s['sign_flip']}, "
                 f"显著 regime 数={s['n_sig']}, regime条件={s['regime_conditional']}, "
                 f"distinct={s['distinct']} (事实/推理)")

    # episode alignment
    L.append("\n**平滑 regime 概率 × 已知宏观段** (描述性交叉核对,模型未对这些段拟合):")
    for name, dom, p, n in _episode_alignment(fit):
        if dom is None:
            L.append(f"    - {name}: 样本无覆盖")
        else:
            L.append(f"    - {name}: 主导 **Regime {dom}** (段内均概率 {p:.2f}, n={n} 月) (事实)")
    ann = _annual_dominant(fit)
    L.append("\n**年度主导 regime 时序** (每年 argmax 众数) (事实):")
    L.append("```")
    yrs = list(ann.index)
    line = "  ".join(f"{y}:r{ann.loc[y]}" for y in yrs)
    # wrap ~8 per line for readability
    cells = line.split("  ")
    for i in range(0, len(cells), 8):
        L.append("  ".join(cells[i:i + 8]))
    L.append("```")
    return L


def _sample_verdict(fit) -> tuple:
    """Per-sample structural verdict: is there a CREDIBLE, distinct regime
    structure? (interpretable, reason). Hard gate: the fit must be trustworthy
    (converged + no degenerate regime + ≥2 non-degenerate regimes); only then do
    we compare coefficients across the non-degenerate regimes."""
    # gate 1: trustworthiness (converged, no degenerate regime, ≥2 survive)
    if not fit.get("trustworthy"):
        why = []
        if fit.get("converged") is False:
            why.append("MLE 未收敛")
        if fit.get("degenerate"):
            why.append(f"含退化 regime({fit.get('n_nondegenerate')}/{fit['k_regimes']} non-deg)")
        if fit.get("n_nondegenerate", 0) < 2:
            why.append("non-degenerate regime <2(无切换)")
        return False, "拟合不可信:" + "、".join(why or ["未通过可信门"])
    # gate 2: distinct coefficients among the non-degenerate regimes
    spread = regime_coeff_spread(fit)  # nondegenerate_only=True by default
    exog = fit["exog"]
    any_sig = any(spread[nm]["sig_in_any"] for nm in exog)
    any_distinct = any(spread[nm]["distinct"] for nm in exog)
    interpretable = bool(any_sig and any_distinct)
    bits = [f"{fit['n_nondegenerate']} 个 non-degenerate 持久 regime、已收敛"]
    bits.append("有驱动在某些 regime 显著" if any_sig else "无驱动显著(|t|<1e6 口径)")
    bits.append("系数跨 regime 翻转/部分显著/量级显著有别(regime 条件锚特征)" if any_distinct
                else "系数跨 regime 无实质区别")
    return interpretable, "; ".join(bits)


def _build_report(df, notes, results, args) -> str:
    L = []
    L.append("# 黄金「锚 + 偏离」— PR #4: Markov-switching 回归(离散时变锚)\n")
    L.append(f"> 区间: {df.index.min().date()} .. {df.index.max().date()} "
             f"({len(df)} 个月)。对应 `docs/gold-anchor-vecm-spec.md` PR #4 节。\n")
    L.append("> 背景: PR #1–#3 否定了所有**线性 / 常参数 / 单断点**锚(双变量、三变量 Johansen、"
             "Gregory-Hansen 断点协整全否)。本 PR 测**离散时变锚**——世界在 2–3 个 regime 间跳,"
             "每个 regime 一套驱动系数。模型对**平稳的 Δln金价**(非 I(1) levels,避免伪回归)跑 "
             "`MarkovRegression`,自变量 Δln(债务/GDP)[+ Δ实利率],**系数与残差方差均 regime 依赖**。\n")

    L.append("\n## 0. 数据与口径\n")
    for k, v in notes.items():
        L.append(f"- **{k}**: {v}")

    L.append("\n## ⚠️ 方法坑(如实标注)\n")
    L.append("- **MS 对初值/局部最优敏感** (事实):似然多峰,本实现用默认 EM 起点 + "
             f"{N_RESTARTS} 次随机重启(`search_reps`,seeded 可复现)取最高似然;不同种子仍可能落不同局部解。")
    L.append("- **regime 标签不可识别** (事实):重排 regime 序号似然不变 → 不能按序号解读,"
             "必须**事后按系数**解释(本报告 regime 标签按显著驱动 + 波动率描述)。")
    L.append("- **退化解硬化(本轮重点)** (事实):真数据 MS 常返回退化解——某 regime 塌到单个离群点"
             "(σ²→0、期望持续≈1 月、系数 |t|~1e13 来自 SE≈0)。硬化:① 系数显著须 **|t|<1e6 且 p<α**"
             "(|t|>1e6 判未识别,不算显著);② regime 退化判据 = σ² < 全样本 Δln金价方差×1e-3 ∨ 期望持续<3 月 "
             "∨ 任一系数未识别;③ K 选择**作废含退化 regime 的解**,取最大「全 non-degenerate」可信 K"
             "(可能 K=1=无切换);④ 区分度/「regime 条件锚」只在 non-degenerate regime 间比较;"
             "⑤ 最优解仍未收敛/退化 → 报告如实标「不可信」,不给「获支持」。")
    L.append("- **短样本下 3 态可能过拟合** (推理):K 由 BIC 选,但 BIC 会靠退化 regime 撑大 K;"
             "故可信 K 用上面的退化门约束,而非纯 BIC。")
    L.append("- **干净实利率仅 2003+** (事实):全样本(1968+)只用 debt-only MS(实利率拼接代理脏);"
             "「实利率进 regime」的检验只在 post-2003 子样本做。")
    L.append("- **Δln 一阶差分丢长期信息** (推理):MS-回归测的是「短期驱动系数是否 regime 依赖」;"
             "让协整向量本身 regime 依赖的 **MS-VECM**(Krolzig 路线)计算重,留作进阶,本 PR 不做。")

    samples_interpretable = []
    for i, (label, _exog, _start) in enumerate(SAMPLES):
        res = results[i]
        L.append(f"\n## {i + 1}. 样本: {label}\n")
        if res["skip"]:
            L.append(f"- **skipped: {res['skip']}** (推理)")
            samples_interpretable.append(None)
            continue
        sel = res["sel"]
        L.append(f"- 差分后样本窗口 {res['window'][0]} .. {res['window'][1]} (n={res['frame_n']}) (事实)")
        if sel["errors"]:
            L.append(f"- K 拟合失败: {sel['errors']} (事实)")
        L.append(f"- BIC: {{K: BIC}} = { {k: round(v, 1) for k, v in sel['bic'].items()} }; "
                 f"BIC 最小 K={sel['bic_selected_k']};各 K 是否可信(收敛+无退化+≥2 non-deg regime): "
                 f"{sel['clean_k']} (事实)")
        if sel["selected_k"] is not None:
            L.append(f"- **可信选 K={sel['selected_k']}**(最大的可信 K;BIC 最小 K 若靠退化 regime 取胜则作废)(推理)")
        else:
            L.append("- ⚠️ **无任一 K 可信(全部未收敛/含退化 regime)→ 无可信 regime 切换,等价 K=1=无切换** (推理)")
        if sel["bic_selected_k"] is None:
            L.append("- **无任一 K 成功拟合 → 该样本无结论** (推理)")
            samples_interpretable.append(None)
            continue
        # display the trusted fit if one exists; else show the BIC fit for
        # diagnosis (clearly tagged degenerate), with a null verdict.
        trusted = sel["selected_k"] is not None
        chosen_k = sel["selected_k"] if trusted else sel["bic_selected_k"]
        fit = sel["fits"][chosen_k]
        if not trusted:
            L.append(f"- 下方展示 BIC 最小 K={chosen_k} 的解**仅作诊断**(不可信,见退化标注):")
        L.extend(_fmt_fit(fit))
        interp, reason = _sample_verdict(fit)
        samples_interpretable.append(interp)
        L.append(f"\n- **本样本裁决 (推理): "
                 f"{'存在可信、显著有别的 regime 结构' if interp else '无可信/显著有别的 regime 结构'}**"
                 f" — {reason}。")
        if trusted and fit["exog"] and "d_real_rate_10y" in fit["exog"]:
            sp = regime_coeff_spread(fit)["d_real_rate_10y"]
            where = ("部分 regime" if sp["regime_conditional"]
                     else "所有 regime" if sp["sig_in_all"] else "无 regime")
            rd = ("是(系数量级随 regime 显著有别)" if sp["distinct"] else "否(各 regime 系数无实质差异)")
            L.append(f"- **「长端是锚」的 regime 条件版 (推理)**: Δ实利率在 **{where}** 显著"
                     f"(可比 regime 数 {sp['n_compared']},显著 regime 数 {sp['n_sig']},符号翻转={sp['sign_flip']},"
                     f"量级显著有别={sp['magnitude_distinct']});regime 依赖={rd}。"
                     "→ 实利率作为驱动是否 regime 依赖,由此判定。")
        elif not trusted and fit["exog"] and "d_real_rate_10y" in fit["exog"]:
            L.append("- **「长端是锚」的 regime 条件版**: 拟合不可信 → **不报**实利率的 regime 落点(避免被退化 regime 机械触发)(推理)。")

    # ── overall verdict / kill condition ──
    L.append("\n## 核心裁决 — 离散时变锚是否成立\n")
    real = [v for v in samples_interpretable if v is not None]
    if not real:
        L.append("- **两样本均无法拟合/无结论 → 离散时变锚未获检验 (推理)。**")
    elif any(real):
        L.append("- **离散时变锚获支持 (推理/事实)**: 至少一个样本出现**可信**(收敛、无退化 regime、"
                 "≥2 个 non-degenerate 持久 regime)且系数显著有别的 regime 结构。"
                 "→ 「锚的配方随 regime 时变」有 MS 证据;此刻最可能的 regime 见上(年度主导时序末年)。")
        L.append("- **下一步 (推测)**: 进阶可上 MS-VECM(协整向量 regime 依赖)或 PR #5 的 TVP/Kalman "
                 "看漂移是**突跳**(支持 regime)还是**平滑**(支持 TVP)。")
    else:
        L.append("- ⚠️ **杀死条件触发 (spec PR #4)**: 两样本经退化门硬化后均**无可信、显著有别的 regime 结构** "
                 "(未收敛 / 退化 regime / 无切换;事实+推理) → **连离散时变锚也不成立**,延续 PR#1–#3 的全否。")
        L.append("- **下一步 (推测)**: 转 **TVP / Kalman 平滑漂移 (PR #5)**;若 TVP 也无结构,"
                 "则黄金「无稳定锚,只剩 regime/叙事」。")

    L.append("\n> Claim types: 系数估计值、BIC、转移矩阵、平滑概率为 (事实);regime 解读、"
             "「可解释/显著有别」、「长端是 regime 条件锚」等判读为 (推理);MS-VECM/TVP 路径为 (推测)。"
             "MS regime 标签不可识别,按系数解释,勿按序号。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1968-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

    print("[1/4] Building monthly gold-anchor panel...")
    panel = build_anchor_panel(start=args.start, end=args.end)
    df = panel.data
    if df.empty:
        raise ValueError(f"empty panel for start={args.start}, end={args.end}")
    print(f"  panel: {df.shape[0]} months × {df.shape[1]} cols, "
          f"{df.index.min().date()}..{df.index.max().date()}")

    print("[2/4] Fitting Markov-switching regressions (K=2,3 per sample)...")
    results = []
    for label, exog, start_override in SAMPLES:
        res = _fit_sample(df, panel.notes, exog, start_override)
        results.append(res)
        if res["skip"]:
            print(f"  {label}: skipped — {res['skip']}")
            continue
        sel = res["sel"]
        print(f"  {label}: n={res['frame_n']}, BIC={ {k: round(v, 1) for k, v in sel['bic'].items()} } "
              f"clean_k={sel['clean_k']} → trusted K={sel['selected_k']} "
              f"(BIC-min K={sel['bic_selected_k']}); errors={sel['errors']}")

    print("[3/4] Saving smoothed regime probabilities + meta...")
    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "start": args.start, "end": args.end, "k_values": list(K_VALUES),
            "n_restarts": N_RESTARTS, "seed": SEED, "samples": {}}
    for (label, _e, _s), res in zip(SAMPLES, results):
        sel = res.get("sel")
        if res["skip"] or sel is None or sel["bic_selected_k"] is None:
            meta["samples"][label] = {"skip": res.get("skip") or "no K fit"}
            continue
        trusted = sel["selected_k"] is not None
        chosen_k = sel["selected_k"] if trusted else sel["bic_selected_k"]
        fit = sel["fits"][chosen_k]
        slug = ("debt_only" if "debt-only" in label else "debt_real_post2003")
        sp_path = os.path.join(DATA_DIR, f"gold_anchor_markov_smoothed_{slug}.csv")
        fit["smoothed_probabilities"].to_csv(sp_path)
        meta["samples"][label] = {
            "displayed_k": fit["k_regimes"], "trusted": trusted,
            "trusted_k": sel["selected_k"], "bic_selected_k": sel["bic_selected_k"],
            "clean_k": sel["clean_k"], "no_switching": sel["no_switching"],
            "converged": fit.get("converged"), "degenerate": fit.get("degenerate"),
            "n_nondegenerate": fit.get("n_nondegenerate"),
            "bic": sel["bic"], "n_obs": fit["n_obs"], "window": res["window"],
            "expected_durations": fit["expected_durations"],
            "transition_matrix": fit["transition_matrix"],
            "smoothed_csv": os.path.basename(sp_path),
        }
        print(f"  saved → {sp_path}")
    with open(os.path.join(DATA_DIR, "gold_anchor_markov_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[4/4] Writing analysis report...")
    today = (args.end or datetime.now().strftime("%Y-%m-%d"))
    report = _build_report(df, panel.notes, results, args)
    out = os.path.join(ANALYSIS_DIR, f"gold_anchor_markov_{today}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved → {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
