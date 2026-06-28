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
    L.append(f"- 选定 **K={fit['k_regimes']}** (BIC={fit['bic']:.1f}, AIC={fit['aic']:.1f}, "
             f"llf={fit['llf']:.1f}, n_obs={fit['n_obs']}, "
             f"switching_variance={fit.get('switching_variance')}, "
             f"n_restarts={fit.get('n_restarts')}, converged={fit['converged']}) (事实)\n")
    L.append("**各 regime 系数表** (Δln金价 ~ 各驱动的 Δ;系数/方差均 regime 依赖):\n")
    for r in fit["regimes"]:
        L.append(f"- **Regime {r['regime']}** — {_regime_label(r, exog)} "
                 f"(期望持续 {r['expected_duration']:.1f} 月) (事实):")
        for nm, cell in r["coeffs"].items():
            sig = "**显著**" if cell["significant"] else "不显著"
            L.append(f"    - {DRIVER_LABEL.get(nm, nm)}: coef={cell['coef']:+.4f} "
                     f"(t={cell['t']:.2f}, p={cell['p']:.3f}, {sig})")
    tm = np.asarray(fit["transition_matrix"])
    L.append("\n**转移矩阵** P[i,j]=P(s_t=i | s_{t-1}=j),列和=1 (事实):")
    L.append("```")
    hdr = "from→  " + "  ".join(f"r{j}" for j in range(fit["k_regimes"]))
    L.append(hdr)
    for i in range(fit["k_regimes"]):
        L.append(f"to r{i}: " + "  ".join(f"{tm[i, j]:.3f}" for j in range(fit["k_regimes"])))
    L.append("```")
    L.append(f"- 期望持续期 (月): {[round(d, 1) for d in fit['expected_durations']]} (事实)")

    # regime-conditional anchor read
    spread = regime_coeff_spread(fit)
    L.append("\n**驱动的 regime 区分度** (是否「regime 条件锚」):")
    for nm in exog:
        s = spread[nm]
        L.append(f"    - {DRIVER_LABEL.get(nm, nm)}: 跨 regime 系数={[round(c, 3) for c in s['coefs']]}, "
                 f"符号翻转={s['sign_flip']}, 显著 regime 数={s['n_sig']}/{fit['k_regimes']}, "
                 f"regime条件(部分显著)={s['regime_conditional']} (事实/推理)")

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
    """Per-sample structural verdict: is there an interpretable, significantly-
    distinct regime structure? (interpretable, reason)."""
    spread = regime_coeff_spread(fit)
    exog = fit["exog"]
    any_sig = any(spread[nm]["sig_in_any"] for nm in exog)
    # distinct = sign flip OR only-some-regimes significant OR a beyond-noise
    # magnitude gap (a driver significant in both regimes but with very different
    # magnitude is still a different anchor recipe per regime).
    any_distinct = any(spread[nm]["distinct"] for nm in exog)
    # persistence: at least one regime expected to last > 6 months (not noise)
    persistent = max(fit["expected_durations"]) > 6.0
    interpretable = bool(any_sig and any_distinct and persistent)
    bits = []
    bits.append("有驱动在某些 regime 显著" if any_sig else "无驱动在任何 regime 显著")
    bits.append("系数跨 regime 翻转/部分显著/量级显著有别(regime 条件锚特征)" if any_distinct
                else "系数跨 regime 无实质区别")
    bits.append(f"最长期望持续 {max(fit['expected_durations']):.1f} 月"
                + ("(够持久)" if persistent else "(过短,疑噪声)"))
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
    L.append("- **短样本下 3 态可能过拟合** (推理):K 由 BIC + 经济可读性选;K=3 若 BIC 不降或某 regime "
             "持续期过短/无显著驱动,判过拟合,退回 K=2。")
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
        L.append(f"- BIC 选阶: {{K: BIC}} = "
                 f"{ {k: round(v, 1) for k, v in sel['bic'].items()} } → 选 **K={sel['selected_k']}** "
                 "(BIC 最小;并核对经济可读性) (事实/推理)")
        if sel["selected_k"] is None:
            L.append("- **无任一 K 成功拟合 → 该样本无结论** (推理)")
            samples_interpretable.append(None)
            continue
        fit = sel["fits"][sel["selected_k"]]
        L.extend(_fmt_fit(fit))
        interp, reason = _sample_verdict(fit)
        samples_interpretable.append(interp)
        L.append(f"\n- **本样本裁决 (推理): "
                 f"{'存在可解释、显著有别的 regime 结构' if interp else '未见可解释/显著有别的 regime 结构'}**"
                 f" — {reason}。")
        if fit["exog"] and "d_real_rate_10y" in fit["exog"]:
            sp = regime_coeff_spread(fit)["d_real_rate_10y"]
            where = ("部分 regime" if sp["regime_conditional"]
                     else "所有 regime" if sp["sig_in_all"] else "无 regime")
            rd = ("是(系数量级随 regime 显著有别)" if sp["distinct"] else "否(各 regime 系数无实质差异)")
            L.append(f"- **「长端是锚」的 regime 条件版 (推理)**: Δ实利率在 **{where}** 显著"
                     f"(显著 regime 数 {sp['n_sig']}/{fit['k_regimes']},符号翻转={sp['sign_flip']},"
                     f"量级显著有别={sp['magnitude_distinct']});regime 依赖={rd}。"
                     "→ 实利率作为驱动是否 regime 依赖,由此判定。")

    # ── overall verdict / kill condition ──
    L.append("\n## 核心裁决 — 离散时变锚是否成立\n")
    real = [v for v in samples_interpretable if v is not None]
    if not real:
        L.append("- **两样本均无法拟合/无结论 → 离散时变锚未获检验 (推理)。**")
    elif any(real):
        L.append("- **离散时变锚获支持 (推理/事实)**: 至少一个样本出现可解释、系数显著有别的 regime 结构"
                 "(系数跨 regime 翻转或 regime 条件显著、regime 持久)。"
                 "→ 「锚的配方随 regime 时变」有 MS 证据;此刻最可能的 regime 见上(年度主导时序末年)。")
        L.append("- **下一步 (推测)**: 进阶可上 MS-VECM(协整向量 regime 依赖)或 PR #5 的 TVP/Kalman "
                 "看漂移是**突跳**(支持 regime)还是**平滑**(支持 TVP)。")
    else:
        L.append("- ⚠️ **杀死条件触发 (spec PR #4)**: 两样本均**找不到**可解释、系数显著有别的 regime 结构 "
                 "(事实/推理) → **连离散时变锚也不成立**。")
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
              f"→ K={sel['selected_k']}; errors={sel['errors']}")

    print("[3/4] Saving smoothed regime probabilities + meta...")
    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "start": args.start, "end": args.end, "k_values": list(K_VALUES),
            "n_restarts": N_RESTARTS, "seed": SEED, "samples": {}}
    for (label, _e, _s), res in zip(SAMPLES, results):
        if res["skip"] or res["sel"]["selected_k"] is None:
            meta["samples"][label] = {"skip": res.get("skip") or "no K fit"}
            continue
        fit = res["sel"]["fits"][res["sel"]["selected_k"]]
        slug = ("debt_only" if "debt-only" in label else "debt_real_post2003")
        sp_path = os.path.join(DATA_DIR, f"gold_anchor_markov_smoothed_{slug}.csv")
        fit["smoothed_probabilities"].to_csv(sp_path)
        meta["samples"][label] = {
            "selected_k": fit["k_regimes"], "bic": res["sel"]["bic"],
            "n_obs": fit["n_obs"], "window": res["window"],
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
