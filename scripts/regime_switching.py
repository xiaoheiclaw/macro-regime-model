#!/usr/bin/env python3
"""
Macro Regime Attribution Engine - Phase 2 (fixed)
Hamilton Markov Regime Switching Model
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR
from lib.data_loader import load_merged

df = load_merged()

print(f"Data: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

report = []
report.append("# Phase 2: Markov Regime Switching Analysis")
report.append(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report.append("")

# ── Helper to safely extract params ────────────────────
def safe_params(res, k_regimes=2):
    """Extract regime params from fitted model"""
    params = {}
    param_names = res.params.index.tolist() if hasattr(res.params, 'index') else []
    for name in param_names:
        params[name] = float(res.params[name])
    return params

# ── Model 1: Oil → 10Y Yield ───────────────────────────
report.append("## Model 1: Oil Price → 10Y Yield (2-Regime Switching)")
report.append("")

oil_ret = df["WTI_crude"].pct_change().dropna() * 100
yield_chg = df["US10Y_yield"].diff().dropna() * 100
common = oil_ret.index.intersection(yield_chg.index)
oil_ret_a = oil_ret.loc[common].dropna()
yield_chg_a = yield_chg.loc[common].dropna()
common2 = oil_ret_a.index.intersection(yield_chg_a.index)
oil_ret_a = oil_ret_a.loc[common2]
yield_chg_a = yield_chg_a.loc[common2]

print(f"Model 1: {len(oil_ret_a)} obs")

try:
    mod1 = MarkovRegression(yield_chg_a.values, k_regimes=2, exog=oil_ret_a.values,
                            switching_variance=True, switching_exog=True)
    res1 = mod1.fit(maxiter=500, disp=False)
    p = safe_params(res1)

    print(f"Model 1 params: {list(p.keys())}")

    for regime in range(2):
        alpha_key = [k for k in p if f'const' in k and f'[{regime}]' in k]
        beta_key = [k for k in p if 'x1' in k and f'[{regime}]' in k]
        sigma_key = [k for k in p if 'sigma' in k and f'[{regime}]' in k]

        alpha = p[alpha_key[0]] if alpha_key else None
        beta = p[beta_key[0]] if beta_key else None
        sigma = p[sigma_key[0]] ** 0.5 if sigma_key else None

        report.append(f"### Regime {regime}")
        if alpha is not None: report.append(f"- α (drift): {alpha:.4f} bp/day")
        if beta is not None: report.append(f"- β (oil→yield): {beta:.4f} bp per 1% oil move")
        if sigma is not None: report.append(f"- σ (vol): {sigma:.4f}")

        if beta is not None:
            if beta > 0.5:
                report.append(f"- 🔴 **通胀传导regime**: 油涨→收益率涨")
            elif beta < -0.5:
                report.append(f"- 🟢 **增长恐慌regime**: 油涨→收益率跌（衰退定价）")
            else:
                report.append(f"- 🟡 **弱关系regime**: 油价对收益率影响不大")
        report.append("")

    # Smoothed probabilities
    sm = res1.smoothed_marginal_probabilities
    if isinstance(sm, np.ndarray):
        regime_probs = pd.DataFrame(sm, index=yield_chg_a.index, columns=[f"P_R{i}" for i in range(2)])
    else:
        regime_probs = pd.DataFrame(sm.values if hasattr(sm, 'values') else sm,
                                     index=yield_chg_a.index, columns=[f"P_R{i}" for i in range(2)])

    report.append("### Current Regime Probability")
    curr = regime_probs.iloc[-1]
    for col in regime_probs.columns:
        report.append(f"- {col}: {curr[col]:.1%}")
    report.append(f"- **当前最可能: Regime {curr.values.argmax()}**")
    report.append("")

    # Transition matrix
    tp_keys = [k for k in p if 'p[' in k]
    if tp_keys:
        report.append("### Transition Matrix")
        for k in sorted(tp_keys):
            prob = p[k]
            duration = 1/(1-prob) if prob < 1 else float('inf')
            report.append(f"- {k} = {prob:.3f} → avg duration ≈ {duration:.0f} days")
        report.append("")

    # Recent regime history
    report.append("### Recent Regime History (last 40 trading days)")
    report.append("```")
    for i in range(-40, 0):
        if abs(i) <= len(regime_probs):
            row = regime_probs.iloc[i]
            date = regime_probs.index[i]
            dominant = row.values.argmax()
            bar_len = int(row.iloc[1] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            report.append(f"{date.strftime('%m/%d')} R{dominant} |{bar}| P(R1)={row.iloc[1]:.0%}")
    report.append("```")
    report.append("")

    regime_probs.to_csv(os.path.join(DATA_DIR, "oil_bond_regime_probs.csv"))

except Exception as e:
    report.append(f"**Model 1 error: {e}**")
    import traceback; traceback.print_exc()
    report.append("")

# ── Model 2: SPX → Bond ────────────────────────────────
report.append("## Model 2: SPX → 10Y Yield (2-Regime Switching)")
report.append("")

if "SPX" in df.columns and df["SPX"].notna().sum() > 100:
    spx_ret = df["SPX"].pct_change().dropna() * 100
    common_s = spx_ret.index.intersection(yield_chg.index)
    spx_a = spx_ret.loc[common_s].dropna()
    ychg_a = yield_chg.loc[common_s].dropna()
    common_s2 = spx_a.index.intersection(ychg_a.index)
    spx_a = spx_a.loc[common_s2]
    ychg_a = ychg_a.loc[common_s2]

    try:
        mod2 = MarkovRegression(ychg_a.values, k_regimes=2, exog=spx_a.values,
                                switching_variance=True, switching_exog=True)
        res2 = mod2.fit(maxiter=500, disp=False)
        p2 = safe_params(res2)

        for regime in range(2):
            beta_key = [k for k in p2 if 'x1' in k and f'[{regime}]' in k]
            sigma_key = [k for k in p2 if 'sigma' in k and f'[{regime}]' in k]
            beta = p2[beta_key[0]] if beta_key else None
            sigma = p2[sigma_key[0]] ** 0.5 if sigma_key else None

            report.append(f"### Regime {regime}")
            if beta is not None: report.append(f"- β (SPX→yield): {beta:.4f}")
            if sigma is not None: report.append(f"- σ: {sigma:.4f}")
            if beta is not None:
                if beta > 0.3:
                    report.append(f"- 📈 **Growth regime**: 股涨→收益率涨（同向，增长乐观）")
                elif beta < -0.3:
                    report.append(f"- 📉 **Decoupled regime**: 股涨→收益率跌（反向，流动性驱动）")
                else:
                    report.append(f"- ⚖️ **Mixed regime**")
            report.append("")

        sm2 = res2.smoothed_marginal_probabilities
        rp2 = pd.DataFrame(sm2 if isinstance(sm2, np.ndarray) else sm2.values,
                           index=ychg_a.index, columns=[f"P_R{i}" for i in range(2)])
        curr2 = rp2.iloc[-1]
        report.append(f"### Current: Regime {curr2.values.argmax()} (P={curr2.max():.1%})")
        report.append("")

        rp2.to_csv(os.path.join(DATA_DIR, "spx_bond_regime_probs.csv"))

    except Exception as e:
        report.append(f"**Model 2 error: {e}**")
        import traceback; traceback.print_exc()

# ── Model 3: Yield Level Regime ─────────────────────────
report.append("")
report.append("## Model 3: 10Y Yield Level Regime (AR(1) Switching)")
report.append("")

try:
    yield_level = df["US10Y_yield"].dropna()
    mod3 = MarkovAutoregression(yield_level.values, k_regimes=2, order=1,
                                 switching_ar=False, switching_variance=True)
    res3 = mod3.fit(maxiter=500, disp=False)
    p3 = safe_params(res3)

    for regime in range(2):
        const_key = [k for k in p3 if 'const' in k and f'[{regime}]' in k]
        sigma_key = [k for k in p3 if 'sigma' in k and f'[{regime}]' in k]
        mu = p3[const_key[0]] if const_key else None
        sigma = p3[sigma_key[0]] ** 0.5 if sigma_key else None

        report.append(f"### Regime {regime}")
        if mu is not None: report.append(f"- Mean level: {mu:.3f}%")
        if sigma is not None: report.append(f"- Vol: {sigma:.4f}")
        if mu is not None:
            if mu > 3.5: report.append(f"- 🔴 **高利率regime**")
            elif mu < 2.0: report.append(f"- 🟢 **低利率regime**")
            else: report.append(f"- 🟡 **中性regime**")
        report.append("")

    sm3 = res3.smoothed_marginal_probabilities
    rp3 = pd.DataFrame(sm3 if isinstance(sm3, np.ndarray) else sm3.values,
                       index=yield_level.index, columns=[f"P_R{i}" for i in range(2)])
    curr3 = rp3.iloc[-1]
    report.append(f"### Current: Regime {curr3.values.argmax()} (P={curr3.max():.1%})")
    report.append("")

except Exception as e:
    report.append(f"**Model 3 error: {e}**")
    import traceback; traceback.print_exc()

# ── Historical Parallels ───────────────────────────────
report.append("")
report.append("## Historical Parallels: Stock-Bond Correlation Flips")
report.append("")

if "SPX" in df.columns:
    spx_r = df["SPX"].pct_change()
    y_chg = df["US10Y_yield"].diff()
    rc = spx_r.rolling(20).corr(y_chg).dropna()
    corr_flip = rc.diff(20)
    big_flips = corr_flip[corr_flip < -0.8].sort_values()

    report.append("### Top Correlation Crashes (20d change < -0.8)")
    report.append("| Date | Δ Corr (20d) | Level | Context |")
    report.append("|---|---|---|---|")

    seen = set()
    for date, chg in big_flips.head(20).items():
        mk = date.strftime("%Y-%m")
        if mk in seen: continue
        seen.add(mk)
        level = rc.loc[date]
        if date.year == 2020 and date.month <= 4: ctx = "🦠 COVID crash"
        elif date.year == 2022 and date.month <= 3: ctx = "🦅 Fed hawkish pivot"
        elif date.year == 2022 and date.month >= 9: ctx = "🇬🇧 UK gilt crisis"
        elif date.year == 2023 and date.month in [3,4]: ctx = "🏦 SVB / regional banks"
        elif date.year == 2023 and date.month in [7,8]: ctx = "📊 Debt ceiling / Fitch downgrade"
        elif date.year == 2025: ctx = "🌐 Trade war / tariffs"
        elif date.year == 2026: ctx = "🇮🇷 Iran conflict / fiscal"
        else: ctx = ""
        report.append(f"| {date.strftime('%Y-%m-%d')} | {chg:.2f} | {level:.2f} | {ctx} |")

    report.append("")

    # Current rank
    current_flip = corr_flip.iloc[-1] if len(corr_flip) > 0 else 0
    rank = (corr_flip < current_flip).sum()
    total = len(corr_flip)
    percentile = rank / total * 100
    report.append(f"**当前翻转: {current_flip:.2f}，历史百分位: {percentile:.1f}%（越低越极端）**")
    report.append("")

# ── Oil-Bond correlation flip ──────────────────────────
report.append("## Oil-Bond Correlation Regime History")
report.append("")

oil_r = df["WTI_crude"].pct_change()
y_chg2 = df["US10Y_yield"].diff()
ob_corr = oil_r.rolling(20).corr(y_chg2).dropna()

report.append("### By Year")
report.append("| Year | Mean Corr | % Days Negative | Interpretation |")
report.append("|---|---|---|---|")
for year in range(2020, 2027):
    mask = ob_corr.index.year == year
    if mask.sum() > 0:
        subset = ob_corr[mask]
        neg_pct = (subset < 0).mean() * 100
        mean_c = subset.mean()
        if neg_pct > 50:
            interp = "增长/衰退主导"
        elif neg_pct > 30:
            interp = "混合"
        else:
            interp = "通胀主导"
        report.append(f"| {year} | {mean_c:.3f} | {neg_pct:.0f}% | {interp} |")

report.append("")

# Last 5 days detail
report.append("### Last 10 Trading Days (daily co-movement)")
report.append("| Date | Oil Δ% | 10Y Δbp | Same Direction? |")
report.append("|---|---|---|---|")
for i in range(-10, 0):
    date = df.index[i]
    o_chg = oil_r.iloc[i] * 100 if pd.notna(oil_r.iloc[i]) else 0
    y_ch = y_chg2.iloc[i] * 100 if pd.notna(y_chg2.iloc[i]) else 0
    same = "✓ 同向" if (o_chg > 0 and y_ch > 0) or (o_chg < 0 and y_ch < 0) else "✗ 反向" if abs(o_chg) > 0.1 and abs(y_ch) > 0.1 else "—"
    report.append(f"| {date.strftime('%m/%d')} | {o_chg:+.2f}% | {y_ch:+.1f}bp | {same} |")

report.append("")

# ── Final Synthesis ─────────────────────────────────────
report.append("## 🎯 综合结论")
report.append("")

output = "\n".join(report)
out_path = os.path.join(ANALYSIS_DIR, "regime_switching_20260331.md")
with open(out_path, "w") as f:
    f.write(output)

print("\n" + "=" * 60)
print(output)
print(f"\n✓ Saved to {out_path}")
