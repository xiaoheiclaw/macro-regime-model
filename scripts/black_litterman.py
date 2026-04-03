#!/usr/bin/env python3
"""
Phase 4a: Black-Litterman Scenario-Based Asset Allocation
Input: scenario probabilities → Output: optimal weights
"""
import os, sys
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR
from lib.data_loader import load_merged
from lib.universe import ASSETS, ASSET_NAMES, MARKET_WEIGHTS
from lib.tuning import load_tuning_params

# ── Load data ───────────────────────────────────────────
df = load_merged()

# Compute returns (for yields, use negative of yield change as "bond return" proxy)
returns = pd.DataFrame(index=df.index)
for a in ASSETS:
    if a in df.columns:
        if a == "US10Y_yield":
            # Bond return proxy: yield down = positive return (duration ~8)
            returns[a] = -df[a].diff() * 8 / 100  # rough bond return
        else:
            returns[a] = df[a].pct_change()

returns = returns.dropna()
available = [a for a in ASSETS if a in returns.columns and returns[a].notna().sum() > 200]
returns = returns[available]

print(f"Asset universe: {available}")
print(f"Return data: {len(returns)} days")

# ── Historical stats ────────────────────────────────────
# Use recent 1Y for covariance (more relevant than full history)
recent = returns.iloc[-252:]
Sigma = recent.cov() * 252  # annualized covariance
mu_hist = recent.mean() * 252  # annualized returns

# Market cap weights (rough proxies for equilibrium)
w_mkt = np.array([MARKET_WEIGHTS.get(a, 0.1) for a in available])
w_mkt = w_mkt / w_mkt.sum()

# Risk aversion parameter
delta = 2.5

# Equilibrium excess returns (implied by market weights)
Pi = delta * Sigma.values @ w_mkt

# ── Scenario Views ──────────────────────────────────────
# Try to load TimesFM-generated views; fall back to manual scenarios
_timesfm_path = os.path.join(DATA_DIR, "timesfm_views.json")
_use_timesfm = False

if os.path.exists(_timesfm_path):
    import time as _time
    _age_hours = (_time.time() - os.path.getmtime(_timesfm_path)) / 3600
    if _age_hours < 24:
        with open(_timesfm_path) as _f:
            _tfm = json.load(_f)
        _scenario = _tfm["scenario"]
        SCENARIOS = {_scenario["name"]: {"prob": _scenario["prob"], "views": _scenario["views"]}}
        # Use TimesFM confidence to build Omega later
        _tfm_confidence = _scenario.get("confidence", {})
        _use_timesfm = True
        print(f"✓ Using TimesFM views (age: {_age_hours:.1f}h)")
    else:
        print(f"⚠ TimesFM views stale ({_age_hours:.0f}h old), using manual scenarios")

if not _use_timesfm:
    _tfm_confidence = {}

# Manual scenarios (used as fallback when TimesFM views unavailable)
if not _use_timesfm:
    SCENARIOS = {
        "A: 冲突降级 (油回落)": {
            "prob": 0.15,
            "views": {
                "WTI_crude": -0.25, "US10Y_yield": 0.15, "SPX": 0.10,
                "Gold": -0.05, "BTC": 0.05, "DXY": 0.03,
            }
        },
        "B: 冲突持续 + 美财政刺激": {
            "prob": 0.35,
            "views": {
                "WTI_crude": 0.10, "US10Y_yield": 0.05, "SPX": -0.05,
                "Gold": 0.20, "BTC": 0.15, "DXY": -0.08,
            }
        },
        "C: 冲突升级 + 无财政": {
            "prob": 0.20,
            "views": {
                "WTI_crude": 0.30, "US10Y_yield": -0.10, "SPX": -0.20,
                "Gold": 0.10, "BTC": -0.15, "DXY": 0.10,
            }
        },
        "D: 全球财政联动 (文章核心)": {
            "prob": 0.30,
            "views": {
                "WTI_crude": 0.05, "US10Y_yield": 0.08, "SPX": 0.00,
                "Gold": 0.30, "BTC": 0.25, "DXY": -0.15,
            }
        },
    }

# ── Black-Litterman implementation ──────────────────────
def black_litterman(Pi, Sigma, P, Q, Omega, tau=0.05):
    """
    Pi: Nx1 equilibrium returns
    Sigma: NxN covariance
    P: KxN pick matrix (views)
    Q: Kx1 view returns
    Omega: KxK uncertainty of views
    """
    N = len(Pi)
    tau_Sigma = tau * Sigma
    
    # BL posterior
    inv_tau_Sigma = np.linalg.inv(tau_Sigma)
    inv_Omega = np.linalg.inv(Omega)
    
    M = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
    bl_returns = M @ (inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q)
    bl_cov = M + Sigma
    
    return bl_returns, bl_cov

# Probability-weighted views
n_assets = len(available)
all_views_returns = []
all_views_weights = []

for scenario_name, scenario in SCENARIOS.items():
    prob = scenario["prob"]
    views = scenario["views"]
    
    # Build expected returns for this scenario
    scenario_returns = np.zeros(n_assets)
    for i, asset in enumerate(available):
        if asset in views:
            scenario_returns[i] = views[asset]
        else:
            scenario_returns[i] = 0
    
    all_views_returns.append(scenario_returns * prob)

# Probability-weighted expected returns
weighted_returns = np.sum(all_views_returns, axis=0)

# Create view matrix: one absolute view per asset
P = np.eye(n_assets)
Q = weighted_returns

# View uncertainty
_tp = load_tuning_params()
_omega_scale = _tp.get("omega_scale", 1.0)
if _use_timesfm and _tfm_confidence:
    # TimesFM: use q10-q90 band width as uncertainty (wider band = less confident)
    view_var = np.array([_tfm_confidence.get(a, 0.05) ** 2 for a in available])
    Omega = np.diag(view_var * _omega_scale + 0.001)
else:
    # Manual scenarios: proportional to disagreement across scenarios
    scenario_rets = np.array([
        [SCENARIOS[s]["views"].get(a, 0) for a in available]
        for s in SCENARIOS
    ])
    scenario_probs = np.array([SCENARIOS[s]["prob"] for s in SCENARIOS])
    view_var = np.average((scenario_rets - weighted_returns) ** 2, weights=scenario_probs, axis=0)
    Omega = np.diag(view_var + 0.001)

# Run BL
bl_returns, bl_cov = black_litterman(Pi, Sigma.values, P, Q, Omega)

# Optimal weights (mean-variance)
inv_bl_cov = np.linalg.inv(delta * bl_cov)
w_optimal = inv_bl_cov @ bl_returns
# Normalize to sum to 1 (allow short)
w_optimal_norm = w_optimal / np.abs(w_optimal).sum()

# Long-only version
w_long = np.maximum(w_optimal, 0)
w_long = w_long / w_long.sum() if w_long.sum() > 0 else w_mkt

# ── VaR / CVaR Functions ────────────────────────────────
Z_95 = norm.ppf(0.95)          # 1.6449
PHI_Z95 = norm.pdf(Z_95)      # φ(z_0.95)
ALPHA = 0.05                   # 1 - confidence

def parametric_var(w, mu, cov, confidence=0.95):
    """Annualised Parametric VaR (positive = loss)."""
    mu_p = w @ mu
    sigma_p = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)
    return -mu_p + z * sigma_p

def parametric_cvar(w, mu, cov, confidence=0.95):
    """Annualised Parametric CVaR (Expected Shortfall)."""
    mu_p = w @ mu
    sigma_p = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)
    phi_z = norm.pdf(z)
    alpha = 1 - confidence
    return -mu_p + sigma_p * phi_z / alpha

def portfolio_stats(w, mu, cov):
    """Return dict with ret, vol, sharpe, var95, cvar95."""
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = ret / vol if vol > 0 else 0.0
    var95 = parametric_var(w, mu, cov)
    cvar95 = parametric_cvar(w, mu, cov)
    return dict(ret=ret, vol=vol, sharpe=sharpe, var95=var95, cvar95=cvar95)

# ── CVaR-constrained optimisation ───────────────────────
CVAR_THRESHOLD = 0.15  # 15% annual CVaR cap

def min_cvar_optimise(mu, cov, cvar_threshold=CVAR_THRESHOLD):
    """Minimise CVaR subject to long-only, sum=1, CVaR < threshold."""
    n = len(mu)
    
    def objective(w):
        return parametric_cvar(w, mu, cov)
    
    def cvar_constraint(w):
        return cvar_threshold - parametric_cvar(w, mu, cov)
    
    constraints = [
        {"type": "eq",  "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": cvar_constraint},
    ]
    bounds = [(0.0, 1.0)] * n
    
    # Try multiple starting points for robustness
    best_result = None
    best_cvar = np.inf
    x0_candidates = [
        np.ones(n) / n,                       # equal weight
        w_mkt.copy(),                          # market weight
        w_long.copy(),                         # BL long-only
    ]
    
    for x0 in x0_candidates:
        res = minimize(objective, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
        if res.success and res.fun < best_cvar:
            best_cvar = res.fun
            best_result = res
    
    if best_result is None:
        # Fallback: relax threshold to 25% and retry
        constraints[1] = {"type": "ineq", "fun": lambda w: 0.25 - parametric_cvar(w, mu, cov)}
        best_result = minimize(objective, np.ones(n) / n, method="SLSQP",
                               bounds=bounds, constraints=constraints,
                               options={"maxiter": 1000, "ftol": 1e-12})
    
    w_opt = best_result.x
    w_opt = np.maximum(w_opt, 0)
    w_opt /= w_opt.sum()
    return w_opt

w_min_cvar = min_cvar_optimise(bl_returns, bl_cov)
print(f"\nMin-CVaR weights computed.")

# ── Portfolio comparison stats ──────────────────────────
stats_mkt    = portfolio_stats(w_mkt,      bl_returns, bl_cov)
stats_sharpe = portfolio_stats(w_long,     bl_returns, bl_cov)
stats_cvar   = portfolio_stats(w_min_cvar, bl_returns, bl_cov)

print(f"  Market   → Ret={stats_mkt['ret']*100:+.1f}% Vol={stats_mkt['vol']*100:.1f}% VaR={stats_mkt['var95']*100:.1f}% CVaR={stats_mkt['cvar95']*100:.1f}%")
print(f"  BL Sharpe→ Ret={stats_sharpe['ret']*100:+.1f}% Vol={stats_sharpe['vol']*100:.1f}% VaR={stats_sharpe['var95']*100:.1f}% CVaR={stats_sharpe['cvar95']*100:.1f}%")
print(f"  BL CVaR  → Ret={stats_cvar['ret']*100:+.1f}% Vol={stats_cvar['vol']*100:.1f}% VaR={stats_cvar['var95']*100:.1f}% CVaR={stats_cvar['cvar95']*100:.1f}%")

# ── Efficient Frontier (dual-objective) ─────────────────
def compute_efficient_frontier(mu, cov, n_points=30):
    """Generate frontier: for each target return, find min-vol and record CVaR."""
    n = len(mu)
    ret_min = 0.0
    ret_max = float(np.max(mu))
    target_rets = np.linspace(ret_min, ret_max * 0.95, n_points)
    
    rows = []
    for t_ret in target_rets:
        def obj_vol(w):
            return np.sqrt(w @ cov @ w)
        
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=t_ret: w @ mu - tr},
        ]
        bounds = [(0.0, 1.0)] * n
        
        res = minimize(obj_vol, np.ones(n) / n, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
        if res.success:
            w_opt = res.x
            w_opt = np.maximum(w_opt, 0)
            w_opt /= w_opt.sum()
            vol = np.sqrt(w_opt @ cov @ w_opt)
            cvar = parametric_cvar(w_opt, mu, cov)
            wdict = {available[i]: round(float(w_opt[i]), 4) for i in range(n)}
            rows.append({
                "target_return": round(float(t_ret), 6),
                "optimal_vol": round(float(vol), 6),
                "optimal_cvar": round(float(cvar), 6),
                "weights_json": json.dumps(wdict),
            })
    return pd.DataFrame(rows)

frontier_df = compute_efficient_frontier(bl_returns, bl_cov)
frontier_path = os.path.join(DATA_DIR, "bl_efficient_frontier.csv")
frontier_df.to_csv(frontier_path, index=False)
print(f"\n✓ Efficient frontier ({len(frontier_df)} pts) → {frontier_path}")

# ── Report ──────────────────────────────────────────────
report = []
report.append("# Phase 4: Black-Litterman Scenario Allocation")
report.append(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report.append("")

report.append("## Scenario Assumptions")
report.append("")
report.append("| Scenario | Prob | SPX | Bond | Gold | Oil | BTC | USD |")
report.append("|---|---|---|---|---|---|---|---|")
for name, s in SCENARIOS.items():
    v = s["views"]
    report.append(f"| {name} | {s['prob']:.0%} | {v.get('SPX',0):+.0%} | {v.get('US10Y_yield',0):+.0%} | {v.get('Gold',0):+.0%} | {v.get('WTI_crude',0):+.0%} | {v.get('BTC',0):+.0%} | {v.get('DXY',0):+.0%} |")

report.append("")
report.append("## Probability-Weighted Expected Returns")
report.append("")
report.append("| Asset | Equilibrium | Scenario View | BL Posterior |")
report.append("|---|---|---|---|")
for i, a in enumerate(available):
    name = ASSET_NAMES.get(a, a)
    report.append(f"| {name} | {Pi[i]*100:+.1f}% | {Q[i]*100:+.1f}% | {bl_returns[i]*100:+.1f}% |")

report.append("")
report.append("## Optimal Allocation")
report.append("")
report.append("| Asset | Market Wt | BL Max Sharpe | BL Min-CVaR |")
report.append("|---|---|---|---|")
for i, a in enumerate(available):
    name = ASSET_NAMES.get(a, a)
    report.append(f"| {name} | {w_mkt[i]:.0%} | {w_long[i]:.0%} | {w_min_cvar[i]:.0%} |")

# Portfolio stats (BL Max Sharpe)
port_ret = w_long @ bl_returns * 100
port_vol = np.sqrt(w_long @ bl_cov @ w_long) * 100
sharpe = port_ret / port_vol if port_vol > 0 else 0

report.append("")
report.append(f"**BL Max Sharpe — Return: {port_ret:+.1f}%, Vol: {port_vol:.1f}%, Sharpe: {sharpe:.2f}**")

# ── VaR / CVaR Comparison ──────────────────────────────
report.append("")
report.append("## 📊 尾部风险对比 (VaR & CVaR)")
report.append("")
report.append("| 指标 | Market Weight | BL Max Sharpe | BL Min-CVaR |")
report.append("|---|---|---|---|")
labels = [
    ("Expected Return", "ret", 100, "+.1f", "%"),
    ("Volatility", "vol", 100, ".1f", "%"),
    ("Sharpe Ratio", "sharpe", 1, ".2f", ""),
    ("VaR (95%)", "var95", 100, ".1f", "%"),
    ("CVaR (95%)", "cvar95", 100, ".1f", "%"),
]
for label, key, scale, fmt, suffix in labels:
    v1 = stats_mkt[key] * scale
    v2 = stats_sharpe[key] * scale
    v3 = stats_cvar[key] * scale
    report.append(f"| {label} | {v1:{fmt}}{suffix} | {v2:{fmt}}{suffix} | {v3:{fmt}}{suffix} |")

report.append("")
report.append(f"> **CVaR 约束阈值: {CVAR_THRESHOLD:.0%}**  ")
report.append(f"> VaR = −μ_p + z_{{0.95}} · σ_p ；CVaR = −μ_p + σ_p · φ(z_{{0.95}}) / α")
report.append("")

# Key takeaways
report.append("## 🎯 配置要点")
report.append("")

# Sort by weight for BL Max Sharpe
sorted_idx = np.argsort(w_long)[::-1]
report.append("**BL Max Sharpe 按权重排序：**")
for i in sorted_idx:
    if w_long[i] > 0.01:
        name = ASSET_NAMES.get(available[i], available[i])
        report.append(f"1. **{name}: {w_long[i]:.0%}** (BL return: {bl_returns[i]*100:+.1f}%)")

report.append("")

# Sort by weight for Min-CVaR
sorted_cvar_idx = np.argsort(w_min_cvar)[::-1]
report.append("**BL Min-CVaR 按权重排序：**")
for i in sorted_cvar_idx:
    if w_min_cvar[i] > 0.01:
        name = ASSET_NAMES.get(available[i], available[i])
        report.append(f"1. **{name}: {w_min_cvar[i]:.0%}** (BL return: {bl_returns[i]*100:+.1f}%)")

report.append("")
report.append("**核心逻辑：** 在当前regime下（增长恐慌+财政预期），")
report.append("Max Sharpe 组合追求风险调整后收益最大化，")
report.append("Min-CVaR 组合在控制极端亏损（95% CVaR）前提下优化配置，")
report.append("后者更适合尾部风险敏感的投资者。")
report.append("")
report.append(f"> Efficient frontier 数据已输出至 `data/bl_efficient_frontier.csv` ({len(frontier_df)} 个数据点)")

output = "\n".join(report)
out_path = os.path.join(ANALYSIS_DIR, "black_litterman_20260331.md")
with open(out_path, "w") as f:
    f.write(output)

print("\n" + "=" * 60)
print(output)
print(f"\n✓ Saved to {out_path}")
