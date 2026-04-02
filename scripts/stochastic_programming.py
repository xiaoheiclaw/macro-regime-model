#!/usr/bin/env python3
"""
Phase 4b: Stochastic Programming with Monte Carlo Regime Path Simulation
=========================================================================
Instead of static BL scenario weighting, we:
  1. Estimate per-regime return distributions (mu, Sigma) with outlier handling
  2. Estimate Markov transition matrix from observed regime sequence
  3. Simulate 1000 forward paths (22 days each) from current regime
  4. Jointly optimise portfolio weights across ALL paths
  5. Compare with existing BL allocations

Two objectives:
  - max_avg_sharpe : maximise average Sharpe across paths
  - min_avg_cvar   : minimise average CVaR(95%) across paths

Improvements over v1:
  - Winsorise returns (1st/99th pctl) to handle Oil April-2020 crash
  - Per-asset max weight cap (50%) to prevent extreme concentration
  - Ledoit-Wolf shrinkage covariance for stability
  - Multiple random starts for both optimisations
  - Dynamic BL weight loading from black_litterman output
"""
import os, sys, time, json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import numba
from numba import njit
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR
from lib.universe import ASSETS, ASSET_NAMES, MARKET_WEIGHTS

ANALYSIS = ANALYSIS_DIR  # alias for compatibility

t0 = time.time()

# ── Config ──────────────────────────────────────────────
N_PATHS      = 1000
HORIZON      = 22       # ~1 month trading days
RF           = 0.04     # risk-free rate (annual, ~SOFR)
ALPHA        = 0.05     # CVaR confidence (95%)
MAX_WEIGHT   = 0.50     # per-asset cap
WINSOR_LO    = 0.01     # winsorisation percentiles
WINSOR_HI    = 0.99
SEED         = 42

NAMES = ASSET_NAMES  # alias for compatibility

# ── 1. Data loading ────────────────────────────────────
print("=" * 70)
print("Phase 4b: Stochastic Programming – Regime Path Optimisation")
print("=" * 70)

from lib.data_loader import load_merged
df = load_merged()
ens = pd.read_csv(os.path.join(DATA_DIR, "ensemble_regime.csv"),
                  index_col=0, parse_dates=True)

# ── Compute returns ─────────────────────────────────────
returns_raw = pd.DataFrame(index=df.index)
for a in ASSETS:
    if a not in df.columns:
        continue
    if a == "US10Y_yield":
        # Bond return proxy: −Δyield × duration(~8)
        returns_raw[a] = -df[a].diff() * 8 / 100
    else:
        returns_raw[a] = df[a].pct_change()

returns_raw = returns_raw.dropna()
available = [a for a in ASSETS if a in returns_raw.columns and returns_raw[a].notna().sum() > 200]
returns_raw = returns_raw[available]
n_assets = len(available)

print(f"\nAssets : {available}")
print(f"Raw returns: {len(returns_raw)} days  ({returns_raw.index[0].date()} → {returns_raw.index[-1].date()})")

# ── Winsorise returns ───────────────────────────────────
# Critical: WTI had -306% daily return in Apr 2020 (negative oil price)
returns = returns_raw.copy()
for col in returns.columns:
    lo = returns[col].quantile(WINSOR_LO)
    hi = returns[col].quantile(WINSOR_HI)
    n_clipped = ((returns[col] < lo) | (returns[col] > hi)).sum()
    returns[col] = returns[col].clip(lo, hi)
    if n_clipped > 0:
        print(f"  Winsorised {col}: {n_clipped} obs clipped to [{lo:.4f}, {hi:.4f}]")

print(f"Returns after winsorisation: {len(returns)} days")

# ── Merge regime labels ─────────────────────────────────
common_idx = returns.index.intersection(ens.index)
returns_r = returns.loc[common_idx].copy()
regime_series = ens.loc[common_idx, "ens_regime"].astype(int)
n_regimes = regime_series.nunique()
regime_labels = sorted(regime_series.unique())
print(f"Regimes: {n_regimes}  labels={regime_labels}  days={len(returns_r)}")

# ── 2. Per-regime parameter estimation ──────────────────
print(f"\n── Per-regime parameter estimation ──")

def ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf linear shrinkage toward scaled identity.
    X: (T, N) centered return matrix
    Returns: (N, N) shrunk covariance
    """
    T, N = X.shape
    S = X.T @ X / T  # sample covariance
    trace_S = np.trace(S)
    mu_target = trace_S / N  # shrinkage target = scaled identity

    # Frobenius norm terms
    delta = S - mu_target * np.eye(N)
    delta_sq = np.sum(delta ** 2)

    # Estimate optimal shrinkage intensity
    X2 = X ** 2
    sum_sq = (X2.T @ X2) / T - S ** 2
    rho_hat = np.sum(sum_sq) / T
    kappa = (rho_hat) / (delta_sq + 1e-12)
    shrinkage = max(0.0, min(1.0, kappa))

    Sigma_shrunk = (1 - shrinkage) * S + shrinkage * mu_target * np.eye(N)
    return Sigma_shrunk, shrinkage

mu_regime = {}      # {regime: np.array of annualised means}
Sigma_regime = {}   # {regime: np.ndarray annualised cov}
shrink_info = {}

for r in regime_labels:
    mask = regime_series == r
    ret_r = returns_r.loc[mask]
    days = len(ret_r)

    mu_regime[r] = ret_r.mean().values * 252  # annualise

    # Ledoit-Wolf shrinkage covariance
    X_centered = (ret_r - ret_r.mean()).values
    cov_shrunk, alpha_shrink = ledoit_wolf_shrinkage(X_centered)
    Sigma_regime[r] = cov_shrunk * 252  # annualise
    shrink_info[r] = alpha_shrink

    print(f"  Regime {r}: {days} days, shrinkage={alpha_shrink:.3f}")
    print(f"    mu_ann = [{', '.join(f'{m:+.1%}' for m in mu_regime[r])}]")
    vols = np.sqrt(np.diag(Sigma_regime[r]))
    print(f"    vol_ann= [{', '.join(f'{v:.1%}' for v in vols)}]")

# ── 3. Transition matrix estimation ────────────────────
print(f"\n── Transition matrix estimation ──")

trans_count = np.zeros((n_regimes, n_regimes))
for i in range(1, len(regime_series)):
    prev = regime_series.iloc[i - 1]
    curr = regime_series.iloc[i]
    trans_count[prev, curr] += 1

# Row-normalise
trans_matrix = trans_count / trans_count.sum(axis=1, keepdims=True)
print("Transition matrix (from observed ens_regime):")
for i in range(n_regimes):
    row = "  ".join(f"{trans_matrix[i, j]:.4f}" for j in range(n_regimes))
    duration = 1.0 / (1.0 - trans_matrix[i, i]) if trans_matrix[i, i] < 1 else np.inf
    print(f"  R{i} → [{row}]   avg duration ≈ {duration:.1f} days")

current_regime = int(regime_series.iloc[-1])
current_date   = regime_series.index[-1]
print(f"\nCurrent regime: R{current_regime}  ({current_date.date()})")

# ── 4. Monte Carlo path simulation ─────────────────────
print(f"\n── Monte Carlo: {N_PATHS} paths × {HORIZON} days ──")
np.random.seed(SEED)

@njit(cache=True)
def simulate_regime_paths(start_regime, trans_mat, n_paths, horizon, u):
    """Simulate regime paths via Markov chain (numba-accelerated)."""
    paths = np.empty((n_paths, horizon), dtype=np.int32)
    cum_trans = np.empty_like(trans_mat)
    n_regimes = trans_mat.shape[0]
    for i in range(n_regimes):
        s = 0.0
        for j in range(n_regimes):
            s += trans_mat[i, j]
            cum_trans[i, j] = s
    for p in range(n_paths):
        paths[p, 0] = start_regime
        for t in range(1, horizon):
            prev = paths[p, t - 1]
            val = u[p, t]
            regime = n_regimes - 1
            for r in range(n_regimes):
                if val < cum_trans[prev, r]:
                    regime = r
                    break
            paths[p, t] = regime
    return paths

t_sim = time.time()
u_sim = np.random.rand(N_PATHS, HORIZON)
regime_paths = simulate_regime_paths(current_regime, trans_matrix, N_PATHS, HORIZON, u_sim)
print(f"  Simulation done in {time.time() - t_sim:.2f}s")

# Path stats
regime_counts = np.zeros((N_PATHS, n_regimes))
for r in regime_labels:
    regime_counts[:, r] = (regime_paths == r).sum(axis=1)
avg_days_in_regime = regime_counts.mean(axis=0)
for r in regime_labels:
    print(f"  R{r}: avg {avg_days_in_regime[r]:.1f} days ({avg_days_in_regime[r] / HORIZON:.0%})")

# ── Pre-compute Cholesky & daily parameters ─────────────
mu_daily  = {r: mu_regime[r] / 252 for r in regime_labels}
Sigma_daily = {r: Sigma_regime[r] / 252 for r in regime_labels}

chol = {}
for r in regime_labels:
    try:
        chol[r] = np.linalg.cholesky(Sigma_daily[r])
    except np.linalg.LinAlgError:
        chol[r] = np.linalg.cholesky(Sigma_daily[r] + np.eye(n_assets) * 1e-8)

# Pre-draw standard normals for reproducibility across optimisations
rng_draws = np.random.randn(N_PATHS, HORIZON, n_assets)

# ── Convert regime parameters to arrays for numba ───────
mu_daily_arr = np.empty((n_regimes, n_assets), dtype=np.float64)
chol_arr = np.empty((n_regimes, n_assets, n_assets), dtype=np.float64)
for r in regime_labels:
    mu_daily_arr[r] = mu_daily[r]
    chol_arr[r] = chol[r]

# ── 5. Path-based portfolio evaluation (numba) ─────────

@njit(cache=True)
def _compute_path_returns(weights, regime_paths, mu_daily_arr, chol_arr, rng_draws, n_regimes):
    """Compute cumulative portfolio return for each path (numba-accelerated)."""
    n_paths = regime_paths.shape[0]
    horizon = regime_paths.shape[1]
    n_assets = weights.shape[0]
    cum = np.ones(n_paths, dtype=np.float64)
    for p in range(n_paths):
        for t in range(horizon):
            r = regime_paths[p, t]
            # daily_ret = mu_daily[r] + rng_draws[p,t,:] @ chol[r].T
            port_ret = 0.0
            for i in range(n_assets):
                asset_ret = mu_daily_arr[r, i]
                for j in range(n_assets):
                    asset_ret += rng_draws[p, t, j] * chol_arr[r, j, i]
                port_ret += asset_ret * weights[i]
            cum[p] *= (1.0 + port_ret)
    return cum - 1.0

# Warm up numba JIT (first call triggers compilation)
_warm = _compute_path_returns(
    np.ones(n_assets, dtype=np.float64) / n_assets,
    regime_paths[:2].astype(np.int32), mu_daily_arr, chol_arr, rng_draws[:2], n_regimes)

def compute_path_returns(weights):
    return _compute_path_returns(
        np.ascontiguousarray(weights, dtype=np.float64),
        regime_paths.astype(np.int32), mu_daily_arr, chol_arr, rng_draws, n_regimes)


def path_portfolio_stats(w):
    """Annualised stats from simulated path distribution."""
    path_rets = compute_path_returns(w)
    ann_factor = 252 / HORIZON
    ann_rets = (1 + path_rets) ** ann_factor - 1

    mean_ret = ann_rets.mean()
    vol      = ann_rets.std()
    sharpe   = (mean_ret - RF) / vol if vol > 1e-10 else 0.0

    # Historical VaR/CVaR from simulated distribution
    sorted_rets = np.sort(ann_rets)
    var_idx = int(np.floor(ALPHA * N_PATHS))
    var_idx = max(var_idx, 1)
    var_95  = -sorted_rets[var_idx]
    cvar_95 = -sorted_rets[:var_idx].mean()

    return dict(ret=mean_ret, vol=vol, sharpe=sharpe,
                var95=var_95, cvar95=cvar_95, path_rets=ann_rets)


# ── 6. Optimisation ────────────────────────────────────
print(f"\n── Optimisation ──")

# Constraints: sum = 1, long-only, max weight per asset
bounds = [(0.0, MAX_WEIGHT)] * n_assets
cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def neg_avg_sharpe(w):
    return -path_portfolio_stats(w)["sharpe"]


def avg_cvar(w):
    return path_portfolio_stats(w)["cvar95"]


# Generate diverse starting points
def make_starts():
    starts = []
    # Equal weight
    starts.append(np.ones(n_assets) / n_assets)
    # Biased towards each asset (within bounds)
    for i in range(n_assets):
        x = np.ones(n_assets) * (1 - MAX_WEIGHT) / (n_assets - 1)
        x[i] = MAX_WEIGHT
        starts.append(x)
    # Defensive: bonds + USD + gold heavy
    d = np.zeros(n_assets)
    for i, a in enumerate(available):
        if a in ("US10Y_yield", "DXY", "Gold"):
            d[i] = 0.28
        else:
            d[i] = 0.16 / max(1, n_assets - 3)
    d /= d.sum()
    starts.append(d)
    # Random
    rng = np.random.RandomState(123)
    for _ in range(4):
        x = rng.dirichlet(np.ones(n_assets))
        x = np.clip(x, 0, MAX_WEIGHT)
        x /= x.sum()
        starts.append(x)
    return starts


starts = make_starts()

# --- max_avg_sharpe ---
print("\n  [1] max_avg_sharpe")
t_opt = time.time()
best_sharpe_val = np.inf
best_w_sharpe = starts[0].copy()

for idx, x0 in enumerate(starts):
    res = minimize(neg_avg_sharpe, x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    if res.success and res.fun < best_sharpe_val:
        best_sharpe_val = res.fun
        best_w_sharpe = res.x.copy()
    if (idx + 1) % 4 == 0:
        print(f"    ... tried {idx + 1}/{len(starts)} starts, best Sharpe = {-best_sharpe_val:.4f}")

w_sp_sharpe = np.maximum(best_w_sharpe, 0)
w_sp_sharpe = np.minimum(w_sp_sharpe, MAX_WEIGHT)
w_sp_sharpe /= w_sp_sharpe.sum()
print(f"  ✓ SP-Sharpe done in {time.time() - t_opt:.1f}s  "
      f"(Sharpe = {-best_sharpe_val:.4f})")

# --- min_avg_cvar ---
print("\n  [2] min_avg_cvar")
t_opt2 = time.time()
best_cvar_val = np.inf
best_w_cvar = starts[0].copy()

for idx, x0 in enumerate(starts):
    res = minimize(avg_cvar, x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    if res.success and res.fun < best_cvar_val:
        best_cvar_val = res.fun
        best_w_cvar = res.x.copy()
    if (idx + 1) % 4 == 0:
        print(f"    ... tried {idx + 1}/{len(starts)} starts, best CVaR = {best_cvar_val:.4f}")

w_sp_cvar = np.maximum(best_w_cvar, 0)
w_sp_cvar = np.minimum(w_sp_cvar, MAX_WEIGHT)
w_sp_cvar /= w_sp_cvar.sum()
print(f"  ✓ SP-CVaR done in {time.time() - t_opt2:.1f}s  "
      f"(CVaR = {best_cvar_val:.4f})")

# ── 7. Load BL weights ─────────────────────────────────
print(f"\n── Loading BL weights ──")

# Try loading from existing stochastic_prog_weights or black_litterman output
bl_weights_loaded = False

# Method 1: Run BL weight extraction inline
try:
    # Re-derive BL weights (same logic as black_litterman.py, condensed)
    recent = returns.iloc[-252:]
    Sigma_bl = recent.cov().values * 252
    mu_hist = recent.mean().values * 252

    w_mkt_raw = np.array([MARKET_WEIGHTS.get(a, 0.1) for a in available])
    w_mkt = w_mkt_raw / w_mkt_raw.sum()

    delta = 2.5
    Pi = delta * Sigma_bl @ w_mkt

    # BL scenario views (same as black_litterman.py)
    SCENARIOS = {
        "A": {"prob": 0.15, "views": {"WTI_crude": -0.25, "US10Y_yield": 0.15, "SPX": 0.10, "Gold": -0.05, "BTC": 0.05, "DXY": 0.03}},
        "B": {"prob": 0.35, "views": {"WTI_crude": 0.10, "US10Y_yield": 0.05, "SPX": -0.05, "Gold": 0.20, "BTC": 0.15, "DXY": -0.08}},
        "C": {"prob": 0.20, "views": {"WTI_crude": 0.30, "US10Y_yield": -0.10, "SPX": -0.20, "Gold": 0.10, "BTC": -0.15, "DXY": 0.10}},
        "D": {"prob": 0.30, "views": {"WTI_crude": 0.05, "US10Y_yield": 0.08, "SPX": 0.00, "Gold": 0.30, "BTC": 0.25, "DXY": -0.15}},
    }

    scenario_rets = np.array([
        [SCENARIOS[s]["views"].get(a, 0) for a in available] for s in SCENARIOS
    ])
    scenario_probs = np.array([SCENARIOS[s]["prob"] for s in SCENARIOS])
    weighted_returns = scenario_probs @ scenario_rets

    P = np.eye(n_assets)
    Q = weighted_returns
    view_var = np.average((scenario_rets - weighted_returns) ** 2,
                          weights=scenario_probs, axis=0)
    Omega = np.diag(view_var + 0.001)
    tau = 0.05
    tau_Sigma = tau * Sigma_bl
    inv_tau_Sigma = np.linalg.inv(tau_Sigma)
    inv_Omega = np.linalg.inv(Omega)
    M = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
    bl_returns = M @ (inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q)
    bl_cov = M + Sigma_bl

    # BL Max Sharpe (long-only)
    inv_bl_cov = np.linalg.inv(delta * bl_cov)
    w_bl_raw = inv_bl_cov @ bl_returns
    w_bl_sharpe = np.maximum(w_bl_raw, 0)
    w_bl_sharpe = w_bl_sharpe / w_bl_sharpe.sum() if w_bl_sharpe.sum() > 0 else w_mkt
    bl_weights_loaded = True
    print(f"  ✓ BL weights derived inline")
    print(f"    BL-Sharpe: [{', '.join(f'{a}={w_bl_sharpe[i]:.0%}' for i,a in enumerate(available))}]")
except Exception as e:
    print(f"  ⚠ BL derivation failed: {e}")
    # Fallback: hardcoded from Phase 4a output
    BL_MAP = {"SPX": 0.25, "US10Y_yield": 0.32, "Gold": 0.23,
              "WTI_crude": 0.11, "BTC": 0.08, "DXY": 0.01}
    w_bl_sharpe = np.array([BL_MAP.get(a, 0.1) for a in available])
    w_bl_sharpe /= w_bl_sharpe.sum()
    w_mkt_raw = np.array([MARKET_WEIGHTS.get(a, 0.1) for a in available])
    w_mkt = w_mkt_raw / w_mkt_raw.sum()
    print(f"  Using hardcoded BL weights (fallback)")

# Market weights
w_market = np.array([MARKET_WEIGHTS.get(a, 0.1) for a in available])
w_market /= w_market.sum()

# ── 8. Comprehensive comparison ────────────────────────
print(f"\n── Strategy Comparison ──")

strategies = {
    "Market":     w_market,
    "BL-Sharpe":  w_bl_sharpe,
    "SP-Sharpe":  w_sp_sharpe,
    "SP-CVaR":    w_sp_cvar,
}

stats = {}
for name, w in strategies.items():
    stats[name] = path_portfolio_stats(w)
    s = stats[name]
    wstr = " ".join(f"{a[:4]}={w[i]:.0%}" for i, a in enumerate(available))
    print(f"  {name:12s}  Ret={s['ret']:+.2%}  Vol={s['vol']:.2%}  "
          f"Sharpe={s['sharpe']:.3f}  VaR={s['var95']:.2%}  CVaR={s['cvar95']:.2%}")
    print(f"    weights: [{wstr}]")

# ── 9. Save weights CSV ────────────────────────────────
rows = []
for name, w in strategies.items():
    row = {"strategy": name}
    for i, a in enumerate(available):
        row[a] = round(float(w[i]), 4)
    s = stats[name]
    row["ann_return"]  = round(s["ret"],    4)
    row["ann_vol"]     = round(s["vol"],    4)
    row["sharpe"]      = round(s["sharpe"], 4)
    row["var_95"]      = round(s["var95"],  4)
    row["cvar_95"]     = round(s["cvar95"], 4)
    rows.append(row)

weights_df = pd.DataFrame(rows)
weights_path = os.path.join(DATA_DIR, "stochastic_prog_weights.csv")
weights_df.to_csv(weights_path, index=False)
print(f"\n✓ Weights → {weights_path}")

# ── 10. Markdown report ────────────────────────────────
R = []
R.append("# Phase 4b: Stochastic Programming Regime-Path Optimisation")
R.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
R.append("")

R.append("## 方法论")
R.append("")
R.append("传统 Black-Litterman 对情景做**静态概率加权**；本模块改用**随机规划**：")
R.append("")
R.append("1. 从当前 ensemble regime 出发")
R.append("2. 用 Markov 转移矩阵蒙特卡洛模拟 1,000 条 22 天（≈1 月）路径")
R.append("3. 每条路径按 regime 序列从条件分布采样资产日收益")
R.append("4. 在所有路径上联合优化组合权重（max Sharpe / min CVaR）")
R.append("")
R.append("**v2 改进：**")
R.append(f"- 收益 winsorisation（{WINSOR_LO:.0%}/{WINSOR_HI:.0%} 分位），消除 2020-04 油价负值极端噪声")
R.append(f"- 单资产上限 {MAX_WEIGHT:.0%}，防止极端集中")
R.append("- Ledoit-Wolf 收缩协方差，提升小样本稳定性")
R.append("- 多起始点优化（>10 组），降低局部最优风险")
R.append("")

# Transition matrix
R.append("## Markov 转移矩阵")
R.append("")
header = "| From \\ To |" + "".join(f" R{j} |" for j in regime_labels)
R.append(header)
R.append("|---|" + "---|" * n_regimes)
for i in regime_labels:
    row = f"| **R{i}** |"
    for j in regime_labels:
        row += f" {trans_matrix[i, j]:.4f} |"
    R.append(row)
R.append("")
for i in regime_labels:
    dur = 1.0 / (1.0 - trans_matrix[i, i]) if trans_matrix[i, i] < 1 else np.inf
    R.append(f"- R{i} 平均持续 **{dur:.1f}** 天")
R.append("")
R.append(f"当前 regime: **R{current_regime}** (截至 {current_date.date()})")
R.append("")

# Path simulation stats
R.append("## 路径模拟统计")
R.append("")
R.append(f"- 模拟路径数: **{N_PATHS}**")
R.append(f"- 前瞻天数: **{HORIZON}** (≈1 个月)")
R.append(f"- 起始 regime: **R{current_regime}**")
R.append("")
R.append("| Regime | 平均天数 | 占比 | 解读 |")
R.append("|---|---|---|---|")
interp = {0: "温和/扩张", 1: "压力/收缩"}
for r in regime_labels:
    pct = avg_days_in_regime[r] / HORIZON
    label = interp.get(r, "—")
    R.append(f"| R{r} | {avg_days_in_regime[r]:.1f} | {pct:.0%} | {label} |")
R.append("")

# Per-regime return profiles
R.append("## Regime 条件收益 (年化, winsorised)")
R.append("")
R.append("| Asset |" + "".join(f" R{r} μ | R{r} σ |" for r in regime_labels))
sep = "|---|" + "---|---|" * n_regimes
R.append(sep)
for i, a in enumerate(available):
    name = NAMES.get(a, a)
    row = f"| {name} |"
    for r in regime_labels:
        m = mu_regime[r][i]
        s = np.sqrt(Sigma_regime[r][i, i])
        row += f" {m:+.1%} | {s:.1%} |"
    R.append(row)
R.append("")
for r in regime_labels:
    R.append(f"- R{r} Ledoit-Wolf shrinkage α = {shrink_info[r]:.3f}")
R.append("")

# Weight comparison table
R.append("## 📊 四策略权重对比")
R.append("")
R.append("| Asset | Market | BL-Sharpe | SP-Sharpe | SP-CVaR |")
R.append("|---|---|---|---|---|")
for i, a in enumerate(available):
    name = NAMES.get(a, a)
    R.append(f"| {name} | {w_market[i]:.0%} | {w_bl_sharpe[i]:.0%} | "
             f"{w_sp_sharpe[i]:.0%} | {w_sp_cvar[i]:.0%} |")
R.append("")

# Performance comparison
R.append("## 📈 四策略绩效对比 (Monte Carlo)")
R.append("")
R.append("| 指标 | Market | BL-Sharpe | SP-Sharpe | SP-CVaR |")
R.append("|---|---|---|---|---|")
metrics_spec = [
    ("Expected Return", "ret",    100, "+.1f", "%"),
    ("Volatility",      "vol",    100, ".1f",  "%"),
    ("Sharpe Ratio",    "sharpe", 1,   ".3f",  ""),
    ("VaR (95%)",       "var95",  100, ".1f",  "%"),
    ("CVaR (95%)",      "cvar95", 100, ".1f",  "%"),
]
for label, key, scale, fmt, suffix in metrics_spec:
    vals = []
    for strat in ["Market", "BL-Sharpe", "SP-Sharpe", "SP-CVaR"]:
        v = stats[strat][key] * scale
        vals.append(f"{v:{fmt}}{suffix}")
    R.append(f"| {label} | {' | '.join(vals)} |")
R.append("")
R.append(f"> 无风险利率 = {RF:.0%} (SOFR proxy)")
R.append(f"> VaR/CVaR 为历史模拟法 (从 {N_PATHS} 条路径分布)")
R.append(f"> 单资产权重上限 = {MAX_WEIGHT:.0%}")
R.append("")

# Key findings
R.append("## 🎯 核心发现")
R.append("")

best_sharpe_strat = max(stats.keys(), key=lambda k: stats[k]["sharpe"])
best_cvar_strat   = min(stats.keys(), key=lambda k: stats[k]["cvar95"])
R.append(f"1. **最优 Sharpe**: {best_sharpe_strat} "
         f"(Sharpe = {stats[best_sharpe_strat]['sharpe']:.3f})")
R.append(f"2. **最低 CVaR**: {best_cvar_strat} "
         f"(CVaR = {stats[best_cvar_strat]['cvar95']:.1%})")
R.append("")

# SP vs BL comparison
R.append("### 随机规划 vs Black-Litterman")
R.append("")
sp_sh = stats["SP-Sharpe"]
bl_sh = stats["BL-Sharpe"]
R.append(f"- **SP-Sharpe vs BL-Sharpe**: "
         f"Return {sp_sh['ret'] - bl_sh['ret']:+.1%}, "
         f"Vol {sp_sh['vol'] - bl_sh['vol']:+.1%}, "
         f"Sharpe {sp_sh['sharpe'] - bl_sh['sharpe']:+.3f}, "
         f"CVaR {sp_sh['cvar95'] - bl_sh['cvar95']:+.1%}")

sp_cv = stats["SP-CVaR"]
mkt   = stats["Market"]
R.append(f"- **SP-CVaR vs Market**: "
         f"Return {sp_cv['ret'] - mkt['ret']:+.1%}, "
         f"Vol {sp_cv['vol'] - mkt['vol']:+.1%}, "
         f"CVaR {sp_cv['cvar95'] - mkt['cvar95']:+.1%}")
R.append("")

# Weight shift analysis
R.append("### 权重偏移分析")
R.append("")
R.append("**SP-Sharpe vs BL-Sharpe 权重差异:**")
for i, a in enumerate(available):
    diff = w_sp_sharpe[i] - w_bl_sharpe[i]
    if abs(diff) > 0.02:
        direction = "↑" if diff > 0 else "↓"
        R.append(f"- {NAMES.get(a, a)}: {direction} {abs(diff):.0%} "
                 f"({w_bl_sharpe[i]:.0%} → {w_sp_sharpe[i]:.0%})")
R.append("")

R.append("**SP-CVaR 防御特征:**")
top_cvar = np.argsort(w_sp_cvar)[::-1]
for idx in top_cvar[:3]:
    R.append(f"- {NAMES.get(available[idx], available[idx])}: "
             f"**{w_sp_cvar[idx]:.0%}**")
R.append("")

# Methodology comparison
R.append("### 方法论差异总结")
R.append("")
R.append("| 维度 | Black-Litterman | 随机规划 |")
R.append("|---|---|---|")
R.append("| 情景处理 | 静态概率加权 | Markov 路径模拟 |")
R.append("| 时间维度 | 单期（无路径依赖） | 多期（22 天路径） |")
R.append("| 转移动态 | 不考虑 regime 转换 | Regime 转移矩阵驱动 |")
R.append("| 协方差估计 | 样本协方差（近 1Y） | Ledoit-Wolf 收缩（按 regime 分组） |")
R.append("| 风险度量 | 参数法 VaR/CVaR | 历史模拟法（路径分布） |")
R.append("| 集中度控制 | 无（可能极端集中） | 单资产上限 50% |")
R.append("| 适用场景 | 稳定 regime 下的战略配置 | Regime 转换频繁时的战术调整 |")
R.append("")

elapsed = time.time() - t0
R.append("---")
R.append(f"*计算耗时: {elapsed:.1f}s | {N_PATHS} 路径 × {HORIZON} 天 | "
         f"winsorised returns | Ledoit-Wolf shrinkage | max_wt={MAX_WEIGHT:.0%}*")

report_text = "\n".join(R)
report_path = os.path.join(ANALYSIS, "stochastic_prog_20260401.md")
with open(report_path, "w") as f:
    f.write(report_text)

print("\n" + "=" * 70)
print(report_text)
print(f"\n✓ Report → {report_path}")
print(f"✓ Weights → {weights_path}")
print(f"✓ Total time: {elapsed:.1f}s")
