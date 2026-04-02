#!/usr/bin/env python
"""
Wasserstein K-Means Regime Detection
=====================================
Phase 2 enhancement: ensemble Markov Switching + Wasserstein K-Means.
Reference: Berkeley MFE MarketMoodRing project.

Method:
  1. Rolling window (20d) → empirical distribution per window
  2. Wasserstein distance matrix between all windows
  3. K-Means on the distance matrix → 2 regimes
  4. Ensemble with Hamilton Markov Switching results
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import numba
from numba import njit
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR

# ── paths ──────────────────────────────────────────────────────────────────
MERGED   = os.path.join(DATA_DIR, "merged_data.csv")
MARKOV   = os.path.join(DATA_DIR, "oil_bond_regime_probs.csv")
OUT_WKM  = os.path.join(DATA_DIR, "wasserstein_regime_probs.csv")
OUT_ENS  = os.path.join(DATA_DIR, "ensemble_regime.csv")
OUT_RPT  = os.path.join(ANALYSIS_DIR, "wasserstein_regime_20260401.md")

# ── parameters ─────────────────────────────────────────────────────────────
WINDOW   = 20      # rolling window size
N_BINS   = 21      # histogram bins for distribution representation
N_REG    = 2       # number of regimes
MAX_ITER = 100     # k-means iterations
N_INIT   = 10      # k-means restarts
SEED     = 42

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Wasserstein K-Means Regime Detection")
print("=" * 60)

df = pd.read_csv(MERGED, parse_dates=["date"])
df.set_index("date", inplace=True)
df.sort_index(inplace=True)

# Compute returns / changes
returns = pd.DataFrame(index=df.index)

# Oil return
if "WTI_crude" in df.columns:
    returns["oil_ret"] = df["WTI_crude"].pct_change()

# 10Y yield change
if "US10Y_yield" in df.columns:
    returns["yield_chg"] = df["US10Y_yield"].diff()

# SPX return
if "SPX" in df.columns:
    returns["spx_ret"] = df["SPX"].pct_change()

returns.dropna(inplace=True)
print(f"\nInput features: {list(returns.columns)}")
print(f"Date range: {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"Observations: {len(returns)}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. WASSERSTEIN K-MEANS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Computing rolling window distributions ──")

def window_to_histogram(window_data: np.ndarray, n_bins: int = N_BINS):
    """Convert a rolling window of multi-feature returns into a combined histogram.
    
    For each feature: compute histogram on a shared grid.
    Concatenate all feature histograms → one distribution vector.
    Normalize to sum to 1 (probability distribution).
    """
    all_hists = []
    for col_idx in range(window_data.shape[1]):
        col_data = window_data[:, col_idx]
        # Use KDE to smooth, then sample back to histogram
        try:
            kde = stats.gaussian_kde(col_data, bw_method="silverman")
            # Grid: cover data range with some padding
            lo, hi = col_data.min(), col_data.max()
            spread = hi - lo
            if spread < 1e-12:
                # Degenerate: all same value
                hist = np.ones(n_bins) / n_bins
            else:
                grid = np.linspace(lo - 0.1 * spread, hi + 0.1 * spread, n_bins + 1)
                centers = 0.5 * (grid[:-1] + grid[1:])
                hist = kde(centers)
                hist = hist / hist.sum()  # normalize
        except Exception:
            hist = np.ones(n_bins) / n_bins
        all_hists.append(hist)
    
    combined = np.concatenate(all_hists)
    combined = combined / combined.sum()
    return combined


# Build distribution for each rolling window
n_features = returns.shape[1]
dist_dim = N_BINS * n_features
values = returns.values
dates_valid = returns.index[WINDOW - 1:]  # dates where we have full window

distributions = []
for i in range(WINDOW - 1, len(values)):
    window = values[i - WINDOW + 1: i + 1]
    h = window_to_histogram(window, N_BINS)
    distributions.append(h)

distributions = np.array(distributions)
print(f"Distribution matrix: {distributions.shape} (windows × bins)")

# ── Wasserstein distance matrix (numba-accelerated) ──
print("── Computing Wasserstein distance matrix (numba-accelerated) ──")

@njit(cache=True)
def wasserstein_1d_cdf(a, b, n_bins):
    """1D Wasserstein-1 distance via CDF difference.
    a, b: 1D histograms of length n_bins (summing to same total).
    Returns W1 = sum |CDF_a - CDF_b| / n_bins.
    """
    cdf_diff = 0.0
    running = 0.0
    for i in range(n_bins):
        running += a[i] - b[i]
        cdf_diff += abs(running)
    return cdf_diff / n_bins

@njit(cache=True)
def pairwise_wasserstein_matrix(distributions, n_features, n_bins):
    """Compute pairwise Wasserstein distance matrix using per-feature 1D W1.
    
    Since cross-feature transport cost >> within-feature cost,
    EMD decomposes into independent 1D problems per feature.
    Total distance = sum of per-feature W1 distances.
    """
    N = distributions.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            total_w = 0.0
            for f in range(n_features):
                start = f * n_bins
                end = start + n_bins
                w = wasserstein_1d_cdf(
                    distributions[i, start:end],
                    distributions[j, start:end],
                    n_bins)
                total_w += w
            dist_matrix[i, j] = total_w
            dist_matrix[j, i] = total_w
    return dist_matrix

N = len(distributions)

# Warm up numba JIT
_ = pairwise_wasserstein_matrix(distributions[:3].copy(), n_features, N_BINS)

import time as _time
_t0 = _time.time()
dist_matrix = pairwise_wasserstein_matrix(
    np.ascontiguousarray(distributions, dtype=np.float64), n_features, N_BINS)
print(f"  Distance matrix computed: {dist_matrix.shape} in {_time.time() - _t0:.2f}s")

# ── K-Means on distance matrix ──
print("── Running K-Means clustering ──")

# K-Medoids style: use precomputed distance matrix
# We do kernel K-Means via double-centering → eigendecomposition → standard K-Means
# Equivalent to MDS embedding then K-Means

from sklearn.cluster import KMeans

# MDS-like embedding: double-centering
D_sq = dist_matrix ** 2
n = D_sq.shape[0]
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ D_sq @ H

# Eigendecomposition - take top components
eigenvalues, eigenvectors = np.linalg.eigh(B)
# Sort descending
idx_sort = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx_sort]
eigenvectors = eigenvectors[:, idx_sort]

# Take positive eigenvalues for embedding
n_components = min(10, np.sum(eigenvalues > 1e-10))
if n_components < 2:
    n_components = 2
embedding = eigenvectors[:, :n_components] * np.sqrt(np.maximum(eigenvalues[:n_components], 0))

print(f"  MDS embedding: {embedding.shape} ({n_components} components)")

# Standard K-Means on embedded space
kmeans = KMeans(n_clusters=N_REG, n_init=N_INIT, max_iter=MAX_ITER, random_state=SEED)
labels = kmeans.fit_predict(embedding)

# ── Determine which cluster = which regime ──
# Convention: Regime 0 = calm/normal, Regime 1 = stress/turbulent
# Heuristic: stress regime has higher volatility (oil + yield)
cluster_vols = []
for c in range(N_REG):
    mask = labels == c
    vol = np.mean(np.std(values[WINDOW - 1:][mask], axis=0))
    cluster_vols.append(vol)

# If cluster 0 has higher vol, swap labels so Regime 1 = stress
if cluster_vols[0] > cluster_vols[1]:
    labels = 1 - labels
    cluster_vols = cluster_vols[::-1]
    print("  Swapped labels: Regime 0=calm, Regime 1=stress")

print(f"  Regime 0 (calm) vol: {cluster_vols[0]:.6f}")
print(f"  Regime 1 (stress) vol: {cluster_vols[1]:.6f}")
print(f"  Regime counts: R0={np.sum(labels==0)}, R1={np.sum(labels==1)}")

# ── Build WKM output DataFrame ──
wkm_df = pd.DataFrame(index=dates_valid)
wkm_df.index.name = "date"
wkm_df["wkm_regime"] = labels

# Compute soft probabilities from distance to cluster centers
centers = kmeans.cluster_centers_
dists_to_centers = np.zeros((len(embedding), N_REG))
for c in range(N_REG):
    dists_to_centers[:, c] = np.linalg.norm(embedding - centers[c], axis=1)

# Convert distances to probabilities (softmax-like)
# Closer = higher probability
inv_dists = 1.0 / (dists_to_centers + 1e-10)
probs = inv_dists / inv_dists.sum(axis=1, keepdims=True)

wkm_df["wkm_P_R0"] = probs[:, 0]
wkm_df["wkm_P_R1"] = probs[:, 1]

# Save WKM results
wkm_df.to_csv(OUT_WKM)
print(f"\n✓ WKM results saved: {OUT_WKM}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE WITH MARKOV SWITCHING
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Ensemble with Markov Switching ──")

markov_df = pd.read_csv(MARKOV, parse_dates=["date"])
markov_df.set_index("date", inplace=True)
markov_df.sort_index(inplace=True)

# Align dates
common_dates = wkm_df.index.intersection(markov_df.index)
print(f"Common dates: {len(common_dates)}")

ens = pd.DataFrame(index=common_dates)
ens.index.name = "date"
ens["markov_P_R0"] = markov_df.loc[common_dates, "P_R0"]
ens["markov_P_R1"] = markov_df.loc[common_dates, "P_R1"]
ens["wkm_P_R0"] = wkm_df.loc[common_dates, "wkm_P_R0"]
ens["wkm_P_R1"] = wkm_df.loc[common_dates, "wkm_P_R1"]

# Markov regime assignment
ens["markov_regime"] = (ens["markov_P_R1"] > 0.5).astype(int)
ens["wkm_regime"] = wkm_df.loc[common_dates, "wkm_regime"]

# Ensemble: weighted average (Markov 0.6, WKM 0.4)
W_MARKOV = 0.6
W_WKM = 0.4
ens["ens_P_R0"] = W_MARKOV * ens["markov_P_R0"] + W_WKM * ens["wkm_P_R0"]
ens["ens_P_R1"] = W_MARKOV * ens["markov_P_R1"] + W_WKM * ens["wkm_P_R1"]
ens["ens_regime"] = (ens["ens_P_R1"] > 0.5).astype(int)

# Agreement analysis
ens["agree"] = (ens["markov_regime"] == ens["wkm_regime"]).astype(int)
agreement_rate = ens["agree"].mean()
print(f"Agreement rate: {agreement_rate:.4f} ({agreement_rate*100:.1f}%)")

# Save ensemble
ens.to_csv(OUT_ENS)
print(f"✓ Ensemble results saved: {OUT_ENS}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS & REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating report ──")

# Regime statistics
def regime_stats(df, regime_col, returns_df):
    """Compute per-regime statistics."""
    aligned = returns_df.loc[df.index]
    stats_list = []
    for r in range(N_REG):
        mask = df[regime_col] == r
        n_days = mask.sum()
        pct = n_days / len(df) * 100
        sub = aligned[mask]
        row = {
            "regime": r,
            "days": int(n_days),
            "pct": f"{pct:.1f}%",
        }
        for col in aligned.columns:
            row[f"{col}_mean"] = f"{sub[col].mean():.6f}"
            row[f"{col}_std"] = f"{sub[col].std():.6f}"
        stats_list.append(row)
    return stats_list

markov_stats = regime_stats(ens, "markov_regime", returns)
wkm_stats = regime_stats(ens, "wkm_regime", returns)
ens_stats = regime_stats(ens, "ens_regime", returns)

# Recent 40 days
recent = ens.tail(40).copy()
current_markov = ens["markov_regime"].iloc[-1]
current_wkm = ens["wkm_regime"].iloc[-1]
current_ens = ens["ens_regime"].iloc[-1]
current_ens_p = ens["ens_P_R1"].iloc[-1]

# Disagreement periods
disagree = ens[ens["agree"] == 0].copy()
n_disagree = len(disagree)

# Find contiguous disagreement periods
disagree_periods = []
if n_disagree > 0:
    dates_dis = disagree.index.tolist()
    start = dates_dis[0]
    prev = dates_dis[0]
    for d in dates_dis[1:]:
        if (d - prev).days > 5:  # gap > 5 days = new period
            disagree_periods.append((start, prev))
            start = d
        prev = d
    disagree_periods.append((start, prev))

# ── Generate markdown report ──
regime_label = {0: "Calm/Risk-On 🟢", 1: "Stress/Risk-Off 🔴"}

report = f"""# Wasserstein K-Means Regime Detection Report
> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Current Ensemble Regime** | **{regime_label[current_ens]}** |
| Ensemble Stress Probability | {current_ens_p:.4f} |
| Markov Regime | {regime_label[current_markov]} |
| WKM Regime | {regime_label[current_wkm]} |
| Method Agreement Rate | **{agreement_rate*100:.1f}%** |
| Total Observations | {len(ens)} |
| Disagreement Days | {n_disagree} ({n_disagree/len(ens)*100:.1f}%) |

## Methodology

### Wasserstein K-Means (WKM)
- **Rolling window**: {WINDOW} days
- **Distribution representation**: KDE-smoothed histogram ({N_BINS} bins × {n_features} features)
- **Features**: {', '.join(returns.columns)}
- **Distance metric**: Earth Mover's Distance (Wasserstein-1) via POT library
- **Clustering**: K-Means on MDS embedding ({n_components} components) of Wasserstein distance matrix
- **Regimes**: {N_REG} (Calm vs Stress, assigned by volatility)

### Ensemble
- Markov Switching weight: {W_MARKOV}
- WKM weight: {W_WKM}
- Final regime: ensemble P(Stress) > 0.5

## Regime Statistics

### Markov Switching
| Regime | Days | % |"""

for col in returns.columns:
    report += f" {col} μ | {col} σ |"
report += "\n|--------|------|---|"
for col in returns.columns:
    report += "--------|--------|"
for s in markov_stats:
    report += f"\n| {regime_label[s['regime']]} | {s['days']} | {s['pct']} |"
    for col in returns.columns:
        report += f" {s[f'{col}_mean']} | {s[f'{col}_std']} |"

report += f"""

### Wasserstein K-Means
| Regime | Days | % |"""
for col in returns.columns:
    report += f" {col} μ | {col} σ |"
report += "\n|--------|------|---|"
for col in returns.columns:
    report += "--------|--------|"
for s in wkm_stats:
    report += f"\n| {regime_label[s['regime']]} | {s['days']} | {s['pct']} |"
    for col in returns.columns:
        report += f" {s[f'{col}_mean']} | {s[f'{col}_std']} |"

report += f"""

### Ensemble
| Regime | Days | % |"""
for col in returns.columns:
    report += f" {col} μ | {col} σ |"
report += "\n|--------|------|---|"
for col in returns.columns:
    report += "--------|--------|"
for s in ens_stats:
    report += f"\n| {regime_label[s['regime']]} | {s['days']} | {s['pct']} |"
    for col in returns.columns:
        report += f" {s[f'{col}_mean']} | {s[f'{col}_std']} |"

# Recent 40 days timeline
report += """

## Recent 40-Day Regime Timeline

| Date | Markov | WKM | Ensemble | Agree? | Ens P(Stress) |
|------|--------|-----|----------|--------|----------------|
"""
for d, row in recent.iterrows():
    m_icon = "🔴" if row["markov_regime"] == 1 else "🟢"
    w_icon = "🔴" if row["wkm_regime"] == 1 else "🟢"
    e_icon = "🔴" if row["ens_regime"] == 1 else "🟢"
    agree = "✅" if row["agree"] == 1 else "❌"
    report += f"| {d.strftime('%Y-%m-%d')} | {m_icon} R{int(row['markov_regime'])} | {w_icon} R{int(row['wkm_regime'])} | {e_icon} R{int(row['ens_regime'])} | {agree} | {row['ens_P_R1']:.4f} |\n"

# Disagreement analysis
report += f"""
## Disagreement Analysis

Total disagreement days: **{n_disagree}** ({n_disagree/len(ens)*100:.1f}% of all dates)

### Major Disagreement Periods
"""

if len(disagree_periods) > 0:
    report += "| Period | Start | End | Duration | Markov Says | WKM Says |\n"
    report += "|--------|-------|-----|----------|-------------|----------|\n"
    for i, (s, e) in enumerate(disagree_periods[-15:], 1):  # show last 15
        dur = (e - s).days + 1
        # What each method says during this period
        period_data = disagree.loc[s:e]
        m_mode = int(period_data["markov_regime"].mode().iloc[0])
        w_mode = int(period_data["wkm_regime"].mode().iloc[0])
        report += f"| {i} | {s.strftime('%Y-%m-%d')} | {e.strftime('%Y-%m-%d')} | ~{dur}d | {regime_label[m_mode]} | {regime_label[w_mode]} |\n"

# Why disagreements happen
report += """
### Why Do Methods Disagree?

1. **Transition sensitivity**: Markov Switching is parametric (mean + variance per regime) 
   and transitions smoothly via probability. WKM looks at full distribution shape — 
   it can detect skewness/kurtosis shifts that Markov misses.

2. **Lag differences**: WKM uses a 20-day rolling window so it inherently smooths over 
   short spikes. Markov can flip regimes on a single extreme day.

3. **Distribution vs moments**: WKM clusters on the entire return distribution 
   (including tails, bimodality). Markov only models the first two moments (mean, variance) 
   per regime. This means WKM may identify "stressed" periods where volatility is moderate 
   but tail risk is elevated.

4. **Boundary effects**: Near regime transitions, both methods show uncertainty — 
   Markov through probability hovering around 0.5, WKM through proximity to cluster boundaries.
"""

# Cluster quality
report += f"""
## Model Diagnostics

| Metric | Value |
|--------|-------|
| K-Means inertia | {kmeans.inertia_:.4f} |
| MDS components used | {n_components} |
| Top eigenvalue ratio | {eigenvalues[0]/eigenvalues[:n_components].sum():.4f} |
| WKM regime balance | R0: {np.sum(labels==0)}, R1: {np.sum(labels==1)} |
| Distance matrix range | [{dist_matrix[dist_matrix>0].min():.6f}, {dist_matrix.max():.6f}] |

## Files Generated

- `data/wasserstein_regime_probs.csv` — WKM regime assignments + soft probabilities
- `data/ensemble_regime.csv` — Full ensemble (Markov + WKM + combined)
- `analysis/wasserstein_regime_20260401.md` — This report
"""

# Write report
os.makedirs(os.path.dirname(OUT_RPT), exist_ok=True)
with open(OUT_RPT, "w") as f:
    f.write(report)
print(f"✓ Report saved: {OUT_RPT}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"  Current Markov regime:   {regime_label[current_markov]}")
print(f"  Current WKM regime:      {regime_label[current_wkm]}")
print(f"  Current Ensemble regime: {regime_label[current_ens]} (P_stress={current_ens_p:.4f})")
print(f"  Agreement rate:          {agreement_rate*100:.1f}%")
print(f"  Disagreement days:       {n_disagree}/{len(ens)}")
print("=" * 60)
