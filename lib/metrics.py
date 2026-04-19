"""
Forecast evaluation metrics.

CRPS (Continuous Ranked Probability Score) and PIT (Probability Integral
Transform) for both sample-based ensembles (KAF scenarios, analog pools) and
closed-form distributions (Gaussian benchmarks).

Definitions follow Gneiting-Raftery (2007); CRPS is the proper scoring rule
for probabilistic point forecasts and is preferred over log-score when the
forecast has discrete support (samples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def crps_sample(
    samples: np.ndarray,
    realized: float,
    weights: np.ndarray | None = None,
) -> float:
    """
    Empirical CRPS for a weighted ensemble of forecast samples.
      CRPS = E|X - y| - 0.5 · E|X - X'|
    where X, X' are i.i.d. draws from the forecast and y is realized.

    O(N²) direct form. Suitable for N ≤ a few thousand.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n == 0:
        return float("nan")
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0:
            return float("nan")
        w = w / s
    e1 = float(np.sum(w * np.abs(samples - realized)))
    diff = np.abs(samples[:, None] - samples[None, :])
    e2 = 0.5 * float(np.sum(w[:, None] * w[None, :] * diff))
    return e1 - e2


def crps_normal(mu: float, sigma: float, realized: float) -> float:
    """
    Closed-form CRPS for Normal(mu, sigma) vs realized y.
      CRPS = σ · (z·(2Φ(z) − 1) + 2φ(z) − 1/√π), where z = (y − μ)/σ.
    """
    if sigma <= 0:
        return float(abs(mu - realized))
    z = (realized - mu) / sigma
    return float(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    )


def pit_sample(
    samples: np.ndarray,
    realized: float,
    weights: np.ndarray | None = None,
) -> float:
    """
    Probability Integral Transform = P(X ≤ y) under the forecast. Uniform on
    [0,1] under a calibrated forecast; deviations diagnose miscalibration.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n == 0:
        return float("nan")
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0:
            return float("nan")
        w = w / s
    return float(np.sum(w * (samples <= realized)))


def pit_normal(mu: float, sigma: float, realized: float) -> float:
    if sigma <= 0:
        return 0.5
    return float(norm.cdf((realized - mu) / sigma))


def energy_score_sample(
    samples: np.ndarray,
    realized: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Multivariate Energy Score (Gneiting 2008) — proper scoring rule for
    probabilistic multivariate forecasts. Generalizes CRPS to vector-valued
    predictands.
      ES = E||X - y||₂ - 0.5 · E||X - X'||₂
    samples: (N, D) ensemble
    realized: (D,) vector
    weights: (N,) optional scenario weights
    """
    samples = np.asarray(samples, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(realized):
        return float("nan")
    n = len(samples)
    if n == 0:
        return float("nan")
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0:
            return float("nan")
        w = w / s
    d_to_y = np.linalg.norm(samples - realized[None, :], axis=1)
    e1 = float(np.sum(w * d_to_y))
    d_pair = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)
    e2 = 0.5 * float(np.sum(w[:, None] * w[None, :] * d_pair))
    return e1 - e2


def energy_score_mvn(
    mu: np.ndarray,
    Sigma: np.ndarray,
    realized: np.ndarray,
    n_samples: int = 5000,
    n_seeds: int = 3,
    base_seed: int = 42,
) -> float:
    """
    Energy Score for Normal(μ, Σ) against realized y, via MC sampling.

    Codex Round-7 Critical: single-seed 500 samples gives MC noise ~O(0.04)
    which is comparable to the skill magnitude we try to detect. Now averages
    over n_seeds independent MCs × n_samples each, so effective sample count
    is n_samples × n_seeds and MC noise drops by √(n_seeds). Default 5000×3
    = 15k effective → MC SE roughly 0.008 vs 0.045 single-seed 500.
    """
    d = len(mu)
    Sigma_reg = Sigma + 1e-8 * np.eye(d)
    scores = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        try:
            samples = rng.multivariate_normal(mu, Sigma_reg, size=n_samples)
        except np.linalg.LinAlgError:
            continue
        scores.append(energy_score_sample(samples, realized))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def moving_block_bootstrap_mean(
    x: np.ndarray,
    block_length: int,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Moving-block bootstrap CI for the mean of an overlapping/autocorrelated
    series. Returns (mean, ci_low_95, ci_high_95). Block length should be
    at least the forward-return horizon to preserve dependence.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block_length:
        return float(np.mean(x)) if n else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts_space = n - block_length + 1
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, starts_space, size=n_blocks)
        resample = np.concatenate([x[s:s + block_length] for s in starts])[:n]
        means[i] = resample.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def moving_block_bootstrap_sharpe_diff(
    r_a: np.ndarray,
    r_b: np.ndarray,
    block_length: int,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Moving-block bootstrap CI for Sharpe(a) − Sharpe(b) on PAIRED return
    series (same asofs). Block length = h preserves overlapping-forward
    dependence. Returns (point_estimate, ci_lo_95, ci_hi_95).
    """
    r_a = np.asarray(r_a, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    mask = np.isfinite(r_a) & np.isfinite(r_b)
    r_a, r_b = r_a[mask], r_b[mask]
    n = len(r_a)
    if n < 2 * block_length:
        return float("nan"), float("nan"), float("nan")

    def sharpe(x):
        s = x.std(ddof=1)
        return float(x.mean() / s) if s > 0 else 0.0

    point = sharpe(r_a) - sharpe(r_b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts_space = n - block_length + 1
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, starts_space, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        diffs[i] = sharpe(r_a[idx]) - sharpe(r_b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def regime_conditional_samples(
    state_series: pd.Series,
    global_template_series: pd.Series,
    asof: pd.Timestamp,
    h: int,
    min_samples: int = 10,
) -> np.ndarray | None:
    """
    Regime-conditional historical bootstrap: at asof, identify current global
    regime (argmax of global template probabilities), then sample h-month
    forward cumulative log returns from all PAST asofs in the same regime.

    global_template_series: date-indexed series of int template IDs
        (argmax already taken — use global_templates['template_id']).
    Returns sample array or None if < min_samples matching months.
    """
    if asof not in global_template_series.index:
        return None
    cur_regime = int(global_template_series.loc[asof])
    past_tmpl = global_template_series.loc[:asof].iloc[:-1]
    matching = past_tmpl[past_tmpl == cur_regime].index
    if len(matching) < min_samples:
        return None

    def _fwd(s, hh):
        return s.shift(-1).rolling(hh).sum().shift(-(hh - 1))

    fwd = _fwd(state_series, h)
    samples = fwd.reindex(matching).dropna().values
    if len(samples) < min_samples:
        return None
    return samples.astype(float)


def ar1_forecast(
    series: pd.Series,
    h: int,
    window: int = 120,
) -> tuple[float, float] | None:
    """
    AR(1) h-step-ahead forecast: y_{t+1} = c + φ y_t + ε. Iterates forward h
    steps. Returns (cumulative mean μ_h, cumulative std σ_h) for log-return
    cumulative over h months. OLS on rolling `window` recent observations.
    """
    s = series.dropna().tail(window + 1)
    if len(s) < 36:
        return None
    y = s.values[1:]
    x = s.values[:-1]
    # OLS: y = c + phi * x + eps
    phi = np.cov(x, y, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-12)
    c = float(np.mean(y) - phi * np.mean(x))
    resid = y - (c + phi * x)
    sigma_eps = float(np.std(resid, ddof=1))
    # h-step iterated forecast from last observed
    last = float(s.values[-1])
    mu_t = last
    mu_cum = 0.0
    var_cum = 0.0
    coef_sum = 0.0
    for k in range(1, h + 1):
        mu_t = c + phi * mu_t
        mu_cum += mu_t
        coef_sum = 1.0 + phi * coef_sum  # recursive coefficient accumulation
        var_cum += (coef_sum ** 2) * (sigma_eps ** 2)
    return float(mu_cum), float(np.sqrt(max(var_cum, 0.0)))
