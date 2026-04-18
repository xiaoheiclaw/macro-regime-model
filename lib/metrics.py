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
