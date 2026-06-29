"""Gold regime-dominance classifier + conditional strategy.

The question (not the same as PR #5's trend-timing): can *explicitly classifying*
which factor dominates gold — the classic **real-rate regime** (real rate ↓ →
gold ↑) vs the **de-dollarization regime** (central-bank buying / reserve
diversification pushing gold up even as real rates rise, the post-2022 break) —
and trading the *dominant* factor, beat PR #5's S1 pure-trend blend (already
shown to beat buy-and-hold) on the same track, net of cost, 1968–2026?

Hypothesis (to falsify): the 2022 "break" is a handover of dominance from the
real rate to de-dollarization flows. If explicitly detecting the dominant
factor leads price or sizes better, a regime-conditional strategy should beat
S1. If it does not, S1's price trend already implicitly follows whatever
factor is in charge, and the explicit classifier is redundant complexity.

Fingerprint (ex-ante, monthly, no hindsight labels — every value at t uses
data ≤ t only):
  real-rate-dominant     : rolling corr(Δlog gold, Δreal_rate) significantly < 0
  de-dollarization-dom.  : that corr → 0 / turns positive, OR a trailing-window
                           divergence (Δgold>0 ∧ Δreal_rate>0, the classic
                           negative relation breaking down). An optional
                           central-bank-demand proxy (cb_demand, e.g. net official
                           gold buying / TIC foreign-official flows) confirms it
                           where available (2010+, lagged).

Trade:
  real-rate-dominant months → the real-rate signal drives exposure (hold when
                               the real rate is not rising over the signal window)
  de-dollarization months   → trend/momentum drives exposure (follow price)

Positions are decided at t and held through t+1 (the shared `run_backtest`
engine applies `.shift(1)`). Long-only 0–100%, vol-targeted, net of trading
cost — identical machinery and panel to S1, so the head-to-head is same-track.

This module reuses `build_timing_panel` (gold + real_rate_10y + usd_broad +
tbill leg) and the `run_backtest` / metrics engine from `lib.gold_trend_timing`
(PR #1–#5, already on main). It adds NO new data fetching of its own.

Data honesty — central-bank gold buying: WGC quarterly net-purchase data starts
2010+ and is lagged; there is no clean FRED series. `cb_demand` is therefore an
OPTIONAL injectable proxy. When None (the default), the gold–real-rate
relationship is the SOLE fingerprint across the full sample — the documented
fallback for pre-2010 history and for any run without a CB-demand feed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from lib.gold_trend_timing import (  # noqa: F401  (re-exported for convenience)
    DEFAULT_LOOKBACKS,
    DEFAULT_TARGET_VOL,
    DEFAULT_VOL_WINDOW,
    trend_exposure,
    vol_scale,
)

ANNUAL = 12

# ── Standard parameters (NOT tuned — picked from conventional values; the
#    sensitivity band over corr_window ∈ {24, 36, 48} is reported by the runner) ──
DEFAULT_CORR_WINDOW = 36        # months for the rolling gold–real-rate relation
DEFAULT_RR_SIGNAL_WINDOW = 12   # real-rate "not rising" lookback (== regime_gate)
DEFAULT_DOMINANCE_THRESHOLD = 0.5  # hard regime-label cut on the probability

# Standard thresholds for mapping the fingerprint to a probability. corr is
# "significantly negative" at -0.3 (a conventional moderate effect) and
# "broken/positive" at 0.0 — the linear map between them is the real-rate→
# de-dollarization axis. Divergence share is benchmarked against the ~0.25
# expected if Δgold and Δreal_rate were independent (0.5·0.5), so a window
# where gold and the real rate rise together more often than chance is the
# de-dollarization signal; 0.5 is "a majority of months diverge".
DEFAULT_CORR_NEG = -0.3
DEFAULT_CORR_BREAK = 0.0
DEFAULT_DIV_CHANCE = 0.25   # P(Δgold>0 ∧ Δrr>0) if independent
DEFAULT_DIV_HI = 0.5        # majority-of-window divergence → strong signal


def _log_diff(s: pd.Series) -> pd.Series:
    """First difference of the log — the monthly log return for a price."""
    return np.log(s).diff()


def rolling_gold_realrate_corr(
    gold_nominal: pd.Series,
    real_rate: pd.Series,
    window: int = DEFAULT_CORR_WINDOW,
) -> pd.Series:
    """Rolling correlation of monthly Δlog gold vs Δreal_rate, using data ≤ t
    only (a trailing `.rolling(window)`). NaN until the window fills.

    Δgold is the log return (a price is multiplicative); Δreal_rate is the
    level difference (basis points are an absolute quantity). corr() is
    invariant to scale/sign of either input, so only the *relation* matters."""
    if window <= 0:
        # window<=0 makes rolling read current/future data → look-ahead
        raise ValueError(f"window must be a positive integer, got {window}")
    dg = _log_diff(gold_nominal)
    drr = real_rate.diff()
    return dg.rolling(window, min_periods=window).corr(drr)


def divergence_share(
    gold_nominal: pd.Series,
    real_rate: pd.Series,
    window: int = DEFAULT_CORR_WINDOW,
) -> pd.Series:
    """Trailing share of months in the window with Δgold>0 AND Δreal_rate>0 —
    i.e. gold rising *with* a rising real rate, the classic negative relation
    breaking down. Uses data ≤ t only. NaN until the window fills.

    Under the classic real-rate regime this share is low (gold falls when the
    real rate rises); under de-dollarization gold rises anyway, so the share
    climbs toward/above the ~0.25 chance level."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    dg = _log_diff(gold_nominal)
    drr = real_rate.diff()
    both_up = (dg > 0) & (drr > 0)
    return both_up.astype(float).rolling(window, min_periods=window).mean()


def level_divergence(
    gold_nominal: pd.Series,
    real_rate: pd.Series,
    window: int = DEFAULT_CORR_WINDOW,
) -> pd.Series:
    """Trailing-window *level* divergence: 1.0 when, over the past `window`
    months, gold is HIGHER and the real rate is also HIGHER — i.e. gold rose
    even though the real rate rose, the classic negative LEVEL relation broken.
    Uses data ≤ t only (`gold[t]` vs `gold[t-window]`). NaN until the window
    fills.

    This is the strong reading of the de-dollarization fingerprint: the
    post-2022 break is a *level* divergence (gold up ~46% while the 10y real
    rate climbed from −1% to +2%), which a per-month change-correlation can
    miss because the monthly co-movement can stay negative even as the levels
    pull apart. Δgold uses the log level (price is multiplicative); Δreal_rate
    is the level difference (rates are absolute)."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    logp = np.log(gold_nominal)
    gold_up = logp - logp.shift(window)        # trailing log return over window
    rr_up = real_rate - real_rate.shift(window)  # trailing real-rate change
    both = (gold_up > 0) & (rr_up > 0)
    out = both.astype(float)
    out[gold_up.isna() | rr_up.isna()] = np.nan  # warm-up → NaN (no default)
    return out


def dominance_probability(
    gold_nominal: pd.Series,
    real_rate: pd.Series,
    window: int = DEFAULT_CORR_WINDOW,
    *,
    corr_neg: float = DEFAULT_CORR_NEG,
    corr_break: float = DEFAULT_CORR_BREAK,
    div_chance: float = DEFAULT_DIV_CHANCE,
    div_hi: float = DEFAULT_DIV_HI,
    cb_demand: Optional[pd.Series] = None,
) -> pd.Series:
    """Ex-ante probability ∈ [0,1] that gold is in the **de-dollarization-
    dominant** regime at each month t (1 = de-dollarization, 0 = real-rate).

    Three structural sub-signals are combined by element-wise max (any one
    firing is sufficient evidence — they capture the same breakdown from
    different angles), plus an optional CB-demand confirm:
      p_corr : rolling Δ-correlation mapped linearly from corr_neg (→0,
               real-rate dominant) to corr_break (→1, relation broken/positive).
      p_div  : per-month divergence share mapped from div_chance (→0, the
               independent baseline) to div_hi (→1, majority of months diverge).
      p_level: trailing-window *level* divergence (gold up AND real rate up over
               the window) → 1.0. This is the signal that actually fires in the
               post-2022 break, which the change-based p_corr/p_div can miss
               because the monthly co-movement stayed negative even as the
               levels pulled apart.
      cb_demand (optional): a monthly net-buying proxy. A trailing-mean
               positive reading (sustained official accumulation) confirms
               de-dollarization at full strength; it can only *raise* the
               probability, never manufacture it where the gold–real-rate
               relation does not cooperate.

    All pieces are trailing rollings / forward shifts only → no look-ahead.
    NaN where the window has not yet filled (so warm-up trims cleanly)."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    if corr_break <= corr_neg:
        raise ValueError(
            f"corr_break ({corr_break}) must be > corr_neg ({corr_neg}) "
            "for a monotonic real-rate→de-dollarization axis"
        )
    if not (div_chance < div_hi):
        raise ValueError(
            f"div_chance ({div_chance}) must be < div_hi ({div_hi})"
        )

    corr = rolling_gold_realrate_corr(gold_nominal, real_rate, window)
    p_corr = ((corr - corr_neg) / (corr_break - corr_neg)).clip(0.0, 1.0)

    div = divergence_share(gold_nominal, real_rate, window)
    p_div = ((div - div_chance) / (div_hi - div_chance)).clip(0.0, 1.0)

    p_level = level_divergence(gold_nominal, real_rate, window)

    # element-wise max; all three share the same ~window-month warm-up, so the
    # NaN union is just that warm-up (np.maximum propagates NaN → trimmed later).
    p = np.maximum(np.maximum(p_corr, p_div), p_level)

    if cb_demand is not None:
        cb = cb_demand.reindex(gold_nominal.index)
        cb_mean = cb.rolling(window, min_periods=window).mean()
        # sustained net buying (trailing mean > 0) → confirm at full strength;
        # can only raise p, never lower it. cb_mean NaN where not yet filled.
        cb_pos = (cb_mean > 0).astype(float)
        cb_pos[cb_mean.isna()] = np.nan
        p = np.maximum(p, cb_pos)

    return p


def regime_label(
    prob: pd.Series,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> pd.Series:
    """Hard regime label per month: 1 = de-dollarization-dominant, 0 =
    real-rate-dominant. NaN where `prob` is NaN (no history yet) — the label
    does NOT silently default to either regime on missing data."""
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0,1), got {threshold}")
    label = (prob >= threshold).astype(float)
    label[prob.isna()] = np.nan
    label.name = "regime_label"
    return label


def s3_dominance(
    panel: pd.DataFrame,
    prob: pd.Series,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    rr_window: int = DEFAULT_RR_SIGNAL_WINDOW,
) -> pd.Series:
    """Regime-conditional long-only position ∈ [0,1], decided at t, held t+1.

      real-rate-dominant months (prob→0) → real-rate signal: hold when the real
                                             rate is NOT rising over rr_window
      de-dollarization months (prob→1)   → trend signal: the 3/6/12 blend

    The two exposures are blended by the probability (a smooth handoff, not a
    hard switch), then vol-targeted and clipped to [0,1] — the same sizing and
    cash-leg machinery as S1, so the comparison is purely "which signal, when".

    Warm-up stays NaN (before the corr window, the rr window, the longest
    trend lookback, and the vol window all have history) so `run_backtest`
    trims those months rather than crediting a cash stub."""
    if rr_window <= 0:
        raise ValueError(f"rr_window must be a positive integer, got {rr_window}")
    rr = panel["real_rate_10y"]
    rr_chg = rr - rr.shift(rr_window)        # forward shift → reads the past
    rr_falling = (rr_chg <= 0).astype(float)  # 1 when real rate not rising
    rr_falling[rr_chg.isna()] = np.nan        # warm-up NaN

    trend = trend_exposure(panel["gold_nominal"], lookbacks)  # NaN during warm-up
    vs = vol_scale(panel["gold_ret"], target_vol, vol_window)  # NaN during warm-up

    raw = (1.0 - prob) * rr_falling + prob * trend
    return (raw * vs).clip(lower=0.0, upper=1.0)


def regime_timeline(
    label: pd.Series,
    freq: str = "YE",
) -> pd.DataFrame:
    """Mean de-dollarization share per period (default: per year). A value of
    0.0 = wholly real-rate-dominant period, 1.0 = wholly de-dollarization.
    Read as the timeline that should show the post-2022 handover."""
    s = label.dropna()
    if s.empty:
        return pd.DataFrame(columns=["de-dollarization_share", "n_months"])
    grp = s.resample(freq)
    out = pd.DataFrame({
        "de-dollarization_share": grp.mean(),
        "n_months": grp.size(),
    })
    out.index.name = label.index.name or "date"
    return out
