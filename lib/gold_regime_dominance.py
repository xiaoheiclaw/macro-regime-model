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
                           negative relation breaking down), OR sustained net
                           central-bank buying. An optional cb_demand proxy
                           (WGC net official purchases / TIC foreign-official
                           flows) is a FOURTH, co-equal disjunct (NOT a mere
                           confirm): publication-lagged, sparse feeds forward-
                           filled to monthly, and firing only when a majority of
                           the window's available readings are positive.

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
    both_up = ((dg > 0) & (drr > 0)).astype(float)
    # the first diff() is NaN; NaN>0 is False, which would otherwise count the
    # no-data month as a (biased) non-divergence and let the first window report
    # one slot early. Mark it NaN so min_periods=window requires `window` real
    # observations — honouring the "NaN until the window fills" contract.
    both_up[dg.isna() | drr.isna()] = np.nan
    return both_up.rolling(window, min_periods=window).mean()


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
    # the level signal uses only the two endpoints (t, t-window); a missing
    # observation INSIDE the window would otherwise be invisible, letting a
    # level read fire on an incomplete window. Require every month in
    # [t-window, t] to have BOTH series present.
    valid = gold_nominal.notna() & real_rate.notna()
    complete = valid.rolling(window + 1, min_periods=window + 1).sum().eq(window + 1)
    out[gold_up.isna() | rr_up.isna() | ~complete.fillna(False)] = np.nan
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
    cb_lag_months: int = 1,
    cb_min_share: float = 0.5,
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
      cb_demand (optional): a net-buying proxy (WGC quarterly central-bank
               purchases, TIC foreign-official flows, …) treated as a FOURTH,
               co-equal de-dollarization fingerprint — the spec's third
               disjunct "central-bank net buying picking up" (the fingerprint is
               an OR: corr-breakdown ∨ divergence ∨ CB-buying). Sustained net
               official buying is itself sufficient evidence of de-dollarization
               (the thesis is structural reserve demand independent of rates),
               so it is combined with the rate-relation signals by the SAME
               element-wise max — it can raise the probability even where the
               rate relation alone reads real-rate-dominant (p low). It is NOT a
               mere "confirm" of an existing de-dollarization call. It is masked
               to NaN wherever the base probability is NaN, so it never
               fabricates a regime on months with no underlying panel (warm-up /
               no real rate). The series is shifted forward by ``cb_lag_months``
               (default 1) before use to model publication lag — WGC
               central-bank-buying data is released with a quarter+ delay, so the
               month-t figure is NOT known at decision time t. Pass already-
               lagged data with ``cb_lag_months=0`` if aligned to availability.
               Sparse (e.g. quarterly) feeds are forward-filled to the monthly
               panel before rolling. "Sustained" is decided by ``cb_min_share``
               (default 0.5): the fingerprint fires only when at least that
               share of the window's AVAILABLE readings are positive — so a
               single large spike among zeros does NOT stamp a full-strength
               regime over the whole window.

    All pieces are trailing rollings / forward shifts only → no look-ahead.
    NaN where the window has not yet filled (so warm-up trims cleanly)."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    if cb_lag_months < 0:
        # a negative lag would shift CB data BACKWARD → read future releases
        raise ValueError(f"cb_lag_months must be >= 0, got {cb_lag_months}")
    if not (0.0 < cb_min_share <= 1.0):
        raise ValueError(f"cb_min_share must be in (0, 1], got {cb_min_share}")
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

    # skip-na row-wise max so "any one sub-signal firing is sufficient": a NaN
    # in one component (e.g. p_corr undefined when Δreal_rate has zero variance
    # over the window) must NOT wipe out a valid p_div/p_level. Only when ALL
    # three are NaN (the shared warm-up) is the result NaN → trimmed downstream.
    comps = pd.concat([p_corr, p_div, p_level], axis=1)
    p = comps.max(axis=1, skipna=True)
    p[comps.isna().all(axis=1)] = np.nan

    if cb_demand is not None:
        # shift forward by the publication lag so the month-t figure is only
        # used cb_lag_months later (WGC data is released with a quarter+ delay).
        # CB feeds land on arbitrary timestamps (quarter-end, mid-month release,
        # PeriodIndex) — reindexing them directly onto the month-end panel would
        # silently drop every non-month-end point (→ all-NaN → never fires).
        # Normalize to month-end FIRST: PeriodIndex → timestamp, then collapse
        # any intra-month points to their month-end via resample("ME").last().
        cb = cb_demand.sort_index().copy()
        if isinstance(cb.index, pd.PeriodIndex):
            # how="end": a quarterly Period 2022Q1 must land at its END
            # (2022-03-31), NOT its start (2022-01-31). The default "start"
            # mapping is a look-ahead — Q1's reading would be usable in Feb,
            # before the quarter closes. normalize() strips the 23:59:59 so
            # resample("ME").last() collapses cleanly to the month-end.
            cb.index = cb.index.to_timestamp("M", how="end").normalize()
        cb = cb.resample("ME").last()
        cb_raw = cb.reindex(gold_nominal.index).shift(cb_lag_months)
        # forward-fill so a (quarterly) reading carries through the inter-report
        # months; rolling min_periods=window still needs `window` post-fill
        # months, so coverage is auditable rather than silently never-firing.
        cb = cb_raw.ffill()
        n_avail = cb.rolling(window, min_periods=window).count()
        n_pos = (cb > 0).astype(float).rolling(window, min_periods=window).sum()
        # sustained net buying = a MAJORITY of the available readings in the
        # window are positive (not a bare mean>0, which one large spike could
        # satisfy alone). share = pos / available; fire when ≥ cb_min_share.
        with np.errstate(invalid="ignore", divide="ignore"):
            share = n_pos / n_avail
        cb_pos = (share >= cb_min_share).astype(float)
        cb_pos[n_avail.isna()] = np.nan
        # CB buying is a FOURTH co-equal de-dollarization fingerprint (spec's ∨):
        # combined by the same element-wise max, so it raises p even where the
        # rate relation alone reads real-rate-dominant. Masked to NaN wherever
        # base p is NaN, so it never fabricates a regime with no underlying panel
        # (warm-up / no real rate). max never lowers p.
        cb_signal = cb_pos.where(p.notna())
        p = pd.concat([p, cb_signal], axis=1).max(axis=1, skipna=True)

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
    # align prob to the panel up front: a prob series with missing/extra months
    # would otherwise let pandas union-align the arithmetic below, producing
    # off-panel dates or surprise NaN that the backtest then mis-aligns on.
    prob = prob.reindex(panel.index)
    # prob is a regime *probability* — out-of-range values are a caller error
    # that the final [0,1] exposure clip would otherwise silently mask.
    bad = prob.dropna()
    if not bad.between(0.0, 1.0).all():
        raise ValueError(
            "prob must be in [0, 1] (regime probability); got out-of-range "
            f"values, e.g. {list(bad[~bad.between(0.0, 1.0)].round(4).items())[:3]}"
        )
    rr = panel["real_rate_10y"]
    rr_chg = rr - rr.shift(rr_window)        # trailing change: rr[t] - rr[t-rr_window]
    rr_falling = (rr_chg <= 0).astype(float)  # 1 when real rate not rising
    rr_falling[rr_chg.isna()] = np.nan        # warm-up NaN

    trend = trend_exposure(panel["gold_nominal"], lookbacks)  # NaN during warm-up
    vs = vol_scale(panel["gold_ret"], target_vol, vol_window)  # NaN during warm-up

    # weight-masked blend: a zero-weight branch must contribute exactly 0, even
    # if its signal is NaN there (plain `0 * NaN` is NaN and would poison the
    # result — e.g. at prob=1 SD must equal S1 trend even where rr_falling is
    # NaN from a long rr_window or a missing real rate). Each branch keeps its
    # signal only where its weight is non-zero, else 0; so the result is NaN
    # ONLY when a *non-zero-weight* branch's signal is genuinely missing.
    w_rr = 1.0 - prob
    rr_part = rr_falling.where(w_rr != 0.0, 0.0) * w_rr
    trend_part = trend.where(prob != 0.0, 0.0) * prob
    raw = rr_part + trend_part
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
