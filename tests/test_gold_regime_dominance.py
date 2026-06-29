"""Tests for the gold regime-dominance classifier + conditional strategy.

Offline by construction: every test injects synthetic `anchor_fn`/`fetch_fn`
into `build_timing_panel` (the same injection seam PR #5's tests use), so no
network/FRED/datasets.io dependency.

Covers:
  1. no look-ahead — truncating the future leaves past probabilities unchanged
  2. fingerprint logic — classic negative gold/real-rate relation → real-rate
     dominant; gold rising with a rising real rate → de-dollarization
  3. regime-conditional switch — prob→1 reproduces S1 trend; prob→0 reproduces
     the real-rate "not rising" signal; intermediate is a smooth blend
  4. same-track comparability — S0/S1/SD all run through the shared engine on
     a common investable window; SD positions stay in [0,1]
  5. missing-data fallback — no real rate → all-NaN (honest, no default label);
     cb_demand=None is the default path; a positive cb_demand only raises prob
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_trend_timing import (
    build_timing_panel,
    compute_metrics,
    run_backtest,
    s0_buy_hold,
    s1_trend,
    vol_scale,
)
from lib.gold_regime_dominance import (
    DEFAULT_RR_SIGNAL_WINDOW,
    DEFAULT_TARGET_VOL,
    DEFAULT_VOL_WINDOW,
    divergence_share,
    dominance_probability,
    level_divergence,
    regime_label,
    regime_timeline,
    rolling_gold_realrate_corr,
    s3_dominance,
)


# ── fixture: a synthetic but well-behaved monthly panel via injection ────
def _panel(gold=None, real_rate=None, n: int = 180, seed: int = 0, tbill: float = 4.0):
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    if gold is None:
        rng = np.random.RandomState(seed)
        gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    if real_rate is None:
        rng2 = np.random.RandomState(seed + 1)
        real_rate = pd.Series(2.0 + np.cumsum(rng2.randn(n) * 0.1), index=idx)
    gold, real_rate = gold.align(real_rate, join="outer")
    usd = pd.Series(np.linspace(100.0, 110.0, n), index=idx)
    anchor = SimpleNamespace(
        data=pd.DataFrame({"gold_nominal": gold, "real_rate_10y": real_rate}, index=idx)
    )

    def fetch_fn(series_id, start="1968-01-01"):
        if series_id in ("TWEXBMTH", "DTWEXBGS"):
            return usd
        if series_id == "TB3MS":
            return pd.Series(tbill, index=idx)
        return pd.Series(np.nan, index=idx)

    tp = build_timing_panel(
        start="1980-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    return tp.data


# ── 1. no look-ahead ─────────────────────────────────────────────────────
def test_no_lookahead_truncation_invariance():
    """A classifier value at t must depend only on data ≤ t. Truncating the
    series after month N must leave the first N months' probabilities identical
    (NaN where they were NaN)."""
    n = 200
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(7)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    real_rate = pd.Series(2.0 + np.cumsum(rng.randn(n) * 0.1), index=idx)

    window = 36
    full = dominance_probability(gold, real_rate, window=window)

    cut = 120
    trunc = dominance_probability(gold.iloc[:cut], real_rate.iloc[:cut], window=window)

    a = full.iloc[:cut].to_numpy()
    b = trunc.to_numpy()
    # equal_nan: warm-up NaNs must match too
    assert a.shape == b.shape
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_allclose(a[mask], b[mask], rtol=0, atol=0)
    # and the warm-up NaN positions agree
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


def test_corr_uses_only_trailing_window():
    """rolling_gold_realrate_corr at t depends only on [t-window+1, t]."""
    n = 80
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    gold = pd.Series(np.exp(np.cumsum(np.linspace(0.01, 0.02, n))), index=idx)
    rr = pd.Series(np.cumsum(np.linspace(0.1, 0.2, n)), index=idx)
    window = 24
    full = rolling_gold_realrate_corr(gold, rr, window)
    # corrupt everything after month 60 — the first 60 months must be untouched
    rr2 = rr.copy()
    rr2.iloc[60:] = rr2.iloc[60:] * 1e6
    full2 = rolling_gold_realrate_corr(gold, rr2, window)
    np.testing.assert_array_equal(np.isnan(full.iloc[:60]), np.isnan(full2.iloc[:60]))
    a = full.iloc[:60].to_numpy()
    b = full2.iloc[:60].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_allclose(a[mask], b[mask], rtol=0, atol=0)


# ── 2. fingerprint logic ─────────────────────────────────────────────────
def test_fingerprint_classic_negative_is_real_rate_dominant():
    """Gold rising while the real rate falls (perfect negative co-movement) →
    low de-dollarization probability = real-rate-dominant."""
    n = 120
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)          # gold log-return, all positive
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)  # real rate falls as gold rises
    p = dominance_probability(gold, rr, window=36)
    assert p.dropna().max() < 0.3            # real-rate-dominant throughout


def test_fingerprint_divergence_is_de_dollarization():
    """Gold rising WITH a rising real rate (the classic relation breaks) →
    high de-dollarization probability."""
    n = 120
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(dg * 10.0), index=idx)   # real rate rises WITH gold
    p = dominance_probability(gold, rr, window=36)
    assert p.dropna().min() > 0.7            # de-dollarization throughout


def test_divergence_share_bounds_and_warmup():
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    gold = pd.Series(np.exp(np.cumsum(np.random.RandomState(0).randn(n) * 0.02)), index=idx)
    rr = pd.Series(np.cumsum(np.random.RandomState(1).randn(n) * 0.1) + 2.0, index=idx)
    div = divergence_share(gold, rr, window=24)
    valid = div.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()
    # warm-up (first `window` months) is NaN, not a silently-wrong number — the
    # first diff() NaN is masked so min_periods=window requires `window` real obs
    assert div.iloc[:24].isna().all()


def test_level_divergence_catches_2022_style_break():
    """The post-2022 break is a LEVEL divergence: gold rises over a multi-year
    window while the real rate ALSO rises, even though month-to-month they can
    still co-move negatively. level_divergence must fire (→1) here, and the
    combined probability must call de-dollarization — the change-only corr/div
    signals alone would miss it."""
    n = 80
    idx = pd.date_range("2019-01-31", periods=n, freq="ME")
    window = 24
    rng = np.random.RandomState(3)
    shock = rng.randn(n) * 0.01
    dg = 0.012 + shock           # gold up on average, monthly noise = +shock
    drr = 0.02 - shock * 5.0     # real rate up on average, monthly = -shock (anti)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(drr), index=idx)
    lvl = level_divergence(gold, rr, window=window).dropna()
    assert (lvl == 1.0).all()            # the LEVEL divergence fires…
    p = dominance_probability(gold, rr, window=window)
    assert p.dropna().min() >= 0.99      # …→ combined prob calls de-dollarization


def test_level_divergence_warmup_and_no_lookahead():
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    gold = pd.Series(np.exp(np.cumsum(np.full(n, 0.01))), index=idx)
    rr = pd.Series(np.cumsum(np.full(n, 0.02)), index=idx)
    window = 24
    lvl = level_divergence(gold, rr, window=window)
    assert lvl.iloc[:window].isna().all()        # warm-up NaN, no default
    trunc = level_divergence(gold.iloc[:40], rr.iloc[:40], window=window)
    a, b = lvl.iloc[:40].to_numpy(), trunc.to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    np.testing.assert_allclose(a[mask], b[mask])


def test_level_divergence_interior_gap_yields_nan():
    """P2 regression: level_divergence reads only the two endpoints, so a real-
    rate gap INSIDE the window would otherwise be invisible and let it fire on
    an incomplete window. Now the whole window must be present → NaN at/after a
    mid-window hole, and so must the combined probability."""
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    gold = pd.Series(np.exp(np.cumsum(np.full(n, 0.01))), index=idx)
    rr = pd.Series(np.cumsum(np.full(n, 0.02)), index=idx)
    window = 12
    # punch a hole at month 30 — the windows ending at 30..(30+window-1) straddle it
    rr_hole = rr.copy()
    rr_hole.iloc[30] = np.nan
    lvl = level_divergence(gold, rr_hole, window=window)
    full = level_divergence(gold, rr, window=window)
    # months whose [t-window, t] contains index 30 must be NaN, not a fake 0/1
    affected = lvl.index[30: 30 + window]
    assert lvl.loc[affected].isna().all()
    # unaffected months (whose window avoids the hole) agree with the complete case
    clean = lvl.index[window + 1: 30]
    np.testing.assert_array_equal(np.isnan(lvl.loc[clean].to_numpy()),
                                  np.isnan(full.loc[clean].to_numpy()))
    # the hole also propagates to the combined probability (no fabricated regime)
    p_hole = dominance_probability(gold, rr_hole, window=window)
    p_full = dominance_probability(gold, rr, window=window)
    assert p_hole.loc[affected].isna().all()


def test_nan_in_one_subsignal_does_not_blank_valid_signal():
    """P1-2 regression: a NaN in one sub-signal must NOT wipe out the others. A
    perfectly FLAT real rate makes Δreal_rate exactly 0 → zero variance → rolling
    corr is NaN; the divergence/level signals are well-defined (both 0 here, a
    flat rate is real-rate-dominant). The combined probability must be the valid
    0.0, never NaN (the old np.maximum propagated the NaN and blanked it)."""
    n = 80
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    window = 24
    rng = np.random.RandomState(11)
    gold = pd.Series(np.exp(np.cumsum(0.01 + rng.randn(n) * 0.02)), index=idx)
    rr = pd.Series(2.0, index=idx)             # exactly flat → Δrr ≡ 0.0 (exact)
    corr = rolling_gold_realrate_corr(gold, rr, window)
    assert corr.dropna().empty                 # corr is NaN everywhere (zero var)
    p = dominance_probability(gold, rr, window=window)
    assert p.iloc[window:].notna().all()       # NOT blanked to NaN by the corr NaN
    np.testing.assert_allclose(p.iloc[window:].to_numpy(), 0.0)  # real-rate-dominant


def test_cb_demand_all_nan_preserves_base_probability():
    """P1-2 regression (cb half): an entirely-unavailable cb_demand (all NaN,
    e.g. pre-coverage) must leave the probability identical to the no-cb path —
    cb can only RAISE p, never blank it. The old np.maximum(p, NaN)=NaN broke
    this, wiping every valid month where cb had no data."""
    panel = _panel()
    g, rr = panel["gold_nominal"], panel["real_rate_10y"]
    base = dominance_probability(g, rr, window=36)
    cb_nan = pd.Series(np.nan, index=panel.index)
    with_cb = dominance_probability(g, rr, window=36, cb_demand=cb_nan)
    np.testing.assert_array_equal(np.isnan(base.to_numpy()), np.isnan(with_cb.to_numpy()))
    m = base.notna().to_numpy()
    np.testing.assert_allclose(base.to_numpy()[m], with_cb.to_numpy()[m])


def test_s3_reindexes_prob_to_panel():
    """P2-2 regression: a prob series with extra/missing months must be aligned
    to the panel — the position is indexed exactly on panel.index, never on a
    union that introduces off-panel dates."""
    panel = _panel(n=120)
    prob = dominance_probability(panel["gold_nominal"], panel["real_rate_10y"], window=36)
    # prob carrying an extra out-of-panel month + dropping an in-panel one
    extra = pd.Series(
        {pd.Timestamp("2099-12-31"): 1.0},
    )
    prob_dirty = pd.concat([prob.iloc[5:], extra])
    pos = s3_dominance(panel, prob_dirty)
    assert pos.index.equals(panel.index)            # exactly the panel index
    assert pd.Timestamp("2099-12-31") not in pos.index


# ── 3. regime-conditional switch ─────────────────────────────────────────
def test_s3_prob_one_equals_s1_trend():
    """With prob = 1 (de-dollarization everywhere) the strategy must reduce to
    S1's trend exposure — de-dollarization months follow price."""
    panel = _panel()
    prob_one = pd.Series(1.0, index=panel.index)
    s3 = s3_dominance(panel, prob_one)
    s1 = s1_trend(panel)
    # warm-up agrees (both NaN); where defined they are identical
    np.testing.assert_array_equal(np.isnan(s3), np.isnan(s1))
    np.testing.assert_allclose(s3.dropna().to_numpy(), s1.dropna().to_numpy())


def test_s3_prob_zero_equals_real_rate_signal():
    """With prob = 0 (real-rate dominant everywhere) the strategy must reduce
    to the real-rate 'not rising' signal × vol-scale — real-rate months follow
    the real rate."""
    panel = _panel()
    prob_zero = pd.Series(0.0, index=panel.index)
    s3 = s3_dominance(panel, prob_zero)

    rr = panel["real_rate_10y"]
    rr_chg = rr - rr.shift(DEFAULT_RR_SIGNAL_WINDOW)
    rr_falling = (rr_chg <= 0).astype(float)
    rr_falling[rr_chg.isna()] = np.nan
    vs = vol_scale(panel["gold_ret"], DEFAULT_TARGET_VOL, DEFAULT_VOL_WINDOW)
    expected = (rr_falling * vs).clip(0.0, 1.0)

    np.testing.assert_array_equal(np.isnan(s3), np.isnan(expected))
    np.testing.assert_allclose(s3.dropna().to_numpy(), expected.dropna().to_numpy())


def test_s3_blend_is_between_pure_signals():
    """At prob = 0.5 the exposure must lie between the real-rate-only and the
    trend-only exposures *pointwise* (a convex blend lies between the two
    endpoints — but which endpoint is larger varies month to month, so the
    bound is the element-wise min/max, not a fixed lo≤hi ordering)."""
    panel = _panel()
    mid = pd.Series(0.5, index=panel.index)
    s3_mid = s3_dominance(panel, mid).dropna()
    rr_only = s3_dominance(panel, pd.Series(0.0, index=panel.index)).dropna()
    trend_only = s3_dominance(panel, pd.Series(1.0, index=panel.index)).dropna()
    common = s3_mid.index.intersection(rr_only.index).intersection(trend_only.index)
    m, a, b = s3_mid.loc[common], rr_only.loc[common], trend_only.loc[common]
    lo = pd.concat([a, b], axis=1).min(axis=1)
    hi = pd.concat([a, b], axis=1).max(axis=1)
    assert ((m >= lo - 1e-9) & (m <= hi + 1e-9)).all()


# ── 4. same-track comparability with S1 ──────────────────────────────────
def test_s0_s1_sd_run_on_common_window_and_sd_is_long_only():
    panel = _panel(n=180)
    prob = dominance_probability(panel["gold_nominal"], panel["real_rate_10y"], window=36)
    positions = {
        "S0_buyhold": s0_buy_hold(panel.index),
        "S1_blend": s1_trend(panel),
        "SD_blend": s3_dominance(panel, prob),
    }
    # SD positions respect the long-only 0–100% contract wherever defined
    sd = positions["SD_blend"].dropna()
    assert sd.between(0.0, 1.0).all()

    bts = {
        k: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=10.0)
        for k, p in positions.items()
    }
    assert all(len(bt) > 0 for bt in bts.values())   # every strategy investable
    for bt in bts.values():                          # metrics are computable
        m = compute_metrics(bt)
        assert m["n_months"] > 0
    # a real shared (common investable) window exists across all three —
    # the same-track comparability precondition the runner relies on
    starts = [bt.index.min() for bt in bts.values()]
    ends = [bt.index.max() for bt in bts.values()]
    assert max(starts) <= min(ends)


# ── 5. missing-data fallback ─────────────────────────────────────────────
def test_no_real_rate_yields_all_nan_no_default_label():
    """Without a real rate the classifier cannot adjudicate → all-NaN (never a
    silent default to either regime)."""
    panel = _panel()
    rr_nan = pd.Series(np.nan, index=panel.index)
    p = dominance_probability(panel["gold_nominal"], rr_nan, window=36)
    assert p.isna().all()
    s3 = s3_dominance(panel, p)
    assert s3.isna().all()
    # the engine turns an all-NaN position into an empty (never-invested) book,
    # not a crash and not a fake cash stub
    bt = run_backtest(s3, panel["gold_ret"], panel["tbill_ret"])
    assert len(bt) == 0


def test_cb_demand_cannot_manufacture_regime_without_base_fingerprint():
    """P1 regression: CB demand only CONFIRMS a regime the gold-real-rate
    fingerprint already defines — it must never MANUFACTURE one. With the real
    rate entirely missing (base probability all-NaN), even a strongly positive
    cb_demand must leave the probability all-NaN (the "no real rate → no default
    classification" honesty contract)."""
    panel = _panel()
    rr_nan = pd.Series(np.nan, index=panel.index)
    cb = pd.Series(10.0, index=panel.index)        # sustained heavy net buying
    p = dominance_probability(panel["gold_nominal"], rr_nan, window=36, cb_demand=cb)
    assert p.isna().all()                          # NOT fabricated to 1.0


def test_cb_demand_is_publication_lagged_no_lookahead():
    """P2 regression: cb_demand is shifted by cb_lag_months before use, so a CB
    figure first appearing at month t cannot influence the month-t (or earlier)
    probability. With lag=1, a single CB spike at index k may only affect the
    probability from k+1 onward — index k and before are identical to no-cb."""
    panel = _panel()
    g, rr = panel["gold_nominal"], panel["real_rate_10y"]
    base = dominance_probability(g, rr, window=12)
    k = 100
    cb = pd.Series(0.0, index=panel.index)
    cb.iloc[k] = 1e6                                # one huge spike at month k
    lagged = dominance_probability(g, rr, window=12, cb_demand=cb, cb_lag_months=1)
    # months ≤ k are untouched by a spike that is only "published" at k+1
    a = base.iloc[: k + 1].to_numpy()
    b = lagged.iloc[: k + 1].to_numpy()
    m = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_allclose(a[m], b[m])
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


def test_negative_cb_lag_rejected():
    panel = _panel()
    with pytest.raises(ValueError):
        dominance_probability(panel["gold_nominal"], panel["real_rate_10y"],
                              cb_demand=pd.Series(1.0, index=panel.index),
                              cb_lag_months=-1)


def test_s3_rejects_out_of_range_prob():
    """P3 regression: a prob outside [0,1] is a caller error that the final
    exposure clip would otherwise silently mask — s3_dominance must reject it."""
    panel = _panel()
    bad = pd.Series(1.5, index=panel.index)         # > 1
    with pytest.raises(ValueError):
        s3_dominance(panel, bad)
    bad2 = pd.Series(-0.2, index=panel.index)       # < 0
    with pytest.raises(ValueError):
        s3_dominance(panel, bad2)


def test_cb_demand_none_is_default_and_positive_cb_only_raises_prob():
    """cb_demand=None is the default path (no CB feed). A sustained positive
    cb_demand (net official buying) can only raise the probability, and lifts
    it to full strength where it is positive."""
    panel = _panel()
    g, rr = panel["gold_nominal"], panel["real_rate_10y"]
    base = dominance_probability(g, rr, window=36)               # cb=None path
    cb = pd.Series(0.0, index=panel.index)
    cb.iloc[80:] = 5.0                                           # sustained net buying
    with_cb = dominance_probability(g, rr, window=36, cb_demand=cb)

    common_idx = base.dropna().index
    # monotonic: cb never lowers prob
    assert (with_cb.loc[common_idx] >= base.loc[common_idx] - 1e-12).all()
    # where cb has been positive long enough to fill its window, prob is ~1
    late = with_cb.iloc[-20:].dropna()
    assert (late >= 0.99).all()


def test_cb_buying_is_coequal_fingerprint_raises_real_rate_dominant_month():
    """P2 contract: per the spec's OR fingerprint, sustained CB buying is itself
    sufficient evidence of de-dollarization — it raises the probability even on
    months the rate relation alone reads real-rate-dominant (classic negative
    co-movement, base p≈0). It is a co-equal disjunct, not a mere 'confirm'."""
    n = 120
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    # classic negative: gold up while real rate falls → base prob ≈ 0
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)
    base = dominance_probability(gold, rr, window=36)
    assert base.dropna().max() < 0.3                   # real-rate-dominant
    cb = pd.Series(5.0, index=idx)                     # sustained net buying
    with_cb = dominance_probability(gold, rr, window=36, cb_demand=cb, cb_lag_months=1)
    # CB buying lifts the (otherwise real-rate-dominant) late months to ~1
    late = with_cb.iloc[-20:].dropna()
    assert (late >= 0.99).all()


def test_cb_single_spike_does_not_stamp_full_strength_regime():
    """P2 robustness: a single large positive CB reading surrounded by zeros
    must NOT flip the whole rolling window to full-strength de-dollarization.
    With cb_min_share=0.5 (default), one positive in a 24-month window (share
    ~1/24) is far below the majority bar → no cb-driven raise anywhere."""
    n = 120
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)        # real-rate-dominant
    base = dominance_probability(gold, rr, window=24)
    cb = pd.Series(0.0, index=idx)
    cb.iloc[60] = 1e9                                       # one huge spike, rest zero
    with_cb = dominance_probability(gold, rr, window=24, cb_demand=cb, cb_lag_months=0)
    common = base.dropna().index
    # the spike must not raise any month above the base (real-rate-dominant) call
    np.testing.assert_array_less(
        with_cb.loc[common].to_numpy(), base.loc[common].to_numpy() + 0.01 + 1e-9)


def test_cb_quarterly_sparse_feed_is_forward_filled_and_fires():
    """P2 contract: a sparse (quarterly) CB feed reindexed onto the monthly
    panel is forward-filled so the rolling window can actually fill, and a
    sustained majority-positive quarterly pattern DOES fire de-dollarization
    (the feed is usable, not silently never-triggering)."""
    n = 120
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)        # real-rate-dominant
    base = dominance_probability(gold, rr, window=36)
    # quarterly observations only (every 3rd month), all positive (sustained)
    cb = pd.Series(np.nan, index=idx)
    cb.iloc[::3] = 5.0
    with_cb = dominance_probability(gold, rr, window=36, cb_demand=cb, cb_lag_months=0)
    # the late window is lifted above the (real-rate-dominant) base by CB buying
    late_base = base.iloc[-20:].dropna()
    late_cb = with_cb.loc[late_base.index]
    assert (late_cb >= late_base + 0.5).all()


def test_cb_non_month_end_timestamps_are_normalized_not_dropped():
    """P2 regression: a CB feed on quarter-end / mid-month release dates (not the
    panel's month-end timestamps) used to be silently dropped by reindex → all
    NaN → never fires. The feed is now resampled to month-end first, so a
    sustained positive non-month-end feed still lifts the probability."""
    n = 120
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)            # real-rate-dominant
    base = dominance_probability(gold, rr, window=36)
    # CB readings on the 15th of every month (NOT month-end) — all positive
    cb_idx = pd.date_range("1985-01-15", periods=n, freq="MS") + pd.Timedelta(days=14)
    cb = pd.Series(5.0, index=cb_idx)
    with_cb = dominance_probability(gold, rr, window=36, cb_demand=cb, cb_lag_months=0)
    late_base = base.iloc[-20:].dropna()
    late_cb = with_cb.loc[late_base.index]
    assert (late_cb >= late_base + 0.5).all()                   # normalized → fires


def test_cb_quarterly_period_index_lands_at_quarter_end_no_lookahead():
    """P2 regression: a quarterly PeriodIndex CB feed (2022Q1) must land at its
    QUARTER END (Mar 31), not the period start (Jan 31). The old to_timestamp
    ("M") default used the start → with cb_lag_months=1 a Q1 reading was usable
    in Feb, before the quarter closed — a look-ahead. Now a quarter's reading
    cannot affect any probability at or before (quarter_end + cb_lag_months)."""
    n = 120
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")
    dg = np.linspace(0.005, 0.02, n)
    gold = pd.Series(np.exp(np.cumsum(dg)), index=idx)
    rr = pd.Series(np.cumsum(-dg * 10.0), index=idx)            # real-rate-dominant
    # quarterly PeriodIndex: one positive reading in 1986Q1 (Jan-Mar), rest NaN
    cb = pd.Series(np.nan, index=pd.period_range("1985Q1", periods=40, freq="Q"))
    cb.loc["1986Q1"] = 5.0
    lag = 1
    with_cb = dominance_probability(gold, rr, window=12, cb_demand=cb, cb_lag_months=lag)
    # 1986Q1 ends 1986-03-31; with a 1-month publication lag it is usable only
    # from 1986-04-30 onward. Every month ≤ 1986-03-31 must be UNAFFECTED (equal
    # to the no-cb base) — the reading is not known there.
    base = dominance_probability(gold, rr, window=12)
    pre = with_cb.index <= pd.Timestamp("1986-03-31")
    a, b = base[pre].to_numpy(), with_cb[pre].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_allclose(a[mask], b[mask])


def test_regime_label_and_timeline():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    prob = pd.Series(np.nan, index=idx, dtype=float)
    prob.iloc[40:50] = 0.2          # real-rate-dominant block
    prob.iloc[50:] = 0.9            # de-dollarization block
    label = regime_label(prob)
    assert label.iloc[:40].isna().all()           # warm-up / missing → NaN
    assert label.iloc[40:50].sum() == 0           # 0.2 < 0.5 → real-rate (0)
    assert label.iloc[50:].mean() == 1.0          # 0.9 ≥ 0.5 → de-dollarization (1)
    tl = regime_timeline(label, freq="YE")
    assert "de-dollarization_share" in tl.columns
    assert tl["n_months"].sum() == (~prob.isna()).sum()


def test_segment_investable_months_handles_empty_and_nan():
    """P1 regression: a segment that overlaps the panel but has NO investable
    months yields an all-NaN n_months column; segment_investable_months must
    return 0 (→ the runner skips it) rather than crashing on int(NaN)."""
    from scripts.gold_dominance_backtest import segment_investable_months

    # all-NaN n_months (what metrics_table returns for an empty/no-overlap slice)
    m_nan = pd.DataFrame({"n_months": [float("nan"), float("nan")]},
                         index=["S1_blend", "SD_blend"])
    assert segment_investable_months(m_nan) == 0
    # empty frame
    assert segment_investable_months(pd.DataFrame({"n_months": []})) == 0
    # mixed: the min over the present (non-NaN) values
    m_mixed = pd.DataFrame({"n_months": [40.0, float("nan"), 12.0]})
    assert segment_investable_months(m_mixed) == 12
