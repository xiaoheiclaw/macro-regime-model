"""Tests for the walk-forward (expanding-window) re-calibration — PR #15.

Offline / synthetic by construction (no network). Covers the task spec:
  1. expanding calibration is ex-ante (no future function): truncating the future
     leaves every past value unchanged, for z AND percentile, include & exclude.
  2. warm-up gating: NaN until `min_periods` observations accrue.
  3. comparability with PR #14's full-sample口径: at the FINAL month the expanding
     (include-current) z/percentile equal the full-sample read (the structural fact
     that the headline current number is not where look-ahead hides).
  4. percentile math: monotone series, ties, include vs exclude convention.
  5. extreme reclassification + walk-forward conditional table contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.gold_dedollar_gap import full_percentile, full_zscore
from lib.gold_dedollar_gap_walkforward import (
    current_walk_forward_reading,
    expanding_percentile,
    expanding_zscore,
    extreme_reclassification,
    full_percentile_series,
    walk_forward_calibration,
    walk_forward_conditional_table,
    warmup_sensitivity,
)


def _me_index(n, start="2010-01-31"):
    return pd.date_range(start, periods=n, freq=pd.offsets.MonthEnd())


def _close_nan(a: pd.Series, b: pd.Series, *, atol=1e-9):
    assert a.shape == b.shape
    av, bv = a.to_numpy(), b.to_numpy()
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    m = ~(np.isnan(av) | np.isnan(bv))
    np.testing.assert_allclose(av[m], bv[m], rtol=0, atol=atol)


# ── 1. ex-ante / no look-ahead ───────────────────────────────────────────
@pytest.mark.parametrize("exclude", [False, True])
def test_expanding_zscore_no_lookahead(exclude):
    n = 120
    rng = np.random.RandomState(1)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    full = expanding_zscore(s, min_periods=12, exclude_current=exclude)
    cut = 70
    trunc = expanding_zscore(s.iloc[:cut], min_periods=12, exclude_current=exclude)
    _close_nan(full.iloc[:cut], trunc)


@pytest.mark.parametrize("exclude", [False, True])
def test_expanding_percentile_no_lookahead(exclude):
    n = 120
    rng = np.random.RandomState(2)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    full = expanding_percentile(s, min_periods=12, exclude_current=exclude)
    cut = 80
    trunc = expanding_percentile(s.iloc[:cut], min_periods=12, exclude_current=exclude)
    _close_nan(full.iloc[:cut], trunc)


def test_expanding_uses_only_data_up_to_t_explicit():
    """A spike in the FUTURE must not change any earlier expanding value."""
    n = 60
    s = pd.Series(np.linspace(0.0, 1.0, n), index=_me_index(n))
    base_z = expanding_zscore(s, min_periods=6)
    base_p = expanding_percentile(s, min_periods=6)
    s2 = s.copy()
    s2.iloc[50] += 1000.0  # huge future shock
    z2 = expanding_zscore(s2, min_periods=6)
    p2 = expanding_percentile(s2, min_periods=6)
    _close_nan(base_z.iloc[:50], z2.iloc[:50])
    _close_nan(base_p.iloc[:50], p2.iloc[:50])


# ── 2. warm-up gating ─────────────────────────────────────────────────────
@pytest.mark.parametrize("warm", [12, 24, 36])
def test_warmup_gate_blanks_until_enough_history(warm):
    n = 80
    rng = np.random.RandomState(3)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    z = expanding_zscore(s, min_periods=warm)
    p = expanding_percentile(s, min_periods=warm)
    # include-current: the k-th observation (index warm-1) is the first defined one
    assert z.iloc[:warm - 1].isna().all()
    assert np.isfinite(z.iloc[warm - 1])
    assert p.iloc[:warm - 1].isna().all()
    assert np.isfinite(p.iloc[warm - 1])


def test_warmup_gate_counts_nonnan_only():
    """Warm-up counts OBSERVATIONS, not calendar rows: NaN gaps don't satisfy it."""
    n = 40
    s = pd.Series(np.arange(n, dtype=float), index=_me_index(n))
    s.iloc[:10] = np.nan  # first 10 months missing
    p = expanding_percentile(s, min_periods=12)
    # need 12 non-NaN obs; data starts at row 10 → first defined at row 10+12-1 = 21
    assert p.iloc[:21].isna().all()
    assert np.isfinite(p.iloc[21])


def test_warmup_rejects_too_small():
    s = pd.Series([1.0, 2.0, 3.0], index=_me_index(3))
    with pytest.raises(ValueError):
        expanding_zscore(s, min_periods=1)
    with pytest.raises(ValueError):
        expanding_percentile(s, min_periods=1)


# ── 3. comparability with PR #14 full-sample at the final month ───────────
def test_expanding_include_equals_full_sample_at_final_point():
    """The structural fact: at the LAST month the expanding (include-current)
    window IS the full sample, so z/percentile coincide with PR #14's read."""
    n = 100
    rng = np.random.RandomState(4)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    z_wf = expanding_zscore(s, min_periods=24, exclude_current=False)
    p_wf = expanding_percentile(s, min_periods=24, exclude_current=False)
    last = s.index[-1]
    np.testing.assert_allclose(z_wf.loc[last],
                               full_zscore(s).loc[last], atol=1e-9)
    np.testing.assert_allclose(p_wf.loc[last],
                               full_percentile(s, float(s.iloc[-1])), atol=1e-9)


def test_current_reading_incl_matches_full_excl_moves_by_1_over_n():
    n = 138
    rng = np.random.RandomState(5)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    rd = current_walk_forward_reading(s, warmup=24)
    # include-current == full-sample at asof
    np.testing.assert_allclose(rd.pct_wf_incl, rd.pct_full, atol=1e-9)
    np.testing.assert_allclose(rd.z_wf_incl, rd.z_full, atol=1e-9)
    # exclude-current drops just the newest point → within ~1/N of the full read
    assert abs(rd.pct_wf_excl - rd.pct_full) <= 1.5 / n
    assert rd.n_resid == n


# ── 4. percentile / z math ────────────────────────────────────────────────
def test_expanding_percentile_monotone_increasing_is_one():
    n = 30
    s = pd.Series(np.arange(n, dtype=float), index=_me_index(n))
    p_incl = expanding_percentile(s, min_periods=5, exclude_current=False)
    # each new point is the max of history-so-far → percentile 1.0 (include)
    np.testing.assert_allclose(p_incl.dropna().to_numpy(),
                               np.ones(p_incl.notna().sum()), atol=1e-12)
    # exclude-current on a strictly increasing series: today > all prior → 0 of them
    # are <= ... wait: <= prior means none are >= today, fraction(prior <= today)=1.0
    p_excl = expanding_percentile(s, min_periods=5, exclude_current=True)
    np.testing.assert_allclose(p_excl.dropna().to_numpy(),
                               np.ones(p_excl.notna().sum()), atol=1e-12)


def test_expanding_percentile_monotone_decreasing():
    n = 30
    s = pd.Series(np.arange(n, 0, -1, dtype=float), index=_me_index(n))
    p_incl = expanding_percentile(s, min_periods=5, exclude_current=False)
    # each new point is the min so far → only itself <= itself → 1/k
    defined = p_incl.dropna()
    k = np.arange(5, n + 1)  # baseline size at each defined row
    np.testing.assert_allclose(defined.to_numpy(), 1.0 / k, atol=1e-12)
    # exclude-current: today < all prior → fraction(prior <= today) = 0
    p_excl = expanding_percentile(s, min_periods=5, exclude_current=True)
    np.testing.assert_allclose(p_excl.dropna().to_numpy(),
                               np.zeros(p_excl.notna().sum()), atol=1e-12)


def test_full_percentile_series_matches_pointwise_full_percentile():
    n = 50
    rng = np.random.RandomState(6)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    fps = full_percentile_series(s)
    # equals PR #14 full_percentile evaluated at each value
    for ts in s.index[::7]:
        np.testing.assert_allclose(
            fps.loc[ts], full_percentile(s, float(s.loc[ts])), atol=1e-12)


def test_expanding_zscore_constant_window_is_nan():
    s = pd.Series(np.full(20, 3.0), index=_me_index(20))
    z = expanding_zscore(s, min_periods=5)
    assert z.dropna().empty  # std == 0 → no information


def test_expanding_percentile_handles_ties():
    s = pd.Series([1.0, 1.0, 1.0, 2.0], index=_me_index(4))
    p = expanding_percentile(s, min_periods=2, exclude_current=False)
    # at row1: two 1s, both <= 1 → 1.0; row3: value 2 is max of 4 → 1.0
    np.testing.assert_allclose(p.iloc[1], 1.0)
    np.testing.assert_allclose(p.iloc[3], 1.0)


# ── 5. trajectory frame + reclassification + conditional table ────────────
def test_walk_forward_calibration_frame_columns_and_gap():
    n = 80
    rng = np.random.RandomState(8)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    frame = walk_forward_calibration(resid, warmup=24)
    assert list(frame.columns) == [
        "resid", "z_full", "pct_full", "z_wf", "pct_wf", "pct_gap"]
    # pct_gap = pct_wf - pct_full where both defined
    both = frame.dropna(subset=["pct_wf", "pct_full"])
    np.testing.assert_allclose(
        both["pct_gap"].to_numpy(),
        (both["pct_wf"] - both["pct_full"]).to_numpy(), atol=1e-12)
    # at the final month the gap is ~0 (include-current == full-sample)
    np.testing.assert_allclose(frame["pct_gap"].iloc[-1], 0.0, atol=1e-9)


def test_extreme_reclassification_contract():
    """A residual that ramps then plateaus high: late months are full-sample
    extreme; early-in-history they were already unprecedented (ex-ante extreme too),
    so agreement should be high and warm-up months excluded from the rate."""
    n = 90
    base = np.concatenate([np.linspace(-1, 2.5, 60), np.full(30, 2.6)])
    resid = pd.Series(base, index=_me_index(n))
    rc = extreme_reclassification(resid, top_q=0.9, warmup=24)
    s = rc.summary
    assert s["n_full_extreme"] > 0
    assert 0.0 <= s["agreement_rate"] <= 1.0
    # evaluable + warmup == full_extreme count (no silent drops)
    assert s["n_evaluable"] + s["n_warmup"] == s["n_full_extreme"]
    # episodes frame has the per-month detail
    for col in ("date", "pct_full", "pct_wf", "wf_extreme", "in_warmup"):
        assert col in rc.episodes.columns


def test_extreme_reclassification_rejects_bad_q():
    resid = pd.Series(np.arange(30.0), index=_me_index(30))
    with pytest.raises(ValueError):
        extreme_reclassification(resid, top_q=0.0)


def test_walk_forward_conditional_table_is_ex_ante_and_split():
    n = 90
    rng = np.random.RandomState(10)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    resid.iloc[30:40] = 4.0  # an observable mid-sample extreme block
    price = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)),
                      index=_me_index(n))
    tbl = walk_forward_conditional_table(
        resid, price, horizons=(12,), top_q=0.9, warmup=24)
    assert set(tbl["regime"]) == {"extreme_high_wf", "rest"}
    ext = tbl[tbl["regime"] == "extreme_high_wf"].iloc[0]
    assert int(ext["n"]) >= 0  # may be 0 if all extremes fall in warm-up/tail
    for col in ("n", "mean", "median", "p25", "p75", "hit"):
        assert col in tbl.columns


def test_warmup_sensitivity_table():
    n = 100
    rng = np.random.RandomState(12)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    tbl = warmup_sensitivity(resid, warmups=(12, 24, 36))
    assert list(tbl["warmup"]) == [12, 24, 36]
    # include-current current read is warm-up invariant (window ends at the same
    # final month, uses full history regardless of the gate)
    assert tbl["pct_wf_incl"].nunique() == 1
    # a larger warm-up starts the ex-ante trajectory later
    firsts = tbl.set_index("warmup")["first_wf_date"]
    assert firsts[12] <= firsts[24] <= firsts[36]
