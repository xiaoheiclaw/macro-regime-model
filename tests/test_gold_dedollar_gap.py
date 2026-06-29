"""Tests for the gold vs de-dollarization deviation monitor (PR #13).

Offline by construction: all panels/series are synthetic (no network/FRED).
Covers (per the task spec):
  1. DI construction — signed z-scored composite, equal/custom weights,
     missing-component fallback (drop & renormalize), min_present gating.
  2. deviation — rolling-OLS residual is ex-ante (truncating the future leaves
     past residuals unchanged); + = gold above its DI-implied level.
  3. z-score / percentile — full + leak-free trailing percentile contracts.
  4. historical forward-return bucketing — extreme-high vs rest, forward return
     is genuinely forward (NaN tail), conditioning split is exhaustive.
  5. stationarity handling — rolling local fit; first-difference helper.
  6. missing-component / graceful fallback — DI with one leg all-NaN.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_dedollar_gap import (
    DEFAULT_COMPONENTS,
    adjudicate,
    build_di,
    build_gap_panel,
    compute_deviation,
    conditional_forward_table,
    current_reading,
    forward_log_return,
    full_percentile,
    full_zscore,
    rolling_ols_resid,
    rolling_percentile,
    rolling_zscore,
)


def _close_equal_nan(a: pd.Series, b: pd.Series, *, atol: float = 1e-9) -> None:
    assert a.shape == b.shape
    av, bv = a.to_numpy(), b.to_numpy()
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    mask = ~(np.isnan(av) | np.isnan(bv))
    np.testing.assert_allclose(av[mask], bv[mask], rtol=0, atol=atol)


# ── 1. DI construction ───────────────────────────────────────────────────
def _panel(n=120, seed=0):
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    cb = pd.Series(np.cumsum(np.abs(rng.randn(n))), index=idx)        # rising stock
    share = pd.Series(0.30 - np.cumsum(np.abs(rng.randn(n))) * 1e-3, index=idx)  # falling
    return pd.DataFrame({"cb_cum_excess": cb, "custody_share": share}, index=idx)


def test_di_is_signed_zscore_composite():
    df = _panel()
    res = build_di(df)
    # equal weight over 2 present components
    assert set(res.weights) == {"cb_cum_excess", "custody_share"}
    np.testing.assert_allclose(list(res.weights.values()), [0.5, 0.5])
    # custody is negated: a falling share contributes POSITIVE de-dollarization,
    # so its signed component should be (mostly) increasing → positive correlation
    # with the rising cb component. DI should rise over time (both legs rise).
    di = res.di.dropna()
    assert di.iloc[-1] > di.iloc[0]
    # DI equals the per-row mean of the two signed z components
    manual = res.components.mean(axis=1)
    _close_equal_nan(res.di, manual)


def test_di_custom_weights_renormalize():
    df = _panel()
    res = build_di(df, weights={"cb_cum_excess": 3.0, "custody_share": 1.0})
    np.testing.assert_allclose(res.weights["cb_cum_excess"], 0.75)
    np.testing.assert_allclose(res.weights["custody_share"], 0.25)
    manual = (0.75 * res.components["cb_cum_excess"]
              + 0.25 * res.components["custody_share"])
    _close_equal_nan(res.di, manual)


def test_di_missing_component_fallback():
    """A component that is entirely NaN is dropped and weights renormalize → DI
    still computed from the remaining leg (graceful degradation)."""
    df = _panel()
    df["custody_share"] = np.nan
    res = build_di(df)
    assert res.dropped == ["custody_share"]
    assert list(res.weights) == ["cb_cum_excess"]
    np.testing.assert_allclose(res.weights["cb_cum_excess"], 1.0)
    # DI == the single signed z component
    _close_equal_nan(res.di, res.components["cb_cum_excess"])


def test_di_all_components_missing_raises():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"cb_cum_excess": np.nan, "custody_share": np.nan}, index=idx)
    with pytest.raises(ValueError):
        build_di(df)


def test_di_min_present_gates_partial_rows():
    """With min_present = all, a row missing one leg → DI NaN; relaxing to 1 fills
    it from the available leg."""
    df = _panel(n=40)
    df.loc[df.index[5], "custody_share"] = np.nan
    strict = build_di(df)  # min_present defaults to all (2)
    assert np.isnan(strict.di.loc[df.index[5]])
    relaxed = build_di(df, min_present=1)
    assert np.isfinite(relaxed.di.loc[df.index[5]])
    # the relaxed value at that row equals just the cb signed-z (only leg present)
    np.testing.assert_allclose(
        relaxed.di.loc[df.index[5]],
        relaxed.components["cb_cum_excess"].loc[df.index[5]])


# ── 2. deviation: rolling-OLS residual is ex-ante ────────────────────────
def test_rolling_resid_no_lookahead():
    n, w = 120, 36
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(7)
    di = pd.Series(np.cumsum(rng.randn(n)) * 0.1, index=idx)
    y = pd.Series(2.0 + 1.5 * di.to_numpy() + rng.randn(n) * 0.05, index=idx)
    full = rolling_ols_resid(y, di, w)
    cut = 80
    trunc = rolling_ols_resid(y.iloc[:cut], di.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc, atol=1e-9)


def test_rolling_resid_sign_positive_when_gold_above():
    """If gold is bumped UP above the DI-implied line at the latest point, the
    residual is positive (gold running ahead)."""
    n, w = 60, 36
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    di = pd.Series(np.linspace(-1, 1, n), index=idx)
    y = pd.Series(1.0 + 2.0 * di.to_numpy(), index=idx)  # perfect line → ~0 resid
    resid0 = rolling_ols_resid(y, di, w)
    assert abs(resid0.dropna().iloc[-1]) < 1e-6
    y2 = y.copy()
    y2.iloc[-1] += 0.5  # gold jumps above the fundamentals-implied level
    resid1 = rolling_ols_resid(y2, di, w)
    assert resid1.dropna().iloc[-1] > 0.4


def test_rolling_resid_constant_x_window_is_nan():
    n, w = 50, 24
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    di = pd.Series(np.ones(n), index=idx)  # constant → slope unidentified
    y = pd.Series(np.arange(n, dtype=float), index=idx)
    resid = rolling_ols_resid(y, di, w)
    assert resid.dropna().empty


def test_compute_deviation_reports_both_zscores():
    n = 100
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(11)
    di = pd.Series(np.cumsum(rng.randn(n)) * 0.1, index=idx)
    y = pd.Series(2.0 + 1.2 * di.to_numpy() + rng.randn(n) * 0.1, index=idx)
    dev = compute_deviation(y, di, window=36)
    assert dev.resid.notna().sum() > 0
    assert dev.gap_z_roll.notna().sum() > 0
    assert dev.gap_z_full.notna().sum() > 0
    # full z has ~zero mean / unit std over its support
    z = dev.gap_z_full.dropna()
    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(ddof=0), 1.0, atol=1e-9)


# ── 3. z-score / percentile contracts ────────────────────────────────────
def test_full_zscore_constant_is_zero():
    s = pd.Series([3.0, 3.0, 3.0])
    z = full_zscore(s)
    np.testing.assert_allclose(z.to_numpy(), [0.0, 0.0, 0.0])


def test_full_zscore_preserves_nan():
    s = pd.Series([1.0, np.nan, 3.0, 5.0])
    z = full_zscore(s)
    assert np.isnan(z.iloc[1])
    assert z.dropna().shape[0] == 3


def test_rolling_zscore_trailing_no_lookahead():
    n, w = 80, 24
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(3)
    s = pd.Series(rng.randn(n), index=idx)
    full = rolling_zscore(s, w)
    cut = 60
    trunc = rolling_zscore(s.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_full_percentile_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert full_percentile(s) == 1.0           # latest (4) is the max
    assert full_percentile(s, 2.0) == 0.5      # two of four <= 2
    assert np.isnan(full_percentile(pd.Series([], dtype=float)))


def test_rolling_percentile_leak_free_and_range():
    n, w = 100, 36
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(5)
    s = pd.Series(rng.randn(n), index=idx)
    full = rolling_percentile(s, w)
    cut = 70
    trunc = rolling_percentile(s.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc)
    pv = full.dropna()
    assert pv.min() >= 0.0 and pv.max() <= 1.0


def test_rolling_percentile_flat_window_is_nan():
    idx = pd.date_range("2010-01-31", periods=20, freq="ME")
    s = pd.Series(np.ones(20), index=idx)
    assert rolling_percentile(s, 6).dropna().empty


# ── 4. historical forward-return bucketing ───────────────────────────────
def test_forward_log_return_is_forward():
    idx = pd.date_range("2010-01-31", periods=10, freq="ME")
    price = pd.Series(np.exp(np.arange(10) * 0.1), index=idx)  # +0.1 log/mo
    fwd = forward_log_return(price, 3)
    # forward 3m log return is +0.3 everywhere observable; last 3 are NaN (forward)
    np.testing.assert_allclose(fwd.dropna().to_numpy(), [0.3] * 7, atol=1e-9)
    assert fwd.iloc[-3:].isna().all()


def test_forward_log_return_rejects_nonpositive_horizon():
    idx = pd.date_range("2010-01-31", periods=5, freq="ME")
    with pytest.raises(ValueError):
        forward_log_return(pd.Series(np.arange(1, 6.0), index=idx), 0)


def test_conditional_forward_table_splits_extreme_vs_rest():
    n = 80
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(9)
    # extremes mid-history (a spike) so forward returns are observable for them —
    # a monotone gap would park all extremes at the unobservable tail.
    gap = pd.Series(rng.randn(n), index=idx)
    gap.iloc[20:28] = 5.0                                    # extreme-high block mid-sample
    price = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    tbl = conditional_forward_table(gap, price, horizons=(12,), top_q=0.9)
    regimes = set(tbl["regime"])
    assert regimes == {"extreme_high", "rest"}
    ext_n = int(tbl[tbl["regime"] == "extreme_high"]["n"].iloc[0])
    rest_n = int(tbl[tbl["regime"] == "rest"]["n"].iloc[0])
    # extreme is the top decile of the conditioning months (n observed forward)
    assert 0 < ext_n <= rest_n
    # required summary columns present
    for col in ("n", "mean", "median", "p25", "p75", "hit"):
        assert col in tbl.columns


def test_conditional_forward_table_empty_gap():
    idx = pd.date_range("2010-01-31", periods=10, freq="ME")
    price = pd.Series(np.arange(1, 11.0), index=idx)
    gap = pd.Series(np.nan, index=idx)
    tbl = conditional_forward_table(gap, price, horizons=(12,))
    assert tbl.empty


def test_conditional_forward_table_rejects_bad_q():
    idx = pd.date_range("2010-01-31", periods=10, freq="ME")
    price = pd.Series(np.arange(1, 11.0), index=idx)
    gap = pd.Series(np.linspace(0, 1, 10), index=idx)
    with pytest.raises(ValueError):
        conditional_forward_table(gap, price, top_q=1.5)


# ── 5. current reading + verdict ─────────────────────────────────────────
def _dev_with_latest_pct(target_high=True):
    n = 100
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(13)
    di = pd.Series(np.cumsum(rng.randn(n)) * 0.1, index=idx)
    y = pd.Series(1.0 + 1.0 * di.to_numpy() + rng.randn(n) * 0.05, index=idx)
    dev = compute_deviation(y, di, window=36)
    return dev, di


def test_current_reading_fields():
    dev, di = _dev_with_latest_pct()
    cr = current_reading(dev, di, roll_window=36)
    assert cr.asof is not None
    assert 0.0 <= cr.gap_pct_full <= 1.0
    assert 0.0 <= cr.di_pct_full <= 1.0


def test_adjudicate_labels():
    from lib.gold_dedollar_gap import CurrentReading
    ts = pd.Timestamp("2026-01-31")
    extreme = CurrentReading(asof=ts, gap_z_full=2.5, gap_pct_full=0.97,
                             gap_pct_roll=1.0, di_pct_full=0.9)
    normal = CurrentReading(asof=ts, gap_z_full=0.1, gap_pct_full=0.45,
                            gap_pct_roll=0.5, di_pct_full=0.5)
    elevated = CurrentReading(asof=ts, gap_z_full=0.8, gap_pct_full=0.80,
                              gap_pct_roll=0.8, di_pct_full=0.7)
    unknown = CurrentReading(asof=None, gap_z_full=np.nan, gap_pct_full=np.nan,
                             gap_pct_roll=np.nan, di_pct_full=np.nan)
    assert adjudicate(extreme)[0] == "EXTREME"
    assert adjudicate(normal)[0] == "NORMAL"
    assert adjudicate(elevated)[0] == "ELEVATED"
    assert adjudicate(unknown)[0] == "UNKNOWN"


# ── 6. panel assembly with injected (offline) data ───────────────────────
def test_build_gap_panel_offline_injection():
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.RandomState(21)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(60) * 0.03) + 6.0), index=idx)
    share = pd.Series(0.30 - np.cumsum(np.abs(rng.randn(60))) * 1e-3, index=idx)
    base = pd.DataFrame({"gold_nominal": gold, "custody_share": share}, index=idx)
    fake_dedollar = lambda **kw: SimpleNamespace(data=base, notes={"x": "synthetic"})

    def fake_fetch(series_id, start):
        # CPI: smooth rising index on a daily grid (resampled to ME inside)
        di = pd.date_range("2009-12-31", periods=2000, freq="D")
        return pd.Series(200.0 + np.arange(len(di)) * 0.01, index=di)

    def fake_wgc(start, end):
        return pd.Series(np.cumsum(np.abs(rng.randn(60))), index=idx)

    panel = build_gap_panel(
        start="2010-01", end="2014-12",
        fetch_fn=fake_fetch, dedollar_fn=fake_dedollar, wgc_fn=fake_wgc,
    )
    df = panel.data
    for col in ("gold_nominal", "custody_share", "cpi", "cb_cum_excess",
                "ln_gold", "ln_gold_real"):
        assert col in df.columns
    assert df["cpi"].notna().any()
    assert df["cb_cum_excess"].notna().any()
    # ln_gold_real = ln(gold/cpi)
    sample = df.dropna(subset=["gold_nominal", "cpi"]).index[0]
    np.testing.assert_allclose(
        df.loc[sample, "ln_gold_real"],
        np.log(df.loc[sample, "gold_nominal"] / df.loc[sample, "cpi"]),
        atol=1e-9)
    # end-to-end DI + deviation runs on the assembled panel
    di_res = build_di(df)
    dev = compute_deviation(df["ln_gold"], di_res.di, window=24)
    assert dev.resid.notna().any()
