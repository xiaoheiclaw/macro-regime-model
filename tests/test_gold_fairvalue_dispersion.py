"""Tests for the gold fair-value dispersion signal + S4 dispersion-adjusted timing.

Offline by construction: panel/series are synthetic (no network/FRED/datasets.io).
Covers (per the task spec):
  1. no look-ahead — truncating the future leaves past implied prices / dispersion
     / rank unchanged (the leak-free rolling-calibration contract)
  2. estimator logic — rolling OLS recovers an exact linear fair-value relation;
     ratio mean-reversion reverts to the mean ratio; an estimator is NaN where its
     input is NaN (oil pre-1986)
  3. dispersion — cross-sectional std of the ln-gaps; ≥2 lenses required; a missing
     lens is skipped (not polluting); SHIFT-INVARIANT (a constant offset added to
     every gap is unchanged → the signal trades disagreement, never a single lens's
     level bias)
  4. dispersion rank — leak-free ∈[0,1], monotone in dispersion
  5. S4 positions — ∈[0,1]; high-dispersion weight ≤ low-dispersion; soft→S1 when
     dispersion is at its min; hard tercile mapping
  6. same-track — S0/S1/S4 run through the shared `run_backtest` engine on a common
     investable window; oil pre-1986 skipped yet dispersion still defined from the
     other lenses
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_fairvalue_dispersion import (
    DEFAULT_TIER_HIGH,
    DEFAULT_TIER_LOW,
    DEFAULT_TIER_MID,
    _dispersion_weight,
    _ratio_meanrev_implied,
    _rolling_ols_implied,
    build_dispersion_panel,
    dispersion,
    dispersion_rank,
    estimator_count,
    estimator_coverage,
    estimator_gaps,
    implied_cpi,
    implied_debt_gdp,
    implied_gold_copper,
    implied_gold_oil,
    implied_m2_gdp,
    implied_real_rate,
    s4_dispersion,
)
from lib.gold_trend_timing import run_backtest, s0_buy_hold, s1_trend

CALIB = 60  # smaller window so synthetic tests fill within n=160 (real default 120)


def _close_equal_nan(a: pd.Series, b: pd.Series, *, atol: float = 0.0) -> None:
    """Assert two series agree on both values (where either is non-NaN) and NaN
    positions. The no-lookahead contract requires BOTH."""
    assert a.shape == b.shape
    av, bv = a.to_numpy(), b.to_numpy()
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    mask = ~(np.isnan(av) | np.isnan(bv))
    np.testing.assert_allclose(av[mask], bv[mask], rtol=0, atol=atol)


# ── 1. no look-ahead ────────────────────────────────────────────────────
def test_rolling_ols_implied_no_lookahead():
    n, w = 160, CALIB
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(1)
    x = pd.Series(np.linspace(1.0, 4.0, n) + rng.randn(n) * 0.1, index=idx)
    ln_y = pd.Series(2.0 + 0.7 * x.to_numpy() + rng.randn(n) * 0.05, index=idx)

    full = _rolling_ols_implied(ln_y, x, w)
    cut = 120
    trunc = _rolling_ols_implied(ln_y.iloc[:cut], x.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_ratio_meanrev_implied_no_lookahead():
    n, w = 160, CALIB
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(2)
    ln_x = pd.Series(np.log(np.linspace(10, 40, n)) + rng.randn(n) * 0.05, index=idx)
    ln_y = pd.Series(np.log(np.linspace(200, 600, n)) + rng.randn(n) * 0.04, index=idx)

    full = _ratio_meanrev_implied(ln_y, ln_x, w)
    cut = 110
    trunc = _ratio_meanrev_implied(ln_y.iloc[:cut], ln_x.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_dispersion_rank_no_lookahead():
    n, w = 120, 60
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(3)
    disp = pd.Series(np.abs(rng.randn(n)) + 0.1, index=idx)
    full = dispersion_rank(disp, w)
    cut = 90
    trunc = dispersion_rank(disp.iloc[:cut], w)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_rank_uses_only_trailing_window():
    """dispersion_rank at t depends only on disp[t-w+1..t]. Corrupting the future
    must leave the first 2/3 of the rank series untouched."""
    n, w = 120, 40
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(4)
    disp = pd.Series(np.abs(rng.randn(n)) + 0.1, index=idx)
    full = dispersion_rank(disp, w)
    disp2 = disp.copy()
    disp2.iloc[80:] = disp2.iloc[80:] * 1e4  # corrupt the future
    full2 = dispersion_rank(disp2, w)
    _close_equal_nan(full.iloc[:80], full2.iloc[:80])


# ── 2. estimator logic ──────────────────────────────────────────────────
def test_rolling_ols_recovers_exact_linear_fair_value():
    """If ln_y is an exact linear function of x, the rolling OLS fits it perfectly
    and implied == exp(ln_y) once the window fills."""
    n, w = 90, 40
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    x = pd.Series(np.linspace(1.0, 3.0, n), index=idx)
    ln_y = pd.Series(1.0 + 0.5 * x.to_numpy(), index=idx)  # exact line
    y = np.exp(ln_y)
    implied = _rolling_ols_implied(ln_y, x, w)
    valid = implied.notna()
    # where defined, implied must equal y to high precision (exact in-sample fit)
    np.testing.assert_allclose(implied[valid].to_numpy(), y[valid].to_numpy(), rtol=1e-8)


def test_ratio_meanrev_constant_ratio_implies_today():
    """If the gold/commodity ratio is constant, the trailing-mean ratio equals it,
    so implied = ratio × today's commodity = today's gold."""
    n, w = 80, 30
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    ln_x = pd.Series(np.log(np.linspace(10, 30, n)), index=idx)
    c = 0.9  # constant log-ratio
    ln_y = ln_x + c
    y = np.exp(ln_y)
    implied = _ratio_meanrev_implied(ln_y, ln_x, w)
    np.testing.assert_allclose(implied.dropna().to_numpy(), y[implied.notna()].to_numpy(),
                               rtol=1e-9)


def test_estimator_nan_where_input_nan():
    """Oil-style: leading NaN in the commodity must keep the ratio estimator NaN
    there (and the trailing window must still need w real obs after the NaN ends)."""
    n, w = 100, 40
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    gold = pd.Series(np.exp(np.cumsum(np.linspace(0.01, 0.02, n))), index=idx)
    oil = pd.Series(np.linspace(30, 60, n), index=idx)
    oil.iloc[:30] = np.nan  # "pre-1986" gap
    implied = implied_gold_oil(gold, oil, w)
    # entirely NaN through the gap + the calibration window after oil starts
    assert implied.iloc[:30].isna().all()
    # the first non-NaN lands exactly at the w-th non-NaN oil observation: the
    # trailing window needs w non-NaN oil points, which first happens at the w-th.
    first_valid = implied.first_valid_index()
    assert first_valid == oil.dropna().index[w - 1]


def test_estimators_reject_nonpositive_window():
    idx = pd.date_range("1990-01-31", periods=50, freq="ME")
    y = pd.Series(np.linspace(1, 2, 50), index=idx)
    with pytest.raises(ValueError):
        _rolling_ols_implied(y, y, 0)
    with pytest.raises(ValueError):
        _ratio_meanrev_implied(y, y, -1)


# ── 3. dispersion ───────────────────────────────────────────────────────
def test_dispersion_is_cross_sectional_std_of_gaps():
    """dispersion == manual row-wise std (ddof=1) of the gaps, needing ≥2 lenses."""
    idx = pd.date_range("1990-01-31", periods=4, freq="ME")
    gaps = pd.DataFrame({
        "a": [0.10, np.nan, 0.0, 0.2],
        "b": [0.20, 0.5, 0.4, 0.8],
        "c": [-0.10, -0.1, 0.2, 0.0],
    }, index=idx)
    disp = dispersion(gaps)
    # row 0: std of [0.1,0.2,-0.1] ddof=1
    np.testing.assert_allclose(disp.iloc[0], np.std([0.1, 0.2, -0.1], ddof=1))
    # row 1: only b & c non-NaN → std of [0.5,-0.1]
    np.testing.assert_allclose(disp.iloc[1], np.std([0.5, -0.1], ddof=1))
    assert disp.notna().all()  # every row has ≥2 lenses


def test_dispersion_needs_min_two_estimators():
    idx = pd.date_range("1990-01-31", periods=3, freq="ME")
    gaps = pd.DataFrame({"a": [0.1, 0.2, np.nan], "b": [np.nan, np.nan, 0.5]}, index=idx)
    disp = dispersion(gaps)
    # rows 0,1 have one lens each → NaN; row 2 has one lens → NaN
    assert disp.isna().all()
    with pytest.raises(ValueError):
        dispersion(gaps, min_estimators=1)


def test_dispersion_missing_lens_does_not_pollute():
    """A lens that is NaN on some months is dropped those months; the others still
    yield a valid dispersion (the skip-don't-pollute contract)."""
    idx = pd.date_range("1990-01-31", periods=5, freq="ME")
    a = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    b = pd.Series([0.2, np.nan, 0.4, np.nan, 0.6], index=idx)  # intermittently NaN
    c = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx)
    gaps = pd.DataFrame({"a": a, "b": b, "c": c})
    disp = dispersion(gaps)
    assert disp.notna().all()
    # where b is NaN the dispersion equals std of just {a,c}
    np.testing.assert_allclose(disp.iloc[1], np.std([0.2, 0.1], ddof=1))
    np.testing.assert_allclose(disp.iloc[3], np.std([0.4, 0.3], ddof=1))


def test_dispersion_is_shift_invariant():
    """THE insight property: adding a constant to every gap (every lens equally
    more bullish/bearish) leaves dispersion unchanged — it measures disagreement,
    not any single lens's level bias. So the strategy can never trade one lens."""
    rng = np.random.RandomState(7)
    idx = pd.date_range("1990-01-31", periods=40, freq="ME")
    gaps = pd.DataFrame(rng.randn(40, 4), columns=list("abcd"), index=idx)
    base = dispersion(gaps)
    shifted = dispersion(gaps + 100.0)  # everyone +100% more bullish
    np.testing.assert_allclose(base.dropna().to_numpy(), shifted.dropna().to_numpy())


def test_estimator_gaps_are_log_ratios():
    idx = pd.date_range("1990-01-31", periods=3, freq="ME")
    market = pd.Series([100.0, 100.0, 100.0], index=idx)
    implieds = {"x": pd.Series([110.0, 90.0, 100.0], index=idx)}  # +10%, -10%, 0%
    gaps = estimator_gaps(implieds, market)
    np.testing.assert_allclose(gaps["x"].to_numpy(), np.log([1.1, 0.9, 1.0]))


# ── 4. dispersion rank ──────────────────────────────────────────────────
def test_dispersion_rank_reaches_zero_and_one():
    """The rank maps the trailing-window MIN → 0.0 and MAX → 1.0 — the contract a
    bare .rank(pct=True) (floor 1/window) would violate, and that the soft weight
    (1 − rank) needs to return to FULL trend at the dispersion minimum."""
    n, w = 60, 30
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    disp = pd.Series(0.30, index=idx)
    disp.iloc[40] = 0.01   # window min (inside full windows from t=40 on)
    disp.iloc[50] = 0.99   # window max
    rank = dispersion_rank(disp, w)
    assert rank.dropna().between(0.0, 1.0).all()
    np.testing.assert_allclose(rank.iloc[40], 0.0)   # own value is its window min
    np.testing.assert_allclose(rank.iloc[50], 1.0)   # own value is its window max


def test_dispersion_rank_tied_min_is_zero():
    """method='min': tied minima all take the smallest rank → 0.0 after rescale
    (not an averaged mid-rank), so repeated consensus readings gate identically
    to a single one."""
    n, w = 60, 30
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    disp = pd.Series(np.linspace(0.2, 0.4, n), index=idx)  # rising baseline
    disp.iloc[20] = 0.01
    disp.iloc[40] = 0.01   # ties the min; t=40 sits inside a full window
    rank = dispersion_rank(disp, w)
    np.testing.assert_allclose(rank.iloc[40], 0.0)   # tied min → 0.0, not averaged


def test_dispersion_rank_rejects_window_of_one():
    idx = pd.date_range("1990-01-31", periods=10, freq="ME")
    with pytest.raises(ValueError):
        dispersion_rank(pd.Series(np.arange(10.0), index=idx), window=1)


# ── 5. S4 positions ─────────────────────────────────────────────────────
def _timing_panel(n=160, seed=0):
    """Minimal panel for s1_trend/s4_dispersion: needs gold_nominal + gold_ret."""
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    return pd.DataFrame({
        "gold_nominal": gold,
        "gold_ret": gold.pct_change(fill_method=None),
    }, index=idx)


def test_dispersion_weight_hard_tiers():
    idx = pd.date_range("1990-01-31", periods=6, freq="ME")
    rank = pd.Series([0.0, 0.2, 1.0 / 3, 0.5, 2.0 / 3, 0.9], index=idx)
    w = _dispersion_weight(rank, "hard", (DEFAULT_TIER_LOW, DEFAULT_TIER_MID, DEFAULT_TIER_HIGH))
    assert list(w) == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]


def test_dispersion_weight_soft_is_one_minus_rank():
    idx = pd.date_range("1990-01-31", periods=4, freq="ME")
    rank = pd.Series([0.0, 0.25, 0.75, 1.0], index=idx)
    w = _dispersion_weight(rank, "soft", (1.0, 0.5, 0.0))
    np.testing.assert_allclose(w.to_numpy(), [1.0, 0.75, 0.25, 0.0])


def test_s4_positions_in_unit_interval():
    panel = _timing_panel()
    idx = panel.index
    for rank_seed, mode in [(11, "hard"), (12, "soft")]:
        rng = np.random.RandomState(rank_seed)
        rank = pd.Series(rng.rand(len(idx)), index=idx)
        pos = s4_dispersion(panel, rank, mode=mode)
        valid = pos.dropna()
        assert valid.between(0.0, 1.0).all(), f"{mode} produced out-of-range weights"


def test_s4_high_dispersion_cuts_more_than_low():
    """Holding the panel fixed, a max-dispersion rank must weight ≤ a min-dispersion
    rank everywhere both are defined (high disagreement → less trend exposure)."""
    panel = _timing_panel()
    idx = panel.index
    rank_lo = pd.Series(0.0, index=idx)   # consensus everywhere
    rank_hi = pd.Series(1.0, index=idx)   # max disagreement everywhere
    for mode in ("hard", "soft"):
        lo = s4_dispersion(panel, rank_lo, mode=mode).dropna()
        hi = s4_dispersion(panel, rank_hi, mode=mode).dropna()
        assert (hi <= lo + 1e-12).all(), f"{mode}: high-disp weight exceeded low"


def test_s4_soft_equals_s1_at_min_dispersion():
    """When dispersion is at its trailing min (rank→0) the soft weight →1, so S4
    reproduces S1 exactly (dispersion adds nothing when lenses agree)."""
    panel = _timing_panel()
    rank0 = pd.Series(0.0, index=panel.index)
    s1 = s1_trend(panel)
    s4 = s4_dispersion(panel, rank0, mode="soft")
    _close_equal_nan(s1, s4)


def test_s4_hard_exits_at_max_dispersion():
    """rank→1 (max disagreement) under hard tiers → weight 0 → S4 fully in cash
    wherever S1 was invested (the regime-turn exit)."""
    panel = _timing_panel()
    rank1 = pd.Series(1.0, index=panel.index)
    s4 = s4_dispersion(panel, rank1, mode="hard").dropna()
    np.testing.assert_allclose(s4.to_numpy(), np.zeros(len(s4)))


# ── 6. build_dispersion_panel (injection) + same-track end-to-end ───────
def _anchor_and_fetch(n=240, seed=5, oil_start=0):
    """Synthetic anchor panel + a fetch_fn returning CPI/oil/copper. oil_start>0
    leaves a leading-NaN oil block (pre-1986 analogue)."""
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    debt_gdp = pd.Series(np.linspace(0.4, 1.2, n) + rng.randn(n) * 0.01, index=idx)
    m2_gdp = pd.Series(np.linspace(0.5, 0.9, n) + rng.randn(n) * 0.01, index=idx)
    real_rate = pd.Series(2.0 + np.cumsum(rng.randn(n) * 0.1), index=idx)
    anchor = SimpleNamespace(data=pd.DataFrame({
        "gold_nominal": gold, "debt_gdp": debt_gdp,
        "m2_gdp": m2_gdp, "real_rate_10y": real_rate,
    }, index=idx))

    cpi = pd.Series(np.linspace(100, 200, n) + rng.randn(n) * 0.5, index=idx)
    oil = pd.Series(30 + rng.randn(n) * 3, index=idx)
    if oil_start > 0:            # emulate "pre-1986 oil missing"
        oil.iloc[:oil_start] = np.nan
    copper = pd.Series(100 + rng.randn(n) * 4, index=idx)

    def fetch_fn(series_id, start="1968-01-01"):
        return {"CPIAUCSL": cpi, "MCOILWTICO": oil, "WPUSI019011": copper}.get(
            series_id, pd.Series(np.nan, index=idx))
    return anchor, fetch_fn, idx


def test_build_dispersion_panel_columns_and_oil_gap():
    anchor, fetch_fn, idx = _anchor_and_fetch(n=180, oil_start=40)
    dp = build_dispersion_panel(
        start="1990-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    df = dp.data
    for col in ("gold_nominal", "debt_gdp", "m2_gdp", "real_rate_10y", "cpi", "oil", "copper"):
        assert col in df.columns
    # the leading oil gap survives the build (not silently filled)
    assert df["oil"].iloc[:40].isna().all()
    assert df["oil"].iloc[60:].notna().any()


def test_end_to_end_same_track_with_missing_oil():
    """Full chain on a synthetic panel with a leading oil gap: oil is NaN early but
    dispersion is still defined from the other lenses; S0/S1/S4 run through the
    shared engine on a common window with positions ∈[0,1]."""
    n, w, oil_start = 220, CALIB, 80  # core lenses valid from idx[w]; oil from idx[oil_start+w]
    anchor, fetch_fn, idx = _anchor_and_fetch(n=n, oil_start=oil_start)
    dp = build_dispersion_panel(
        start="1990-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    df = dp.data
    w = CALIB
    implieds = {
        "debt_gdp": implied_debt_gdp(df.gold_nominal, df.debt_gdp, w),
        "real_rate": implied_real_rate(df.gold_nominal, df.real_rate_10y, w),
        "m2_gdp": implied_m2_gdp(df.gold_nominal, df.m2_gdp, w),
        "cpi": implied_cpi(df.gold_nominal, df.cpi, w),
        "gold_oil": implied_gold_oil(df.gold_nominal, df.oil, w),
        "gold_copper": implied_gold_copper(df.gold_nominal, df.copper, w),
    }
    gaps = estimator_gaps(implieds, df.gold_nominal)
    disp = dispersion(gaps)
    rank = dispersion_rank(disp, w)

    # dispersion exists from the CORE lenses after their warmup but BEFORE oil's
    # calibration fills — oil is still NaN there, yet dispersion is defined from
    # the other lenses (the skip-don't-pollute contract at the panel level).
    mid = disp.loc[idx[w + 5]: idx[oil_start + w - 5]].dropna()
    assert len(mid) > 0, "dispersion undefined post-core-warmup — other lenses should count"
    cnt = estimator_count(gaps).loc[mid.index]
    assert (cnt >= 2).all()
    assert implieds["gold_oil"].loc[mid.index].isna().all()

    # same-track: S0 / S1 / S4(hard) / S4(soft) through the shared engine
    panel = df.assign(gold_ret=df.gold_nominal.pct_change(fill_method=None))
    positions = {
        "S0": s0_buy_hold(panel.index),
        "S1": s1_trend(panel),
        "S4_hard": s4_dispersion(panel, rank, mode="hard"),
        "S4_soft": s4_dispersion(panel, rank, mode="soft"),
    }
    bts = {k: run_backtest(p, panel["gold_ret"], pd.Series(0.0, index=panel.index))
           for k, p in positions.items()}
    for k, p in positions.items():
        assert p.dropna().between(0.0, 1.0).all(), f"{k} out of [0,1]"
    # every strategy must produce some tradeable months on a common window
    starts = [bt.index.min() for bt in bts.values()]
    ends = [bt.index.max() for bt in bts.values()]
    assert max(starts) <= min(ends)
    # the common window has real breadth (warm-up didn't eat the sample)
    common_n = (pd.Series(1, index=bts["S1"].index)
                .loc[max(starts):min(ends)]).sum()
    assert common_n >= 12


def test_estimator_coverage_reports_skipped_lens():
    idx = pd.date_range("1990-01-31", periods=10, freq="ME")
    full = pd.Series(np.arange(10.0) + 1, index=idx)
    empty = pd.Series(np.nan, index=idx)  # a lens with no data (e.g. copper unfetchable)
    cov = estimator_coverage({"a": full, "b": empty})
    assert cov.loc["a", "n_months"] == 10
    assert cov.loc["b", "n_months"] == 0
    assert pd.isna(cov.loc["b", "first"])


def test_landmarks_snap_to_month_end():
    """A 'YYYY-MM' landmark must hit that month's END row (the panel is month-end
    indexed), not the prior month. Regression for pd.Timestamp('2011-09')→
    2011-09-01 silently reading the 2011-08 row."""
    from scripts.gold_dispersion_backtest import landmarks_table
    idx = pd.date_range("2007-01-31", periods=180, freq="ME")  # 2007-01 .. 2021-12
    disp = pd.Series(np.arange(180.0), index=idx)
    rank = pd.Series(np.linspace(0.0, 1.0, 180), index=idx)
    n_est = pd.Series(6, index=idx)
    lm = landmarks_table(disp, rank, n_est, idx)
    # '2011-09' → 2011-09-30, not 2011-08-31
    assert lm.loc["2011-09 nominal peak", "month"] == "2011-09"
    assert lm.loc["2011-09 nominal peak", "dispersion"] == disp.loc[pd.Timestamp("2011-09-30")]
    # '2020-03' → 2020-03-31, not 2020-02-29
    assert lm.loc["2020-03 COVID trough", "month"] == "2020-03"
    # '1980-01' predates this 2007-start sample → out_of_sample, NOT the first
    # sample month's reading masquerading as 1980's
    assert lm.loc["1980-01 nominal peak", "month"] == "out_of_sample"
    assert pd.isna(lm.loc["1980-01 nominal peak", "dispersion"])
    assert lm.loc["1980-01 nominal peak", "n_lenses"] == 0


def test_min_int_argparse_type_rejects_below_minimum():
    """--disp-window 1 must fail at PARSE time (argparse type), not crash later
    inside dispersion_rank's (raw-1)/(window-1) div-by-0."""
    import argparse
    from scripts.gold_dispersion_backtest import _min_int
    check = _min_int(2)
    assert check("2") == 2
    assert check("120") == 120
    with pytest.raises(argparse.ArgumentTypeError):
        check("1")
    with pytest.raises(argparse.ArgumentTypeError):
        check("0")
    with pytest.raises(argparse.ArgumentTypeError):
        check("not-an-int")

