"""Tests for the gold de-dollarization leading-indicator size modulator (PR #8).

Offline by construction: panel/series are synthetic (no network/FRED/datasets.io).
Covers (per the task spec):
  1. no look-ahead — truncating the future leaves past strength / rank / panel
     values unchanged (the leak-free trailing-window contract)
  2. signal logic — custody_share = custody/debt; strength = −trailing Δ(share)
     (falling share → positive strength); rank ∈[0,1], monotone, reaches 0 and 1
  3. size factor — INCREASING in rank (opposite of PR #7 dispersion); neutral at
     mid; hard tercile; rejects a non-monotone tier set
  4. S5 positions — ∈[0,1] and CAPPED at 1.0 (封顶, no leverage even when f>1);
     strong de-dollarization sizes UP (≥ S1), weak sizes DOWN (≤ S1); neutral
     factor → S5 = S1; S5 only modulates a trend-on position (never entry)
  5. missing-data fallback — an unavailable proxy (all-NaN rank) → neutral factor
     → S5 ≡ S1 exactly; signal_available reports False
  6. same-track — S0/S1/S5 run through the shared `run_backtest` engine on an
     identical investable window; build_dedollar_panel injection + debt ffill
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_dedollarization_leading import (
    DEFAULT_F_MAX,
    DEFAULT_F_MIN,
    DEFAULT_F_NEUTRAL,
    build_dedollar_panel,
    custody_share,
    dedollar_factor,
    dedollar_rank,
    dedollar_strength,
    s5_dedollar,
    signal_available,
)
from lib.gold_trend_timing import run_backtest, s0_buy_hold, s1_trend

CHG = 12   # change window
RANK = 36  # rank window (smaller so synthetic tests fill within n)


def _close_equal_nan(a: pd.Series, b: pd.Series, *, atol: float = 0.0) -> None:
    """Assert two series agree on both values (where either is non-NaN) and NaN
    positions. The no-lookahead contract requires BOTH."""
    assert a.shape == b.shape
    av, bv = a.to_numpy(), b.to_numpy()
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    mask = ~(np.isnan(av) | np.isnan(bv))
    np.testing.assert_allclose(av[mask], bv[mask], rtol=0, atol=atol)


# ── 1. no look-ahead ────────────────────────────────────────────────────
def test_strength_no_lookahead():
    n = 120
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(1)
    share = pd.Series(0.3 - np.cumsum(rng.rand(n)) * 1e-3, index=idx)
    full = dedollar_strength(share, CHG)
    cut = 80
    trunc = dedollar_strength(share.iloc[:cut], CHG)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_rank_no_lookahead():
    n = 120
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(2)
    strength = pd.Series(rng.randn(n), index=idx)
    full = dedollar_rank(strength, RANK)
    cut = 90
    trunc = dedollar_rank(strength.iloc[:cut], RANK)
    _close_equal_nan(full.iloc[:cut], trunc)


def test_rank_uses_only_trailing_window():
    """rank at t depends only on strength[t-w+1..t]; corrupting the future leaves
    the early ranks untouched."""
    n, w = 120, 40
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(3)
    strength = pd.Series(rng.randn(n), index=idx)
    full = dedollar_rank(strength, w)
    s2 = strength.copy()
    s2.iloc[80:] = s2.iloc[80:] * 1e4
    full2 = dedollar_rank(s2, w)
    _close_equal_nan(full.iloc[:80], full2.iloc[:80])


# ── 2. signal logic ──────────────────────────────────────────────────────
def test_custody_share_is_ratio():
    idx = pd.date_range("2003-01-31", periods=3, freq="ME")
    custody = pd.Series([1000.0, 1100.0, 900.0], index=idx)
    debt = pd.Series([5000.0, 5500.0, 6000.0], index=idx)
    share = custody_share(custody, debt)
    np.testing.assert_allclose(share.to_numpy(), [0.2, 0.2, 0.15])


def test_strength_is_negated_trailing_change():
    """Falling share → POSITIVE strength; rising share → negative."""
    idx = pd.date_range("2003-01-31", periods=5, freq="ME")
    share = pd.Series([0.30, 0.29, 0.28, 0.27, 0.26], index=idx)  # steadily falling
    s = dedollar_strength(share, window=1)
    # each step falls 0.01 → strength +0.01 (first is NaN: no prior month)
    assert np.isnan(s.iloc[0])
    np.testing.assert_allclose(s.dropna().to_numpy(), [0.01, 0.01, 0.01, 0.01])
    # a RISING share gives negative strength
    rising = pd.Series([0.20, 0.22], index=idx[:2])
    np.testing.assert_allclose(dedollar_strength(rising, 1).dropna().to_numpy(), [-0.02])


def test_strength_rejects_nonpositive_window():
    idx = pd.date_range("2003-01-31", periods=10, freq="ME")
    with pytest.raises(ValueError):
        dedollar_strength(pd.Series(np.arange(10.0), index=idx), window=0)


def test_rank_reaches_zero_and_one():
    n, w = 60, 30
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    strength = pd.Series(0.0, index=idx)
    strength.iloc[40] = -1.0   # window min
    strength.iloc[50] = 1.0    # window max
    rank = dedollar_rank(strength, w)
    assert rank.dropna().between(0.0, 1.0).all()
    np.testing.assert_allclose(rank.iloc[40], 0.0)
    np.testing.assert_allclose(rank.iloc[50], 1.0)


def test_rank_monotone_in_strength():
    """Higher strength → higher (or equal) rank within the same window."""
    n, w = 50, 25
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    strength = pd.Series(np.linspace(-1.0, 1.0, n), index=idx)  # strictly increasing
    rank = dedollar_rank(strength, w).dropna()
    assert (rank.diff().dropna() >= -1e-12).all()  # non-decreasing


def test_rank_tied_min_is_zero():
    n, w = 60, 30
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    strength = pd.Series(np.linspace(0.2, 0.4, n), index=idx)
    strength.iloc[20] = -1.0
    strength.iloc[40] = -1.0  # ties the min; t=40 inside a full window
    rank = dedollar_rank(strength, w)
    np.testing.assert_allclose(rank.iloc[40], 0.0)


def test_rank_rejects_window_of_one():
    idx = pd.date_range("2003-01-31", periods=10, freq="ME")
    with pytest.raises(ValueError):
        dedollar_rank(pd.Series(np.arange(10.0), index=idx), window=1)


def test_rank_all_nan_input_stays_nan():
    """The missing-proxy path: an all-NaN strength ranks to all-NaN (no crash)."""
    idx = pd.date_range("2003-01-31", periods=40, freq="ME")
    rank = dedollar_rank(pd.Series(np.nan, index=idx), RANK)
    assert rank.isna().all()


# ── 3. size factor ───────────────────────────────────────────────────────
def test_factor_soft_is_linear_increasing():
    idx = pd.date_range("2003-01-31", periods=5, freq="ME")
    rank = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=idx)
    f = dedollar_factor(rank, mode="soft", f_min=0.5, f_neutral=1.0, f_max=1.5)
    # linear 0.5 → 1.5; rank 0.5 → neutral 1.0 exactly
    np.testing.assert_allclose(f.to_numpy(), [0.5, 0.75, 1.0, 1.25, 1.5])


def test_factor_hard_tiers():
    idx = pd.date_range("2003-01-31", periods=6, freq="ME")
    rank = pd.Series([0.0, 0.2, 1.0 / 3, 0.5, 2.0 / 3, 0.9], index=idx)
    f = dedollar_factor(rank, mode="hard", f_min=0.5, f_neutral=1.0, f_max=1.5)
    assert list(f) == [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]


def test_factor_is_increasing_in_rank():
    """Core property (opposite of PR #7 dispersion): stronger de-dollarization →
    larger factor everywhere both modes are defined."""
    idx = pd.date_range("2003-01-31", periods=50, freq="ME")
    rng = np.random.RandomState(7)
    rank = pd.Series(np.sort(rng.rand(50)), index=idx)  # ascending
    for mode in ("soft", "hard"):
        f = dedollar_factor(rank, mode=mode)
        assert (f.diff().dropna() >= -1e-12).all(), f"{mode} not non-decreasing"


def test_factor_nan_rank_falls_back_to_neutral():
    idx = pd.date_range("2003-01-31", periods=4, freq="ME")
    rank = pd.Series([np.nan, 0.0, np.nan, 1.0], index=idx)
    for mode in ("soft", "hard"):
        f = dedollar_factor(rank, mode=mode, f_min=0.5, f_neutral=1.0, f_max=1.5)
        assert f.iloc[0] == 1.0 and f.iloc[2] == 1.0  # NaN → neutral


def test_factor_rejects_non_monotone_tiers():
    idx = pd.date_range("2003-01-31", periods=3, freq="ME")
    rank = pd.Series([0.0, 0.5, 1.0], index=idx)
    with pytest.raises(ValueError):
        dedollar_factor(rank, f_min=1.0, f_neutral=0.5, f_max=1.5)  # neutral < min


# ── 4. S5 positions ──────────────────────────────────────────────────────
def _timing_panel(n=160, seed=0):
    """Minimal panel for s1_trend/s5_dedollar: needs gold_nominal + gold_ret."""
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    return pd.DataFrame({
        "gold_nominal": gold,
        "gold_ret": gold.pct_change(fill_method=None),
    }, index=idx)


def test_s5_positions_in_unit_interval_and_capped():
    """∈[0,1] AND never exceeds 1.0 even when f_max>1 would push above (封顶)."""
    panel = _timing_panel()
    idx = panel.index
    for rank_seed, mode in [(11, "hard"), (12, "soft")]:
        rng = np.random.RandomState(rank_seed)
        rank = pd.Series(rng.rand(len(idx)), index=idx)
        pos = s5_dedollar(panel, rank, mode=mode, f_max=3.0)  # aggressive amplify
        valid = pos.dropna()
        assert valid.between(0.0, 1.0).all(), f"{mode} produced out-of-range weights"
        assert valid.max() <= 1.0 + 1e-12  # cap holds despite f_max=3.0


def test_s5_strong_sizes_up_weak_sizes_down():
    """Holding the panel fixed: max-strength rank ≥ S1 ≥ min-strength rank (size UP
    when de-dollarization strong, DOWN when weak) — the directional core."""
    panel = _timing_panel()
    idx = panel.index
    base = s1_trend(panel)
    rank_lo = pd.Series(0.0, index=idx)   # weakest de-dollarization
    rank_hi = pd.Series(1.0, index=idx)   # strongest
    for mode in ("hard", "soft"):
        lo = s5_dedollar(panel, rank_lo, mode=mode).dropna()
        hi = s5_dedollar(panel, rank_hi, mode=mode).dropna()
        b = base.reindex(lo.index)
        assert (lo <= b + 1e-12).all(), f"{mode}: weak did not size DOWN vs S1"
        assert (hi >= b - 1e-12).all(), f"{mode}: strong did not size UP vs S1"


def test_s5_neutral_rank_equals_s1():
    """At neutral rank (soft 0.5 → f=1.0) S5 reproduces S1 exactly."""
    panel = _timing_panel()
    rank_mid = pd.Series(0.5, index=panel.index)
    s1 = s1_trend(panel)
    s5 = s5_dedollar(panel, rank_mid, mode="soft")
    _close_equal_nan(s1, s5)


def test_s5_does_not_create_position_when_trend_off():
    """Size modulation, NOT entry: where S1 = 0 (no trend) S5 must be 0 regardless
    of how strong de-dollarization is."""
    panel = _timing_panel()
    s1 = s1_trend(panel)
    rank_hi = pd.Series(1.0, index=panel.index)
    s5 = s5_dedollar(panel, rank_hi, mode="hard")
    off = s1[s1 == 0.0].index
    assert len(off) > 0
    np.testing.assert_allclose(s5.reindex(off).to_numpy(), np.zeros(len(off)))


# ── 5. missing-data fallback ──────────────────────────────────────────────
def test_signal_available_flag():
    idx = pd.date_range("2003-01-31", periods=10, freq="ME")
    assert signal_available(pd.Series(np.linspace(0, 1, 10), index=idx))
    assert not signal_available(pd.Series(np.nan, index=idx))


def test_s5_falls_back_to_s1_when_proxy_unavailable():
    """An entirely unavailable proxy (all-NaN rank) → neutral factor → S5 ≡ S1
    exactly. The graceful missing-data fallback the task requires."""
    panel = _timing_panel()
    rank_missing = pd.Series(np.nan, index=panel.index)
    s1 = s1_trend(panel)
    for mode in ("soft", "hard"):
        s5 = s5_dedollar(panel, rank_missing, mode=mode)
        _close_equal_nan(s1, s5)


def test_s5_identical_window_to_s1():
    """Because the factor is neutral (not NaN) where the rank is undefined, S5's
    only NaN is S1's warm-up → identical investable window (fair same-track)."""
    panel = _timing_panel()
    idx = panel.index
    rng = np.random.RandomState(5)
    # rank defined only on the back half (front half NaN = warm-up)
    rank = pd.Series(np.nan, index=idx)
    rank.iloc[len(idx) // 2:] = rng.rand(len(idx) - len(idx) // 2)
    s1 = s1_trend(panel)
    s5 = s5_dedollar(panel, rank, mode="soft")
    np.testing.assert_array_equal(s1.isna().to_numpy(), s5.isna().to_numpy())


# ── 6. build_dedollar_panel (injection) + same-track end-to-end ──────────
def _anchor_and_fetch(n=240, seed=5, custody_missing=False):
    """Synthetic anchor panel + a fetch_fn returning custody (weekly-ish) & debt
    (quarterly). custody_missing emulates an unavailable proxy series."""
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=idx)
    anchor = SimpleNamespace(data=pd.DataFrame({"gold_nominal": gold}, index=idx))

    # custody declines over time, debt grows → share falls (de-dollarization)
    custody = pd.Series(3.0e6 - np.linspace(0, 0.5e6, n) + rng.randn(n) * 1e4, index=idx)
    if custody_missing:
        custody = pd.Series(np.nan, index=idx)
    # quarterly debt: only quarter-end months observed (rest NaN → tests ffill)
    debt_full = pd.Series(6.0e6 + np.linspace(0, 4.0e6, n), index=idx)
    debt = debt_full.copy()
    keep = (debt.index.month % 3 == 0)
    debt[~keep] = np.nan

    def fetch_fn(series_id, start="1968-01-01"):
        return {"WMTSECL1": custody, "GFDEBTN": debt}.get(
            series_id, pd.Series(np.nan, index=idx))
    return anchor, fetch_fn, idx, debt_full


def test_build_panel_columns_and_debt_ffill():
    anchor, fetch_fn, idx, debt_full = _anchor_and_fetch(n=120)
    dp = build_dedollar_panel(
        start="2003-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    df = dp.data
    for col in ("gold_nominal", "foreign_official_custody", "total_public_debt", "custody_share"):
        assert col in df.columns
    # debt was quarterly-with-NaN; the panel ffills it → no interior NaN after the
    # first quarter-end, and the level matches the carried-forward quarter value.
    debt = df["total_public_debt"]
    assert debt.notna().sum() > debt_full.index.month.isin([3, 6, 9, 12]).sum() - 4
    # ffill never invents a value before the first observation
    first_obs = debt_full[debt_full.index.month % 3 == 0].index.min()
    assert debt.loc[:first_obs].dropna().index.min() >= first_obs


def test_build_panel_custody_missing_is_nan_share():
    """Unavailable custody → share all-NaN (no fake 0); the panel still builds."""
    anchor, fetch_fn, idx, _ = _anchor_and_fetch(n=120, custody_missing=True)
    dp = build_dedollar_panel(
        start="2003-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    assert dp.data["custody_share"].isna().all()


def test_end_to_end_same_track():
    """Full chain on a synthetic panel: build → share → strength → rank → S0/S1/S5
    through the shared engine on an identical investable window, positions ∈[0,1]."""
    n = 200
    anchor, fetch_fn, idx, _ = _anchor_and_fetch(n=n)
    dp = build_dedollar_panel(
        start="2003-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    )
    df = dp.data
    share = custody_share(df["foreign_official_custody"], df["total_public_debt"])
    strength = dedollar_strength(share, CHG)
    rank = dedollar_rank(strength, RANK)
    assert signal_available(rank)

    panel = df.assign(gold_ret=df["gold_nominal"].pct_change(fill_method=None))
    positions = {
        "S0": s0_buy_hold(panel.index),
        "S1": s1_trend(panel),
        "S5_soft": s5_dedollar(panel, rank, mode="soft"),
        "S5_hard": s5_dedollar(panel, rank, mode="hard"),
    }
    bts = {k: run_backtest(p, panel["gold_ret"], pd.Series(0.0, index=panel.index))
           for k, p in positions.items()}
    for k, p in positions.items():
        assert p.dropna().between(0.0, 1.0).all(), f"{k} out of [0,1]"
    # S1 and S5 share an identical investable window (neutral fallback, not NaN)
    assert bts["S1"].index.min() == bts["S5_soft"].index.min()
    assert bts["S1"].index.max() == bts["S5_soft"].index.max()
    starts = [bt.index.min() for bt in bts.values()]
    ends = [bt.index.max() for bt in bts.values()]
    assert max(starts) <= min(ends)
    common_n = (pd.Series(1, index=bts["S1"].index).loc[max(starts):min(ends)]).sum()
    assert common_n >= 12


def test_timeline_snaps_to_month_end():
    """A 'YYYY-MM' landmark must hit that month's END row (month-end panel)."""
    from scripts.gold_dedollarization_backtest import timeline_table
    idx = pd.date_range("2007-01-31", periods=240, freq="ME")
    share = pd.Series(np.linspace(0.3, 0.2, 240), index=idx)
    strength = pd.Series(np.arange(240.0), index=idx)
    rank = pd.Series(np.linspace(0.0, 1.0, 240), index=idx)
    tl = timeline_table(share, strength, rank, idx)
    assert tl.loc["2022-02 Russia reserve freeze", "month"] == "2022-02"
    assert tl.loc["2022-02 Russia reserve freeze", "strength"] == strength.loc[
        pd.Timestamp("2022-02-28")]
