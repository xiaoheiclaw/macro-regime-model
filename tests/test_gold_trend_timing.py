"""Unit tests for the gold long-only trend-timing backtest.

Covers the failure modes that matter for an honest backtest:
  • no look-ahead (signals shift correctly into the next period)
  • vol targeting (long-only, capped 0–100%)
  • cost accounting (per-rebalance turnover × bps)
  • buy-and-hold baseline reduces to the asset return
  • metric correctness (CAGR / vol / Sharpe / MaxDD / Calmar / hit / turnover)
  • regime-gate logic incl. the missing-dollar fallback

All synthetic — no network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from lib.gold_trend_timing import (
    ANNUAL,
    compute_metrics,
    momentum_signal,
    regime_gate,
    run_backtest,
    s0_buy_hold,
    s1_trend,
    s2_trend_regime,
    trend_exposure,
    vol_scale,
    _splice_dollar,
)


def _midx(n: int, start="2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


# ── no look-ahead ──────────────────────────────────────────────────────
def test_momentum_signal_uses_only_past():
    # Price strictly rising → trailing momentum > 0 once enough history.
    idx = _midx(8)
    price = pd.Series([100, 101, 102, 103, 104, 105, 106, 107.0], index=idx)
    sig = momentum_signal(price, lookback=3)
    # first 3 are NaN (no t-3 reference), then all 1.0 (rising)
    assert sig.iloc[:3].isna().all()
    assert (sig.iloc[3:] == 1.0).all()


def test_momentum_signal_detects_decline():
    idx = _midx(6)
    price = pd.Series([100, 99, 98, 97, 96, 95.0], index=idx)
    sig = momentum_signal(price, lookback=2)
    assert (sig.dropna() == 0.0).all()


def test_backtest_no_lookahead_shift():
    # A position taken at t must only affect the return at t+1, never at t.
    idx = _midx(4)
    gold_ret = pd.Series([0.10, 0.20, 0.30, 0.40], index=idx)
    tbill = pd.Series(0.0, index=idx)
    # full weight only at the 2nd month (decision), zero elsewhere
    pos = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=0.0)
    # first row (idx[0]) is dropped: no prior position → no return. Reference
    # by date label, not position, to be explicit about the shift.
    # held = pos.shift(1): month idx[2] holds the weight set at idx[1].
    assert bt["held"].loc[idx[2]] == 1.0
    # the gold return captured at idx[2] is 0.30, NOT idx[1]'s 0.20
    assert bt["gross_ret"].loc[idx[2]] == pytest.approx(0.30)
    # decision month idx[1] still holds the *prior* weight (0) → 0 return
    assert bt["gross_ret"].loc[idx[1]] == pytest.approx(0.0)


# ── vol targeting ──────────────────────────────────────────────────────
def test_vol_scale_caps_at_one():
    # Tiny realised vol → target/realised > 1 → capped at 1.0.
    idx = _midx(12)
    ret = pd.Series(np.full(12, 0.001), index=idx)  # near-zero vol
    vs = vol_scale(ret, target_vol=0.10, window=6)
    assert (vs.dropna() <= 1.0).all()
    assert vs.dropna().iloc[-1] == pytest.approx(1.0)


def test_vol_scale_scales_down_high_vol():
    # Construct returns with a known monthly std, check target/realised.
    idx = _midx(24)
    rng = np.random.default_rng(0)
    monthly_sigma = 0.10  # → annualised ~0.346, target 0.10 → scale ~0.29
    ret = pd.Series(rng.normal(0, monthly_sigma, 24), index=idx)
    vs = vol_scale(ret, target_vol=0.10, window=6)
    rv = ret.rolling(6).std() * np.sqrt(ANNUAL)
    expected = (0.10 / rv).clip(0, 1)
    pd.testing.assert_series_equal(vs.dropna(), expected.dropna())
    assert (vs.dropna() < 1.0).all()  # high vol → always scaled down


# ── cost accounting ────────────────────────────────────────────────────
def test_cost_accounting_turnover():
    idx = _midx(5)
    gold_ret = pd.Series(0.0, index=idx)
    tbill = pd.Series(0.0, index=idx)
    # held = pos.shift(1): [nan, 0, 1, 0, 1]
    pos = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=10.0)
    # idx[0] dropped (held NaN). held = pos.shift(1): idx1=0 (first valid, trade
    # from cash but weight 0 → turnover 0), idx2=1 (Δ1), idx3=0 (Δ1), idx4=1 (Δ1)
    assert bt["turnover"].loc[idx[1]] == pytest.approx(0.0)  # 0 weight, no trade
    assert bt["turnover"].loc[idx[2]] == pytest.approx(1.0)
    assert bt["turnover"].loc[idx[3]] == pytest.approx(1.0)
    # cost = turnover * 10bps; with zero returns, net = -cost
    assert bt["cost"].loc[idx[2]] == pytest.approx(10e-4)
    assert bt["net_ret"].loc[idx[2]] == pytest.approx(-10e-4)


def test_missing_intermediate_return_raises():
    # A return gap inside the invested span is a data hole a backtest cannot
    # honestly model — it must raise, not silently drop the month (which would
    # also drop that month's turnover/cost and under-count it).
    idx = _midx(5)
    gold_ret = pd.Series([0.0, 0.0, np.nan, 0.0, 0.0], index=idx)  # idx[2] gap
    tbill = pd.Series(0.0, index=idx)
    pos = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    with pytest.raises(ValueError, match="traded span"):
        run_backtest(pos, gold_ret, tbill, cost_bps=10.0)


def test_leading_warmup_nan_is_ok():
    # A leading NaN return (e.g. month-0 pct_change) sits *before* the first
    # position and must not trip the gap check.
    idx = _midx(4)
    gold_ret = pd.Series([np.nan, 0.01, 0.02, 0.03], index=idx)
    tbill = pd.Series(0.0, index=idx)
    pos = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx)  # held = [nan,0,1,1]
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=0.0)
    # span starts at idx[1] (first non-NaN held); month-0 excluded, no raise
    assert bt.index[0] == idx[1]
    assert not bt["net_ret"].isna().any()


def test_run_backtest_rejects_out_of_range_weights():
    idx = _midx(4)
    gold_ret = pd.Series([0.01, 0.02, 0.0, 0.01], index=idx)
    tbill = pd.Series(0.0, index=idx)
    # held = pos.shift(1) → leverage >1 inside the span
    over = pd.Series([0.0, 1.5, 1.5, 1.5], index=idx)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run_backtest(over, gold_ret, tbill, cost_bps=0.0)
    # negative (short) weight
    short = pd.Series([0.0, -0.3, -0.3, -0.3], index=idx)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run_backtest(short, gold_ret, tbill, cost_bps=0.0)


def test_cost_zero_bps_is_costless():
    idx = _midx(5)
    gold_ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    tbill = pd.Series(0.001, index=idx)
    pos = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=idx)
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=0.0)
    assert (bt["cost"] == 0.0).all()
    pd.testing.assert_series_equal(bt["net_ret"], bt["gross_ret"], check_names=False)


# ── buy-and-hold baseline ──────────────────────────────────────────────
def test_buy_hold_reduces_to_asset_return():
    idx = _midx(6)
    gold_ret = pd.Series([0.05, -0.02, 0.03, 0.04, -0.01, 0.02], index=idx)
    tbill = pd.Series(0.001, index=idx)
    pos = s0_buy_hold(idx)
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=0.0)
    # held is always 1 → gross == gold_ret (on the valid, shifted window)
    held_gold = bt["gross_ret"]
    pd.testing.assert_series_equal(
        held_gold, gold_ret.reindex(bt.index), check_names=False
    )
    # exactly one trade (initial entry from cash)
    assert bt["turnover"].iloc[0] == pytest.approx(1.0)
    assert (bt["turnover"].iloc[1:] == 0.0).all()


# ── metric correctness ─────────────────────────────────────────────────
def test_metrics_constant_return():
    idx = _midx(ANNUAL)  # 12 months
    r = 0.01
    bt = pd.DataFrame({
        "net_ret": pd.Series(r, index=idx),
        "tbill_ret": pd.Series(0.0, index=idx),
        "turnover": pd.Series(0.0, index=idx),
    })
    m = compute_metrics(bt)
    assert m["cagr"] == pytest.approx((1 + r) ** 12 - 1, rel=1e-9)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert m["hit_rate"] == pytest.approx(1.0)
    assert m["n_months"] == 12


def test_metrics_max_drawdown():
    idx = _midx(4)
    # +50%, then -50% (cum: 1.5, 0.75), recover... worst dd = 0.75/1.5 - 1 = -0.5
    net = pd.Series([0.5, -0.5, 0.0, 0.0], index=idx)
    bt = pd.DataFrame({
        "net_ret": net,
        "tbill_ret": pd.Series(0.0, index=idx),
        "turnover": pd.Series(0.0, index=idx),
    })
    m = compute_metrics(bt)
    assert m["max_dd"] == pytest.approx(-0.5, abs=1e-9)
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.5)


def test_metrics_drawdown_from_opening_month():
    # A drawdown that starts in the very first month must be captured against
    # the implicit starting wealth of 1.0 (regression: was understated to 0).
    idx = _midx(2)
    net = pd.Series([-0.5, 0.0], index=idx)
    bt = pd.DataFrame({"net_ret": net, "tbill_ret": pd.Series(0.0, index=idx),
                       "turnover": pd.Series(0.0, index=idx)})
    m = compute_metrics(bt)
    assert m["max_dd"] == pytest.approx(-0.5, abs=1e-9)


def test_metrics_sharpe_excess_over_rf():
    idx = _midx(24)
    rng = np.random.default_rng(1)
    net = pd.Series(rng.normal(0.01, 0.02, 24), index=idx)
    rf = pd.Series(0.003, index=idx)
    bt = pd.DataFrame({"net_ret": net, "tbill_ret": rf,
                       "turnover": pd.Series(0.0, index=idx)})
    m = compute_metrics(bt)
    excess = net - 0.003
    expected = (excess.mean() * 12) / (excess.std(ddof=1) * np.sqrt(12))
    assert m["sharpe"] == pytest.approx(expected, rel=1e-9)


def test_metrics_annualised_turnover():
    idx = _midx(24)  # 2 years
    bt = pd.DataFrame({
        "net_ret": pd.Series(0.0, index=idx),
        "tbill_ret": pd.Series(0.0, index=idx),
        "turnover": pd.Series(0.5, index=idx),  # 0.5/mo × 24 = 12 total / 2y = 6
    })
    m = compute_metrics(bt)
    assert m["ann_turnover"] == pytest.approx(6.0)


# ── regime gate logic ──────────────────────────────────────────────────
def test_regime_gate_favourable_when_falling():
    idx = _midx(15)
    # real rate falling, dollar falling → favourable (1.0)
    rr = pd.Series(np.linspace(2.0, 0.0, 15), index=idx)
    usd = pd.Series(np.linspace(120.0, 100.0, 15), index=idx)
    gate = regime_gate(rr, usd, window=12)
    assert gate.iloc[-1] == 1.0


def test_regime_gate_unfavourable_when_both_rising():
    idx = _midx(15)
    rr = pd.Series(np.linspace(0.0, 2.0, 15), index=idx)   # rising
    usd = pd.Series(np.linspace(100.0, 120.0, 15), index=idx)  # strengthening
    gate = regime_gate(rr, usd, window=12)
    assert gate.iloc[-1] == 0.0


def test_regime_gate_unfavourable_when_one_rising():
    idx = _midx(15)
    rr = pd.Series(np.linspace(0.0, 2.0, 15), index=idx)   # rising → adverse
    usd = pd.Series(np.linspace(120.0, 100.0, 15), index=idx)  # falling → ok
    gate = regime_gate(rr, usd, window=12)
    assert gate.iloc[-1] == 0.0  # AND of conditions


def test_regime_gate_missing_dollar_falls_back_to_real_rate():
    idx = _midx(15)
    rr = pd.Series(np.linspace(2.0, 0.0, 15), index=idx)   # falling → ok
    usd = pd.Series(np.nan, index=idx)                      # no dollar data
    gate = regime_gate(rr, usd, window=12)
    # missing dollar treated favourable → gate driven by real-rate alone → 1.0
    assert gate.iloc[-1] == 1.0


def test_regime_gate_structural_prehistory_fails_open():
    # USD only exists from idx[10] on (e.g. pre-1973). Before it starts, the
    # gate must fall back to real-rate-only (fail open), not force an exit.
    idx = _midx(20)
    rr = pd.Series(np.linspace(2.0, 0.0, 20), index=idx)  # falling → favourable
    usd = pd.Series(np.nan, index=idx)
    usd.iloc[10:] = np.linspace(100.0, 90.0, 10)
    gate = regime_gate(rr, usd, window=3)
    assert gate.iloc[5] == 1.0  # before USD exists → real-rate-only favourable


def test_regime_gate_interior_gap_fails_closed():
    # A hole *after* the series starts must NOT be treated as favourable — the
    # gate should exit (0) rather than hold on unknown macro state.
    idx = _midx(10)
    rr = pd.Series(np.linspace(2.0, 0.0, 10), index=idx)        # falling
    usd = pd.Series(np.linspace(110.0, 100.0, 10), index=idx)   # falling
    usd.iloc[6] = np.nan  # interior hole
    gate = regime_gate(rr, usd, window=3)
    assert gate.iloc[6] == 0.0   # change references the hole → fail closed
    assert gate.iloc[5] == 1.0   # clean month → favourable


# ── strategy wiring / integration on synthetic panel ───────────────────
def _synth_panel(n=60):
    idx = _midx(n)
    rng = np.random.default_rng(7)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0.005, 0.04, n)), index=idx)
    return pd.DataFrame({
        "gold_nominal": price,
        "gold_ret": price.pct_change(),
        "real_rate_10y": pd.Series(np.linspace(1.0, 0.5, n), index=idx),
        "usd_broad": pd.Series(np.linspace(100, 110, n), index=idx),
        "tbill_ret": pd.Series(0.002, index=idx),
    })


def test_positions_long_only_bounded():
    panel = _synth_panel()
    s1 = s1_trend(panel)
    s2 = s2_trend_regime(panel)
    # warm-up is NaN (not 0); the realised exposure is bounded [0,1]
    assert s1.dropna().between(0.0, 1.0).all()
    assert s2.dropna().between(0.0, 1.0).all()
    # S2 ≤ S1 everywhere they are both defined (gate only removes exposure)
    both = s1.notna() & s2.notna()
    assert (s2[both] <= s1[both] + 1e-12).all()


def test_s1_warmup_is_nan_not_cash():
    # Before all lookbacks + the vol window have history, S1 must be NaN
    # (not 0 = cash), so run_backtest trims rather than crediting T-bill.
    panel = _synth_panel(n=40)
    s1 = s1_trend(panel, lookbacks=(3, 6, 12), vol_window=6)
    # month 0..11 cannot have the 12m signal → NaN
    assert s1.iloc[:12].isna().all()
    assert s1.iloc[12:].notna().all()


def test_s2_full_exit_when_gate_off():
    panel = _synth_panel()
    # force regime adverse: real rate strictly rising, dollar strictly rising
    n = len(panel)
    panel = panel.copy()
    panel["real_rate_10y"] = pd.Series(np.linspace(0.0, 3.0, n), index=panel.index)
    panel["usd_broad"] = pd.Series(np.linspace(90.0, 130.0, n), index=panel.index)
    s2 = s2_trend_regime(panel, gate_off_exposure=0.0)
    # after the regime window, gate is off → exposure forced to 0
    assert s2.iloc[-1] == 0.0


def test_trend_exposure_blend_levels():
    idx = _midx(20)
    price = pd.Series(100 * np.cumprod(np.r_[[1.0], np.full(19, 1.01)]), index=idx)
    te = trend_exposure(price, [3, 6, 12])
    # all rising → blend hits 1.0 once all lookbacks have history
    assert te.dropna().iloc[-1] == pytest.approx(1.0)
    assert te.dropna().between(0.0, 1.0).all()


def test_trend_exposure_blend_warmup_is_nan():
    # During warm-up (longest lookback not yet ready) the blend must be NaN,
    # not driven to 1.0 by the short-lookback signal alone.
    idx = _midx(14)
    price = pd.Series(100 * np.cumprod(np.r_[[1.0], np.full(13, 1.01)]), index=idx)
    te = trend_exposure(price, [3, 6, 12])
    # at idx[11] the 12m signal has no t-12 reference → blend NaN
    assert np.isnan(te.iloc[11])
    # at idx[12] all three lookbacks are ready and rising → full 1.0
    assert te.iloc[12] == pytest.approx(1.0)


# ── dollar splice ──────────────────────────────────────────────────────
def test_splice_dollar_level_continuous():
    old_idx = pd.date_range("1990-01-31", periods=24, freq="ME")
    new_idx = pd.date_range("1991-01-31", periods=24, freq="ME")
    old = pd.Series(np.linspace(100, 110, 24), index=old_idx)
    new = pd.Series(np.linspace(50, 60, 24), index=new_idx)  # different base
    spliced = _splice_dollar(old, new)
    join = old_idx.intersection(new_idx).max()  # join at END of overlap
    # the old backbone is kept through the join month, verbatim
    assert spliced.loc[join] == pytest.approx(old.loc[join])
    # months after the old series ends come from the rebased new series, and at
    # the join the rebased new equals the old level (continuity)
    scale = old.loc[join] / new.loc[join]
    after = new_idx[new_idx > join][0]
    assert spliced.loc[after] == pytest.approx(new.loc[after] * scale)
    # old series fully present up to its end (not truncated at overlap start)
    assert spliced.loc[old_idx[-1]] == pytest.approx(old.loc[old_idx[-1]])
    assert spliced.index.is_monotonic_increasing
    assert not spliced.index.has_duplicates


def test_splice_dollar_no_overlap_is_union():
    old_idx = pd.date_range("1990-01-31", periods=6, freq="ME")
    new_idx = pd.date_range("1991-01-31", periods=6, freq="ME")  # disjoint
    old = pd.Series(np.arange(6.0), index=old_idx)
    new = pd.Series(np.arange(100.0, 106.0), index=new_idx)
    spliced = _splice_dollar(old, new)
    # disjoint → simple union, both segments preserved verbatim
    assert len(spliced) == 12
    assert spliced.loc[old_idx[0]] == pytest.approx(0.0)
    assert spliced.loc[new_idx[0]] == pytest.approx(100.0)


def test_trend_exposure_empty_lookbacks_raises():
    price = pd.Series([100.0, 101.0], index=_midx(2))
    with pytest.raises(ValueError, match="non-empty"):
        trend_exposure(price, [])


# ── script-level: common_span / metrics_table robustness ───────────────
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "gold_trend_backtest",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "gold_trend_backtest.py"),
)
gtb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(gtb)


def _bt(idx_dates):
    idx = pd.DatetimeIndex(idx_dates)
    return pd.DataFrame({"net_ret": pd.Series(0.01, index=idx),
                         "tbill_ret": pd.Series(0.0, index=idx),
                         "turnover": pd.Series(0.0, index=idx)})


def test_common_span_none_when_a_strategy_is_empty():
    full = _bt(pd.date_range("2000-01-31", periods=12, freq="ME"))
    empty = full.iloc[0:0]
    cstart, cend = gtb.common_span({"S0": full, "S1": empty})
    assert cstart is None and cend is None


def test_common_span_none_when_no_overlap():
    a = _bt(pd.date_range("2000-01-31", periods=6, freq="ME"))
    b = _bt(pd.date_range("2010-01-31", periods=6, freq="ME"))  # disjoint
    cstart, cend = gtb.common_span({"S0": a, "S1": b})
    assert cstart is None and cend is None


def test_metrics_table_all_nan_when_no_common_window():
    a = _bt(pd.date_range("2000-01-31", periods=6, freq="ME"))
    b = a.iloc[0:0]
    tbl = gtb.metrics_table({"S0": a, "S1": b})
    # no crash; every cell NaN for the key metrics
    assert tbl["sharpe"].isna().all()


def test_common_span_intersection_when_aligned():
    a = _bt(pd.date_range("2000-01-31", periods=24, freq="ME"))
    b = _bt(pd.date_range("2000-06-30", periods=24, freq="ME"))
    cstart, cend = gtb.common_span({"S0": a, "S1": b})
    assert cstart == pd.Timestamp("2000-06-30")
    assert cend == pd.Timestamp("2001-12-31")
