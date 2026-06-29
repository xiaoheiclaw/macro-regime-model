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


def test_turnover_survives_missing_intermediate_return():
    # A real position change must still be charged even if an adjacent month's
    # return is missing (regression: masking held first dropped the diff/cost).
    idx = _midx(5)
    gold_ret = pd.Series([0.0, 0.0, np.nan, 0.0, 0.0], index=idx)  # idx[2] missing
    tbill = pd.Series(0.0, index=idx)
    pos = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    bt = run_backtest(pos, gold_ret, tbill, cost_bps=10.0)
    # idx[2] (held=1, return NaN) is dropped from output...
    assert idx[2] not in bt.index
    # ...but the trade into idx[3] (held 1→0) is still charged, not NaN-eaten.
    assert bt["turnover"].loc[idx[3]] == pytest.approx(1.0)
    assert bt["cost"].loc[idx[3]] == pytest.approx(10e-4)


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
    assert s1.between(0.0, 1.0).all()
    assert s2.between(0.0, 1.0).all()
    # S2 ≤ S1 everywhere (gate only removes exposure)
    assert (s2 <= s1 + 1e-12).all()


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
    join = old_idx.intersection(new_idx).min()
    # at the join the spliced value equals the OLD level (new rebased to old)
    assert spliced.loc[join] == pytest.approx(old.loc[join])
    # monotonic index, no gaps/dups
    assert spliced.index.is_monotonic_increasing
    assert not spliced.index.has_duplicates
