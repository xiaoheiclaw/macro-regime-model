"""Unit tests for the S1 post-2000 sub-period verdict module.

Covers the new analytics this module adds on top of `lib.gold_trend_timing`:
  • lived-experience drawdown caliber: longest underwater run, longest
    consecutive-loss streak, discrete trade count
  • majority-vote S1 signal (warm-up NaN contract, ≤ full exposure)
  • fair common window across strategies
  • segment metric slicing on the shared window (same caliber for S0 & S1)
  • paired in-sample net-diff significance (mean / t-stat / bootstrap CI)
  • a full integration run across the cost grid and all variants

All synthetic — no network (the panel is built by hand / via injected stubs).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_s1_subperiod import (
    COST_GRID,
    METRIC_COLS,
    POST2000_SEGMENT,
    S0_LABEL,
    S1_VARIANTS,
    SUBPERIOD_SEGMENTS,
    build_positions,
    common_window,
    extended_metrics,
    longest_underwater,
    max_consecutive_loss_months,
    paired_net_diff_stats,
    s1_majority_vote,
    segment_metrics,
    segment_window,
    trade_count,
    variant_position,
    verdict,
)
from lib.gold_trend_timing import run_backtest, s0_buy_hold


def _midx(n: int, start="2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _bt_from_net(net: pd.Series) -> pd.DataFrame:
    """Minimal backtest frame compute_metrics/extended_metrics accept."""
    return pd.DataFrame({
        "net_ret": net,
        "tbill_ret": pd.Series(0.0, index=net.index),
        "turnover": pd.Series(0.0, index=net.index),
        "held": pd.Series(1.0, index=net.index),
    })


# ── longest underwater run ─────────────────────────────────────────────────
def test_longest_underwater_basic():
    idx = _midx(5)
    # +10%, -20% (underwater), -5% (still under), +50% (recovers above peak), 0
    net = pd.Series([0.10, -0.20, -0.05, 0.50, 0.0], index=idx)
    # wealth: 1.10, 0.88, 0.836, 1.254, 1.254 ; peak hit at month0 (1.10),
    # months 1&2 underwater (below 1.10), month3 recovers above → 2 months
    assert longest_underwater(net) == 2


def test_longest_underwater_opening_drawdown_counts():
    # A drawdown that opens in month 1 must count against starting wealth 1.0.
    idx = _midx(3)
    net = pd.Series([-0.10, -0.10, 0.0], index=idx)  # never recovers to 1.0
    assert longest_underwater(net) == 3


def test_longest_underwater_never_underwater():
    idx = _midx(4)
    net = pd.Series([0.01, 0.02, 0.01, 0.03], index=idx)  # monotone up
    assert longest_underwater(net) == 0


def test_longest_underwater_empty():
    assert longest_underwater(pd.Series(dtype="float64")) == 0


# ── consecutive loss months ─────────────────────────────────────────────────
def test_max_consecutive_loss_months():
    idx = _midx(7)
    # losses run: single, then a run of 3
    net = pd.Series([-0.01, 0.02, -0.01, -0.02, -0.03, 0.01, -0.01], index=idx)
    assert max_consecutive_loss_months(net) == 3


def test_max_consecutive_loss_months_zero_is_not_a_loss():
    idx = _midx(4)
    net = pd.Series([0.0, -0.01, 0.0, -0.01], index=idx)  # zeros break the streak
    assert max_consecutive_loss_months(net) == 1


# ── discrete trade count ─────────────────────────────────────────────────────
def test_trade_count_buy_hold_is_one():
    idx = _midx(6)
    gold = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03, 0.01], index=idx)
    tbill = pd.Series(0.0, index=idx)
    bt = run_backtest(s0_buy_hold(idx), gold, tbill, cost_bps=0.0)
    assert trade_count(bt) == 1  # single entry from cash, never exits


def test_trade_count_counts_entries_and_exits():
    idx = _midx(6)
    gold = pd.Series(0.0, index=idx)
    tbill = pd.Series(0.0, index=idx)
    # held = pos.shift(1): [nan, 1, 0, 1, 0, 1] → invested flips:
    #   open invested(+1), 1→0(+1), 0→1(+1), 1→0(+1), 0→1(+1) = 5
    pos = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], index=idx)
    bt = run_backtest(pos, gold, tbill, cost_bps=0.0)
    assert trade_count(bt) == 5


def test_trade_count_starts_in_cash():
    idx = _midx(5)
    gold = pd.Series(0.0, index=idx)
    tbill = pd.Series(0.0, index=idx)
    # held = pos.shift(1): [nan, 0, 0, 1, 0] → starts in cash (no opening trade),
    # one entry (0→1) and one exit (1→0) = 2
    pos = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0], index=idx)
    bt = run_backtest(pos, gold, tbill, cost_bps=0.0)
    assert trade_count(bt) == 2


def test_trade_count_empty():
    assert trade_count(pd.DataFrame()) == 0


# ── extended_metrics carries base + new fields, same caliber ────────────────
def test_extended_metrics_has_all_fields():
    idx = _midx(24)
    rng = np.random.default_rng(3)
    net = pd.Series(rng.normal(0.005, 0.03, 24), index=idx)
    bt = _bt_from_net(net)
    m = extended_metrics(bt)
    for k in ("sharpe", "calmar", "cagr", "max_dd", "hit_rate", "ann_turnover",
              "n_months", "longest_underwater_m", "max_consec_loss_m", "n_trades"):
        assert k in m
    assert m["n_months"] == 24


# ── majority vote ────────────────────────────────────────────────────────────
def _synth_panel(n=80, seed=7):
    idx = _midx(n, start="1990-01-31")
    rng = np.random.default_rng(seed)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0.004, 0.04, n)), index=idx)
    return pd.DataFrame({
        "gold_nominal": price,
        "gold_ret": price.pct_change(fill_method=None),
        "real_rate_10y": pd.Series(np.linspace(1.0, 0.5, n), index=idx),
        "usd_broad": pd.Series(np.linspace(100, 110, n), index=idx),
        "tbill_ret": pd.Series(0.002, index=idx),
    })


def test_majority_vote_binary_and_bounded():
    panel = _synth_panel()
    # vol_window=3 < the 12m trend lookback so the trend warm-up is the binding
    # one; asserting against the function's own first_valid_index keeps this
    # robust to DEFAULT_VOL_WINDOW changes (codex P2).
    vote = s1_majority_vote(panel, lookbacks=(3, 6, 12), vol_window=3)
    sized = vote.dropna()
    assert len(sized) > 0
    assert sized.between(0.0, 1.0).all()
    # warm-up is contiguous NaN then no holes: once valid, valid to the end
    fv = vote.first_valid_index()
    assert fv is not None
    assert vote.loc[:fv].iloc[:-1].isna().all()   # everything before first-valid is NaN
    assert vote.loc[fv:].notna().all()            # no interior NaN after warm-up
    # trend warm-up must be ≥ the longest lookback (12) — never full exposure early
    assert vote.iloc[:12].isna().all()


def test_majority_vote_all_rising_is_full_when_vol_low():
    # strictly rising price + tiny vol → unanimous long, vol-scale ~1 → ~full.
    # Longer sample + short vol_window so a valid tail is guaranteed (no empty
    # dropna()).
    idx = _midx(30, start="1990-01-31")
    price = pd.Series(100 * np.cumprod(np.r_[[1.0], np.full(29, 1.005)]), index=idx)
    panel = pd.DataFrame({
        "gold_nominal": price,
        "gold_ret": price.pct_change(fill_method=None),
    })
    vote = s1_majority_vote(panel, lookbacks=(3, 6, 12), target_vol=10.0, vol_window=3)
    sized = vote.dropna()
    assert len(sized) > 0
    assert sized.iloc[-1] == pytest.approx(1.0)  # huge target → vs capped 1, unanimous long


def test_variant_position_dispatch_and_bad_kind():
    panel = _synth_panel()
    blend = variant_position(panel, "blend", (3, 6, 12))
    vote = variant_position(panel, "vote", (3, 6, 12))
    assert blend.dropna().between(0.0, 1.0).all()
    assert vote.dropna().between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="unknown variant kind"):
        variant_position(panel, "nonsense", (3,))


# ── common window ────────────────────────────────────────────────────────────
def test_common_window_intersection():
    a = _bt_from_net(pd.Series(0.01, index=_midx(24, "2000-01-31")))
    b = _bt_from_net(pd.Series(0.01, index=_midx(24, "2000-06-30")))
    cstart, cend = common_window({"S0": a, "S1": b})
    assert cstart == pd.Timestamp("2000-06-30")
    assert cend == pd.Timestamp("2001-12-31")


def test_common_window_none_when_empty():
    a = _bt_from_net(pd.Series(0.01, index=_midx(6)))
    cstart, cend = common_window({"S0": a, "S1": a.iloc[0:0]})
    assert cstart is None and cend is None


# ── segment metrics on the shared window ─────────────────────────────────────
def test_segment_metrics_same_months_for_all():
    # S0 invested from month1, S1-like invested from month13 (leading NaN held).
    idx = _midx(60, start="1995-01-31")
    gold = pd.Series(np.linspace(0.01, 0.02, 60), index=idx)
    tbill = pd.Series(0.001, index=idx)
    s0 = run_backtest(s0_buy_hold(idx), gold, tbill, cost_bps=0.0)
    pos_late = pd.Series(1.0, index=idx)
    pos_late.iloc[:12] = np.nan  # warm-up like S1
    s1 = run_backtest(pos_late, gold, tbill, cost_bps=0.0)
    bts = {S0_LABEL: s0, "S1_x": s1}
    tbl = segment_metrics(bts, "1995-01-01", "1999-12-31")
    # both rows present, S0 first, identical month count (fair window trims S0)
    assert list(tbl.index)[0] == S0_LABEL
    assert tbl.loc[S0_LABEL, "n_months"] == tbl.loc["S1_x", "n_months"]
    assert set(METRIC_COLS).issubset(tbl.columns)


def test_segment_metrics_no_overlap_is_nan_row():
    idx = _midx(24, start="2000-01-31")
    bt = _bt_from_net(pd.Series(0.01, index=idx))
    bts = {S0_LABEL: bt, "S1_x": bt}
    tbl = segment_metrics(bts, "1970-01-01", "1975-12-31")  # before the data
    # empty slice → compute_metrics returns NaN metrics (n_months NaN), not a crash
    assert tbl["n_months"].isna().all()
    assert tbl["sharpe"].isna().all()


# ── paired significance ──────────────────────────────────────────────────────
def test_paired_net_diff_constant_gap_excludes_zero():
    idx = _midx(60)
    a = _bt_from_net(pd.Series(0.02, index=idx))   # S1
    b = _bt_from_net(pd.Series(0.01, index=idx))   # S0 → diff +1%/mo constant
    st = paired_net_diff_stats(a, b, n_boot=500, seed=0)
    assert st["mean_monthly"] == pytest.approx(0.01)
    assert st["ann_mean"] == pytest.approx(0.12)
    assert st["ci_excludes_zero"] is True
    assert st["ci_lo"] == pytest.approx(0.12, abs=1e-9)  # zero-variance bootstrap


def test_paired_net_diff_zero_mean_includes_zero():
    idx = _midx(120)
    rng = np.random.default_rng(11)
    common = rng.normal(0.0, 0.02, 120)
    a = _bt_from_net(pd.Series(common + rng.normal(0.0, 0.001, 120), index=idx))
    b = _bt_from_net(pd.Series(common, index=idx))
    st = paired_net_diff_stats(a, b, n_boot=1000, seed=1)
    # near-zero mean diff → CI straddles zero
    assert st["ci_excludes_zero"] is False


def test_paired_net_diff_too_short():
    idx = _midx(1)
    a = _bt_from_net(pd.Series(0.01, index=idx))
    st = paired_net_diff_stats(a, a)
    assert st["n"] == 1
    assert st["ci_excludes_zero"] is False
    assert np.isnan(st["mean_monthly"])


# ── integration: full run across cost grid & variants, same caliber as S0 ────
def test_integration_all_costs_and_variants():
    # ~36 years monthly so the post-2000 window is populated.
    n = 440
    idx = _midx(n, start="1990-01-31")
    rng = np.random.default_rng(42)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0.003, 0.045, n)), index=idx)
    panel = pd.DataFrame({
        "gold_nominal": price,
        "gold_ret": price.pct_change(fill_method=None),
        "real_rate_10y": pd.Series(np.linspace(1.0, -0.5, n), index=idx),
        "usd_broad": pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx),
        "tbill_ret": pd.Series(0.002, index=idx),
    })
    positions = build_positions(panel)
    # S0 + all S1 variants present
    assert S0_LABEL in positions
    assert all(f"S1_{lbl}" in positions for lbl, _, _ in S1_VARIANTS)

    for cost in COST_GRID:
        bts = {
            lbl: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=cost)
            for lbl, p in positions.items()
        }
        common = common_window(bts)
        assert common[0] is not None
        for name, s, e in SUBPERIOD_SEGMENTS:
            tbl = segment_metrics(bts, s, e, common=common)
            # every strategy scored on the same month count within the segment
            counts = tbl["n_months"].unique()
            assert len(counts) == 1, f"{name} @ {cost}bps: unequal month counts {counts}"
            assert set(METRIC_COLS).issubset(tbl.columns)

    # post-2000 paired read runs without error and returns the expected keys
    bts10 = {lbl: run_backtest(p, panel["gold_ret"], panel["tbill_ret"], cost_bps=10.0)
             for lbl, p in positions.items()}
    st = paired_net_diff_stats(bts10["S1_blend_3_6_12"], bts10[S0_LABEL], n_boot=300)
    assert {"ann_mean", "t_stat", "ci_lo", "ci_hi", "ci_excludes_zero"}.issubset(st)


def test_segments_cover_post2000_window():
    names = {n for n, _, _ in SUBPERIOD_SEGMENTS}
    assert POST2000_SEGMENT in names
    assert {"1968-1980", "1980-2000", "2000-2011", "2011-2015", "2016-2026"} <= names


def test_segment_window_is_single_source_of_truth():
    # The paired-CI window and the report tables both resolve bounds here, so
    # they cannot desync. Bounds must match the constant verbatim.
    s, e = segment_window(POST2000_SEGMENT)
    assert (POST2000_SEGMENT, s, e) in SUBPERIOD_SEGMENTS
    with pytest.raises(KeyError, match="unknown segment"):
        segment_window("not-a-segment")


# ── trade_count boundary (prev_held) — codex P2a ─────────────────────────────
def test_trade_count_prev_held_carried_position_is_zero():
    # A position carried in from before the slice (prev_held invested) and held
    # throughout did NOT trade inside the window → 0, not a spurious boundary 1.
    idx = _midx(6)
    bt = pd.DataFrame({"held": pd.Series(1.0, index=idx)})
    assert trade_count(bt, prev_held=1.0) == 0
    # no prior info (start of data, from cash) → the genuine opening entry counts
    assert trade_count(bt, prev_held=None) == 1


def test_trade_count_prev_held_exit_at_boundary_counts():
    # Held cash through the slice but invested the month before → one exit at the
    # boundary (the trade that flattened the carried position) counts.
    idx = _midx(4)
    bt = pd.DataFrame({"held": pd.Series(0.0, index=idx)})
    assert trade_count(bt, prev_held=1.0) == 1
    assert trade_count(bt, prev_held=0.0) == 0  # cash→cash, no trade


def test_segment_metrics_s0_midsample_has_zero_trades():
    # S0 (buy-and-hold) sliced to a mid-sample segment must show 0 trades — it
    # entered once, before the segment, and never traded inside it.
    idx = _midx(120, start="1990-01-31")
    gold = pd.Series(np.linspace(0.01, 0.015, 120), index=idx)
    tbill = pd.Series(0.001, index=idx)
    s0 = run_backtest(s0_buy_hold(idx), gold, tbill, cost_bps=0.0)
    bts = {S0_LABEL: s0}
    common = common_window(bts)
    # first segment (window start) → the genuine entry counts as 1
    first = segment_metrics(bts, "1990-01-01", "1994-12-31", common=common)
    assert first.loc[S0_LABEL, "n_trades"] == 1
    # a later, mid-sample segment → carried in, 0 trades inside
    later = segment_metrics(bts, "1996-01-01", "1999-12-31", common=common)
    assert later.loc[S0_LABEL, "n_trades"] == 0


# ── paired stats: HAC + moving-block bootstrap — codex P2b ────────────────────
def test_paired_net_diff_reports_block_and_hac_params():
    idx = _midx(64)
    rng = np.random.default_rng(5)
    a = _bt_from_net(pd.Series(rng.normal(0.01, 0.02, 64), index=idx))
    b = _bt_from_net(pd.Series(rng.normal(0.005, 0.02, 64), index=idx))
    st = paired_net_diff_stats(a, b, n_boot=400, seed=0)
    assert st["block_len"] >= 1 and st["hac_lag"] >= 1
    # default block ≈ √n, hac lag ≈ n^(1/3)
    assert st["block_len"] == max(1, round(np.sqrt(64)))
    assert st["hac_lag"] == max(1, round(64 ** (1.0 / 3.0)))


def test_paired_net_diff_validates_and_clamps_params():
    idx = _midx(40)
    a = _bt_from_net(pd.Series(0.01, index=idx))
    b = _bt_from_net(pd.Series(0.005, index=idx))
    # n_boot must be positive
    with pytest.raises(ValueError, match="n_boot"):
        paired_net_diff_stats(a, b, n_boot=0)
    # explicit non-positive block_len / hac_lag rejected
    with pytest.raises(ValueError, match="block_len"):
        paired_net_diff_stats(a, b, block_len=0)
    with pytest.raises(ValueError, match="hac_lag"):
        paired_net_diff_stats(a, b, hac_lag=0)
    # block_len > n is clamped to n (no negative max_start / rng crash)
    st = paired_net_diff_stats(a, b, n_boot=200, block_len=999)
    assert st["block_len"] == st["n"] == 40
    assert pd.notna(st["ci_lo"]) and pd.notna(st["ci_hi"])


def test_paired_hac_se_adds_positive_autocovariance():
    # Deterministic (no RNG): a monotone ramp is strongly positively
    # autocorrelated, so adding positive-lag Bartlett terms must strictly
    # increase the HAC se over the γ0-only (lag-0 = IID) se. This tests the
    # mechanism directly rather than relying on a single random AR(1) draw.
    from lib.gold_s1_subperiod import _bartlett_hac_se_mean
    x = np.arange(40, dtype="float64")          # strict positive autocorrelation
    se_lag0 = _bartlett_hac_se_mean(x, lag=0)    # γ0 only ≡ IID se of the mean
    se_lag6 = _bartlett_hac_se_mean(x, lag=6)    # + positive lagged autocovariances
    assert se_lag6 > se_lag0


def test_paired_hac_lag_clamped_and_reported():
    # An over-large hac_lag is clamped to n-1 and the RETURNED hac_lag reflects
    # the actual value used (codex P3), not the raw input.
    idx = _midx(20)
    a = _bt_from_net(pd.Series(0.01, index=idx))
    b = _bt_from_net(pd.Series(0.004, index=idx))
    st = paired_net_diff_stats(a, b, n_boot=200, hac_lag=999)
    assert st["hac_lag"] == st["n"] - 1 == 19


# ── verdict() branch coverage — codex P1 (verdict now lives in lib) ──────────
_SEG_COLS = ["sharpe", "calmar", "cagr", "max_dd", "longest_underwater_m",
             "max_consec_loss_m", "ann_turnover", "n_trades", "hit_rate", "n_months"]


def _seg_table(s0, s1):
    """Build a POST2000 metrics table with S0, the primary S1, and every S1
    variant (all variants mirror the primary so the robustness count is stable).
    `s0`/`s1` are dicts with at least sharpe/calmar/cagr/max_dd."""
    rows = {S0_LABEL: s0}
    for lbl, _, _ in S1_VARIANTS:
        rows[f"S1_{lbl}"] = dict(s1)
    df = pd.DataFrame(rows).T
    for c in _SEG_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[_SEG_COLS].astype(float)


def _seg_by_cost(s0_10, s1_10, s0_25, s1_25):
    return {
        10.0: {POST2000_SEGMENT: _seg_table(s0_10, s1_10)},
        25.0: {POST2000_SEGMENT: _seg_table(s0_25, s1_25)},
    }


def _paired(ann_mean, ci_lo, ci_hi, n=200):
    excl = ci_lo > 0 or ci_hi < 0
    return {10.0: {"n": n, "ann_mean": ann_mean, "t_stat": 1.0, "ci_lo": ci_lo,
                   "ci_hi": ci_hi, "ci_excludes_zero": excl,
                   "block_len": 5, "hac_lag": 3}}


def test_verdict_both_axes_significant():
    s0 = {"sharpe": 0.5, "calmar": 0.3, "cagr": 0.05, "max_dd": -0.4}
    s1 = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    out = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(0.04, 0.01, 0.07))
    assert "on both axes" in out and "STILL HAS EDGE" in out


def test_verdict_both_axes_but_not_significant():
    # P1 regression: S1 higher CAGR (ret True) but CI includes zero must NOT say
    # "GIVES UP raw CAGR" — it should land in the ①a 'leans positive' branch.
    s0 = {"sharpe": 0.5, "calmar": 0.3, "cagr": 0.05, "max_dd": -0.4}
    s1 = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    out = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(0.02, -0.01, 0.05))
    assert "LEANS POSITIVE on both axes" in out
    assert "GIVES UP" not in out  # must not contradict the data


def test_verdict_risk_reducer_gives_up_return():
    # S1 wins risk-adjusted but LOSES on CAGR → ①′ mixed risk-reducer.
    s0 = {"sharpe": 0.74, "calmar": 0.28, "cagr": 0.111, "max_dd": -0.393}
    s1 = {"sharpe": 0.86, "calmar": 0.51, "cagr": 0.095, "max_dd": -0.187}
    out = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(-0.019, -0.045, 0.007))
    assert "risk-reducer post-2000, NOT a return-enhancer" in out
    assert "GIVES UP raw CAGR" in out


def test_verdict_decayed_when_not_risk_adjusted_winner():
    # S1 loses risk-adjusted (lower Sharpe) → ② decayed, regardless of CAGR.
    s0 = {"sharpe": 0.9, "calmar": 0.6, "cagr": 0.10, "max_dd": -0.2}
    s1 = {"sharpe": 0.6, "calmar": 0.3, "cagr": 0.12, "max_dd": -0.3}
    out = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(-0.01, -0.05, 0.03))
    assert "DECAYED" in out


def test_verdict_requires_both_costs_for_risk_adjusted_win():
    # Wins risk-adjusted @10bps but NOT @25bps → not a clean ra win → ② decayed.
    s0_10 = {"sharpe": 0.5, "calmar": 0.3, "cagr": 0.05, "max_dd": -0.4}
    s1_10 = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    s0_25 = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}  # S0 ahead @25
    s1_25 = {"sharpe": 0.6, "calmar": 0.3, "cagr": 0.05, "max_dd": -0.3}
    out = verdict(_seg_by_cost(s0_10, s1_10, s0_25, s1_25),
                      _paired(0.02, 0.005, 0.04))
    assert "DECAYED" in out


def test_verdict_guards_nan_metrics():
    s0 = {"sharpe": float("nan"), "calmar": 0.3, "cagr": 0.05, "max_dd": -0.4}
    s1 = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    out = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(0.04, 0.01, 0.07))
    assert "cannot adjudicate" in out.lower()


def test_verdict_guards_nan_at_25bps_not_silent_decay():
    # 25bps metrics drive the verdict too; a NaN there must yield "cannot
    # adjudicate", NOT a silent False→DECAYED (codex P2).
    good = {"sharpe": 0.9, "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    s0 = {"sharpe": 0.5, "calmar": 0.3, "cagr": 0.05, "max_dd": -0.4}
    bad25 = {"sharpe": float("nan"), "calmar": 0.7, "cagr": 0.09, "max_dd": -0.2}
    out = verdict(_seg_by_cost(s0, good, s0, bad25), _paired(0.02, 0.005, 0.04))
    assert "cannot adjudicate" in out.lower()
    # also guard a NaN CAGR at 25bps (used by the raw-return axis)
    bad25c = {"sharpe": 0.9, "calmar": 0.7, "cagr": float("nan"), "max_dd": -0.2}
    out2 = verdict(_seg_by_cost(s0, good, s0, bad25c), _paired(0.02, 0.005, 0.04))
    assert "cannot adjudicate" in out2.lower()


def test_verdict_guards_short_or_nan_paired_sample():
    # The paired stats drive the significance branch — a degenerate paired sample
    # (n<2 or NaN mean/CI) must yield "cannot adjudicate", not a "leans/mixed"
    # conclusion built on NaN comparisons (codex P2).
    s0 = {"sharpe": 0.74, "calmar": 0.28, "cagr": 0.111, "max_dd": -0.39}
    s1 = {"sharpe": 0.86, "calmar": 0.51, "cagr": 0.095, "max_dd": -0.19}
    out_short = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(-0.019, -0.045, 0.007, n=1))
    assert "cannot adjudicate" in out_short.lower()
    nan = float("nan")
    out_nan = verdict(_seg_by_cost(s0, s1, s0, s1), _paired(nan, nan, nan))
    assert "cannot adjudicate" in out_nan.lower()


def test_default_cost_is_a_grid_point():
    # The headline/verdict cost must be one of the cost-grid points (the script
    # indexes bt_by_cost[DEFAULT_COST_BPS]); guard the cross-module contract.
    from lib.gold_trend_timing import DEFAULT_COST_BPS
    assert DEFAULT_COST_BPS in COST_GRID
    assert 10.0 in COST_GRID and 25.0 in COST_GRID  # verdict reads both explicitly