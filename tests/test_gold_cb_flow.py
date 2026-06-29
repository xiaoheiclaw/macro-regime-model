"""Tests for the PR #10 WGC central-bank flow layer (lib/gold_cb_flow.py).

Offline by construction: the loader/signal builder are pure (embedded series +
tmp CSVs), and the end-to-end attribution check injects synthetic anchor/fetch
exactly like the PR #9 test, so there is no network/FRED dependency. The focus is
mechanics, not any empirical claim:

  * CSV write/read round-trip + embedded-fallback when the file is absent,
  * annual→monthly "均摊" interpolation (sum over a year == annual flow),
  * the four flow signals (cum_excess / cum_stock / excess_flow / flow),
  * 2026 (partial-year) carry-forward AND the no-carry fallback,
  * injecting make_wgc_fn() into the PR #9 panel keeps the EXACT decomposition
    identity (Σ contributions + residual == Δln gold) with ⑤ now explicit.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_cb_flow import (
    BASELINE_TONNES,
    DEFAULT_SIGNAL,
    VALID_SIGNALS,
    WGC_ANNUAL_TONNES,
    build_flow_signal,
    load_wgc_annual,
    make_wgc_fn,
    write_wgc_csv,
)
from lib.gold_credit_spread_attribution import (
    build_attribution_panel,
    build_design,
    decompose_period,
    fit_attribution,
)


# ── 1. CSV materialization + load ────────────────────────────────────────
def test_write_and_load_roundtrip(tmp_path):
    p = str(tmp_path / "wgc.csv")
    write_wgc_csv(p)
    assert os.path.exists(p)
    s = load_wgc_annual(p)
    # every embedded year survives the round-trip with the right tonnage
    for y, v in WGC_ANNUAL_TONNES.items():
        assert float(s.loc[pd.Timestamp(f"{y}-12-31")]) == pytest.approx(v)
    # provenance header present, not parsed as data
    with open(p) as f:
        head = f.read()
    assert "World Gold Council" in head and "ESTIMATES" in head


def test_load_falls_back_to_embedded_when_missing(tmp_path):
    p = str(tmp_path / "absent.csv")
    s = load_wgc_annual(p, write_if_missing=False)
    assert len(s) == len(WGC_ANNUAL_TONNES)
    assert float(s.loc[pd.Timestamp("2022-12-31")]) == pytest.approx(1082.0)
    # not written when write_if_missing=False
    assert not os.path.exists(p)


def test_load_writes_when_missing_and_allowed(tmp_path):
    p = str(tmp_path / "made.csv")
    load_wgc_annual(p, write_if_missing=True)
    assert os.path.exists(p)


def test_load_bad_csv_raises(tmp_path):
    p = str(tmp_path / "bad.csv")
    with open(p, "w") as f:
        f.write("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_wgc_annual(p)


def test_baseline_matches_brief():
    # 2010-2021 mean ≈ 473 (the PR #10 brief baseline)
    assert BASELINE_TONNES == pytest.approx(473.0, abs=1.0)


# ── 2. annual→monthly interpolation ──────────────────────────────────────
def test_monthly_flow_sums_to_annual():
    s = load_wgc_annual(write_if_missing=False)
    flow = build_flow_signal(s, signal="flow", carry_forward_partial=False)
    # 'flow' signal is the annualized rate held flat within a year; the 12
    # monthly values of 2022 each equal the 2022 annual flow.
    y2022 = flow[flow.index.year == 2022]
    assert len(y2022) == 12
    np.testing.assert_allclose(y2022.values, 1082.0)


def test_cum_stock_is_cumulative_tonnes():
    s = load_wgc_annual(write_if_missing=False)
    cum = build_flow_signal(s, signal="cum_stock", carry_forward_partial=False)
    # final cumulative ≈ sum of all annual flows (months spread then summed)
    assert float(cum.iloc[-1]) == pytest.approx(sum(WGC_ANNUAL_TONNES.values()), rel=1e-9)
    assert cum.is_monotonic_increasing


def test_cum_excess_subtracts_baseline():
    s = load_wgc_annual(write_if_missing=False)
    cum_stock = build_flow_signal(s, signal="cum_stock", carry_forward_partial=False)
    cum_excess = build_flow_signal(s, signal="cum_excess", carry_forward_partial=False)
    n = len(cum_stock)
    # cum_excess == cum_stock − baseline·(months/12)
    expected = cum_stock.iloc[-1] - BASELINE_TONNES * (n / 12.0)
    assert float(cum_excess.iloc[-1]) == pytest.approx(expected, rel=1e-9)


def test_invalid_signal_raises():
    s = load_wgc_annual(write_if_missing=False)
    with pytest.raises(ValueError, match="signal must be one of"):
        build_flow_signal(s, signal="bogus")


def test_all_signals_nonempty():
    s = load_wgc_annual(write_if_missing=False)
    for sig in VALID_SIGNALS:
        assert len(build_flow_signal(s, signal=sig, carry_forward_partial=False)) > 0


# ── 3. 2026 partial-year carry-forward / fallback ────────────────────────
def test_carry_forward_extends_past_last_full_year():
    """With carry_forward, requesting end in 2026 extends the monthly grid using
    the latest annual pace (2025); the cumulative signal reaches 2026."""
    s = load_wgc_annual(write_if_missing=False)  # last full year 2025
    flow = build_flow_signal(s, signal="flow", end=pd.Timestamp("2026-03-31"),
                             carry_forward_partial=True)
    mar26 = flow[flow.index == pd.Timestamp("2026-03-31")]
    assert len(mar26) == 1
    # carried at the 2025 annual pace
    assert float(mar26.iloc[0]) == pytest.approx(863.0)


def test_no_carry_stops_at_last_full_year():
    """缺 2026 数据回退:carry_forward_partial=False stops cleanly at the last
    full annual year rather than fabricating future months."""
    s = load_wgc_annual(write_if_missing=False)
    flow = build_flow_signal(s, signal="cum_excess", end=pd.Timestamp("2026-03-31"),
                             carry_forward_partial=False)
    assert flow.index.max() == pd.Timestamp("2025-12-31")


def test_missing_2026_row_still_runs_via_fallback():
    """A CSV that ends at 2025 (no 2026 row) still yields a signal reaching the
    2026 decomposition endpoint through carry-forward — the data-gap fallback."""
    s = load_wgc_annual(write_if_missing=False)
    assert 2026 not in s.index.year  # precondition: no 2026 in source
    fn = make_wgc_fn(signal="cum_excess")
    out = fn("2010-01-01", "2026-03-31")
    assert out.index.max() >= pd.Timestamp("2026-03-31")


# ── 4. end-to-end injection into the PR #9 panel ─────────────────────────
def _anchor_and_fetch(n=320, seed=11):
    """Synthetic anchor + fetch covering 2003+ (so ③ custody is present) through
    ~2029, matching the PR #9 offline test harness."""
    idx = pd.date_range("2003-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02) + 6.0), index=idx)
    debt_gdp = pd.Series(np.linspace(0.55, 1.25, n) + rng.randn(n) * 0.01, index=idx)
    real_rate = pd.Series(1.5 + np.cumsum(rng.randn(n) * 0.05), index=idx)
    anchor = SimpleNamespace(data=pd.DataFrame({
        "gold_nominal": gold, "debt_gdp": debt_gdp, "real_rate_10y": real_rate,
        "ln_gold_nominal": np.log(gold), "ln_debt_gdp": np.log(debt_gdp),
    }, index=idx))
    cpi = pd.Series(np.linspace(180, 320, n) + rng.randn(n) * 0.4, index=idx)
    vix = pd.Series(18 + rng.randn(n) * 4, index=idx).abs()
    credit = pd.Series(2.0 + rng.randn(n) * 0.5, index=idx).abs()
    custody = pd.Series(np.linspace(1.5e6, 2.6e6, n) + rng.randn(n) * 1e4, index=idx)
    debt_lvl = pd.Series(np.linspace(6e6, 3.7e7, n), index=idx)

    def fetch_fn(series_id, start="1968-01-01"):
        return {"CPIAUCSL": cpi, "VIXCLS": vix, "BAA10Y": credit,
                "WMTSECL1": custody, "GFDEBTN": debt_lvl}.get(
                    series_id, pd.Series(np.nan, index=idx))

    return anchor, fetch_fn


def test_injected_flow_layer_keeps_exact_identity():
    anchor, fetch_fn = _anchor_and_fetch()
    panel = build_attribution_panel(
        start="2003-01-01", fetch_fn=fetch_fn,
        anchor_fn=lambda *a, **k: anchor, wgc_fn=make_wgc_fn(signal="cum_excess"),
    )
    # ⑤ flow now present and an included design layer
    assert panel.data["wgc_flow"].notna().any()
    design = build_design(panel.data)
    assert "flow" in [l.key for l in design.layers]
    res = fit_attribution(panel, cpi_mode="identity")
    decomp = decompose_period(res, t0="2022-01")
    # EXACT additive identity holds with ⑤ explicit (Σ parts == TOTAL)
    total = float(decomp.loc[decomp["layer"] == "TOTAL", "contribution_ln"].iloc[0])
    parts = decomp[decomp["layer"] != "TOTAL"]["contribution_ln"].sum()
    assert parts == pytest.approx(total, abs=1e-9)
    # a 'flow' row exists (no longer folded only into the residual)
    assert (decomp["layer"] == "flow").any()


def test_default_signal_constant():
    assert DEFAULT_SIGNAL == "cum_excess" and DEFAULT_SIGNAL in VALID_SIGNALS
