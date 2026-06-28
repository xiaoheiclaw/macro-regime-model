"""Offline tests for the gold-anchor data layer & cointegration gate.

All tests use synthetic series (injected fetchers) — no network / FRED key
required. The statistical assertions check that the test wrappers give the
*right verdict* on series with known integration / cointegration structure.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.gold_anchor import (
    build_anchor_panel,
    classify_integration,
    integration_table,
    johansen_test,
    unit_root_tests,
)

def _white_noise(n=400, seed=12345):
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    return pd.Series(np.random.default_rng(seed).standard_normal(n), index=idx)


def _random_walk(n=400, drift=0.0, seed=1):
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    steps = np.random.default_rng(seed).standard_normal(n) + drift
    return pd.Series(np.cumsum(steps), index=idx)


# ── unit-root tests ────────────────────────────────────────────────────
def test_unit_root_tests_keys():
    out = unit_root_tests(_white_noise())
    for k in ["adf_pvalue", "pp_pvalue", "kpss_pvalue", "n"]:
        assert k in out


def test_white_noise_is_I0():
    res = classify_integration(_white_noise(500))
    assert res["verdict"] == "I(0)"


def test_random_walk_is_I1():
    res = classify_integration(_random_walk(500, seed=7))
    assert res["verdict"] == "I(1)"
    # first difference of a random walk is white noise → stationary
    assert res["diff_stationary"] is True


def test_integration_table_shape():
    df = pd.DataFrame(
        {"a": _white_noise(300).values, "b": _random_walk(300, seed=3).values},
        index=pd.date_range("1980-01-31", periods=300, freq="ME"),
    )
    tab = integration_table(df, ["a", "b"])
    assert list(tab.index) == ["a", "b"]
    assert tab.loc["a", "verdict"] == "I(0)"
    assert tab.loc["b", "verdict"] == "I(1)"
    for col in ["adf_p", "pp_p", "kpss_p", "n"]:
        assert col in tab.columns


# ── cointegration ──────────────────────────────────────────────────────
def test_johansen_detects_cointegration():
    # G = beta*A + stationary noise, A = random walk → cointegrated, rank 1
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    a = np.cumsum(np.random.default_rng(11).standard_normal(n))
    beta_true = 1.0
    g = beta_true * a + np.random.default_rng(22).standard_normal(n) * 0.3
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_anchor": a}, index=idx)
    j = johansen_test(df, ["ln_gold_nominal", "ln_anchor"])
    assert j["rank"] >= 1
    assert abs(j["beta"] - beta_true) < 0.2


def test_johansen_no_cointegration_for_independent_walks():
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    a = np.cumsum(np.random.default_rng(101).standard_normal(n))
    g = np.cumsum(np.random.default_rng(202).standard_normal(n))
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_anchor": a}, index=idx)
    j = johansen_test(df, ["ln_gold_nominal", "ln_anchor"])
    assert j["rank"] == 0


def test_johansen_full_rank_flagged_and_capped():
    # two stationary (white-noise) series → full rank; must NOT read as anchor
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    g = np.random.default_rng(5).standard_normal(n)
    a = np.random.default_rng(6).standard_normal(n)
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_anchor": a}, index=idx)
    j = johansen_test(df, ["ln_gold_nominal", "ln_anchor"])
    assert j["full_rank_stationary"] is True
    assert j["rank"] <= 1  # capped to n-1
    assert j["raw_trace_rank"] == 2 or j["raw_maxeig_rank"] == 2


# ── panel builder (injected synthetic fetchers) ────────────────────────
def _synthetic_fred(series_id, start="1968-01-01"):
    """Return a synthetic series with the right native frequency per series."""
    # crc32 is a stable cross-process seed (hash() is randomized per process)
    rng = np.random.default_rng(zlib.crc32(series_id.encode()))
    if series_id == "GFDEBTN":  # quarterly debt, $M, rising
        idx = pd.date_range("1968-01-01", "2025-12-31", freq="QS")
        return pd.Series(np.linspace(3e5, 3.5e7, len(idx)) * (1 + 0.01 * rng.standard_normal(len(idx))), index=idx)
    if series_id == "GDP":  # quarterly GDP, $B
        idx = pd.date_range("1968-01-01", "2025-12-31", freq="QS")
        return pd.Series(np.linspace(1000, 28000, len(idx)), index=idx)
    if series_id == "M2SL":  # monthly M2, $B
        idx = pd.date_range("1968-01-01", "2025-12-31", freq="MS")
        return pd.Series(np.linspace(500, 21000, len(idx)), index=idx)
    if series_id == "WALCL":  # weekly Fed assets, $M, from 2002
        idx = pd.date_range("2002-12-18", "2025-12-31", freq="W")
        return pd.Series(np.linspace(7e5, 9e6, len(idx)), index=idx)
    if series_id == "DFII10":  # daily TIPS from 2003
        idx = pd.date_range("2003-01-02", "2025-12-31", freq="B")
        return pd.Series(rng.standard_normal(len(idx)) * 0.5 + 1.0, index=idx)
    if series_id == "GS10":  # monthly nominal 10y
        idx = pd.date_range("1968-01-01", "2025-12-31", freq="MS")
        return pd.Series(rng.standard_normal(len(idx)) * 0.5 + 5.0, index=idx)
    if series_id == "CPIAUCSL":  # monthly CPI index, rising
        idx = pd.date_range("1968-01-01", "2025-12-31", freq="MS")
        return pd.Series(np.linspace(34, 320, len(idx)), index=idx)
    raise ValueError(f"unexpected series {series_id}")


def _synthetic_gold(start="1968-01-01"):
    idx = pd.date_range("1968-01-31", "2025-12-31", freq="ME")
    return pd.Series(np.linspace(35, 2000, len(idx)), index=idx)


def test_build_panel_structure():
    panel = build_anchor_panel(fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold)
    df = panel.data
    for col in ["gold_nominal", "debt_gdp", "m2_gdp", "fed_gdp", "real_rate_10y",
                "ln_gold_nominal", "ln_debt_gdp", "ln_m2_gdp", "ln_fed_gdp"]:
        assert col in df.columns, f"missing {col}"
    # monthly index — assert real month-end frequency, never silently None
    assert pd.infer_freq(df.index) in {"ME", "M"}
    # ratios use GDP as denominator → debt_gdp = debt($B) / gdp($B), order ~1
    assert df["debt_gdp"].dropna().median() > 0
    # ln transform consistency
    valid = df["debt_gdp"].dropna()
    assert np.allclose(np.log(valid), df.loc[valid.index, "ln_debt_gdp"])


def test_build_panel_notes_and_splice():
    panel = build_anchor_panel(fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold)
    for key in ["real_rate_splice", "units", "frequency", "coverage", "fed_gdp_coverage"]:
        assert key in panel.notes
    # real rate exists both before and after the TIPS splice (2003)
    rr = panel.data["real_rate_10y"].dropna()
    assert (rr.index < pd.Timestamp("2003-01-01")).any()
    assert (rr.index >= pd.Timestamp("2003-01-01")).any()


def test_panel_coverage_handles_all_nan_column():
    # end before WALCL starts (2002) → fed_gdp/ln_fed_gdp all NaN; coverage
    # note must not crash on NaT min/max (P1 regression guard).
    panel = build_anchor_panel(
        start="1968-01-01", end="2000-12-31",
        fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold,
    )
    assert panel.data["fed_gdp"].notna().sum() == 0
    assert "no observations (n=0)" in panel.notes["coverage"]
    assert isinstance(panel.notes["fed_gdp_coverage"], str)


def test_units_rescaled_to_billions():
    panel = build_anchor_panel(fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold)
    # debt rescaled $M→$B: with synthetic debt up to 3.5e7 $M = 3.5e4 $B and
    # gdp up to 2.8e4 $B, debt/GDP should be order ~1, not ~1000.
    assert panel.data["debt_gdp"].dropna().max() < 10


def test_start_contract_enforced_for_misbehaving_fetcher():
    # fetchers that ignore `start` must not leak pre-start rows into the panel
    panel = build_anchor_panel(
        start="1990-01-01", fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold,
    )
    assert panel.data.index.min() >= pd.Timestamp("1990-01-01")


def test_johansen_validates_inputs():
    idx = pd.date_range("1980-01-31", periods=40, freq="ME")
    df = pd.DataFrame({"ln_gold_nominal": np.arange(40.0), "ln_anchor": np.arange(40.0)}, index=idx)
    with pytest.raises(ValueError):
        johansen_test(df, ["ln_gold_nominal"])  # <2 columns
    with pytest.raises(ValueError):
        johansen_test(df, ["ln_gold_nominal", "ln_anchor"], alpha=0.123)  # bad alpha
    short = df.iloc[:10]
    with pytest.raises(ValueError):
        johansen_test(short, ["ln_gold_nominal", "ln_anchor"])  # too few rows
    bad = df.copy()
    bad.iloc[5, 0] = np.inf
    with pytest.raises(ValueError):
        johansen_test(bad, ["ln_gold_nominal", "ln_anchor"])  # non-finite


def test_should_run_johansen_gate():
    from scripts.gold_anchor_analysis import should_run_johansen

    assert should_run_johansen("I(1)", "I(1)") is True
    # stationary or ambiguous anchor (or gold) must NOT enter Johansen
    assert should_run_johansen("I(1)", "I(0)") is False
    assert should_run_johansen("I(1)", "ambiguous") is False
    assert should_run_johansen("I(0)", "I(1)") is False
    assert should_run_johansen("I(1)", "insufficient data") is False


def test_full_rank_marks_invalid_coint():
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    g = np.random.default_rng(5).standard_normal(n)
    a = np.random.default_rng(6).standard_normal(n)
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_anchor": a}, index=idx)
    j = johansen_test(df, ["ln_gold_nominal", "ln_anchor"])
    assert j["full_rank_stationary"] is True
    assert j["valid_coint"] is False
    assert j["coint_rank"] == 0
    assert j["beta"] is None


def test_end_month_boundary_includes_month_end():
    # end="2025-12" must include 2025-12-31, not exclude the whole month
    panel = build_anchor_panel(
        start="1968-01-01", end="2025-12",
        fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold,
    )
    assert pd.Timestamp("2025-12-31") in panel.data.index
    assert panel.data.index.max() <= pd.Timestamp("2025-12-31")
