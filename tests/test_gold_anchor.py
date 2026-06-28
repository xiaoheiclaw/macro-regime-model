"""Offline tests for the gold-anchor data layer & cointegration gate.

All tests use synthetic series (injected fetchers) — no network / FRED key
required. The statistical assertions check that the test wrappers give the
*right verdict* on series with known integration / cointegration structure.
"""
import zlib

import numpy as np
import pandas as pd
import pytest

# project root is added to sys.path by tests/conftest.py
import lib.gold_anchor as ga
from lib.gold_anchor import (
    build_anchor_panel,
    classify_integration,
    combined_verdict,
    estimate_vecm,
    integration_segments,
    integration_table,
    johansen_robustness,
    johansen_test,
    select_var_order,
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


def test_unit_root_tests_rejects_bad_input():
    from lib.gold_anchor import unit_root_tests
    idx = pd.date_range("1980-01-31", periods=50, freq="ME")
    with pytest.raises(ValueError):
        unit_root_tests(pd.Series([1.0] * 50, index=idx))            # constant
    with pytest.raises(ValueError):
        unit_root_tests(pd.Series(np.arange(5.0)))                   # too few
    bad = pd.Series(np.arange(50.0), index=idx)
    bad.iloc[3] = np.inf
    with pytest.raises(ValueError):
        unit_root_tests(bad)                                         # non-finite


def test_integration_table_reports_invalid_data():
    idx = pd.date_range("1980-01-31", periods=60, freq="ME")
    df = pd.DataFrame({"const": np.ones(60), "rw": _random_walk(60, seed=4).values}, index=idx)
    tab = integration_table(df, ["const", "rw"])
    assert tab.loc["const", "verdict"] == "invalid data"  # constant → caught
    assert tab.loc["rw", "verdict"] in {"I(1)", "ambiguous"}


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
        {"a": _white_noise(600).values, "b": _random_walk(600, seed=3).values},
        index=pd.date_range("1980-01-31", periods=600, freq="ME"),
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


def test_gold_hash_mismatch_hard_fails(monkeypatch):
    # if the pinned source content changes (sha256 != expected) → hard fail
    monkeypatch.setattr(ga, "_http_get_text", lambda url, timeout=30.0: "Date,Price\n2000-01,1.0\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ga.fetch_gold_monthly()


def test_combined_verdict_dual_regression():
    # both regressions are computed and reconciled. A random walk is cleanly
    # I(1) under c and ct; white noise is never I(1) (so it can't enter Johansen)
    # — its exact label may be I(0) or ambiguous depending on the KPSS draw.
    rw = combined_verdict(_random_walk(500, seed=2))
    assert rw["combined"] == "I(1)" and rw["c"] == "I(1)" and rw["ct"] == "I(1)"
    wn = combined_verdict(_white_noise(500, seed=1))
    assert wn["c"] == wn["ct"]            # dual regression consistent
    assert wn["combined"] != "I(1)"       # stationary noise must not read I(1)


def test_pairwise_gate_skips_stationary_anchor():
    from scripts.gold_anchor_analysis import should_run_johansen
    gold = combined_verdict(_random_walk(400, seed=9))      # I(1)
    anchor = combined_verdict(_white_noise(400, seed=10))   # I(0)
    assert should_run_johansen(gold["combined"], anchor["combined"]) is False


def test_select_var_order_and_robustness():
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    a = np.cumsum(np.random.default_rng(11).standard_normal(n))
    g = a + np.random.default_rng(22).standard_normal(n) * 0.3  # cointegrated, β≈1
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_anchor": a}, index=idx)
    sel = select_var_order(df, ["ln_gold_nominal", "ln_anchor"], max_lags=4)
    # k_ar_diff is lagged *differences*: VAR(p) -> p-1, so VAR(1) -> 0
    assert sel["var_order"] >= 1
    assert sel["k_ar_diff"] == max(0, sel["var_order"] - 1)
    rob = johansen_robustness(df, ["ln_gold_nominal", "ln_anchor"])
    assert len(rob) == 12  # 4 lags × 3 det_orders
    # genuine cointegration → rank>=1 dominates the grid
    assert (rob["coint_rank"].dropna().astype(int) >= 1).mean() >= 0.5

    # independent walks → rank 0 dominates (a stray spurious cell is allowed;
    # the grid exists precisely to expose that as instability, not a true anchor)
    g2 = np.cumsum(np.random.default_rng(101).standard_normal(n))
    a2 = np.cumsum(np.random.default_rng(202).standard_normal(n))
    df2 = pd.DataFrame({"ln_gold_nominal": g2, "ln_anchor": a2}, index=idx)
    rob2 = johansen_robustness(df2, ["ln_gold_nominal", "ln_anchor"])
    vals2 = rob2["coint_rank"].dropna().astype(int)
    assert (vals2 == 0).mean() >= 0.5


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


def test_to_monthly_ffill_extends_to_quarter_end():
    from lib.gold_anchor import _to_monthly
    # quarterly obs dated at quarter START (FRED convention); the Q4 value at
    # 2025-10-01 must fill Oct/Nov/Dec, not just Oct.
    idx = pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"])
    m = _to_monthly(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx), "ffill")
    assert pd.Timestamp("2025-12-31") in m.index
    assert m.loc["2025-11-30"] == 4.0 and m.loc["2025-12-31"] == 4.0


def test_empty_window_returns_empty_panel_without_crashing():
    # window entirely after the data → empty panel (script guards on .empty)
    panel = build_anchor_panel(
        start="2030-01-01", end="2031-01-01",
        fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold,
    )
    assert panel.data.empty


def test_end_month_boundary_includes_month_end():
    # end="2025-12" must include 2025-12-31, not exclude the whole month
    panel = build_anchor_panel(
        start="1968-01-01", end="2025-12",
        fetch_fn=_synthetic_fred, gold_fetch_fn=_synthetic_gold,
    )
    assert pd.Timestamp("2025-12-31") in panel.data.index
    assert panel.data.index.max() <= pd.Timestamp("2025-12-31")


# ── PR #2: trivariate anchor [ln gold, ln(debt/GDP), real rate] ─────────
def _trivariate_cointegrated(n=500, b1=1.2, b2=-0.5, seed=31):
    """gold = b1·debt + b2·real + stationary AR(1) error, with debt and real
    both random walks (I(1)) so the long-run β2 on the real rate is identified.
    Returns a DataFrame [ln_gold_nominal, ln_debt_gdp, real_rate_10y]."""
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    debt = np.cumsum(rng.standard_normal(n)) * 0.4 + 0.05 * np.arange(n)
    real = np.cumsum(rng.standard_normal(n)) * 0.2
    err = np.zeros(n)
    for t in range(1, n):  # mean-reverting equilibrium error → cointegration
        err[t] = 0.55 * err[t - 1] + rng.standard_normal() * 0.25
    gold = b1 * debt + b2 * real + err
    return pd.DataFrame(
        {"ln_gold_nominal": gold, "ln_debt_gdp": debt, "real_rate_10y": real},
        index=idx,
    )


def test_johansen_trivariate_reports_two_betas():
    df = _trivariate_cointegrated()
    cols = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]
    j = johansen_test(df, cols, k_ar_diff=1)
    assert j["coint_rank"] == 1              # unique vector → β interpretable
    # betas carries (β_debt, β_real) for the 3-var system
    assert j["betas"] is not None and len(j["betas"]) == 2
    assert abs(j["betas"][0] - 1.2) < 0.4    # β_debt ≈ 1.2 (true)
    assert abs(j["betas"][1] - (-0.5)) < 0.4  # β_real ≈ -0.5 (true) — the PR's core add
    assert j["betas"][0] == j["beta"]        # β alias = first anchor coef
    # robustness grid carries the betas list per cell
    rob = johansen_robustness(df, cols)
    assert "betas" in rob.columns
    good = [b for b in rob["betas"].tolist() if b is not None]
    assert good and all(len(b) == 2 for b in good)


def test_johansen_trivariate_betas_contract():
    # structural invariant (version-stable, NOT a brittle exact-rank assertion):
    # betas is populated iff a valid cointegrating relation exists, and then has
    # exactly n-1 entries. Independent random walks should *usually* give rank 0,
    # but we assert the contract, not the precise LAPACK/statsmodels outcome.
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {
            "ln_gold_nominal": np.cumsum(np.random.default_rng(1).standard_normal(n)),
            "ln_debt_gdp": np.cumsum(np.random.default_rng(2).standard_normal(n)),
            "real_rate_10y": np.cumsum(np.random.default_rng(3).standard_normal(n)),
        },
        index=idx,
    )
    j = johansen_test(df, list(df.columns), k_ar_diff=1)
    assert (j["betas"] is None) == (not j["valid_coint"])  # invariant
    if j["betas"] is not None:
        assert len(j["betas"]) == 2


def test_johansen_trivariate_rank_gt1_no_single_beta():
    # one common stochastic trend shared by all three I(1) series → TWO
    # independent cointegrating vectors → coint_rank=2. The single eigenvector is
    # then a non-unique basis vector, so β/βs must be None (not a misleading number).
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    f = np.cumsum(np.random.default_rng(40).standard_normal(n))  # common trend
    rng = np.random.default_rng(41)
    g = f + rng.standard_normal(n) * 0.3
    d = 2.0 * f + rng.standard_normal(n) * 0.3
    r = 0.5 * f + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"ln_gold_nominal": g, "ln_debt_gdp": d, "real_rate_10y": r}, index=idx)
    j = johansen_test(df, list(df.columns), k_ar_diff=1)
    # contract holds for ANY rank outcome: β/βs populated iff coint_rank == 1
    assert (j["betas"] is not None) == (j["coint_rank"] == 1)
    assert (j["beta"] is not None) == (j["coint_rank"] == 1)
    if j["coint_rank"] > 1:
        assert j["beta"] is None and j["betas"] is None


def test_integration_segments_splits_real_rate():
    # build a series: I(0) white noise pre-2003, I(1) random walk post-2003.
    idx = pd.date_range("1990-01-31", periods=480, freq="ME")
    pre_mask = idx < pd.Timestamp("2003-01-01")
    vals = np.empty(len(idx))
    vals[pre_mask] = np.random.default_rng(7).standard_normal(pre_mask.sum())
    rw = np.cumsum(np.random.default_rng(8).standard_normal((~pre_mask).sum()))
    vals[~pre_mask] = rw
    s = pd.Series(vals, index=idx)
    segs = integration_segments(
        s, {"full": (None, None), "post_2003": ("2003-01-01", None)}
    )
    assert set(segs) == {"full", "post_2003"}
    for name in segs:
        assert segs[name]["combined"] in {"I(0)", "I(1)", "ambiguous"}
        assert segs[name]["c"] is not None and segs[name]["ct"] is not None
    assert segs["post_2003"]["n"] < segs["full"]["n"]
    assert segs["post_2003"]["start"] >= "2003-01-01"


def test_estimate_vecm_recovers_long_and_short_run():
    df = _trivariate_cointegrated(b1=1.2, b2=-0.5)
    cols = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]
    v = estimate_vecm(df, cols, k_ar_diff=2, coint_rank=1, det_order=0)
    # long-run β2 (real rate) recovered with the right sign
    assert set(v["betas"]) == {"ln_debt_gdp", "real_rate_10y"}
    assert abs(v["betas"]["ln_debt_gdp"]["beta"] - 1.2) < 0.5
    assert v["betas"]["real_rate_10y"]["beta"] < 0      # b2 = -0.5 < 0
    # error-correction speed λ < 0 and significant (constructed mean reversion)
    assert v["ec_speed"]["lambda"] < 0
    assert v["ec_speed"]["corrects"] is True
    # short-run block exists and includes a Δreal_rate term in the gold eq
    assert len(v["short_run"]) == 2 * len(cols)  # 2 lags × 3 vars
    rr_terms = [t for t in v["short_run"] if t["var"] == "real_rate_10y"]
    assert len(rr_terms) == 2
    assert v["beta_normalized"][0] == 1.0


def test_johansen_beta_requires_valid_and_unique_coint():
    # β/βs must be None unless there is a VALID (not full-rank-stationary) AND
    # UNIQUE (rank==1) cointegrating relation. Two stationary white-noise series
    # → full_rank_stationary → valid_coint False → β None even if a raw rank
    # would otherwise be ≥1.
    n = 500
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {"ln_gold_nominal": np.random.default_rng(5).standard_normal(n),
         "ln_debt_gdp": np.random.default_rng(6).standard_normal(n)},
        index=idx,
    )
    j = johansen_test(df, list(df.columns))
    assert j["valid_coint"] is False
    assert j["beta"] is None and j["betas"] is None
    # the contract, stated directly:
    assert (j["beta"] is None) == (not (j["valid_coint"] and j["coint_rank"] == 1))


def test_estimate_vecm_validates_inputs():
    df = _trivariate_cointegrated()
    cols = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]
    with pytest.raises(ValueError):
        estimate_vecm(df, cols, k_ar_diff=0)      # no short-run block
    with pytest.raises(ValueError):
        estimate_vecm(df, ["ln_gold_nominal"], k_ar_diff=1)  # <2 cols
    with pytest.raises(ValueError):
        estimate_vecm(df.iloc[:20], cols, k_ar_diff=1)       # too few rows
    with pytest.raises(ValueError):
        estimate_vecm(df, cols, k_ar_diff=1, coint_rank=2)   # rank>1 → no unique β


def test_estimate_vecm_alpha_threshold_propagates():
    # significance uses the caller's alpha, and alpha_level is echoed back.
    df = _trivariate_cointegrated()
    cols = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]
    v = estimate_vecm(df, cols, k_ar_diff=2, coint_rank=1, alpha=0.01)
    assert v["alpha_level"] == 0.01
    for b in v["betas"].values():
        assert b["significant"] == (b["p"] < 0.01)
    assert v["ec_speed"]["significant"] == (v["ec_speed"]["p"] < 0.01)


def test_estimate_vecm_det_orders_do_not_crash():
    # every det_order in the Johansen grid maps to a VALID statsmodels VECM
    # deterministic string (no "nc" crash). det_order=-1 ("n") in particular.
    df = _trivariate_cointegrated()
    cols = ["ln_gold_nominal", "ln_debt_gdp", "real_rate_10y"]
    for det in (-1, 0, 1):
        v = estimate_vecm(df, cols, k_ar_diff=1, coint_rank=1, det_order=det)
        assert set(v["betas"]) == {"ln_debt_gdp", "real_rate_10y"}
        assert v["det_order"] == det
        # negated-β t-stat sign matches the reported (negated) β
        for b in v["betas"].values():
            assert (b["t"] >= 0) == (b["beta"] >= 0) or b["t"] == 0.0
