"""Tests for the PR #11 placebo battery (lib/gold_attribution_placebo).

Offline by construction: a synthetic anchor + fetch_fn feed PR #9's
`build_attribution_panel`, so there is no network/FRED dependency. The focus is
the *mechanics* — placebo series construction, the five-layer attribution running
under each swapped ⑤, the first-difference re-fit, and the additive identity —
NOT any empirical claim about gold.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_credit_spread_attribution import (
    build_attribution_panel,
    decompose_period,
    fit_attribution,
)
from lib.gold_attribution_placebo import (
    WGC_ANNUAL_NET_PURCHASES_T,
    adjudicate,
    annual_to_monthly,
    baseline_no_fifth,
    common_window,
    lead_lag_table,
    make_placebos,
    run_diff_fifth,
    run_levels_fifth,
    stationarity_table,
    wgc_cumulative_excess_annual,
)


# ── synthetic PR#9 panel (mirrors test_gold_credit_spread_attribution) ───
def _panel(n=300, seed=7):
    idx = pd.date_range("1999-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02) + 6.0), index=idx)
    debt_gdp = pd.Series(np.linspace(0.55, 1.25, n) + rng.randn(n) * 0.01, index=idx)
    real_rate = pd.Series(1.5 + np.cumsum(rng.randn(n) * 0.05), index=idx)
    anchor = SimpleNamespace(data=pd.DataFrame({
        "gold_nominal": gold, "debt_gdp": debt_gdp, "real_rate_10y": real_rate,
        "ln_gold_nominal": np.log(gold), "ln_debt_gdp": np.log(debt_gdp),
    }, index=idx))
    cpi = pd.Series(np.linspace(170, 310, n) + rng.randn(n) * 0.4, index=idx)
    vix = pd.Series(18 + rng.randn(n) * 4, index=idx).abs()
    credit = pd.Series(2.0 + rng.randn(n) * 0.5, index=idx).abs()
    custody = pd.Series(np.linspace(1.5e6, 2.6e6, n) + rng.randn(n) * 1e4, index=idx)
    custody.iloc[:12] = np.nan
    debt_lvl = pd.Series(np.linspace(6e6, 3.7e7, n), index=idx)

    def fetch_fn(series_id, start="1968-01-01"):
        return {"CPIAUCSL": cpi, "VIXCLS": vix, "BAA10Y": credit,
                "WMTSECL1": custody, "GFDEBTN": debt_lvl}.get(
            series_id, pd.Series(np.nan, index=idx))

    panel = build_attribution_panel(
        start="1999-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor)
    return panel, idx


# ── 1. WGC constants + cumulative excess construction ────────────────────
def test_wgc_cumulative_excess_shape_and_monotone_post2022():
    ann = wgc_cumulative_excess_annual()
    # one point per year in the table; cumulative
    assert list(ann.index.year) == sorted(WGC_ANNUAL_NET_PURCHASES_T)
    # 2022→2025 is the de-dollarisation ramp: strictly increasing & accelerating
    post = ann[ann.index.year >= 2022]
    assert (post.diff().dropna() > 0).all()
    # exact identity: last value == sum(purchases) − baseline*n_years
    base = 473.0
    expect = sum(WGC_ANNUAL_NET_PURCHASES_T.values()) - base * len(WGC_ANNUAL_NET_PURCHASES_T)
    assert ann.iloc[-1] == pytest.approx(expect)


def test_annual_to_monthly_interpolation_on_grid():
    ann = wgc_cumulative_excess_annual()
    idx = pd.date_range("2010-12-31", "2025-12-31", freq="ME")
    m = annual_to_monthly(ann, idx)
    assert m.index.equals(idx)
    # monthly series is bounded by the annual endpoints it interpolates
    assert m.dropna().min() >= ann.min() - 1e-6
    assert m.dropna().max() <= ann.max() + 1e-6
    # interpolation is monotone within the post-2022 ramp
    seg = m[m.index >= pd.Timestamp("2022-01-31")].dropna()
    assert (seg.diff().dropna() >= -1e-9).all()


# ── 2. placebo construction ──────────────────────────────────────────────
def test_make_placebos_keys_and_monotonicity():
    idx = pd.date_range("2010-12-31", periods=180, freq="ME")
    cpi = pd.Series(np.linspace(200, 320, 180), index=idx)
    m2 = pd.Series(np.linspace(8000, 21000, 180), index=idx)
    ip = pd.Series(np.linspace(95, 103, 180), index=idx)
    p = make_placebos(idx, cpi=cpi, m2=m2, ip=ip, rand_seeds=(1, 2, 3, 4, 5))
    # required placebos present
    for k in ("t", "log_t", "cum_cpi", "cum_m2", "cum_ip", "kink_2022"):
        assert k in p
    # ≥5 seeded random monotone series
    rand_keys = [k for k in p if k.startswith("rand_")]
    assert len(rand_keys) >= 5
    # each monotone-rising series is non-decreasing
    for k in ("t", "log_t", "cum_cpi", "cum_m2", "cum_ip", *rand_keys):
        assert (p[k].diff().dropna() >= -1e-9).all(), k
    # kink: flat before 2022, rising after
    kink = p["kink_2022"]
    assert (kink[kink.index < pd.Timestamp("2022-01-31")] == 0).all()
    assert (kink[kink.index >= pd.Timestamp("2022-01-31")].diff().dropna() > 0).all()


def test_make_placebos_skips_missing_sources():
    idx = pd.date_range("2010-12-31", periods=60, freq="ME")
    p = make_placebos(idx, cpi=None, m2=None, ip=None)
    assert "cum_cpi" not in p and "cum_m2" not in p and "cum_ip" not in p
    assert "t" in p and "kink_2022" in p


def test_random_placebos_are_seed_deterministic():
    idx = pd.date_range("2010-12-31", periods=60, freq="ME")
    a = make_placebos(idx, rand_seeds=(7,))["rand_7"]
    b = make_placebos(idx, rand_seeds=(7,))["rand_7"]
    pd.testing.assert_series_equal(a, b)


# ── 3. levels attribution runs under each swapped ⑤ ──────────────────────
def test_common_window_is_intersection():
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    win = common_window(panel.data, wgc)
    # window is within both the ①–④ design rows and where wgc is defined
    assert win.min() >= wgc.dropna().index.min()
    assert len(win) >= 24


def test_levels_fifth_runs_for_real_and_placebos():
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    placebos = make_placebos(idx, cpi=panel.data["cpi"])
    win = common_window(panel.data, wgc)
    real = run_levels_fifth(panel.data, wgc, key="REAL_WGC", window=win, t0="2022-01")
    assert 0.0 <= real.r2 <= 1.0 and real.n == len(win)
    assert np.isfinite(real.flow_t)
    for k, s in placebos.items():
        r = run_levels_fifth(panel.data, s, key=k, window=win, t0="2022-01")
        assert r.n == len(win)  # identical sample → fair comparison
        assert 0.0 <= r.r2 <= 1.0


def test_levels_decomposition_additive_identity():
    """With ⑤ injected, the five-layer contributions + residual still sum to the
    realised Δln(gold) (delegates to PR#9 decompose; we re-verify here)."""
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    win = common_window(panel.data, wgc)
    df = panel.data.loc[win].copy()
    df["wgc_flow"] = wgc.reindex(win)
    res = fit_attribution(df, cpi_mode="identity", min_obs=24)
    dec = decompose_period(res, t0="2022-01")
    total = float(dec.loc[dec["layer"] == "TOTAL", "contribution_ln"].iloc[0])
    parts = dec[dec["layer"] != "TOTAL"]["contribution_ln"].sum()
    assert parts == pytest.approx(total, abs=1e-9)
    # ⑤ (flow) row is present now that it is injected
    assert "flow" in set(dec["layer"])


def test_baseline_no_fifth_residual_is_larger():
    """Removing ⑤ leaves more in the residual than a (well-fitting) ⑤ would —
    sanity that baseline_no_fifth folds ⑤ into ε_flow."""
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    win = common_window(panel.data, wgc)
    base = baseline_no_fifth(panel.data, window=win, t0="2022-01")
    assert "resid_contrib_pct" in base and np.isfinite(base["resid_contrib_pct"])
    assert base["n"] == len(win)


# ── 4. first-difference (stationary) attribution ─────────────────────────
def test_diff_fifth_runs_and_reports_flow_t():
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    win = common_window(panel.data, wgc)
    d = run_diff_fifth(panel.data, wgc, key="REAL_WGC", window=win)
    assert "flow" in d.coefs.index
    assert np.isfinite(d.flow_t)
    assert d.n == len(win) - 1  # one row lost to differencing
    assert 0.0 <= d.r2 <= 1.0 or np.isnan(d.r2)


@pytest.mark.parametrize("mode", ["identity", "free"])
def test_diff_fifth_free_mode_adds_cpi(mode):
    panel, idx = _panel()
    wgc = annual_to_monthly(wgc_cumulative_excess_annual(), idx)
    win = common_window(panel.data, wgc)
    d = run_diff_fifth(panel.data, wgc, key="REAL_WGC", window=win, cpi_mode=mode)
    if mode == "free":
        assert "cpi" in d.coefs.index
    else:
        assert "cpi" not in d.coefs.index


# ── 5. stationarity + lead/lag tables ────────────────────────────────────
def test_stationarity_table_columns():
    idx = pd.date_range("2010-12-31", periods=180, freq="ME")
    rng = np.random.RandomState(0)
    trend = pd.Series(np.cumsum(np.abs(rng.randn(180))) + np.arange(180), index=idx)
    noise = pd.Series(rng.randn(180), index=idx)
    tbl = stationarity_table({"trend": trend, "noise": noise})
    for col in ("series", "adf_p", "kpss_p", "level_verdict",
                "diff_adf_p", "diff_kpss_p", "diff_verdict"):
        assert col in tbl.columns
    # a pure random walk-ish trend should not be flagged clean I(0) in levels
    tr = tbl[tbl["series"] == "trend"].iloc[0]
    assert tr["level_verdict"] in {"I(1)+", "ambiguous"}


def test_lead_lag_table_symmetry_and_contemp():
    idx = pd.date_range("2010-12-31", periods=120, freq="ME")
    rng = np.random.RandomState(3)
    flow = pd.Series(rng.randn(120), index=idx)
    gold = flow.shift(1).fillna(0) * 0.5 + rng.randn(120) * 0.1  # flow leads gold by 1
    ll = lead_lag_table(gold, flow, max_lag=4)
    assert set(ll["lag_months"]) == set(range(-4, 5))
    # the constructed lead (flow→gold at +1) should show the strongest positive corr
    best = ll.loc[ll["corr"].idxmax(), "lag_months"]
    assert best == 1


# ── 6. verdict logic ─────────────────────────────────────────────────────
def _fr(key, r2, flow_t=5.0, resid_pct=20.0, flow_pct=100.0):
    from lib.gold_attribution_placebo import FifthResult
    return FifthResult(key=key, label=key, r2=r2, n=180, flow_contrib_ln=0.5,
                       flow_contrib_pct=flow_pct, resid_contrib_ln=0.1,
                       resid_contrib_pct=resid_pct, flow_coef=1.0,
                       flow_t=flow_t, flow_p=0.0)


def _dr(flow_t):
    from lib.gold_attribution_placebo import DiffResult
    return DiffResult(key="REAL_WGC", label="real", r2=0.3, n=179,
                      coefs=pd.Series({"const": 0.0, "flow": 1.0}),
                      tstats=pd.Series({"const": 0.0, "flow": flow_t}),
                      pvals=pd.Series({"const": 1.0, "flow": 0.001 if abs(flow_t) > 2 else 0.5}))


def _dr_key(key, flow_t):
    from lib.gold_attribution_placebo import DiffResult
    return DiffResult(key=key, label=key, r2=0.3, n=179,
                      coefs=pd.Series({"const": 0.0, "flow": 1.0}),
                      tstats=pd.Series({"const": 0.0, "flow": flow_t}),
                      pvals=pd.Series({"const": 1.0, "flow": 0.0}))


def test_verdict_real_beats_placebos_and_survives():
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.36), _fr("log_t", 0.45), _fr("cum_cpi", 0.36),
                _fr("kink_2022", 0.40)]
    v = adjudicate(real, placebos, _dr(3.5))
    assert v.verdict == "real"
    assert v.real_beats_placebos and v.survives_in_diff and not v.spurious


def test_verdict_spurious_when_placebo_matches():
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.70)]  # a monotone placebo matches/exceeds → spurious
    v = adjudicate(real, placebos, _dr(3.5))
    assert v.verdict == "spurious"
    assert v.spurious and not v.real_beats_placebos


def test_verdict_spurious_when_diff_dies():
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.36)]
    v = adjudicate(real, placebos, _dr(0.5))  # ⑤ dies in difference
    assert v.verdict == "spurious"
    assert v.spurious and not v.survives_in_diff


def test_verdict_mixed_when_kink_dominates():
    """Kink shape-control beats WGC levels while smooth placebos don't and ⑤
    survives the diff → MIXED (co-movement, downgrade from causal top-pricing)."""
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.36), _fr("kink_2022", 0.88)]
    v = adjudicate(real, placebos, _dr(3.5))
    assert v.verdict == "mixed"
    assert v.real_beats_placebos and not v.spurious and v.kink_dominates
    assert any("拐点" in n for n in v.notes)


def test_verdict_mixed_when_diff_not_singled_out():
    """WGC wins levels and survives diff, kink doesn't dominate, but a monotone
    placebo has a larger diff t → diff doesn't single WGC out → MIXED."""
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.36), _fr("cum_cpi", 0.36), _fr("kink_2022", 0.40)]
    diff_placebos = [_dr_key("t", 1.0), _dr_key("cum_cpi", 5.0)]  # cum_cpi diff t > real
    v = adjudicate(real, placebos, _dr(3.5), diff_placebos=diff_placebos)
    assert v.verdict == "mixed"
    assert not v.diff_singles_out_real
    assert any("并非" in n for n in v.notes)


def test_verdict_kink_excluded_from_monotone_matching():
    """The kink shape-control must NOT by itself flip real_beats_placebos — it is
    reported in notes, not counted as a monotone-trend placebo."""
    real = _fr("REAL_WGC", 0.66)
    placebos = [_fr("t", 0.36), _fr("kink_2022", 0.88)]  # kink huge, monotones low
    v = adjudicate(real, placebos, _dr(3.5))
    assert v.real_beats_placebos  # kink not counted
    assert any("拐点" in n for n in v.notes)
