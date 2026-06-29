"""Tests for the PR #9 ex-post layered gold attribution.

Offline by construction: every test injects a synthetic `anchor_fn` + `fetch_fn`
(and optional `wgc_fn`), so there is no network / FRED-key dependency. The focus
is the *mechanics* of the decomposition, not any empirical claim:

  * five-layer proxy construction (correct signs, composites, coverage),
  * regression output shape (coef/tstat/std-coef per included layer),
  * the EXACT additive identity: Σ contributions + ε_flow residual == Δln(gold),
  * a missing-data layer (⑤ WGC absent / ④ component empty) is skipped cleanly,
  * cpi_mode='identity' pins layer ① to Δln(CPI).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.gold_credit_spread_attribution import (
    LAYERS,
    build_attribution_panel,
    build_design,
    decompose_period,
    fit_attribution,
    rolling_coefs,
    stacked_contribution_path,
    verdict,
)


# ── synthetic data ──────────────────────────────────────────────────────
def _anchor_and_fetch(n=300, seed=7, custody_gap=12, with_credit=True):
    """Synthetic anchor panel (gold/debt_gdp/real_rate + ln_*) and a fetch_fn
    that serves CPI / VIX / credit / custody / debt. `custody_gap` leaves leading
    NaNs in WMTSECL1 (the pre-2003 analogue). `with_credit=False` returns an empty
    credit series (to exercise the layer-skip path)."""
    idx = pd.date_range("1999-01-31", periods=n, freq="ME")
    rng = np.random.RandomState(seed)
    gold = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02) + 6.0), index=idx)
    debt_gdp = pd.Series(np.linspace(0.55, 1.25, n) + rng.randn(n) * 0.01, index=idx)
    real_rate = pd.Series(1.5 + np.cumsum(rng.randn(n) * 0.05), index=idx)
    anchor = SimpleNamespace(data=pd.DataFrame({
        "gold_nominal": gold,
        "debt_gdp": debt_gdp,
        "real_rate_10y": real_rate,
        "ln_gold_nominal": np.log(gold),
        "ln_debt_gdp": np.log(debt_gdp),
    }, index=idx))

    cpi = pd.Series(np.linspace(170, 310, n) + rng.randn(n) * 0.4, index=idx)
    vix = pd.Series(18 + rng.randn(n) * 4, index=idx).abs()
    credit = pd.Series(2.0 + rng.randn(n) * 0.5, index=idx).abs()
    if not with_credit:
        credit = pd.Series(np.nan, index=idx)
    custody = pd.Series(np.linspace(1.5e6, 2.6e6, n) + rng.randn(n) * 1e4, index=idx)
    if custody_gap > 0:
        custody.iloc[:custody_gap] = np.nan
    debt_lvl = pd.Series(np.linspace(6e6, 3.7e7, n), index=idx)

    def fetch_fn(series_id, start="1968-01-01"):
        return {
            "CPIAUCSL": cpi, "VIXCLS": vix, "BAA10Y": credit,
            "WMTSECL1": custody, "GFDEBTN": debt_lvl,
        }.get(series_id, pd.Series(np.nan, index=idx))

    return anchor, fetch_fn, idx


def _panel(**kw):
    anchor, fetch_fn, idx = _anchor_and_fetch(**kw)
    return build_attribution_panel(
        start="1999-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    ), idx


# ── 1. panel / proxy construction ───────────────────────────────────────
def test_panel_columns_and_signs():
    panel, idx = _panel()
    df = panel.data
    for col in ("ln_gold", "ln_cpi", "neg_real_rate", "ln_debt_gdp",
                "neg_custody_share", "vix", "credit_spread", "custody_share"):
        assert col in df.columns
    # sign flips: neg_real_rate == -real_rate_10y; neg_custody_share == -share
    np.testing.assert_allclose(df["neg_real_rate"].dropna(),
                               -df["real_rate_10y"].dropna())
    np.testing.assert_allclose(df["neg_custody_share"].dropna(),
                               -df["custody_share"].dropna())
    # custody leading gap survives (not silently filled)
    assert df["custody_share"].iloc[:12].isna().all()
    # layer ⑤ flow unavailable by default → note documents the fold-into-residual
    assert df["wgc_flow"].isna().all()
    assert "folded into the ε_flow residual" in panel.notes["wgc_flow"]
    assert "ex_post_boundary" in panel.notes


def test_design_matrix_layers_and_zscore():
    panel, _ = _panel()
    design = build_design(panel.data)
    # ⑤ flow dropped (optional, no data); ①②③④ retained
    keys = [l.key for l in design.layers]
    assert keys == ["cpi", "real", "sov", "tail"]
    assert "flow" in design.skipped
    # multi-component composites are ~standardized (mean≈0)
    assert abs(float(design.X["sov"].mean())) < 1e-6
    assert abs(float(design.X["tail"].mean())) < 1e-6
    # single-component layers pass through in natural units
    np.testing.assert_allclose(design.X["real"], design.ln_gold.index.map(
        lambda t: panel.data["neg_real_rate"].loc[t]).astype(float), rtol=0, atol=1e-9)


# ── 2. regression shape ─────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["identity", "free"])
def test_fit_shape(mode):
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode=mode)
    # every included non-const layer has a std-coef; identity pins cpi coef to 1
    for k in ("real", "sov", "tail"):
        assert k in res.std_coefs.index
        assert np.isfinite(res.tstats[k])
    if mode == "identity":
        assert res.coefs["cpi"] == pytest.approx(1.0)
        assert "cpi" not in res.tstats.index   # cpi not a regressor in identity
    else:
        assert "cpi" in res.coefs.index
    assert 0.0 <= res.r2 <= 1.0
    assert res.n > 100
    assert np.isfinite(res.cond_number)


# ── 3. EXACT decomposition identity (核心: 加总=总涨幅) ───────────────────
@pytest.mark.parametrize("mode", ["identity", "free"])
def test_decomposition_sums_to_total(mode):
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode=mode)
    decomp = decompose_period(res, t0="2005-01")
    total = float(decomp.loc[decomp["layer"] == "TOTAL", "contribution_ln"].iloc[0])
    parts = decomp[~decomp["layer"].isin(["TOTAL"])]["contribution_ln"].sum()
    assert parts == pytest.approx(total, abs=1e-9)
    # percentages of the contributing rows (excl TOTAL) sum to ~100
    pct = decomp[decomp["layer"] != "TOTAL"]["contribution_pct_of_total"].sum()
    assert pct == pytest.approx(100.0, abs=1e-6)


def test_identity_layer1_is_delta_ln_cpi():
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode="identity")
    decomp = decompose_period(res, t0="2005-01")
    d = res.design
    d0 = d.ln_cpi.index[d.ln_cpi.index >= pd.Timestamp("2005-01-01")][0]
    d1 = d.ln_cpi.index[-1]
    expected = float(d.ln_cpi.loc[d1] - d.ln_cpi.loc[d0])
    got = float(decomp.loc[decomp["layer"] == "cpi", "contribution_ln"].iloc[0])
    assert got == pytest.approx(expected, abs=1e-9)


def test_stacked_path_sums_to_cumulative_dln_gold():
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode="identity")
    path = stacked_contribution_path(res, t0="2005-01")
    layer_keys = [l.key for l in res.design.layers] + ["flow_resid"]
    recon = path[layer_keys].sum(axis=1)
    np.testing.assert_allclose(recon.to_numpy(), path["total_dln_gold"].to_numpy(),
                               atol=1e-9)
    # first row is the baseline (cumulative move == 0)
    assert path["total_dln_gold"].iloc[0] == pytest.approx(0.0, abs=1e-9)


# ── 4. missing-data layers skipped cleanly ───────────────────────────────
def test_missing_required_layer_raises_by_default():
    """④ tail (required) with no credit data → build_design / fit raise rather
    than silently producing a 4-layer 'attribution' (codex P2)."""
    panel, _ = _panel(with_credit=False)
    with pytest.raises(ValueError, match="required layer 'tail'"):
        build_design(panel.data)
    with pytest.raises(ValueError, match="required layer 'tail'"):
        fit_attribution(panel, cpi_mode="identity")


def test_missing_required_layer_degrades_when_allowed():
    """allow_missing_required=True drops ④, flags it incomplete, and still keeps
    the decomposition exact over the reduced layer set."""
    panel, _ = _panel(with_credit=False)
    design = build_design(panel.data, allow_missing_required=True)
    keys = [l.key for l in design.layers]
    assert "tail" in design.skipped and keys == ["cpi", "real", "sov"]
    assert design.incomplete == ["tail"]
    res = fit_attribution(panel, cpi_mode="identity", allow_missing_required=True)
    assert res.incomplete == ["tail"]
    decomp = decompose_period(res, t0="2005-01")
    total = float(decomp.loc[decomp["layer"] == "TOTAL", "contribution_ln"].iloc[0])
    parts = decomp[decomp["layer"] != "TOTAL"]["contribution_ln"].sum()
    assert parts == pytest.approx(total, abs=1e-9)


def test_no_overlapping_window_raises():
    """Layers each have data but in disjoint windows → empty dropna → ValueError
    (not a silent degenerate lstsq) (codex P2)."""
    anchor, fetch_fn, idx = _anchor_and_fetch(n=120)
    # blank out credit on the first half and VIX on the second half → after the
    # joint dropna across all required components, no rows survive.
    half = len(idx) // 2
    df = build_attribution_panel(
        start="1999-01-01", fetch_fn=fetch_fn, anchor_fn=lambda *a, **k: anchor
    ).data
    df.loc[df.index[:half], "credit_spread"] = np.nan
    df.loc[df.index[half:], "vix"] = np.nan
    with pytest.raises(ValueError, match="overlapping rows"):
        build_design(df)


def test_out_of_range_window_raises():
    """A decomposition window outside the sample raises rather than silently
    snapping to the opposite end (codex P2)."""
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode="identity")
    with pytest.raises(ValueError, match="after the last available row"):
        decompose_period(res, t0="2099-01")
    with pytest.raises(ValueError, match="before the first available row"):
        decompose_period(res, t0="2005-01", t1="1980-01")


def test_verdict_empty_body_uniform_schema():
    """verdict() on a CPI-only decomposition returns the full key schema (no
    KeyError downstream) (codex P2)."""
    decomp = pd.DataFrame([
        {"layer": "cpi", "label": "①", "coef": 1.0, "delta_proxy": 0.1,
         "contribution_ln": 0.1, "contribution_pct_of_total": 100.0},
        {"layer": "TOTAL", "label": "T", "coef": np.nan, "delta_proxy": np.nan,
         "contribution_ln": 0.1, "contribution_pct_of_total": 100.0},
    ])
    v = verdict(decomp)
    for k in ("sovereign_took_over", "top_layer", "top_label",
              "sov_contribution_ln", "ranking"):
        assert k in v
    assert v["ranking"] == [] and v["top_layer"] is None
    assert v["sovereign_took_over"] is False


def test_wgc_flow_layer_included_when_injected():
    anchor, fetch_fn, idx = _anchor_and_fetch()
    rng = np.random.RandomState(1)
    flow = pd.Series(np.linspace(50, 400, len(idx)) + rng.randn(len(idx)) * 5, index=idx)

    def wgc_fn(start, end=None):
        return flow

    panel = build_attribution_panel(
        start="1999-01-01", fetch_fn=fetch_fn,
        anchor_fn=lambda *a, **k: anchor, wgc_fn=wgc_fn,
    )
    assert panel.data["wgc_flow"].notna().any()
    design = build_design(panel.data)
    assert "flow" in [l.key for l in design.layers]
    res = fit_attribution(panel, cpi_mode="identity")
    decomp = decompose_period(res, t0="2005-01")
    total = float(decomp.loc[decomp["layer"] == "TOTAL", "contribution_ln"].iloc[0])
    parts = decomp[decomp["layer"] != "TOTAL"]["contribution_ln"].sum()
    assert parts == pytest.approx(total, abs=1e-9)


# ── 5. rolling coefficients + verdict plumbing ───────────────────────────
def test_rolling_coefs_shape():
    panel, _ = _panel()
    rc = rolling_coefs(panel, window=60, cpi_mode="identity")
    assert set(rc.columns) == {"real", "sov", "tail"}
    assert len(rc) > 0 and rc.notna().any().any()


def test_verdict_keys():
    panel, _ = _panel()
    res = fit_attribution(panel, cpi_mode="identity")
    decomp = decompose_period(res, t0="2005-01")
    v = verdict(decomp)
    assert set(["sovereign_took_over", "top_layer", "ranking"]).issubset(v.keys())
    assert isinstance(v["sovereign_took_over"], bool)


def test_invalid_cpi_mode_raises():
    panel, _ = _panel()
    with pytest.raises(ValueError):
        fit_attribution(panel, cpi_mode="bogus")
    # rolling_coefs validates cpi_mode too (codex P3)
    with pytest.raises(ValueError):
        rolling_coefs(panel, cpi_mode="bogus")


def test_dependency_import_smoke():
    """The module depends on lib.gold_anchor (PR#1, already on main). Assert the
    reused symbols import cleanly so a missing base-repo file fails loudly here
    rather than at script runtime (codex P1 — gold_anchor is a base-repo file,
    not part of this PR's diff)."""
    from lib.gold_anchor import build_anchor_panel, fetch_fred_series
    assert callable(build_anchor_panel) and callable(fetch_fred_series)


def test_missing_cpi_always_raises_even_when_degraded():
    """① CPI is structurally non-degradable: missing ln_cpi raises even with
    allow_missing_required=True (codex P2)."""
    panel, _ = _panel()
    df = panel.data.copy()
    df["ln_cpi"] = np.nan
    with pytest.raises(ValueError, match="ln_cpi"):
        build_design(df, allow_missing_required=True)
