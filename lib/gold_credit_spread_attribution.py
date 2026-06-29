"""Gold "fiat credit-spread" *ex-post* layered attribution (PR #9).

The thesis under test (NOT a forecast — see the boundary note below): gold's
2022→2026 surge is the market re-pricing a **sovereign-credit / de-dollarisation
regime** rather than the usual real-rate cycle. We test it by decomposing the
log gold price into five additive layers and asking which layer *accounts for*
the realised 2022-01→latest appreciation.

The log-linear identity (spec):

    ln(P_gold) = ln(P_CPI)              # ① inflation purchasing-power baseline
               + β·(−r_real)            # ② real-rate premium (DFII10)
               + γ·SovRisk              # ③ sovereign-credit spread
               + δ·TailRisk             # ④ tail-insurance spread
               + ε_flow                 # ⑤ flow disturbance (central-bank buying)

Why this is *ex-post* attribution, not an *ex-ante* model
---------------------------------------------------------
PR #1–#8 already established (and codex confirmed) that every gold↔macro anchor
relation is **regime-dependent and non-extrapolable** — there is no stable
cointegrating vector, the rolling betas wander, and out-of-sample the per-asset
"wins" reverse. So this module makes **no forecast, runs no trading backtest,
and never touches S1**. It does one thing: take the *realised* log return over a
chosen window and split it across the five layers using a full-sample OLS fit, to
*describe* which layer co-moved with gold over that window. The fit coefficients
are explicitly allowed to use the whole sample (this is explanation, not
prediction). We additionally report rolling coefficients precisely to *show* the
instability — i.e. to demonstrate why this decomposition must not be read as a
predictive model.

The exact-decomposition trick
-----------------------------
For any OLS fit  y_t = α + Σ_k b_k · X_{k,t} + e_t , the change between two dates
is an *exact* identity (α cancels):

    y_{t1} − y_{t0} = Σ_k b_k·(X_{k,t1} − X_{k,t0}) + (e_{t1} − e_{t0})

So layer-k's contribution to the realised move is simply  b_k·ΔX_k , and the
contributions plus the residual change sum to the total move with zero slack.
With `cpi_mode="identity"` we impose the purchasing-power identity (b_CPI ≡ 1) by
regressing the *real* log price  ln(gold) − ln(CPI)  on layers ②–⑤; then layer ①'s
contribution is exactly Δln(CPI) and the rest decompose the above-inflation move.
With `cpi_mode="free"` ln(CPI) is just another regressor.

Layer proxies (each a single monthly series; heterogeneous composites z-scored
over the fit sample so a single coefficient is meaningful):
  ① ln(CPI)                                  CPIAUCSL
  ② −real_rate_10y (DFII10 splice, PR #1)    [from build_anchor_panel]
  ③ mean[ z(ln(debt/GDP)), z(−custody_share) ]  debt/GDP (anchor) + WMTSECL1/GFDEBTN
  ④ mean[ z(VIX), z(credit_spread) ]          VIXCLS + BAA10Y (Moody's Baa−10y)
  ⑤ central-bank net purchases               WGC (injected; default unavailable)

Data honesty: the ICE BofA OAS series (BAMLH0A0HYM2 / BAMLC0A0CM) only carry
~3y of history on the public FRED mirror (licensing), so the credit-spread proxy
uses the public-domain Moody's Baa−10y spread (BAA10Y, 1990+). The WGC quarterly
central-bank flow series has no FRED feed; layer ⑤ is therefore *injected* via
`wgc_fn` and, when absent, is reported as "unavailable" and folded into ε_flow.

This module reuses `build_anchor_panel` (gold + debt/GDP + real_rate, PR #1) and
the `fetch_fred_series` seam; it adds CPI / VIX / credit-spread / custody pulls
and touches no PR #1–#8 code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from lib.gold_anchor import build_anchor_panel, fetch_fred_series

# ── FRED ids this module owns ──────────────────────────────────────────
DEFAULT_CPI_ID = "CPIAUCSL"          # CPI-U index, monthly 1949+
DEFAULT_VIX_ID = "VIXCLS"            # CBOE VIX close, daily 1990+
DEFAULT_CREDIT_ID = "BAA10Y"         # Moody's Baa − 10y Treasury, daily 1990+ (public)
DEFAULT_CUSTODY_ID = "WMTSECL1"      # Fed H.4.1 Treasuries in custody for foreign
                                     # official accounts, $M weekly 2002-12+
DEFAULT_DEBT_ID = "GFDEBTN"          # Federal debt total, $M quarterly (custody denom)

DEFAULT_ATTRIB_START = "2022-01"     # the regime-change window under test


@dataclass(frozen=True)
class Layer:
    """One attribution layer: a label plus the panel columns that compose it.

    `components` lists raw panel columns; a multi-component layer is the equal-
    weight mean of the z-scored components (z over the fit sample). `optional`
    layers are silently dropped (and folded into ε_flow) when their data is
    absent."""

    key: str
    label: str
    components: Sequence[str]
    optional: bool = False


# Ordered five-layer spec. `neg_real_rate` / `neg_custody_share` are sign-flipped
# in the panel so that, for EVERY component, a higher value = "more gold-bullish"
# (more inflation, lower real rates, more sovereign risk, more tail stress).
LAYERS: List[Layer] = [
    Layer("cpi", "① 通胀购买力基准 (CPI)", ["ln_cpi"]),
    Layer("real", "② 实利率升水 (−DFII10)", ["neg_real_rate"]),
    Layer("sov", "③ 主权信用利差 (debt/GDP + 外官托管)", ["ln_debt_gdp", "neg_custody_share"]),
    Layer("tail", "④ 尾部保险利差 (VIX + 信用利差)", ["vix", "credit_spread"]),
    Layer("flow", "⑤ 流量扰动 (央行净购金, WGC)", ["wgc_flow"], optional=True),
]


@dataclass
class AttributionPanel:
    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)
    coverage: Dict[str, str] = field(default_factory=dict)


def _to_monthly_mean(s: pd.Series) -> pd.Series:
    """Resample any FRED series to a month-end (ME) mean. Local helper for the
    pulls this module owns (mirrors the dispersion module rather than importing a
    private name)."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    return s.resample("ME").mean()


def _to_monthly_ffill(s: pd.Series, fill_period: str = "Q") -> pd.Series:
    """Forward-fill a lower-frequency stock to month-end within its native period
    (mirrors `gold_anchor._to_monthly(how='ffill')` for the debt denominator)."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    start = s.index.min().to_period("M").to_timestamp("M")
    end = s.index.max().to_period(fill_period).to_timestamp(how="end").normalize()
    end = end.to_period("M").to_timestamp("M")
    idx = pd.date_range(start, end, freq="ME")
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def build_attribution_panel(
    start: str = "1990-01-01",
    end: Optional[str] = None,
    *,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    anchor_fn: Callable[..., object] = build_anchor_panel,
    wgc_fn: Optional[Callable[[str, Optional[str]], pd.Series]] = None,
    cpi_id: str = DEFAULT_CPI_ID,
    vix_id: str = DEFAULT_VIX_ID,
    credit_id: str = DEFAULT_CREDIT_ID,
    custody_id: str = DEFAULT_CUSTODY_ID,
    debt_id: str = DEFAULT_DEBT_ID,
) -> AttributionPanel:
    """Assemble the monthly panel the five attribution layers read.

    Reuses `build_anchor_panel` for gold_nominal + debt_gdp + real_rate_10y
    (PR #1, not re-derived); adds CPI, VIX, credit spread, and the foreign-
    official custody share on the same month-end grid. `wgc_fn` (central-bank
    net purchases) is optional: when None, layer ⑤ is reported unavailable.

    Injection seams: `fetch_fn` covers all FRED pulls; `anchor_fn` is the panel
    builder (stub with an object exposing `.data`); `wgc_fn(start, end)` returns a
    dated flow Series."""
    base = anchor_fn(start=start, end=end, fetch_fn=fetch_fn).data  # type: ignore[attr-defined]
    df = base[["gold_nominal", "debt_gdp", "real_rate_10y", "ln_gold_nominal", "ln_debt_gdp"]].copy()
    df = df.rename(columns={"ln_gold_nominal": "ln_gold"})
    idx = df.index
    notes: Dict[str, str] = {}
    coverage: Dict[str, str] = {}

    # ① inflation baseline
    cpi = _to_monthly_mean(fetch_fn(cpi_id, start)).reindex(idx)
    df["cpi"] = cpi
    df["ln_cpi"] = np.log(df["cpi"].where(df["cpi"] > 0))

    # ② real-rate premium — sign-flip so higher = lower real rate = bullish
    df["neg_real_rate"] = -df["real_rate_10y"]

    # ③ sovereign credit: foreign-official custody share = WMTSECL1 / GFDEBTN
    #    (both $M; declining share = de-dollarisation = rising sovereign risk).
    custody = _to_monthly_mean(fetch_fn(custody_id, start)).reindex(idx)
    debt_lvl = _to_monthly_ffill(fetch_fn(debt_id, start)).reindex(idx)
    share = custody / debt_lvl.where(debt_lvl > 0)
    df["custody_share"] = share
    df["neg_custody_share"] = -share

    # ④ tail-insurance: VIX + investment-grade credit spread
    df["vix"] = _to_monthly_mean(fetch_fn(vix_id, start)).reindex(idx)
    df["credit_spread"] = _to_monthly_mean(fetch_fn(credit_id, start)).reindex(idx)

    # ⑤ flow: WGC central-bank net purchases (injected; quarterly→ME ffill)
    if wgc_fn is not None:
        try:
            flow = wgc_fn(start, end)
            df["wgc_flow"] = _to_monthly_ffill(pd.Series(flow)).reindex(idx)
        except Exception as e:  # pragma: no cover - defensive
            notes["wgc_error"] = f"{type(e).__name__}: {e}"
            df["wgc_flow"] = np.nan
    else:
        df["wgc_flow"] = np.nan

    if end is not None:
        df = df[df.index <= pd.Period(end, freq="M").to_timestamp("M")]

    # coverage per raw component
    def _cov(col: str) -> str:
        s = df[col].dropna()
        if s.empty:
            return "no observations (n=0)"
        return f"{s.index.min().date()}..{s.index.max().date()} (n={len(s)})"

    for col in ["gold_nominal", "ln_cpi", "neg_real_rate", "ln_debt_gdp",
                "neg_custody_share", "vix", "credit_spread", "wgc_flow"]:
        coverage[col] = _cov(col)

    notes["credit_spread_choice"] = (
        f"credit_spread = {credit_id} (Moody's Baa − 10y Treasury, public-domain, "
        "1990+). The ICE BofA OAS series (BAMLH0A0HYM2/BAMLC0A0CM) carry only ~3y "
        "on the public FRED mirror (licensing) — unusable for a 2022-baseline span."
    )
    notes["custody_share"] = (
        f"custody_share = {custody_id} / {debt_id} (foreign-official Treasuries in "
        "Fed custody ÷ total federal debt). WMTSECL1 starts 2002-12, so layer ③'s "
        "de-dollarisation component is a post-2003 proxy."
    )
    notes["wgc_flow"] = (
        "central-bank net gold purchases (WGC quarterly, 2010+) have no FRED feed; "
        "inject via wgc_fn. Absent here → layer ⑤ folded into the ε_flow residual."
        if wgc_fn is None else "WGC flow injected via wgc_fn."
    )
    notes["ex_post_boundary"] = (
        "EX-POST attribution only: the OLS fit uses the full sample to *describe* "
        "which layer co-moved with realised gold; it is NOT a forecast and runs no "
        "trading backtest (PR #1–#8 proved the anchors are regime-dependent / "
        "non-extrapolable). Rolling coefficients are reported to expose that "
        "instability, not to time the market."
    )
    return AttributionPanel(data=df, notes=notes, coverage=coverage)


# ── design-matrix construction ─────────────────────────────────────────
def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def available_layers(df: pd.DataFrame, layers: Sequence[Layer] = LAYERS) -> List[Layer]:
    """Layers whose every component column exists and has >=1 non-NaN value over
    the panel; optional layers with missing data are dropped."""
    out: List[Layer] = []
    for lyr in layers:
        ok = all(c in df.columns and df[c].notna().any() for c in lyr.components)
        if ok:
            out.append(lyr)
        elif not lyr.optional:
            # required layer with no data: keep it so the caller sees the gap,
            # but it will be dropped in build_design with a recorded skip note.
            out.append(lyr)
    return out


def build_design(
    df: pd.DataFrame,
    layers: Sequence[Layer] = LAYERS,
) -> "DesignMatrix":
    """Build the per-layer proxy matrix on the rows where ALL included layers'
    components are present. Single-component layers pass through in natural units;
    multi-component layers become the equal-weight mean of z-scored components
    (z over the retained sample). Layers with no data are skipped and recorded."""
    used: List[Layer] = []
    skipped: Dict[str, str] = {}
    needed_cols: List[str] = []
    for lyr in layers:
        missing = [c for c in lyr.components if c not in df.columns or df[c].notna().sum() == 0]
        if missing:
            skipped[lyr.key] = f"missing/empty components: {missing}"
            continue
        used.append(lyr)
        needed_cols.extend(lyr.components)

    # unique column set (ln_cpi is both a base col and the ① component)
    cols: List[str] = []
    for c in ["ln_gold", "ln_cpi"] + needed_cols:
        if c not in cols:
            cols.append(c)
    sub = df[cols].dropna()
    proxies: Dict[str, pd.Series] = {}
    for lyr in used:
        if len(lyr.components) == 1:
            proxies[lyr.key] = sub[lyr.components[0]].astype(float)
        else:
            comp = pd.concat([_zscore(sub[c]) for c in lyr.components], axis=1)
            proxies[lyr.key] = comp.mean(axis=1)
    X = pd.DataFrame(proxies, index=sub.index)
    return DesignMatrix(X=X, ln_gold=sub["ln_gold"], ln_cpi=sub["ln_cpi"],
                        layers=used, skipped=skipped)


@dataclass
class DesignMatrix:
    X: pd.DataFrame                 # one column per included layer (proxy series)
    ln_gold: pd.Series
    ln_cpi: pd.Series
    layers: List[Layer]
    skipped: Dict[str, str] = field(default_factory=dict)


# ── OLS fit ─────────────────────────────────────────────────────────────
@dataclass
class AttributionResult:
    coefs: pd.Series          # by layer key (+ 'const')
    std_coefs: pd.Series      # standardized betas (by layer key)
    tstats: pd.Series
    pvals: pd.Series
    r2: float
    n: int
    cond_number: float
    cpi_mode: str
    design: DesignMatrix
    fitted: pd.Series
    resid: pd.Series


def _ols(y: np.ndarray, X: np.ndarray):
    """Plain OLS with HC0-free classical t-stats. Returns (beta, tstat, pval, r2)."""
    from scipy import stats

    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    pval = 2.0 * stats.t.sf(np.abs(tstat), dof)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan
    return beta, tstat, pval, r2, resid


def fit_attribution(
    panel: AttributionPanel | pd.DataFrame,
    *,
    layers: Sequence[Layer] = LAYERS,
    cpi_mode: str = "identity",
) -> AttributionResult:
    """Full-sample OLS attribution fit.

    `cpi_mode="identity"`: impose the purchasing-power identity b_CPI≡1 by
    regressing (ln_gold − ln_cpi) on layers ②–⑤; layer ①'s contribution is then
    exactly Δln(CPI). `cpi_mode="free"`: regress ln_gold on ln_cpi + ②–⑤.
    """
    df = panel.data if isinstance(panel, AttributionPanel) else panel
    if cpi_mode not in ("identity", "free"):
        raise ValueError(f"cpi_mode must be 'identity' or 'free', got {cpi_mode!r}")
    design = build_design(df, layers)

    if cpi_mode == "identity":
        y = (design.ln_gold - design.ln_cpi).to_numpy()
        reg_layers = [l for l in design.layers if l.key != "cpi"]
    else:
        y = design.ln_gold.to_numpy()
        reg_layers = list(design.layers)

    Xcols = [design.X[l.key].to_numpy() for l in reg_layers]
    Xmat = np.column_stack([np.ones(len(y))] + Xcols) if Xcols else np.ones((len(y), 1))
    beta, tstat, pval, r2, resid = _ols(y, Xmat)

    names = ["const"] + [l.key for l in reg_layers]
    coefs = pd.Series(beta, index=names)
    tstats = pd.Series(tstat, index=names)
    pvals = pd.Series(pval, index=names)
    # standardized betas: b_k * sd(X_k)/sd(y)
    sd_y = float(np.std(y, ddof=0)) or 1.0
    std_vals = {}
    for l in reg_layers:
        std_vals[l.key] = float(coefs[l.key] * design.X[l.key].std(ddof=0) / sd_y)
    if cpi_mode == "identity":
        coefs["cpi"] = 1.0  # imposed identity
        std_coef_cpi = float(1.0 * design.ln_cpi.std(ddof=0) / sd_y)
        std_vals["cpi"] = std_coef_cpi
    std_coefs = pd.Series(std_vals)

    # condition number of the regressor block (collinearity diagnostic)
    if Xcols:
        Z = np.column_stack([_zscore(design.X[l.key]).to_numpy() for l in reg_layers])
        sv = np.linalg.svd(Z, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    else:
        cond = np.nan

    fitted = pd.Series(Xmat @ beta, index=design.X.index)
    resid_s = pd.Series(resid, index=design.X.index)
    return AttributionResult(
        coefs=coefs, std_coefs=std_coefs, tstats=tstats, pvals=pvals, r2=r2,
        n=len(y), cond_number=cond, cpi_mode=cpi_mode, design=design,
        fitted=fitted, resid=resid_s,
    )


# ── period decomposition ────────────────────────────────────────────────
def _nearest_row(s: pd.Series, when: str, side: str) -> pd.Timestamp:
    """Index label at/after (side='start') or at/before (side='end') a period."""
    target = pd.Period(when, freq="M").to_timestamp("M")
    idx = s.dropna().index
    if side == "start":
        cand = idx[idx >= pd.Period(when, freq="M").to_timestamp()]
        return cand[0] if len(cand) else idx[0]
    cand = idx[idx <= target]
    return cand[-1] if len(cand) else idx[-1]


def decompose_period(
    result: AttributionResult,
    t0: str = DEFAULT_ATTRIB_START,
    t1: Optional[str] = None,
) -> pd.DataFrame:
    """Exact additive decomposition of the realised Δln(gold) over [t0, t1].

    contribution_k = coef_k · (proxy_k[t1] − proxy_k[t0]); for cpi_mode='identity'
    layer ① = Δln(CPI). Returns one row per layer plus 'ε_flow (residual)' and
    'TOTAL'; contributions + residual sum to Δln(gold) with ~0 slack."""
    design = result.design
    d0 = _nearest_row(design.ln_gold, t0, "start")
    d1 = _nearest_row(design.ln_gold, t1, "end") if t1 else design.ln_gold.dropna().index[-1]

    total = float(design.ln_gold.loc[d1] - design.ln_gold.loc[d0])
    rows = []
    explained = 0.0
    for lyr in design.layers:
        if lyr.key == "cpi" and result.cpi_mode == "identity":
            delta = float(design.ln_cpi.loc[d1] - design.ln_cpi.loc[d0])
            coef = 1.0
            contrib = delta
        else:
            coef = float(result.coefs.get(lyr.key, np.nan))
            delta = float(design.X[lyr.key].loc[d1] - design.X[lyr.key].loc[d0])
            contrib = coef * delta
        explained += contrib
        rows.append({
            "layer": lyr.key, "label": lyr.label, "coef": coef,
            "delta_proxy": delta, "contribution_ln": contrib,
            "contribution_pct_of_total": (contrib / total * 100.0) if total else np.nan,
        })
    resid_contrib = total - explained
    rows.append({
        "layer": "flow_resid", "label": "⑤/ε_flow (残差: 流量+未解释)",
        "coef": np.nan, "delta_proxy": np.nan, "contribution_ln": resid_contrib,
        "contribution_pct_of_total": (resid_contrib / total * 100.0) if total else np.nan,
    })
    rows.append({
        "layer": "TOTAL", "label": "总 Δln(gold)", "coef": np.nan,
        "delta_proxy": np.nan, "contribution_ln": total,
        "contribution_pct_of_total": 100.0 if total else np.nan,
    })
    out = pd.DataFrame(rows)
    out.attrs["t0"] = str(d0.date())
    out.attrs["t1"] = str(d1.date())
    out.attrs["total_pct_return"] = float(np.expm1(total) * 100.0)
    return out


def stacked_contribution_path(
    result: AttributionResult,
    t0: str = DEFAULT_ATTRIB_START,
    t1: Optional[str] = None,
) -> pd.DataFrame:
    """Per-month cumulative contribution of each layer since t0, for the stacked
    area chart. Column sum (+ residual) equals cumulative Δln(gold) at every t."""
    design = result.design
    d0 = _nearest_row(design.ln_gold, t0, "start")
    mask = design.X.index >= d0
    if t1:
        mask &= design.X.index <= _nearest_row(design.ln_gold, t1, "end")
    idx = design.X.index[mask]

    out = pd.DataFrame(index=idx)
    cum_expl = pd.Series(0.0, index=idx)
    for lyr in design.layers:
        if lyr.key == "cpi" and result.cpi_mode == "identity":
            series = (design.ln_cpi.loc[idx] - design.ln_cpi.loc[d0])
        else:
            coef = float(result.coefs.get(lyr.key, np.nan))
            series = coef * (design.X[lyr.key].loc[idx] - design.X[lyr.key].loc[d0])
        out[lyr.key] = series
        cum_expl = cum_expl + series
    total = design.ln_gold.loc[idx] - design.ln_gold.loc[d0]
    out["flow_resid"] = total - cum_expl
    out["total_dln_gold"] = total
    return out


# ── rolling coefficients (instability evidence) ──────────────────────────
def rolling_coefs(
    panel: AttributionPanel | pd.DataFrame,
    *,
    window: int = 60,
    layers: Sequence[Layer] = LAYERS,
    cpi_mode: str = "identity",
) -> pd.DataFrame:
    """Rolling-window OLS coefficients per layer, to *show* the regime-dependence
    (PR #1–#4: no stable anchor). Each row t is the fit on the trailing `window`
    months ending at t. NOT used for prediction — instability is the point."""
    df = panel.data if isinstance(panel, AttributionPanel) else panel
    design = build_design(df, layers)
    if cpi_mode == "identity":
        y_full = design.ln_gold - design.ln_cpi
        reg_layers = [l for l in design.layers if l.key != "cpi"]
    else:
        y_full = design.ln_gold
        reg_layers = list(design.layers)
    keys = [l.key for l in reg_layers]
    idx = design.X.index
    rows = {}
    for i in range(window - 1, len(idx)):
        sl = slice(i - window + 1, i + 1)
        yv = y_full.iloc[sl].to_numpy()
        Xv = np.column_stack([np.ones(window)] + [design.X[k].iloc[sl].to_numpy() for k in keys])
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        rows[idx[i]] = dict(zip(keys, beta[1:]))
    return pd.DataFrame.from_dict(rows, orient="index")


def verdict(decomp: pd.DataFrame) -> Dict[str, object]:
    """Core ruling: among the non-inflation layers (②③④⑤+resid), is sovereign
    (③) the largest *positive* contributor to the 2022→latest move?"""
    body = decomp[~decomp["layer"].isin(["cpi", "TOTAL"])].copy()
    body = body.dropna(subset=["contribution_ln"])
    if body.empty:
        return {"sovereign_took_over": False, "reason": "no non-inflation layers"}
    top = body.loc[body["contribution_ln"].idxmax()]
    sov = body[body["layer"] == "sov"]
    sov_contrib = float(sov["contribution_ln"].iloc[0]) if not sov.empty else np.nan
    took_over = bool(top["layer"] == "sov" and top["contribution_ln"] > 0)
    return {
        "sovereign_took_over": took_over,
        "top_layer": str(top["layer"]),
        "top_label": str(top["label"]),
        "top_contribution_ln": float(top["contribution_ln"]),
        "sov_contribution_ln": sov_contrib,
        "ranking": body.sort_values("contribution_ln", ascending=False)[
            ["layer", "contribution_ln", "contribution_pct_of_total"]
        ].to_dict("records"),
    }
