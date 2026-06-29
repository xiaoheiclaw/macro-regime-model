"""Placebo battery for the PR #11 gold-attribution claim (decisive credibility test).

Background
----------
PR #9 decomposes 2022→2026 log gold into five additive layers
(① inflation / ② real-rate / ③ sovereign / ④ tail / ⑤ flow). With layer ⑤
**absent**, the four measurable macro layers explain *less than nothing* of the
post-2022 surge — the ε_flow residual is ~127% of the move (the −real-rate layer
is *negative*: real rates rose, gold rose anyway).

PR #11 then *injected* layer ⑤ = **cumulative excess central-bank gold purchases**
(WGC), and reported the residual collapsing (R² 0.32→0.67, ⑤ "claiming" the bulk
of the move). codex's review raised the decisive objection: cumulative excess
stock is a **monotone-rising trend variable**, gold post-2022 is *also* monotone
rising, and in a short sample *any* monotone trend can soak up the residual and
lift R² — exactly the "debt/GDP spurious correlation" this research already
killed (PR #1). So a residual collapse is **not** evidence of a real
central-bank-buying signal until we rule out trend-fitting.

This module runs that rule-out. It is **ex-post, non-predictive, non-trading** —
it reuses PR #9's `build_attribution_panel` / `fit_attribution` / `decompose_period`
verbatim and only swaps the ⑤ column for a battery of candidate series:

  REAL    cumulative excess WGC purchases (the PR #11 series)
  (a) t        linear time trend
  (b) log(t)   log time trend
  (c) rand     ≥5 seeded random monotone series  cumsum(|N(0,1)|)
  (d) cum CPI  cumulative ln(CPI)
  (e) cum M2   cumulative ln(M2)
  (f) cum IP   cumulative industrial production (unrelated monotone macro)
  (g) kink     flat-until-2022 then linear ramp  ← *shape* control, not a
               smooth trend: isolates whether the WGC fit is about the 2022
               regime *kink* rather than any unique economic content.

Decisive question (per candidate, on an IDENTICAL sample window): can the placebo
reach a comparable R² / residual collapse / ⑤-contribution as the real WGC?

Arbiters
--------
1. **Levels placebo table** — R², residual %, ⑤ contribution %, ⑤ t.
2. **Stationarity** (ADF+KPSS, reusing `gold_anchor.unit_root_tests`): a level
   regression of one I(1)/I(2) trend on another is the textbook spurious setup.
3. **Difference regression** (Δln gold ~ Δlayers): if ⑤'s explanatory power is
   *real* (not a level-trend artefact) it survives in the stationary first-
   difference; if it evaporates, that is the classic spurious-regression
   signature.
4. **Lag/lead** (variance attribution only, NOT causal): does lagged purchase
   *flow* lead gold, or does gold lead purchases (endogeneity direction)?

Honest caveats baked in: WGC annual→monthly is a coarse interpolation; the
common window is short (2010-12+ where WGC exists); endogeneity is reported as
variance attribution, not identified causality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from lib.gold_credit_spread_attribution import (
    build_design,
    decompose_period,
    fit_attribution,
)

# ── WGC central-bank net purchases (tonnes/yr) ──────────────────────────
# Source: World Gold Council "Gold Demand Trends" annual central-bank net
# purchases. Values are WGC estimates and are revised year-to-year (±tens of
# tonnes); 2025 is a partial/estimated full-year figure. Treat as a coarse
# annual series, not a precise stock.
WGC_ANNUAL_NET_PURCHASES_T: Dict[int, float] = {
    2010: 79, 2011: 481, 2012: 569, 2013: 629, 2014: 584, 2015: 580,
    2016: 395, 2017: 379, 2018: 656, 2019: 605, 2020: 255, 2021: 463,
    2022: 1082, 2023: 1037, 2024: 1045, 2025: 863,
}
# Pre-2022 "normal" annual demand baseline (tonnes). The cumulative *excess*
# stock = cumsum(annual − baseline); 473 ≈ the 2010-2021 mean, chosen so the
# stock is ~flat pre-2022 and ramps only in the 2022→ de-dollarisation wave.
WGC_BASELINE_T: float = 473.0

WGC_SOURCE_NOTE = (
    "WGC annual central-bank net purchases (tonnes), World Gold Council 'Gold "
    "Demand Trends'. Estimates, revised yearly (±tens of t); 2025 partial/est. "
    f"Cumulative excess stock = cumsum(annual − {WGC_BASELINE_T:.0f}t baseline), "
    "annual→monthly by time interpolation (coarse)."
)


def wgc_cumulative_excess_annual(
    annual: Optional[Dict[int, float]] = None,
    baseline: float = WGC_BASELINE_T,
) -> pd.Series:
    """Annual cumulative *excess* central-bank gold stock (tonnes), year-end dated.

    excess_y = Σ_{k≤y} (purchases_k − baseline). Flat-ish pre-2022, ramps after.
    """
    annual = WGC_ANNUAL_NET_PURCHASES_T if annual is None else annual
    years = sorted(annual)
    cum = np.cumsum([float(annual[y]) - float(baseline) for y in years])
    idx = pd.to_datetime([f"{y}-12-31" for y in years])
    return pd.Series(cum, index=idx, dtype="float64", name="wgc_cum_excess")


def annual_to_monthly(annual: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Interpolate a year-end annual series onto a monthly index by time
    interpolation (linear in calendar time), then clip to `idx`. Coarse by
    construction — central-bank stock is only known annually."""
    annual = annual.sort_index()
    union = annual.index.union(idx)
    return annual.reindex(union).interpolate("time").reindex(idx)


# ── placebo series builders (all aligned to a monthly index) ─────────────
def make_placebos(
    idx: pd.DatetimeIndex,
    *,
    cpi: Optional[pd.Series] = None,
    m2: Optional[pd.Series] = None,
    ip: Optional[pd.Series] = None,
    rand_seeds: Sequence[int] = (11, 22, 33, 44, 55),
    kink_date: str = "2022-01-31",
) -> "Dict[str, pd.Series]":
    """Build the placebo ⑤-candidate series on `idx`.

    Each placebo is a series that is monotone-rising (or, for the kink, flat then
    rising) — the property codex flagged as able to spuriously soak up residual.

    * ``t`` linear time trend 1..N
    * ``log_t`` log time trend
    * ``rand_<seed>`` cumsum(|N(0,1)|) — random *monotone* walk, one per seed
    * ``cum_cpi`` cumulative ln(CPI)   (needs ``cpi`` level series)
    * ``cum_m2``  cumulative ln(M2)    (needs ``m2`` level series)
    * ``cum_ip``  cumulative IP        (needs ``ip`` level series)
    * ``kink_2022`` 0 until ``kink_date`` then a linear ramp — a *shape* control
      that mimics the WGC stock's flat→ramp profile WITHOUT any economic content.

    cum_* placebos are skipped (omitted from the dict) when their source series
    is None, so the caller can run offline without those FRED pulls.
    """
    n = len(idx)
    out: Dict[str, pd.Series] = {}
    out["t"] = pd.Series(np.arange(1, n + 1, dtype="float64"), index=idx)
    out["log_t"] = pd.Series(np.log(np.arange(1, n + 1, dtype="float64")), index=idx)
    for sd in rand_seeds:
        rng = np.random.RandomState(int(sd))
        out[f"rand_{sd}"] = pd.Series(
            np.cumsum(np.abs(rng.randn(n))), index=idx, dtype="float64"
        )
    if cpi is not None:
        out["cum_cpi"] = np.log(cpi.reindex(idx).astype(float)).cumsum()
    if m2 is not None:
        out["cum_m2"] = np.log(m2.reindex(idx).astype(float)).cumsum()
    if ip is not None:
        out["cum_ip"] = ip.reindex(idx).astype(float).cumsum()
    post = idx >= pd.Timestamp(kink_date)
    kink = pd.Series(0.0, index=idx)
    kink.loc[post] = np.arange(1, int(post.sum()) + 1, dtype="float64")
    out["kink_2022"] = kink
    return out


PLACEBO_LABELS: Dict[str, str] = {
    "REAL_WGC": "真·WGC 累计超额购金存量",
    "t": "(a) 线性时间趋势 t",
    "log_t": "(b) log(t)",
    "cum_cpi": "(d) 累计 ln(CPI)",
    "cum_m2": "(e) 累计 ln(M2)",
    "cum_ip": "(f) 累计工业产值 IP",
    "kink_2022": "(g) 2022 拐点 (先平后升, 形态对照)",
}


def placebo_label(key: str) -> str:
    if key in PLACEBO_LABELS:
        return PLACEBO_LABELS[key]
    if key.startswith("rand_"):
        return f"(c) 随机单调 cumsum|N(0,1)| seed={key.split('_')[1]}"
    return key


# ── levels attribution under a swapped ⑤ ─────────────────────────────────
ATTR_COLS = [
    "ln_gold", "ln_cpi", "neg_real_rate", "ln_debt_gdp",
    "neg_custody_share", "vix", "credit_spread", "wgc_flow",
]


def common_window(
    panel_df: pd.DataFrame, fifth: pd.Series, *, min_obs: int = 24
) -> pd.DatetimeIndex:
    """Rows where layers ①–④ are all present AND the ⑤ candidate is present.

    Fixing this once and reusing it for EVERY candidate is what makes the placebo
    comparison fair — otherwise a placebo with longer history is fit on more data
    and its R² is not comparable to WGC's (a trap we hit while exploring)."""
    base = build_design(panel_df)  # ①–④ on their common dropna (⑤ optional → dropped)
    idx = base.X.index.intersection(fifth.dropna().index)
    if len(idx) < min_obs:
        raise ValueError(
            f"common window has only {len(idx)} rows (<{min_obs}); ①–④ and the "
            "⑤ candidate have no sufficient overlap."
        )
    return idx


@dataclass
class FifthResult:
    key: str
    label: str
    r2: float
    n: int
    flow_contrib_ln: float
    flow_contrib_pct: float
    resid_contrib_ln: float
    resid_contrib_pct: float
    flow_coef: float
    flow_t: float
    flow_p: float


def run_levels_fifth(
    panel_df: pd.DataFrame,
    fifth: pd.Series,
    *,
    key: str,
    window: pd.DatetimeIndex,
    t0: str = "2022-01",
    t1: Optional[str] = None,
    cpi_mode: str = "identity",
    min_obs: int = 24,
) -> FifthResult:
    """Fit the five-layer attribution with ⑤ = `fifth` on `window`, decompose
    [t0,t1], and return the ⑤ / residual diagnostics. Pure reuse of PR #9."""
    df = panel_df.loc[window].copy()
    df["wgc_flow"] = fifth.reindex(window).astype(float)
    res = fit_attribution(df, cpi_mode=cpi_mode, min_obs=min_obs)
    dec = decompose_period(res, t0=t0, t1=t1)
    fl = dec[dec["layer"] == "flow"]
    rd = dec[dec["layer"] == "flow_resid"]
    return FifthResult(
        key=key,
        label=placebo_label(key),
        r2=float(res.r2),
        n=int(res.n),
        flow_contrib_ln=float(fl["contribution_ln"].iloc[0]) if not fl.empty else np.nan,
        flow_contrib_pct=float(fl["contribution_pct_of_total"].iloc[0]) if not fl.empty else np.nan,
        resid_contrib_ln=float(rd["contribution_ln"].iloc[0]),
        resid_contrib_pct=float(rd["contribution_pct_of_total"].iloc[0]),
        flow_coef=float(res.coefs.get("flow", np.nan)),
        flow_t=float(res.tstats.get("flow", np.nan)),
        flow_p=float(res.pvals.get("flow", np.nan)),
    )


def baseline_no_fifth(
    panel_df: pd.DataFrame,
    *,
    window: pd.DatetimeIndex,
    t0: str = "2022-01",
    t1: Optional[str] = None,
    cpi_mode: str = "identity",
    min_obs: int = 24,
) -> Dict[str, float]:
    """Four-layer attribution (⑤ folded into the residual) on `window` — the
    "before" the placebo battery shrinks. Returns R² and residual % of the move
    (the "127%→?" anchor)."""
    df = panel_df.loc[window].copy()
    df["wgc_flow"] = np.nan  # ⑤ absent → optional layer dropped
    res = fit_attribution(df, cpi_mode=cpi_mode, min_obs=min_obs)
    dec = decompose_period(res, t0=t0, t1=t1)
    rd = dec[dec["layer"] == "flow_resid"]
    return {
        "r2": float(res.r2),
        "n": int(res.n),
        "resid_contrib_ln": float(rd["contribution_ln"].iloc[0]),
        "resid_contrib_pct": float(rd["contribution_pct_of_total"].iloc[0]),
    }


# ── shared OLS (mirrors gold_credit_spread_attribution._ols, kept local so we
#    do not import a private name) ────────────────────────────────────────
def _ols(y: np.ndarray, X: np.ndarray):
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


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if (np.isfinite(sd) and sd > 0) else s * 0.0


def _diff_design(panel_df: pd.DataFrame, fifth: pd.Series,
                 window: pd.DatetimeIndex, cpi_mode: str) -> "tuple":
    """First-difference design matching the levels five-layer spec.

    Composites ③ (debt/GDP ⊕ −custody) and ④ (VIX ⊕ credit) are built as the
    equal-weight mean of z-scored *differences* (mirrors the levels z-composite,
    just on Δ). LHS for cpi_mode='identity' is Δln(gold) − Δln(cpi); 'free' adds
    Δln(cpi) as a regressor. Returns (y, Xmat, names)."""
    df = panel_df.loc[window].copy()
    df["wgc_flow"] = fifth.reindex(window).astype(float)
    dd = df[ATTR_COLS].dropna().diff().dropna()
    real = dd["neg_real_rate"]
    sov = (_zscore(dd["ln_debt_gdp"]) + _zscore(dd["neg_custody_share"])) / 2.0
    tail = (_zscore(dd["vix"]) + _zscore(dd["credit_spread"])) / 2.0
    flow = dd["wgc_flow"]
    if cpi_mode == "identity":
        y = (dd["ln_gold"] - dd["ln_cpi"]).to_numpy()
        cols = [("real", real), ("sov", sov), ("tail", tail), ("flow", flow)]
    else:
        y = dd["ln_gold"].to_numpy()
        cols = [("cpi", dd["ln_cpi"]), ("real", real), ("sov", sov),
                ("tail", tail), ("flow", flow)]
    names = ["const"] + [c[0] for c in cols]
    Xmat = np.column_stack([np.ones(len(y))] + [c[1].to_numpy() for c in cols])
    return y, Xmat, names, dd.index


@dataclass
class DiffResult:
    key: str
    label: str
    r2: float
    n: int
    coefs: pd.Series
    tstats: pd.Series
    pvals: pd.Series

    @property
    def flow_t(self) -> float:
        return float(self.tstats.get("flow", np.nan))

    @property
    def flow_p(self) -> float:
        return float(self.pvals.get("flow", np.nan))


def run_diff_fifth(
    panel_df: pd.DataFrame,
    fifth: pd.Series,
    *,
    key: str,
    window: pd.DatetimeIndex,
    cpi_mode: str = "identity",
) -> DiffResult:
    """First-difference five-layer attribution. The arbiter: if ⑤ is real it
    survives here (Δ is stationary); if it was a level-trend artefact it dies."""
    y, Xmat, names, _ = _diff_design(panel_df, fifth, window, cpi_mode)
    beta, tstat, pval, r2, _ = _ols(y, Xmat)
    return DiffResult(
        key=key, label=placebo_label(key), r2=float(r2), n=int(len(y)),
        coefs=pd.Series(beta, index=names),
        tstats=pd.Series(tstat, index=names),
        pvals=pd.Series(pval, index=names),
    )


# ── stationarity ─────────────────────────────────────────────────────────
def stationarity_table(
    series_map: Dict[str, pd.Series], *, min_obs: int = 20
) -> pd.DataFrame:
    """ADF + KPSS on each series' levels AND first difference (reusing
    `gold_anchor.unit_root_tests`). A level regression of one I(1)/I(2) trend on
    another is the spurious setup; this table documents which inputs are
    non-stationary in levels."""
    from lib.gold_anchor import unit_root_tests

    rows = []
    for name, s in series_map.items():
        x = pd.Series(s).dropna().astype(float)
        row: Dict[str, object] = {"series": name, "n": int(len(x))}
        try:
            lvl = unit_root_tests(x, min_obs=min_obs)
            row.update(
                adf_p=lvl["adf_pvalue"], kpss_p=lvl["kpss_pvalue"],
                level_verdict=_io_verdict(lvl["adf_pvalue"], lvl["kpss_pvalue"]),
            )
        except ValueError as e:
            row.update(adf_p=np.nan, kpss_p=np.nan, level_verdict=f"n/a ({type(e).__name__})")
        try:
            d = unit_root_tests(x.diff().dropna(), min_obs=min_obs)
            row.update(
                diff_adf_p=d["adf_pvalue"], diff_kpss_p=d["kpss_pvalue"],
                diff_verdict=_io_verdict(d["adf_pvalue"], d["kpss_pvalue"]),
            )
        except ValueError as e:
            row.update(diff_adf_p=np.nan, diff_kpss_p=np.nan,
                       diff_verdict=f"n/a ({type(e).__name__})")
        rows.append(row)
    return pd.DataFrame(rows)


def _io_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    """Compact I(0)/I(1)/ambiguous label from ADF (null=unit root) + KPSS
    (null=stationary)."""
    adf_rej = np.isfinite(adf_p) and adf_p < alpha
    kpss_rej = np.isfinite(kpss_p) and kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "I(0)"
    if (not adf_rej) and kpss_rej:
        return "I(1)+"  # unit root not rejected & stationarity rejected
    return "ambiguous"


# ── lag/lead (variance attribution, NOT causal identification) ───────────
def lead_lag_table(
    gold_dln: pd.Series,
    flow: pd.Series,
    *,
    max_lag: int = 6,
) -> pd.DataFrame:
    """Cross-correlation of Δln(gold) with leads/lags of the purchase *flow*
    (Δ of the cumulative stock = the annual flow). Positive lag k = flow leads
    gold by k months (purchases→price); negative = gold leads flow (price→
    purchases, the endogeneity direction).

    This is **variance attribution only** — interpolated annual data is far too
    smooth and the sample far too short to identify causal direction; reported to
    show *which* direction co-moves, not to claim causality."""
    g = pd.Series(gold_dln).dropna()
    f = pd.Series(flow).reindex(g.index)
    df = pd.concat([g.rename("g"), f.rename("f")], axis=1).dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        shifted = df["f"].shift(k)  # k>0: past flow vs current gold (flow leads)
        pair = pd.concat([df["g"], shifted], axis=1).dropna()
        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 3 else np.nan
        rows.append({
            "lag_months": k,
            "direction": ("flow→gold" if k > 0 else "gold→flow" if k < 0 else "contemp."),
            "corr": corr,
            "n": int(len(pair)),
        })
    return pd.DataFrame(rows)


# ── verdict ──────────────────────────────────────────────────────────────
@dataclass
class PlaceboVerdict:
    verdict: str                 # "spurious" | "mixed" | "real"
    real_beats_placebos: bool    # real levels R² > every monotone placebo
    survives_in_diff: bool       # real ⑤ |t| ≥ threshold in first-difference
    diff_singles_out_real: bool  # real diff |t| ≥ every monotone placebo diff |t|
    kink_dominates: bool         # 2022 shape-control matches/beats real levels R²
    spurious: bool               # back-compat: verdict == "spurious"
    best_placebo_key: Optional[str]
    best_placebo_r2: float
    real_r2: float
    real_diff_t: float
    kink_r2: float
    reason: str
    notes: List[str] = field(default_factory=list)


def adjudicate(
    real: FifthResult,
    placebos: Sequence[FifthResult],
    real_diff: DiffResult,
    *,
    diff_placebos: Optional[Sequence[DiffResult]] = None,
    diff_t_threshold: float = 2.0,
    r2_margin: float = 0.05,
) -> PlaceboVerdict:
    """Core three-way ruling.

    Two distinct controls, judged separately:
      * **monotone-trend placebos** (a-f): smooth I(1)/I(2) trends. If any matches
        the real WGC *levels* R², the residual collapse is a trend artefact.
      * **kink shape-control** (g): flat-until-2022 then ramp. It carries a real
        2022 break, so it is NOT a monotone-trend placebo; instead it isolates
        whether the WGC levels win is just *the 2022 kink shape* (generic to
        anything that turned up in 2022) rather than unique purchase content.

    The first-difference (stationary) re-fit is the arbiter against level
    spuriousness; we also check whether it *singles out* WGC vs the monotone
    placebos' own diff t.

    verdict:
      * **spurious** — a monotone placebo matches WGC levels R², OR WGC ⑤ dies in
        the difference. The +121% claim is a trend-fitting artefact.
      * **mixed** — WGC beats the smooth placebos AND survives the difference, BUT
        the kink shape-control matches/beats it in levels OR the difference does
        not single WGC out from the placebos. Real *contemporaneous co-movement*,
        but the levels win is shape-driven and the layer cannot be read as causal
        "central-bank top-pricing".
      * **real** — WGC beats every placebo (smooth AND kink), survives the
        difference, and the difference singles it out. Genuine, distinct signal.
    """
    monotone = [p for p in placebos if p.key != "kink_2022"]
    kink = next((p for p in placebos if p.key == "kink_2022"), None)
    best = max(monotone, key=lambda p: p.r2) if monotone else None
    placebo_matches = bool(best and best.r2 >= real.r2 - r2_margin)
    survives = bool(np.isfinite(real_diff.flow_t) and abs(real_diff.flow_t) >= diff_t_threshold)
    kink_dominates = bool(kink and kink.r2 >= real.r2 - r2_margin)

    # does the difference single out WGC over the monotone placebos' own diff t?
    diff_singles_out = True
    if diff_placebos:
        mono_diff_t = [abs(d.flow_t) for d in diff_placebos
                       if d.key not in ("kink_2022", "REAL_WGC") and np.isfinite(d.flow_t)]
        if mono_diff_t:
            diff_singles_out = bool(abs(real_diff.flow_t) >= max(mono_diff_t) - 1e-9)

    if placebo_matches or not survives:
        verdict = "spurious"
    elif kink_dominates or not diff_singles_out:
        verdict = "mixed"
    else:
        verdict = "real"

    notes: List[str] = []
    if kink is not None:
        if kink_dominates:
            notes.append(
                f"形态对照(g)2022拐点 R²={kink.r2:.3f} ≥ 真WGC R²={real.r2:.3f}:"
                "一个**无经济含义**的「先平后升」阶梯就能复现甚至超过真WGC的水平拟合 —— "
                "真WGC在水平口径下拟合的主要是**2022制度拐点的形态**(凡 2022 后转折的序列皆可),"
                "而非央行购金独有的内容(推理)。"
            )
        else:
            notes.append(f"形态对照(g)2022拐点 R²={kink.r2:.3f} < 真WGC R²={real.r2:.3f}。")
    if diff_placebos and not diff_singles_out:
        notes.append(
            f"差分口径下真WGC的⑤ t={real_diff.flow_t:+.2f} **并非**最高(累计CPI/M2 等 placebo 的差分 t 更大),"
            "即平稳口径并未把真WGC从一众宏观趋势中单独挑出 —— 其差分显著性主要来自 2022-24 购金流量与金价"
            "同期共振,而非独有结构(推理)。"
        )

    if verdict == "spurious":
        if placebo_matches and not survives:
            reason = (
                f"placebo「{placebo_label(best.key)}」R²={best.r2:.3f} 已匹配真WGC R²={real.r2:.3f},"
                f"且真WGC的⑤在差分口径消失(t={real_diff.flow_t:.2f}) —— 双重伪回归特征,+121%是趋势拟合假象。"
            )
        elif placebo_matches:
            reason = (
                f"placebo「{placebo_label(best.key)}」R²={best.r2:.3f} 匹配真WGC R²={real.r2:.3f} —— "
                "单调趋势即可复现残差塌缩,+121%认领不可信(伪回归)。"
            )
        else:
            reason = (
                f"真WGC的⑤在差分(平稳)口径消失(t={real_diff.flow_t:.2f}):levels显著但差分不显著 = "
                "典型伪回归特征,+121%认领是水平趋势拟合假象。"
            )
    elif verdict == "mixed":
        reason = (
            f"单调趋势 placebo 无一逼近真WGC水平 R²(最高 {best.r2:.3f} vs 真 {real.r2:.3f}),"
            f"且真WGC的⑤在差分口径存活(t={real_diff.flow_t:.2f}) —— **不是**纯单调趋势伪回归;"
            "**但**真WGC的水平优势主要由 2022 制度拐点形态驱动(见下注),且差分口径未把它从其它宏观趋势中"
            "单独挑出。**裁决:+121% 残差认领含真实的 2022-24 同期共振成分,但应从『央行购金顶价(因果)』"
            "降级为『同期共振相关』** —— 水平 R² 抬升的主体是趋势/形态拟合,而非已识别的购金因果。"
        )
    else:  # real
        reason = (
            f"真WGC水平 R²={real.r2:.3f} 胜过所有 placebo(含形态对照),差分口径存活且把自身单独挑出"
            f"(t={real_diff.flow_t:.2f}) —— +121% 残差认领**含真实且可区分的信号**(但仍不构成因果识别)。"
        )

    return PlaceboVerdict(
        verdict=verdict,
        real_beats_placebos=not placebo_matches,
        survives_in_diff=survives,
        diff_singles_out_real=diff_singles_out,
        kink_dominates=kink_dominates,
        spurious=(verdict == "spurious"),
        best_placebo_key=best.key if best else None,
        best_placebo_r2=best.r2 if best else np.nan,
        real_r2=real.r2,
        real_diff_t=real_diff.flow_t,
        kink_r2=kink.r2 if kink else np.nan,
        reason=reason,
        notes=notes,
    )
