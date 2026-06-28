"""Gold "anchor + deviation" model — data layer & cointegration life-or-death gate.

Implements steps 0–1 of docs/gold-anchor-vecm-spec.md:

  Step 0 (data)  : pull monthly FRED panel — nominal gold, three anchor
                   candidates (all divided by GDP), and a long-end real rate.
  Step 1a (I/d)  : ADF + PP + KPSS unit-root tests → I(0)/I(1) verdict table.
  Step 1b (coint): Johansen (trace + max-eigen) on [ln gold, ln(anchor/GDP)] to
                   decide cointegration rank — does the anchor *hold* (rank>=1)
                   or is it a spurious common trend (rank=0)? Estimate the
                   cointegrating vector and long-run elasticity beta.

Design notes
------------
* All anchors are divided by GDP (the debasement mechanism is "fiat claims
  relative to real output"); raw stock levels would re-create the
  "two rising lines colliding" spurious correlation the spec warns about.
* Gold is kept *nominal*; price level is absorbed by CPI/anchor (spec §0).
* The 10y TIPS real rate (DFII10) only starts 1997. For a longer sample we
  splice a pre-1997 proxy = 10y nominal (GS10) − trailing 12m CPI inflation
  and record the splice break explicitly (a known weakness, spec §0).
* Data fetching is injectable (``fetch_fn``) so tests run offline on synthetic
  series with no network/key dependency.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# ── FRED series catalog ────────────────────────────────────────────────
# unit notes: GFDEBTN & WALCL in $Millions; GDP & M2SL in $Billions.
# We rescale everything to $Billions so ratios are interpretable (the scale
# only shifts the ln level by a constant, so it does not affect unit-root /
# cointegration verdicts or beta — but readable levels help sanity checks).
#
# NOTE on gold: the spec names FRED `GOLDAMGBD228NLBM` (LBMA AM fix), but FRED
# discontinued the LBMA gold series in 2023 (licensing). We instead use the
# public-domain datasets.io monthly gold price (Measuring Worth / LBMA),
# 1833→present, which matches our monthly panel frequency directly.
FRED_SERIES = {
    "debt": "GFDEBTN",                   # Federal debt total, $M, quarterly
    "gdp": "GDP",                        # Nominal GDP, $B SAAR, quarterly
    "m2": "M2SL",                        # M2 money stock, $B, monthly
    "fed_assets": "WALCL",               # Fed total assets, $M, weekly
    "tips_10y": "DFII10",                # 10y TIPS real yield, %, daily (2003+)
    "nominal_10y": "GS10",               # 10y Treasury nominal yield, %, monthly
    "cpi": "CPIAUCSL",                   # CPI-U, index, monthly
}

GOLD_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
)

# series reported in $Millions that we rescale to $Billions
_MILLIONS_TO_BILLIONS = {"debt", "fed_assets"}


@dataclass
class AnchorPanel:
    """Monthly panel + provenance notes."""

    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)


# ── FRED fetching ──────────────────────────────────────────────────────
def _fred_api_key() -> Optional[str]:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    for p in ["~/.fred_api_key", "~/.config/fred/api_key"]:
        fp = os.path.expanduser(p)
        if os.path.exists(fp):
            return open(fp).read().strip()
    return None


def fetch_fred_series(series_id: str, start: str = "1968-01-01") -> pd.Series:
    """Fetch a single FRED series via fredapi, falling back to the public CSV
    endpoint when no API key is available. Returns a float Series indexed by
    date (NaNs dropped)."""
    key = _fred_api_key()
    if key:
        try:
            from fredapi import Fred

            s = Fred(api_key=key).get_series(series_id, observation_start=start)
            s = pd.Series(s, dtype="float64").dropna()
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as e:
            # don't swallow silently — surface the cause, then try CSV fallback
            warnings.warn(
                f"fredapi failed for {series_id} ({type(e).__name__}: {e}); "
                "falling back to public CSV endpoint",
                stacklevel=2,
            )

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_gold_monthly(start: str = "1968-01-01") -> pd.Series:
    """Monthly nominal gold (USD/oz) from the public-domain datasets.io gold
    price dataset (Measuring Worth / LBMA), 1833→present. Used in place of the
    discontinued FRED LBMA series. Returns a float Series at month-end."""
    df = pd.read_csv(GOLD_CSV_URL)
    s = pd.Series(df["Price"].values, index=pd.PeriodIndex(df["Date"], freq="M").to_timestamp("M"))
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s[s.index >= pd.Timestamp(start)]


def _to_monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    """Resample a series to month-end. ``mean`` for noisy daily levels/rates,
    ``last`` for end-of-period stocks, ``ffill`` for lower-frequency series."""
    s = s.sort_index()
    if how == "ffill":
        # reindex onto a month-end grid, carrying the last known value forward
        idx = pd.date_range(s.index.min(), s.index.max(), freq="ME")
        return s.reindex(s.index.union(idx)).ffill().reindex(idx)
    rule = s.resample("ME")
    return rule.mean() if how == "mean" else rule.last()


def build_anchor_panel(
    start: str = "1968-01-01",
    end: Optional[str] = None,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    gold_fetch_fn: Callable[[str], pd.Series] = fetch_gold_monthly,
) -> AnchorPanel:
    """Build the monthly gold-anchor panel.

    Columns: gold_nominal, debt_gdp, m2_gdp, fed_gdp, real_rate_10y, plus
    ln_* transforms used by the tests. ``fetch_fn`` (FRED) and ``gold_fetch_fn``
    are injectable for offline tests.
    """
    raw: Dict[str, pd.Series] = {}
    notes: Dict[str, str] = {}
    for name, sid in FRED_SERIES.items():
        s = fetch_fn(sid, start)
        if name in _MILLIONS_TO_BILLIONS:
            s = s / 1000.0  # $M → $B
        raw[name] = s
    raw["gold_nominal"] = gold_fetch_fn(start)

    # frequency alignment
    gold = _to_monthly(raw["gold_nominal"], "last")   # already monthly
    debt = _to_monthly(raw["debt"], "ffill")          # quarterly stock
    gdp = _to_monthly(raw["gdp"], "ffill")            # quarterly flow (SAAR)
    m2 = _to_monthly(raw["m2"], "last")               # monthly stock
    fed = _to_monthly(raw["fed_assets"], "last")      # weekly stock
    tips = _to_monthly(raw["tips_10y"], "mean")       # daily rate
    nom10 = _to_monthly(raw["nominal_10y"], "mean")   # monthly rate
    cpi = _to_monthly(raw["cpi"], "last")             # monthly index

    # anchor ratios (all ÷ GDP)
    debt_gdp = debt / gdp
    m2_gdp = m2 / gdp
    fed_gdp = fed / gdp

    # real-rate splice: DFII10 (10y TIPS) where available; pre-TIPS proxy =
    # GS10 − trailing-12m CPI YoY. Splice break = first TIPS observation.
    tips_start = tips.first_valid_index()
    cpi_yoy = cpi.pct_change(12) * 100.0
    proxy_real = nom10 - cpi_yoy
    real_rate = tips.copy()
    if tips_start is not None:
        pre = proxy_real[proxy_real.index < tips_start]
        real_rate = pd.concat([pre[~pre.index.isin(real_rate.dropna().index)], real_rate.dropna()]).sort_index()
        splice_break = tips_start.date()
    else:
        real_rate = proxy_real
        splice_break = "n/a (TIPS unavailable)"
    notes["real_rate_splice"] = (
        f"real_rate_10y = DFII10 (10y TIPS) from {splice_break}; "
        "pre-TIPS proxy = GS10 − trailing-12m CPI YoY (realized, not ex-ante). "
        "Splice break is a known weakness (spec §0)."
    )

    df = pd.DataFrame(
        {
            "gold_nominal": gold,
            "debt_gdp": debt_gdp,
            "m2_gdp": m2_gdp,
            "fed_gdp": fed_gdp,
            "real_rate_10y": real_rate,
        }
    )
    # ln transforms (real rate can be negative → not logged)
    for col in ["gold_nominal", "debt_gdp", "m2_gdp", "fed_gdp"]:
        df[f"ln_{col}"] = np.log(df[col])

    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]

    # provenance / coverage notes
    def _cov(s: pd.Series) -> str:
        sv = s.dropna()
        if len(sv) == 0:
            return "no observations (n=0)"
        return f"{sv.index.min().date()}..{sv.index.max().date()} (n={len(sv)})"

    notes["units"] = "debt & fed_assets rescaled $M→$B; gdp,m2 already $B."
    notes["frequency"] = (
        "monthly (ME). daily→mean (gold, rates), weekly/monthly stocks→last, "
        "quarterly (debt, gdp)→ffill within quarter."
    )
    fed_raw = raw["fed_assets"].dropna()
    fed_start = fed_raw.index.min().date() if len(fed_raw) else "n/a"
    notes["fed_gdp_coverage"] = (
        f"WALCL starts {fed_start} "
        "(2002+; effectively a post-2008 anchor — short sample, use with care)."
    )
    notes["coverage"] = "; ".join(f"{c}:{_cov(df[c])}" for c in df.columns)

    return AnchorPanel(data=df, notes=notes)


# ── Unit-root tests (step 0) ───────────────────────────────────────────
def unit_root_tests(series: pd.Series, regression: str = "c") -> Dict[str, float]:
    """Run ADF + PP + KPSS on a series (levels). Returns stats & p-values.

    ADF/PP null = unit root (I(1)); KPSS null = stationarity (I(0)).
    """
    from statsmodels.tsa.stattools import InterpolationWarning, adfuller, kpss
    from arch.unitroot import PhillipsPerron

    x = pd.Series(series).dropna().astype(float)
    out: Dict[str, float] = {"n": int(len(x))}

    adf = adfuller(x, regression=regression, autolag="AIC")
    out["adf_stat"], out["adf_pvalue"] = float(adf[0]), float(adf[1])

    pp = PhillipsPerron(x, trend=regression)
    out["pp_stat"], out["pp_pvalue"] = float(pp.stat), float(pp.pvalue)

    # KPSS uses 'c' or 'ct'; map ADF 'nc'→'c'. statsmodels emits a benign
    # InterpolationWarning when the stat is outside the p-value lookup table.
    kreg = "ct" if regression == "ct" else "c"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kstat, kp, *_ = kpss(x, regression=kreg, nlags="auto")
    out["kpss_stat"], out["kpss_pvalue"] = float(kstat), float(kp)
    return out


def classify_integration(series: pd.Series, alpha: float = 0.05) -> Dict[str, object]:
    """Classify a series as I(0) / I(1) / ambiguous by combining ADF, PP, KPSS
    on the levels and (for I(1) confirmation) the first difference.

    Verdict logic on levels:
      * ADF & PP reject unit root (p<alpha) AND KPSS fails to reject
        stationarity (p>alpha) → I(0).
      * ADF & PP fail to reject AND KPSS rejects → I(1) (confirm via diff).
      * disagreement → "ambiguous" (boundary case — spec flags real rate here).
    """
    lvl = unit_root_tests(series)
    adf_rej = lvl["adf_pvalue"] < alpha
    pp_rej = lvl["pp_pvalue"] < alpha
    kpss_rej = lvl["kpss_pvalue"] < alpha

    stationary_votes = int(adf_rej) + int(pp_rej) + int(not kpss_rej)
    if stationary_votes >= 2 and (adf_rej or pp_rej):
        # ADF/PP reject unit root AND KPSS fails to reject stationarity → I(0)
        verdict = "I(0)"
    elif (not adf_rej) and (not pp_rej) and kpss_rej:
        # ADF/PP fail to reject unit root AND KPSS rejects stationarity → I(1)
        verdict = "I(1)"
    else:
        # everything else (incl. ADF/PP/KPSS all fail to reject = low power /
        # inconclusive, and ADF↔PP↔KPSS disagreement) is genuinely ambiguous
        verdict = "ambiguous"

    res: Dict[str, object] = {"verdict": verdict, "levels": lvl}
    # confirm I(1): the first difference should look I(0)
    diff = pd.Series(series).dropna().diff().dropna()
    if len(diff) > 12:
        d = unit_root_tests(diff)
        res["diff"] = d
        res["diff_stationary"] = bool(d["adf_pvalue"] < alpha or d["pp_pvalue"] < alpha)
    return res


def integration_table(df: pd.DataFrame, columns, min_obs: int = 20) -> pd.DataFrame:
    """Build the I(0)/I(1) verdict table for the given columns. Columns with
    fewer than ``min_obs`` non-NaN observations are reported as "insufficient
    data" instead of being fed to the (crash-prone) test routines."""
    rows = []
    for col in columns:
        n_obs = int(df[col].notna().sum())
        if n_obs < min_obs:
            rows.append(
                {"series": col, "verdict": "insufficient data", "adf_p": np.nan,
                 "pp_p": np.nan, "kpss_p": np.nan, "diff_stationary": None, "n": n_obs}
            )
            continue
        c = classify_integration(df[col])
        lvl = c["levels"]
        rows.append(
            {
                "series": col,
                "verdict": c["verdict"],
                "adf_p": round(lvl["adf_pvalue"], 4),
                "pp_p": round(lvl["pp_pvalue"], 4),
                "kpss_p": round(lvl["kpss_pvalue"], 4),
                "diff_stationary": c.get("diff_stationary"),
                "n": lvl["n"],
            }
        )
    return pd.DataFrame(rows).set_index("series")


# ── Cointegration (step 1) ─────────────────────────────────────────────
# Johansen critical-value columns: [90%, 95%, 99%]
_CV_COL = {0.10: 0, 0.05: 1, 0.01: 2}


def johansen_test(
    df: pd.DataFrame,
    columns,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Run the Johansen cointegration test on the given columns.

    det_order: -1 no deterministic, 0 constant in CE (default), 1 linear trend.
    Returns trace/max-eigen stats vs critical values, the inferred rank, the
    cointegrating vector, and the implied long-run elasticity beta
    (normalized on the first column, i.e. gold).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    sub = df[list(columns)].dropna()
    res = coint_johansen(sub.values, det_order, k_ar_diff)
    cvc = _CV_COL[alpha]
    n = len(columns)

    raw_trace_rank = 0
    for r in range(n):
        if res.lr1[r] > res.cvt[r, cvc]:
            raw_trace_rank = r + 1
        else:
            break
    raw_maxeig_rank = 0
    for r in range(n):
        if res.lr2[r] > res.cvm[r, cvc]:
            raw_maxeig_rank = r + 1
        else:
            break

    # A valid cointegration rank is in [0, n-1]; raw rank == n means the system
    # is full-rank (the series themselves look stationary / model assumptions
    # don't hold), which is NOT "the anchor holds". Cap and flag it.
    full_rank_stationary = (raw_trace_rank == n) or (raw_maxeig_rank == n)
    trace_rank = min(raw_trace_rank, n - 1)
    maxeig_rank = min(raw_maxeig_rank, n - 1)

    # cointegrating vector = first eigenvector, normalized on gold (col 0)
    vec = res.evec[:, 0]
    norm = vec / vec[0]
    # G = α + β·A  ⇒  G - β·A ~ I(0); evec row encodes G + c1*A,
    # so β = -c1 (coefficient on the anchor moved to the RHS).
    beta = float(-norm[1]) if n >= 2 else float("nan")

    return {
        "columns": list(columns),
        "trace_stat": [float(x) for x in res.lr1],
        "trace_cv": [float(x) for x in res.cvt[:, cvc]],
        "maxeig_stat": [float(x) for x in res.lr2],
        "maxeig_cv": [float(x) for x in res.cvm[:, cvc]],
        "raw_trace_rank": int(raw_trace_rank),
        "raw_maxeig_rank": int(raw_maxeig_rank),
        "trace_rank": int(trace_rank),
        "maxeig_rank": int(maxeig_rank),
        "rank": int(min(trace_rank, maxeig_rank)),
        "full_rank_stationary": bool(full_rank_stationary),
        "coint_vector_normalized": [float(x) for x in norm],
        "beta": beta,
        "alpha_level": alpha,
        "n_obs": int(len(sub)),
    }
