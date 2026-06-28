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
* All anchors are divided by **nominal** GDP (FRED `GDP`), i.e. the conventional
  debt-to-GDP / M2-to-GDP ratio — a nominal stock over a nominal flow. The
  debasement motivation is "fiat claims relative to output"; we keep both
  numerator and denominator nominal (a real-GDP `GDPC1` denominator would mix a
  real flow with a nominal stock). Dividing by output is what stops raw stock
  levels from re-creating the "two rising lines colliding" spurious correlation.
* Gold is kept *nominal*; price level is absorbed by CPI/anchor (spec §0).
* The 10y TIPS real rate (FRED DFII10, constant maturity) only starts 2003.
  For a longer sample we splice a pre-TIPS proxy = 10y nominal (GS10) −
  trailing 12m CPI inflation and record the splice break explicitly (a known
  weakness, spec §0). The spec's "1997" refers to TIPS issuance generally, not
  this DFII10 series.
* Data fetching is injectable (``fetch_fn``) so tests run offline on synthetic
  series with no network/key dependency.
"""
from __future__ import annotations

import hashlib
import io
import json
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

# Pinned to a specific commit SHA (NOT `main`) for reproducibility: the content
# is frozen, so the analysis is replayable and an upstream change cannot
# silently move the result. EXPECTED_GOLD_SHA256 is verified on every fetch and
# a mismatch hard-fails. To refresh the data, bump both the SHA and the hash.
GOLD_CSV_COMMIT = "95f96689baad2bc097ace55805cc9492b560d2ba"  # datasets/gold-prices
GOLD_CSV_URL = (
    f"https://raw.githubusercontent.com/datasets/gold-prices/{GOLD_CSV_COMMIT}"
    "/data/monthly.csv"
)
EXPECTED_GOLD_SHA256 = "4ad63a96612effbb57f900c008e011d6c446f795f4722ee8056a2a702725f397"

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
            with open(fp, encoding="utf-8") as f:
                return f.read().strip()
    return None


def _http_get_text(url: str, timeout: float = 30.0) -> str:
    """GET a URL with an explicit timeout + HTTP status check, returning the
    body text. Avoids pandas hanging forever or silently parsing an error
    page as CSV on a network blip."""
    import httpx

    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"download failed for {url}: {type(e).__name__}: {e}") from e
    return r.text


def fetch_fred_series(series_id: str, start: str = "1968-01-01") -> pd.Series:
    """Fetch a single FRED series. With an API key, uses the FRED observations
    JSON endpoint; otherwise the public CSV endpoint. Both go through
    _http_get_text (explicit timeout + status check). Returns a float Series
    indexed by date (NaNs dropped)."""
    key = _fred_api_key()
    if key:
        try:
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={key}&file_type=json"
                f"&observation_start={start}"
            )
            obs = json.loads(_http_get_text(url)).get("observations", [])
            s = pd.Series({o["date"]: o["value"] for o in obs})
            s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as e:
            # don't swallow silently — surface the cause, then try CSV fallback
            warnings.warn(
                f"FRED API failed for {series_id} ({type(e).__name__}: {e}); "
                "falling back to public CSV endpoint",
                stacklevel=2,
            )

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    text = _http_get_text(url)
    s = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_gold_monthly(start: str = "1968-01-01") -> pd.Series:
    """Monthly nominal gold (USD/oz) from the public-domain datasets.io gold
    price dataset (Measuring Worth / LBMA), 1833→present. Used in place of the
    discontinued FRED LBMA series. Returns a float Series at month-end whose
    ``.attrs`` carry source_url / sha256 / n_rows for reproducibility auditing
    (the upstream is a moving `main` branch, so the content hash is the audit
    handle — a changed snapshot is visible rather than silent)."""
    text = _http_get_text(GOLD_CSV_URL)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != EXPECTED_GOLD_SHA256:
        raise ValueError(
            "gold CSV sha256 mismatch — pinned source content changed.\n"
            f"  url      : {GOLD_CSV_URL}\n"
            f"  expected : {EXPECTED_GOLD_SHA256}\n"
            f"  got      : {digest}\n"
            "If this is an intentional data refresh, bump GOLD_CSV_COMMIT and "
            "EXPECTED_GOLD_SHA256 together."
        )
    df = pd.read_csv(io.StringIO(text))
    missing = {"Date", "Price"} - set(df.columns)
    if missing:
        raise ValueError(
            f"gold CSV missing columns: {sorted(missing)} (got {list(df.columns)}); "
            f"source schema may have changed: {GOLD_CSV_URL}"
        )
    s = pd.Series(df["Price"].values, index=pd.PeriodIndex(df["Date"], freq="M").to_timestamp("M"))
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s.index >= pd.Timestamp(start)]
    s.attrs = {"source_url": GOLD_CSV_URL, "sha256": digest, "n_rows": int(len(df))}
    return s


def _parse_month_boundary(x: str, side: str) -> pd.Timestamp:
    """Parse a date string. If only year-month (or year) is given, snap to the
    month's start (side="start") or month-end (side="end") so that
    ``end="2025-12"`` *includes* 2025-12-31 rather than excluding the month."""
    ts = pd.Timestamp(x)
    if len(str(x).split("-")) < 3:  # no explicit day → snap to month edge
        per = pd.Period(ts, freq="M")
        return per.to_timestamp("M") if side == "end" else per.to_timestamp()
    return ts


def _to_monthly(s: pd.Series, how: str = "mean", fill_period: str = "Q") -> pd.Series:
    """Resample a series to month-end. ``mean`` for noisy daily levels/rates,
    ``last`` for end-of-period stocks, ``ffill`` for lower-frequency series.

    For ``ffill`` the value of the last observation is carried to the END of
    its native period (``fill_period``, default "Q"), so e.g. a Q4 GDP/debt
    reading dated 2025-10-01 fills Oct/Nov/Dec rather than only Oct."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    if how == "ffill":
        start = s.index.min().to_period("M").to_timestamp("M")
        # extend the grid to the month-end of the last obs's full period (quarter)
        end = s.index.max().to_period(fill_period).to_timestamp(how="end").normalize()
        end = end.to_period("M").to_timestamp("M")
        idx = pd.date_range(start, end, freq="ME")
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
    # clean machine-readable splice cutoff so callers can split the real-rate
    # series into spliced-full vs clean-TIPS subsamples for the I(d) double read.
    notes["real_rate_tips_start"] = (
        tips_start.date().isoformat() if tips_start is not None else "n/a"
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
    # ln transforms (real rate can be negative → not logged). A price/ratio that
    # is <=0 is a data error; mask it to NaN (so it drops out of the tests)
    # rather than producing -inf/NaN, and record how many were dropped.
    nonpositive = {}
    for col in ["gold_nominal", "debt_gdp", "m2_gdp", "fed_gdp"]:
        bad = df[col].notna() & (df[col] <= 0)
        if bool(bad.any()):
            nonpositive[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
        df[f"ln_{col}"] = np.log(df[col])

    # enforce the [start, end] contract on the final panel — injected fetchers
    # may not honor `start`, and lower-freq ffill grids can predate it.
    # year-month inputs snap to month edges (end="2025-12" includes 2025-12-31).
    if not df.empty:
        df = df[df.index >= _parse_month_boundary(start, "start")]
        if end is not None:
            df = df[df.index <= _parse_month_boundary(end, "end")]

    # provenance / coverage notes
    def _cov(s: pd.Series) -> str:
        sv = s.dropna()
        if len(sv) == 0:
            return "no observations (n=0)"
        return f"{sv.index.min().date()}..{sv.index.max().date()} (n={len(sv)})"

    notes["units"] = "debt & fed_assets rescaled $M→$B; gdp,m2 already $B."
    notes["anchor_definition"] = (
        "anchors = stock / NOMINAL GDP (FRED GDP, $B SAAR) — conventional "
        "debt-to-GDP / M2-to-GDP / Fed-to-GDP ratios (nominal over nominal)."
    )
    notes["frequency"] = (
        "monthly (ME). gold/monthly→last; daily rates→mean; weekly stocks→last; "
        "quarterly (debt, gdp)→ffill within quarter."
    )
    fed_raw = raw["fed_assets"].dropna()
    fed_start = fed_raw.index.min().date() if len(fed_raw) else "n/a"
    notes["fed_gdp_coverage"] = (
        f"WALCL starts {fed_start} "
        "(2002+; effectively a post-2008 anchor — short sample, use with care)."
    )
    notes["coverage"] = "; ".join(f"{c}:{_cov(df[c])}" for c in df.columns)
    gattrs = getattr(raw.get("gold_nominal", pd.Series(dtype=float)), "attrs", {})
    notes["gold_source"] = (
        f"{gattrs.get('source_url', 'n/a')} sha256={gattrs.get('sha256', 'n/a')} "
        f"n_rows={gattrs.get('n_rows', 'n/a')} (moving main branch — hash is the "
        "reproducibility handle; a changed snapshot is auditable, not silent)."
    )
    if nonpositive:
        notes["nonpositive_dropped"] = "; ".join(f"{k}:{v}" for k, v in nonpositive.items())

    return AnchorPanel(data=df, notes=notes)


# ── Unit-root tests (step 0) ───────────────────────────────────────────
def unit_root_tests(series: pd.Series, regression: str = "c", min_obs: int = 12) -> Dict[str, float]:
    """Run ADF + PP + KPSS on a series (levels). Returns stats & p-values.

    ADF/PP null = unit root (I(1)); KPSS null = stationarity (I(0)).
    Validates the input (finite, enough non-NaN points, non-constant) and
    raises a descriptive ValueError instead of letting statsmodels/arch fail
    with a cryptic error.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from arch.unitroot import PhillipsPerron

    x = pd.Series(series).dropna().astype(float)
    name = getattr(series, "name", None)
    if len(x) < min_obs:
        raise ValueError(f"unit_root_tests: series {name!r} has {len(x)} obs (<{min_obs})")
    if not np.isfinite(x.values).all():
        raise ValueError(f"unit_root_tests: series {name!r} contains non-finite values")
    if x.nunique() <= 1:
        raise ValueError(f"unit_root_tests: series {name!r} is constant")
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


def _is_stationary(tests: Dict[str, float], alpha: float = 0.05) -> bool:
    """Strict I(0) test: ADF & PP both reject a unit root AND KPSS fails to
    reject stationarity."""
    return (
        tests["adf_pvalue"] < alpha
        and tests["pp_pvalue"] < alpha
        and tests["kpss_pvalue"] >= alpha
    )


def classify_integration(series: pd.Series, alpha: float = 0.05,
                         regression: str = "c") -> Dict[str, object]:
    """Classify a series as I(0) / I(1) / ambiguous by combining ADF, PP, KPSS
    on the levels and (for I(1) confirmation) the first difference.

    ``regression`` ("c" = constant, "ct" = constant+trend) is applied to the
    LEVELS test. The first difference is always tested with a constant ("c"),
    since a genuinely I(1) series differenced is mean-stationary, not trending.

    Verdict logic on levels:
      * ADF & PP reject unit root (p<alpha) AND KPSS fails to reject
        stationarity (p>alpha) → I(0).
      * ADF & PP fail to reject AND KPSS rejects → tentatively I(1), but only
        kept if the first difference is itself stationary (ADF & PP reject +
        KPSS not reject). Otherwise (e.g. I(2) / structural break) → ambiguous,
        so it is not fed into Johansen.
      * disagreement / all-fail (low power) → "ambiguous".
    """
    lvl = unit_root_tests(series, regression=regression)
    adf_rej = lvl["adf_pvalue"] < alpha
    pp_rej = lvl["pp_pvalue"] < alpha
    kpss_rej = lvl["kpss_pvalue"] < alpha

    if _is_stationary(lvl, alpha):
        verdict = "I(0)"
    elif (not adf_rej) and (not pp_rej) and kpss_rej:
        verdict = "I(1)"  # tentative — confirmed via the difference below
    else:
        verdict = "ambiguous"

    res: Dict[str, object] = {"verdict": verdict, "levels": lvl, "regression": regression}
    # confirm I(1): the first difference must itself look stationary (same
    # strict ADF+PP+KPSS rule, constant only). If not, the I(1) label is downgraded.
    diff = pd.Series(series).dropna().diff().dropna()
    if len(diff) > 12:
        d = unit_root_tests(diff, regression="c")
        diff_stationary = _is_stationary(d, alpha)
        res["diff"] = d
        res["diff_stationary"] = bool(diff_stationary)
        if verdict == "I(1)" and not diff_stationary:
            res["verdict"] = "ambiguous"  # not a clean I(1): I(2)/break suspect
    return res


def combined_verdict(series: pd.Series, alpha: float = 0.05, min_obs: int = 20) -> Dict[str, object]:
    """I(d) verdict using BOTH regressions: 'c' (constant) and 'ct'
    (constant+trend). Agreement → that verdict; disagreement → 'ambiguous';
    invalid input (constant/non-finite/too short) → 'invalid data'. Returns
    {combined, c, ct, c_res, ct_res}."""
    if int(pd.Series(series).notna().sum()) < min_obs:
        return {"combined": "insufficient data", "c": None, "ct": None,
                "c_res": None, "ct_res": None}
    out = {}
    for reg in ("c", "ct"):
        try:
            out[reg] = classify_integration(series, alpha, regression=reg)
        except ValueError:
            out[reg] = None
    vc = out["c"]["verdict"] if out["c"] else "invalid data"
    vct = out["ct"]["verdict"] if out["ct"] else "invalid data"
    if vc == "invalid data" or vct == "invalid data":
        combined = "invalid data"
    elif vc == vct:
        combined = vc
    else:
        combined = "ambiguous"
    return {"combined": combined, "c": vc, "ct": vct, "c_res": out["c"], "ct_res": out["ct"]}


def integration_table(df: pd.DataFrame, columns, min_obs: int = 20) -> pd.DataFrame:
    """I(0)/I(1) verdict table for the given columns, run under BOTH 'c' and
    'ct' level regressions. ``verdict`` = the combined (agree-or-ambiguous)
    verdict; ``adf_p``/``pp_p``/``kpss_p`` are the 'c' p-values (with ``*_ct``
    counterparts). Columns with <min_obs non-NaN obs → "insufficient data";
    constant/non-finite columns → "invalid data" (no crash)."""
    rows = []
    for col in columns:
        n_obs = int(df[col].notna().sum())
        cv = combined_verdict(df[col], min_obs=min_obs)
        row = {"series": col, "verdict": cv["combined"], "verdict_c": cv["c"],
               "verdict_ct": cv["ct"], "n": n_obs}
        c_res, ct_res = cv["c_res"], cv["ct_res"]
        for tag, res in (("", c_res), ("_ct", ct_res)):
            lvl = res["levels"] if res else None
            row[f"adf_p{tag}"] = round(lvl["adf_pvalue"], 4) if lvl else np.nan
            row[f"pp_p{tag}"] = round(lvl["pp_pvalue"], 4) if lvl else np.nan
            row[f"kpss_p{tag}"] = round(lvl["kpss_pvalue"], 4) if lvl else np.nan
        row["diff_stationary"] = c_res.get("diff_stationary") if c_res else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def integration_segments(series, segments: Dict[str, tuple], alpha: float = 0.05,
                         min_obs: int = 20) -> Dict[str, Dict[str, object]]:
    """Run the dual-regression (c + ct) I(d) verdict on named sub-windows of a
    single series. ``segments`` maps a label → (start, end) bounds (either may
    be None for open-ended). Returns label → {combined, c, ct, n, start, end}
    — lightweight, JSON-serializable fields only (the full c_res/ct_res test
    objects are intentionally NOT returned; callers only need the verdicts).

    Used to read the long-end real rate on two regimes separately: the full
    spliced series (GS10−CPI proxy pre-TIPS + DFII10) vs the clean post-TIPS
    DFII10 subsample. PR #1 judged the (full) real rate I(0) and dropped it from
    the cointegrating vector; the split makes that verdict auditable per segment
    rather than letting the splice break drive a single ambiguous label.
    """
    s = pd.Series(series).dropna().astype(float)
    out: Dict[str, Dict[str, object]] = {}
    for name, bounds in segments.items():
        start, end = bounds
        seg = s
        if start is not None:
            seg = seg[seg.index >= pd.Timestamp(start)]
        if end is not None:
            seg = seg[seg.index <= pd.Timestamp(end)]
        cv = combined_verdict(seg, alpha=alpha, min_obs=min_obs)
        out[name] = {
            "combined": cv["combined"], "c": cv["c"], "ct": cv["ct"],
            "n": int(len(seg)),
            "start": (seg.index.min().date().isoformat() if len(seg) else None),
            "end": (seg.index.max().date().isoformat() if len(seg) else None),
        }
    return out


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

    cols = list(columns)
    n = len(cols)
    if n < 2:
        raise ValueError(f"johansen_test needs >=2 columns, got {cols}")
    if alpha not in _CV_COL:
        raise ValueError(f"alpha must be one of {sorted(_CV_COL)}, got {alpha}")

    sub = df[cols].dropna()
    min_obs = max(30, k_ar_diff + 5)
    if len(sub) < min_obs:
        raise ValueError(
            f"johansen_test needs >={min_obs} complete rows for {cols}, "
            f"got {len(sub)} after dropna"
        )
    if not np.isfinite(sub.values).all():
        raise ValueError(f"johansen_test got non-finite values in {cols}")

    res = coint_johansen(sub.values, det_order, k_ar_diff)
    cvc = _CV_COL[alpha]

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
    # don't hold), which is NOT "the anchor holds". Flag it and force the
    # cointegration rank to 0 so callers that key off coint_rank/valid_coint
    # never read full-rank as a valid relation.
    full_rank_stationary = (raw_trace_rank == n) or (raw_maxeig_rank == n)
    trace_rank = min(raw_trace_rank, n - 1)
    maxeig_rank = min(raw_maxeig_rank, n - 1)
    coint_rank = 0 if full_rank_stationary else min(trace_rank, maxeig_rank)
    valid_coint = coint_rank >= 1

    # cointegrating vector = first eigenvector, normalized on gold (col 0)
    vec = res.evec[:, 0]
    if abs(vec[0]) < 1e-12:
        raise ValueError(
            f"johansen_test: cannot normalize cointegrating vector on gold — "
            f"near-zero loading {vec[0]:.2e} for {cols} (n_obs={len(sub)})"
        )
    norm = vec / vec[0]
    # G = α + β·A  ⇒  G - β·A ~ I(0); evec row encodes G + c_i·A_i, so the
    # long-run coefficient on anchor i is β_i = -c_i (moved to the RHS). For a
    # bivariate system β = β_1; for the trivariate anchor [gold, debt, real
    # rate] `betas` carries (β_debt, β_real). β/βs are interpretable ONLY when
    # coint_rank == 1: with rank>1 the single eigenvector is not unique, so the
    # first column is an arbitrary basis vector — report None, not a misleading β.
    # β interpretable only for a VALID (valid_coint rules out full-rank-stationary)
    # and UNIQUE (coint_rank == 1; rank>1 → first eigenvector is a non-unique
    # basis vector) cointegrating relation.
    interpretable = valid_coint and (coint_rank == 1) and (n >= 2)
    beta = float(-norm[1]) if interpretable else None
    betas = [float(-norm[i]) for i in range(1, n)] if interpretable else None

    return {
        "columns": cols,
        "trace_stat": [float(x) for x in res.lr1],
        "trace_cv": [float(x) for x in res.cvt[:, cvc]],
        "maxeig_stat": [float(x) for x in res.lr2],
        "maxeig_cv": [float(x) for x in res.cvm[:, cvc]],
        "raw_trace_rank": int(raw_trace_rank),
        "raw_maxeig_rank": int(raw_maxeig_rank),
        "trace_rank": int(trace_rank),
        "maxeig_rank": int(maxeig_rank),
        "coint_rank": int(coint_rank),
        "rank": int(coint_rank),  # back-compat alias for coint_rank
        "valid_coint": bool(valid_coint),
        "full_rank_stationary": bool(full_rank_stationary),
        "coint_vector_normalized": [float(x) for x in norm],
        "beta": beta,
        "betas": betas,
        "alpha_level": alpha,
        "k_ar_diff": int(k_ar_diff),
        "det_order": int(det_order),
        "n_obs": int(len(sub)),
    }


def select_var_order(df: pd.DataFrame, columns, max_lags: int = 4) -> Dict[str, int]:
    """Select the VAR(p) level order on the complete-case (pairwise dropna)
    subsample by AIC, and derive the Johansen lagged-difference count
    k_ar_diff = p - 1 (statsmodels coint_johansen uses lagged *differences*,
    so VAR(1) -> k_ar_diff=0). Returns {var_order, k_ar_diff, n_obs}."""
    from statsmodels.tsa.api import VAR

    sub = df[list(columns)].dropna()
    if len(sub) < max_lags + 10:
        return {"var_order": 1, "k_ar_diff": 0, "n_obs": int(len(sub))}
    sel = VAR(sub.values).select_order(maxlags=max_lags)
    p = int(getattr(sel, "aic", 1) or 1)
    p = max(1, p)
    return {"var_order": p, "k_ar_diff": max(0, p - 1), "n_obs": int(len(sub))}


def johansen_robustness(df: pd.DataFrame, columns, lags=(1, 2, 3, 4),
                        det_orders=(-1, 0, 1), alpha: float = 0.05) -> pd.DataFrame:
    """Run Johansen across a grid of k_ar_diff (lags) × det_order and report
    coint_rank / valid_coint / β per cell. This is the robustness answer to
    'does the rank verdict survive lag/deterministic-term choices?'."""
    rows = []
    for det in det_orders:
        for lag in lags:
            rec = {"det_order": det, "k_ar_diff": lag}
            try:
                j = johansen_test(df, columns, det_order=det, k_ar_diff=lag, alpha=alpha)
                rec.update(coint_rank=j["coint_rank"], valid_coint=j["valid_coint"],
                           beta=(None if j["beta"] is None else round(j["beta"], 3)),
                           betas=(None if j["betas"] is None else [round(b, 3) for b in j["betas"]]),
                           n_obs=j["n_obs"])
            # trivariate systems hit singular/non-PD matrices more often → a bad
            # cell must degrade to an `error` row, never crash the whole grid.
            except (ValueError, np.linalg.LinAlgError) as e:
                rec.update(coint_rank=None, valid_coint=None, beta=None,
                           betas=None, n_obs=None, error=f"{type(e).__name__}: {str(e)[:40]}")
            rows.append(rec)
    return pd.DataFrame(rows)


# ── VECM (step 2: long-run vector + short-run error correction) ─────────
# det_order (Johansen) → statsmodels VECM `deterministic` string. Valid VECM
# strings are {"n","co","ci","lo","li"} (and combinations); -1 (no deterministic)
# is "n" — NOT "nc" (undocumented, fragile). det_order=0 (constant restricted to
# the CE) ↔ "ci"; 1 (linear trend) ↔ "cili" (constant + linear trend inside CE).
_DET_ORDER_TO_VECM = {-1: "n", 0: "ci", 1: "cili"}


def estimate_vecm(
    df: pd.DataFrame,
    columns,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    det_order: int = 0,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Estimate a VECM and split the anchor into its long-run and short-run
    pieces — the step-2 question "is the real rate IN the anchor or in the
    deviation?".

    Returns, for the gold equation (row 0):
      * ``betas`` — long-run cointegrating coefficients β_i = -beta_i (gold
        normalized to 1), with t/p values → β2 (real rate) significant ⇒ the
        real rate is part of the long-run anchor.
      * ``ec_speed`` (λ = α on gold) + t/p → λ<0 & significant ⇒ genuine error
        correction back to the anchor.
      * ``short_run`` — Δ coefficients in the gold equation (per var × lag) with
        t/p → a significant short-run Δreal_rate ⇒ the real rate drives the
        deviation around the anchor.

    Requires ``k_ar_diff>=1`` so at least one short-run difference block exists
    (a VAR(1)→k_ar_diff=0 Johansen lag carries no Δ dynamics; the caller bumps
    it to 1 for the VECM and notes the bump).
    """
    from statsmodels.tsa.vector_ar.vecm import VECM

    cols = list(columns)
    n = len(cols)
    if n < 2:
        raise ValueError(f"estimate_vecm needs >=2 columns, got {cols}")
    if k_ar_diff < 1:
        raise ValueError(
            f"estimate_vecm needs k_ar_diff>=1 for short-run terms, got {k_ar_diff}"
        )
    # single-anchor decomposition: we report ONE long-run vector (β1/β2) + one
    # error-correction loading λ, which is only meaningful when the cointegrating
    # space is 1-D. rank>1 → multiple vectors, no unique "anchor" → caller must
    # handle separately (mirrors the johansen_test rank>1 → β=None contract).
    if coint_rank != 1:
        raise ValueError(
            f"estimate_vecm currently supports only coint_rank==1 (single anchor "
            f"vector); got {coint_rank}. rank>1 has no unique β1/β2 to report."
        )
    if det_order not in _DET_ORDER_TO_VECM:
        raise ValueError(f"det_order must be one of {sorted(_DET_ORDER_TO_VECM)}")

    sub = df[cols].dropna()
    min_obs = max(40, n * (k_ar_diff + 3))
    if len(sub) < min_obs:
        raise ValueError(
            f"estimate_vecm needs >={min_obs} complete rows for {cols}, "
            f"got {len(sub)} after dropna"
        )
    if not np.isfinite(sub.values).all():
        raise ValueError(f"estimate_vecm got non-finite values in {cols}")

    deterministic = _DET_ORDER_TO_VECM[det_order]
    res = VECM(sub.values, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
               deterministic=deterministic).fit()

    # cointegrating vector: first `n` rows are the variable loadings (any
    # deterministic terms come after). statsmodels already normalizes the first
    # `coint_rank` rows of β to the identity, so for rank==1 res.beta[0]==1 and
    # the reported t/p line up with the coefficients as-is. Assert that before
    # reusing t/p — if a future statsmodels stops auto-normalizing, β would
    # become a ratio and the raw t/p would no longer correspond (P2).
    beta_vec = np.asarray(res.beta)[:n, 0].astype(float)
    if abs(beta_vec[0] - 1.0) > 1e-6:
        raise ValueError(
            f"estimate_vecm: VECM β not normalized to gold=1 (got {beta_vec[0]:.4g}); "
            "reusing statsmodels t/p would be inconsistent — refit/normalize needed."
        )
    beta_norm = beta_vec  # already gold-normalized (beta_vec[0]==1)
    tb = np.asarray(res.tvalues_beta)[:n, 0].astype(float)
    pb = np.asarray(res.pvalues_beta)[:n, 0].astype(float)
    betas = {}
    for i in range(1, n):
        # β is the raw VECM loading negated (moved to the RHS): G = -Σ β_i·X_i.
        # Negate the t-stat too so its sign matches the reported β (t=coef/se,
        # se>0 → sign flips with the coef). The two-sided p-value is unchanged.
        betas[cols[i]] = {
            "beta": float(-beta_norm[i]),
            "t": float(-tb[i]),
            "p": float(pb[i]),
            "significant": bool(pb[i] < alpha),
        }

    # deterministic term(s) restricted to the cointegration relation (the ECT
    # intercept/trend), so the user can reconstruct ECT = G - Σβ_i·X_i + c.
    # "ci" → one constant; "cili" → constant + linear trend; "n" → none.
    dcc = getattr(res, "det_coef_coint", None)
    dcc = np.asarray(dcc) if dcc is not None else None
    coint_det = [float(x) for x in dcc[:, 0]] if (dcc is not None and dcc.size) else None

    ec_speed = {
        "lambda": float(np.asarray(res.alpha)[0, 0]),
        "t": float(np.asarray(res.tvalues_alpha)[0, 0]),
        "p": float(np.asarray(res.pvalues_alpha)[0, 0]),
    }
    ec_speed["significant"] = bool(ec_speed["p"] < alpha)
    ec_speed["corrects"] = bool(ec_speed["lambda"] < 0 and ec_speed["significant"])

    # short-run Γ for the gold equation (row 0). Γ is (n, n*k_ar_diff) ordered
    # [lag1: var0..var_{n-1}, lag2: ...].
    gamma = np.asarray(res.gamma)
    tg = np.asarray(res.tvalues_gamma)
    pg = np.asarray(res.pvalues_gamma)
    short_run = []
    for lag in range(1, k_ar_diff + 1):
        for i, c in enumerate(cols):
            j = (lag - 1) * n + i
            short_run.append({
                "var": c, "lag": lag,
                "coef": float(gamma[0, j]),
                "t": float(tg[0, j]),
                "p": float(pg[0, j]),
                "significant": bool(pg[0, j] < alpha),
            })

    return {
        "columns": cols,
        "coint_rank": int(coint_rank),
        "k_ar_diff": int(k_ar_diff),
        "det_order": int(det_order),
        "deterministic": deterministic,
        "alpha_level": float(alpha),
        "n_obs": int(len(sub)),
        "beta_normalized": [float(x) for x in beta_norm],
        "betas": betas,
        "coint_det": coint_det,
        "ec_speed": ec_speed,
        "short_run": short_run,
    }


# ── Gregory-Hansen cointegration with a single endogenous break (step 3) ─
# PR #3. PR #1 (bivariate Johansen) and PR #2 (trivariate Johansen) both fail to
# find a STABLE constant-parameter cointegrating relation. The next falsifiable
# question (spec §2 cross-validation): is the anchor *segmented* — i.e. does a
# cointegrating relation exist once we allow ONE structural break (level shift /
# regime shift) at an unknown date? Gregory & Hansen (1996, J. Econometrics 70)
# answer this with a residual-based test whose null is "no cointegration" and
# whose alternative is "cointegration with a regime shift at an unknown break".
#
# Procedure (GH 1996):
#   For every candidate break index in the trimmed interior τ∈[trim, 1−trim]:
#     fit the cointegrating regression with a break dummy φ_t = 1{t > [Tτ]}:
#       model "C"  (level shift)  : y1 = μ1 + μ2·φ + αᵀy2 + e
#       model "C/T"(level+trend)  : y1 = μ1 + μ2·φ + β·t + αᵀy2 + e
#       model "C/S"(regime shift) : y1 = μ1 + μ2·φ + α1ᵀy2 + α2ᵀ(y2·φ) + e
#     then test the residual e for a unit root (ADF*, Phillips Zt*, Zα*).
#   The GH statistic for each flavor = the SMALLEST (most negative) statistic
#   across all break points; the argmin is the estimated break. Reject "no
#   cointegration" when the GH statistic < the GH critical value (which is more
#   negative than the standard ADF/PP CVs because we minimized over breaks).
#
# Critical values: Gregory & Hansen (1996) Table 1, indexed by model and m =
# number of I(1) regressors (m=1 bivariate, m=2 trivariate). ADF* and Zt* share
# one CV table; Zα* has its own.
_GH_CV = {
    "C": {  # level shift
        "adf_zt": {
            1: {0.01: -5.13, 0.05: -4.61, 0.10: -4.34},
            2: {0.01: -5.44, 0.05: -4.92, 0.10: -4.69},
            3: {0.01: -5.77, 0.05: -5.28, 0.10: -5.02},
            4: {0.01: -6.05, 0.05: -5.56, 0.10: -5.31},
        },
        "zalpha": {
            1: {0.01: -50.07, 0.05: -40.48, 0.10: -36.19},
            2: {0.01: -57.28, 0.05: -47.96, 0.10: -43.22},
            3: {0.01: -63.64, 0.05: -53.58, 0.10: -48.65},
            4: {0.01: -70.27, 0.05: -59.40, 0.10: -54.38},
        },
    },
    "C/T": {  # level shift with trend
        "adf_zt": {
            1: {0.01: -5.45, 0.05: -4.99, 0.10: -4.72},
            2: {0.01: -5.80, 0.05: -5.29, 0.10: -5.03},
            3: {0.01: -6.05, 0.05: -5.57, 0.10: -5.33},
            4: {0.01: -6.36, 0.05: -5.83, 0.10: -5.59},
        },
        "zalpha": {
            1: {0.01: -57.01, 0.05: -47.65, 0.10: -43.34},
            2: {0.01: -64.77, 0.05: -53.92, 0.10: -48.94},
            3: {0.01: -70.15, 0.05: -59.76, 0.10: -54.94},
            4: {0.01: -76.10, 0.05: -65.44, 0.10: -60.12},
        },
    },
    "C/S": {  # regime shift (level + slope)
        "adf_zt": {
            1: {0.01: -5.47, 0.05: -4.95, 0.10: -4.68},
            2: {0.01: -5.97, 0.05: -5.50, 0.10: -5.23},
            3: {0.01: -6.51, 0.05: -6.00, 0.10: -5.75},
            4: {0.01: -6.92, 0.05: -6.41, 0.10: -6.17},
        },
        "zalpha": {
            1: {0.01: -57.17, 0.05: -47.04, 0.10: -41.85},
            2: {0.01: -68.21, 0.05: -58.33, 0.10: -52.85},
            3: {0.01: -80.15, 0.05: -68.94, 0.10: -63.42},
            4: {0.01: -90.84, 0.05: -78.87, 0.10: -72.75},
        },
    },
}


def gregory_hansen_min_obs(m: int, max_lag: int = 6) -> int:
    """Minimum complete rows ``gregory_hansen_test`` requires for ``m`` regressors
    and ``max_lag`` ADF augmentation. Exposed so callers gate on the SAME bound
    the test enforces internally (no divergent magic thresholds)."""
    n_params_cs = 2 + 2 * m  # the largest design (C/S)
    return max(40, 4 * (n_params_cs + max_lag))


def _gh_critical_values(model: str, m: int, stat: str) -> Optional[Dict[float, float]]:
    """GH (1996) Table 1 critical values for ``model`` (C / C/T / C/S), ``m``
    I(1) regressors, and ``stat`` ('adf'/'zt' share a table; 'zalpha' its own).
    Returns {0.01,0.05,0.10: cv} or None if m is outside the tabulated 1..4."""
    table = _GH_CV[model]["zalpha" if stat == "zalpha" else "adf_zt"]
    return table.get(m)


def _adf_resid_stat(resid: np.ndarray, max_lag: int) -> tuple:
    """ADF* t-statistic on cointegrating residuals — NO deterministic term
    (regression='n'), since the residuals are mean-zero by OLS construction.
    Lag chosen by AIC up to ``max_lag`` (GH use a data-dependent ADF lag).
    Returns (stat, used_lag)."""
    from statsmodels.tsa.stattools import adfuller

    res = adfuller(np.asarray(resid, dtype=float), maxlag=max_lag,
                   regression="n", autolag="AIC")
    return float(res[0]), int(res[2])


def _phillips_z_resid(resid: np.ndarray, bandwidth: Optional[int] = None) -> tuple:
    """Phillips (1987) Zα and Zt unit-root statistics for the residual AR(1)
    e_t = ρ·e_{t-1} + u_t (NO intercept — residuals are mean-zero by OLS).

    Long-run variance λ̂² uses a Bartlett kernel; default bandwidth is the
    Newey-West rule l = floor(4·(n/100)^{2/9}). Formulas follow Hamilton (1994)
    §17.6 (Phillips-Perron, no-deterministic case):

        Zα = n(ρ̂−1) − ½(λ̂²−γ̂₀)·n²/M
        Zt = √(γ̂₀/λ̂²)·t_ρ − ((λ̂²−γ̂₀)/(2λ̂))·(n/√M)

    where M = Σ e_{t-1}², γ̂₀ = (1/n)Σû², t_ρ = (ρ̂−1)√M/s_ols.
    Returns (z_alpha, z_t, rho, bandwidth)."""
    e = np.asarray(resid, dtype=float)
    y, ylag = e[1:], e[:-1]
    M = float(ylag @ ylag)
    if M <= 0:
        raise ValueError("phillips_z: degenerate residual (Σe_{t-1}²≤0)")
    rho = float(y @ ylag) / M
    u = y - rho * ylag
    n = len(u)
    ssr = float(u @ u)
    if ssr <= 0 or n < 3:
        raise ValueError("phillips_z: degenerate residual autoregression")
    gamma0 = ssr / n
    s_ols = np.sqrt(ssr / (n - 1))
    t_rho = (rho - 1.0) * np.sqrt(M) / s_ols
    if bandwidth is None:
        bandwidth = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    bandwidth = max(0, int(bandwidth))
    lam2 = gamma0
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1.0)
        cov = float(u[j:] @ u[:-j]) / n
        lam2 += 2.0 * w * cov
    if lam2 <= 0:
        lam2 = gamma0  # guard: kernel can dip ≤0 in tiny samples
    z_alpha = n * (rho - 1.0) - 0.5 * (lam2 - gamma0) * n * n / M
    z_t = (np.sqrt(gamma0 / lam2) * t_rho
           - ((lam2 - gamma0) / (2.0 * np.sqrt(lam2))) * (n / np.sqrt(M)))
    return float(z_alpha), float(z_t), float(rho), int(bandwidth)


def _gh_design(y2: np.ndarray, k: int, model: str) -> np.ndarray:
    """Build the GH regressor matrix for a break AFTER row index ``k`` (0-based;
    dummy φ_t = 1 for t > k). ``y2`` is (T, m) of I(1) regressors.
      C   : [1, φ, y2]
      C/T : [1, φ, t, y2]
      C/S : [1, φ, y2, φ·y2]"""
    T = y2.shape[0]
    phi = (np.arange(T) > k).astype(float)
    const = np.ones(T)
    if model == "C":
        return np.column_stack([const, phi, y2])
    if model == "C/T":
        trend = np.arange(1, T + 1, dtype=float)
        return np.column_stack([const, phi, trend, y2])
    if model == "C/S":
        return np.column_stack([const, phi, y2, y2 * phi[:, None]])
    raise ValueError(f"gregory_hansen: unknown model {model!r} (use C / C/T / C/S)")


def gregory_hansen_test(
    df: pd.DataFrame,
    y_col: str,
    x_cols,
    model: str = "C/S",
    trim: float = 0.15,
    max_lag: int = 6,
    bandwidth: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Gregory-Hansen (1996) residual-based cointegration test allowing ONE
    endogenous structural break.

    H0: no cointegration. H1: cointegration with a level/regime shift at an
    unknown break. For each candidate break in the trimmed interior the
    cointegrating regression is refit with a break dummy and the residual is
    tested for a unit root; the GH statistic is the minimum (most negative)
    ADF*/Zt*/Zα* over all breaks, with the argmin giving the estimated break.

    Parameters
    ----------
    y_col   : dependent (ln gold).
    x_cols  : I(1) regressor column(s); len == m (1 bivariate, 2 trivariate).
    model   : "C" (level shift), "C/T" (level+trend), or "C/S" (regime shift).
    trim    : interior fraction excluded at each end (GH default 0.15).
    max_lag : max ADF* augmentation lag (AIC chooses ≤ this).
    bandwidth: Bartlett bandwidth for Zα*/Zt* (None → Newey-West rule).
    alpha   : reported reject flag uses this level (CVs for all levels returned).

    Returns a dict with, per statistic, the min value, the break index/date, the
    GH critical values, and a reject flag; plus the cointegrating-vector
    coefficients at the ADF*-optimal break (pre/post for a regime shift)."""
    x_cols = list(x_cols)
    m = len(x_cols)
    if m < 1:
        raise ValueError("gregory_hansen_test needs >=1 regressor in x_cols")
    if model not in _GH_CV:
        raise ValueError(f"model must be one of {sorted(_GH_CV)}, got {model!r}")
    if not (0.0 < trim < 0.5):
        raise ValueError(f"trim must be in (0, 0.5), got {trim}")
    # CVs are tabulated only at 1/5/10% — index without an implicit KeyError.
    if alpha not in (0.01, 0.05, 0.10):
        raise ValueError(
            f"alpha must be one of {{0.01, 0.05, 0.10}} (GH Table 1 levels), got {alpha}")

    cols = [y_col] + x_cols
    sub = df[cols].dropna()
    # break_date reporting needs a datetime index; validate up front, not via a
    # late AttributeError deep in _pack().
    if not isinstance(sub.index, pd.DatetimeIndex):
        raise ValueError(
            "gregory_hansen_test requires a DatetimeIndex (break dates are "
            f"reported from it); got {type(sub.index).__name__}")
    if not np.isfinite(sub.values).all():
        raise ValueError(f"gregory_hansen_test got non-finite values in {cols}")
    T = len(sub)
    # need enough rows so each regime can fit the design and the ADF* lags.
    n_params_cs = 2 + 2 * m  # the largest design (C/S)
    min_obs = gregory_hansen_min_obs(m, max_lag)
    if T < min_obs:
        raise ValueError(
            f"gregory_hansen_test needs >={min_obs} complete rows for {cols}, "
            f"got {T} after dropna"
        )

    y = sub[y_col].to_numpy(dtype=float)
    y2 = sub[x_cols].to_numpy(dtype=float)
    idx = sub.index

    lo = int(np.floor(trim * T))
    hi = int(np.ceil((1.0 - trim) * T)) - 1
    # keep a margin so both regimes have rows for the design + ADF lags
    margin = n_params_cs + max_lag + 2
    lo = max(lo, margin)
    hi = min(hi, T - margin - 1)
    if hi <= lo:
        raise ValueError(
            f"gregory_hansen_test: trimmed break window empty (T={T}, trim={trim})"
        )

    best = {
        "adf": {"stat": np.inf, "k": None, "lag": None},
        "zt": {"stat": np.inf, "k": None},
        "zalpha": {"stat": np.inf, "k": None},
    }
    n_breaks = 0
    n_failed = 0
    for k in range(lo, hi + 1):
        X = _gh_design(y2, k, model)
        # OLS via lstsq. A rank-deficient design (collinear break regressors at
        # extreme breaks) does NOT raise — lstsq returns a least-norm solution on
        # an unidentified model. Detect it via the returned rank and skip, so an
        # unidentified break never enters the min-statistic search.
        try:
            coef, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            n_failed += 1
            continue
        if rank < X.shape[1]:
            n_failed += 1
            continue
        resid = y - X @ coef
        try:
            adf_stat, used_lag = _adf_resid_stat(resid, max_lag)
            za, zt, _rho, _bw = _phillips_z_resid(resid, bandwidth)
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1
            continue
        n_breaks += 1
        if adf_stat < best["adf"]["stat"]:
            best["adf"] = {"stat": adf_stat, "k": k, "lag": used_lag}
        if zt < best["zt"]["stat"]:
            best["zt"] = {"stat": zt, "k": k}
        if za < best["zalpha"]["stat"]:
            best["zalpha"] = {"stat": za, "k": k}

    if n_breaks == 0:
        raise ValueError(
            "gregory_hansen_test: every candidate break regression failed "
            f"(T={T}, model={model})"
        )

    def _pack(stat_key: str) -> Dict[str, object]:
        b = best[stat_key]
        cv = _gh_critical_values(model, m, "zalpha" if stat_key == "zalpha" else "adf")
        stat_val = float(b["stat"])
        reject = bool(cv is not None and stat_val < cv[alpha])
        out = {
            "stat": stat_val,
            "break_index": (None if b["k"] is None else int(b["k"])),
            "break_date": (None if b["k"] is None else idx[b["k"]].date().isoformat()),
            "break_fraction": (None if b["k"] is None else round((b["k"] + 1) / T, 3)),
            "critical_values": cv,
            "reject_no_coint": reject,
        }
        if "lag" in b:
            out["adf_lag"] = b["lag"]
        return out

    results = {s: _pack(s) for s in ("adf", "zt", "zalpha")}

    # cointegrating-vector coefficients at the ADF*-optimal break (the headline
    # statistic). For C/S report pre/post regime slopes; for C/C-T the slope is
    # constant and only the intercept shifts.
    coint = None
    k_star = best["adf"]["k"]
    if k_star is not None:
        Xs = _gh_design(y2, k_star, model)
        coef, _res, rank_s, _sv = np.linalg.lstsq(Xs, y, rcond=None)
    else:
        rank_s = 0
    if k_star is not None and rank_s >= Xs.shape[1]:
        mu1 = float(coef[0])
        mu2 = float(coef[1])
        if model == "C/S":
            betas_pre = [float(c) for c in coef[2:2 + m]]
            betas_post = [float(coef[2 + m + i] + coef[2 + i]) for i in range(m)]
        elif model == "C/T":
            betas_pre = betas_post = [float(c) for c in coef[3:3 + m]]
        else:  # C
            betas_pre = betas_post = [float(c) for c in coef[2:2 + m]]
        coint = {
            "x_cols": x_cols,
            "intercept_pre": mu1,
            "intercept_post": mu1 + mu2,
            "betas_pre": betas_pre,
            "betas_post": betas_post,
        }

    return {
        "model": model,
        "y_col": y_col,
        "x_cols": x_cols,
        "m": int(m),
        "n_obs": int(T),
        "trim": float(trim),
        "alpha_level": float(alpha),
        "start": idx.min().date().isoformat(),
        "end": idx.max().date().isoformat(),
        "n_breaks_evaluated": int(n_breaks),
        "n_breaks_failed": int(n_failed),
        "adf": results["adf"],
        "zt": results["zt"],
        "zalpha": results["zalpha"],
        "coint_vector": coint,
        "any_reject": bool(results["adf"]["reject_no_coint"]
                           or results["zt"]["reject_no_coint"]
                           or results["zalpha"]["reject_no_coint"]),
        "cv_available": bool(_gh_critical_values(model, m, "adf") is not None),
    }
