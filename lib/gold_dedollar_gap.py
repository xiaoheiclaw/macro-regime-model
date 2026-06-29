"""Gold vs de-dollarization fundamentals — *deviation / valuation monitor* (PR #13).

The user's worry (verbatim framing): **has gold run far ahead of the de-
dollarization fundamentals it is supposed to track?** i.e. relative to the
structural de-dollarization bid (central banks buying gold, foreign officials
retreating from USD Treasuries), how expensive is gold *right now*, and is the
2025-2026 hedge being put on at the most expensive point of the narrative?

This module is a **quantitative anchor for a hedging decision — NOT a forecast
and NOT a trading backtest.** It builds a synthetic *de-dollarization index*
(DI) from the same fundamentals the rest of this repo already wires in, then
measures how far the gold price sits *above or below* the level that DI co-moves
with, and where that deviation falls in its own history.

What it reuses (touches no existing code)
-----------------------------------------
* ``lib.gold_cb_flow`` — WGC cumulative **excess** central-bank gold purchases
  (the post-2022 abnormal accumulation stock; PR #10). De-dollarization sign +.
* ``lib.gold_dedollarization_leading.build_dedollar_panel`` — foreign-official
  UST **custody share** = WMTSECL1 / GFDEBTN (PR #8) and gold_nominal. A
  *falling* share = de-dollarization, so the sign is **−**.
* ``lib.gold_anchor`` — ``fetch_fred_series`` seam for CPI (real-purchasing-power
  leg) and the optional broad-dollar index; ``build_anchor_panel`` underlies the
  dedollar panel's gold leg.

The DI (de-dollarization index)
-------------------------------
DI = weighted sum of z-scored components, each signed so that **larger DI = more
de-dollarization**:

  (a) cb_cum_excess   cumulative excess CB gold buying (tonnes-stock)   sign +
  (b) custody_share   foreign-official UST custody / total debt          sign −
  (c) dxy (optional)  broad USD index level                             sign −

Components are z-scored over their common coverage and equal-weighted by default
(weights are reported and perturbed in the robustness band). A component that is
entirely unavailable is **dropped and the remaining weights renormalized** (the
missing-component fallback) — DI degrades gracefully rather than going all-NaN.

The deviation (two口径, both reported)
-------------------------------------
(a) **nominal**: residual of a *rolling* OLS of ln(gold) on DI, then z-scored —
    "how many SD is gold above the level its recent co-movement with DI implies".
(b) **real / purchasing-power**: same on ln(gold / CPI).

Why a ROLLING regression on top of a stationarity caveat: DI and ln(gold) are
both strongly trending (this repo's PR #11 *placebo* test showed a monotone
trend can manufacture a spurious +121% / R²0.67 against gold). A single full-
sample levels regression of two trending series is exactly that trap. A rolling-
window residual measures *local* deviation from the prevailing relationship, and
we additionally report a **first-difference** stationarity read in the robustness
band. The headline is therefore a *relative-valuation* statement, not a
cointegration claim.

Honesty (read before quoting any number)
----------------------------------------
* DI is a **synthetic composite with subjective weights**; its level/scale is an
  artifact of standardization and component choice.
* The CB-purchase leg is **annual WGC estimates interpolated to monthly** (coarse
  shape; see ``lib.gold_cb_flow``); custody is weekly→monthly; CPI monthly. The
  fundamentals are *low-frequency*, so DI moves slowly and sub-annual deviation
  shape is dominated by the gold price.
* The full-sample z-scores / percentiles used for the headline "where are we vs
  history" read are **in-sample / descriptive** by construction (the whole point
  is to rank today against the whole history). The historical "extreme → forward
  return" table is a **conditional description, NOT a prediction or a causal
  claim**: high deviation ≠ a guaranteed drawdown.
* Common-coverage window: custody starts ~2002-12 and WGC excess buying ~2010, so
  a DI that uses both is structurally a **post-2010** object. N for the forward-
  return conditioning is therefore small — reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lib.gold_anchor import fetch_fred_series
from lib.gold_cb_flow import make_wgc_fn
from lib.gold_dedollarization_leading import build_dedollar_panel

# ── component spec ──────────────────────────────────────────────────────────
# (column, sign) — sign orients each leg so that *larger contribution = more
# de-dollarization*. custody_share is negated (a falling foreign-official share
# of the UST market = retreat from USD); dxy is negated (a weaker dollar trend).
ComponentSpec = Tuple[str, float]
DEFAULT_COMPONENTS: Tuple[ComponentSpec, ...] = (
    ("cb_cum_excess", +1.0),
    ("custody_share", -1.0),
)
DXY_COMPONENT: ComponentSpec = ("dxy", -1.0)

DEFAULT_CPI_ID = "CPIAUCSL"        # CPI-U index, monthly
DEFAULT_DXY_ID = "DTWEXBGS"        # Broad USD index (Goods & Services), daily (2006+)

# rolling-window band reported in the robustness section (months).
DEFAULT_REG_WINDOW = 60
WINDOW_BAND = (36, 60, 120)

# forward-return horizons for the historical-extreme conditional table (months).
DEFAULT_HORIZONS = (12, 24, 36)
# deviation is "extreme high" above this full-sample quantile.
DEFAULT_TOP_Q = 0.90
# headline verdict thresholds (full-sample percentile / z of the latest reading).
EXTREME_PCT = 0.90
ELEVATED_PCT = 0.75
HIGH_Z = 1.5


# ── small stats helpers (kept local; mirror the repo's _zscore / leak-free
#    rank conventions rather than importing private names) ─────────────────
def full_zscore(s: pd.Series) -> pd.Series:
    """Full-sample z-score: (s − mean) / std (ddof=0), NaN-preserving. A constant
    or all-NaN series → all zeros / NaN (no div-by-0). Descriptive / in-sample by
    construction — used for the 'vs whole history' headline reads."""
    sv = s.dropna()
    sd = float(sv.std(ddof=0)) if len(sv) else np.nan
    if not (np.isfinite(sd) and sd > 0):
        return s - s  # preserves NaN positions; constant → 0.0
    return (s - float(sv.mean())) / sd


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling z-score: (s − mean_w) / std_w over the trailing `window`
    (min_periods=window → NaN warm-up). Trailing only → ex-ante. A degenerate
    flat window (std=0) yields NaN (no information / no div-by-0)."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    roll = s.rolling(window, min_periods=window)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (s - mu) / sd.where(sd > 0)
    return z


def full_percentile(s: pd.Series, value: Optional[float] = None) -> float:
    """Fraction of the (non-NaN) history that is ≤ `value` (default: the latest
    non-NaN observation). ∈[0,1]; NaN if the series is empty. Descriptive."""
    sv = s.dropna()
    if sv.empty:
        return float("nan")
    if value is None:
        value = float(sv.iloc[-1])
    return float((sv <= value).mean())


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Leak-free trailing percentile rank ∈[0,1] of `s` within its trailing
    `window` (mirrors the PR #8 dedollar_rank construction): 0 = trailing-window
    min, 1 = max. NaN until the window fills and on all-NaN input. A degenerate
    flat window (max==min, no information) → NaN. Trailing → ex-ante."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    roll = s.rolling(window, min_periods=window)
    raw = roll.rank(method="min")
    rank = ((raw - 1.0) / (window - 1.0)).clip(lower=0.0, upper=1.0)
    rmin, rmax = roll.min(), roll.max()
    rank[rmax == rmin] = np.nan
    return rank


def _to_monthly_last(s: pd.Series) -> pd.Series:
    """Month-end last (for a level index like CPI already ~monthly)."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    return s.resample("ME").last()


def _to_monthly_mean(s: pd.Series) -> pd.Series:
    """Month-end mean (for a daily series like the broad dollar index)."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    return s.resample("ME").mean()


# ── panel assembly ──────────────────────────────────────────────────────────
@dataclass
class GapPanel:
    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)


def build_gap_panel(
    start: str = "2002-12",
    end: Optional[str] = None,
    *,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    dedollar_fn: Callable[..., object] = build_dedollar_panel,
    wgc_fn: Optional[Callable[[str, Optional[str]], pd.Series]] = None,
    include_dxy: bool = False,
    cpi_id: str = DEFAULT_CPI_ID,
    dxy_id: str = DEFAULT_DXY_ID,
) -> GapPanel:
    """Assemble the monthly panel the DI + deviation read from.

    Columns: gold_nominal, cpi, cb_cum_excess, custody_share, [dxy], plus
    ln_gold and ln_gold_real = ln(gold / cpi).

    Reuse / injection:
      * gold_nominal + custody_share + the custody/debt legs come from
        ``build_dedollar_panel`` (PR #8) — not re-derived. Inject ``dedollar_fn``
        (an object exposing ``.data``) in tests to avoid the network.
      * cb_cum_excess comes from ``make_wgc_fn(signal='cum_excess')`` (PR #10);
        inject ``wgc_fn`` to override.
      * ``fetch_fn`` covers ONLY this module's own pulls (CPI, optional DXY); it is
        NOT forwarded to the dedollar panel's own (anchor) fetchers, so a caller
        stubbing CPI cannot corrupt the base gold/custody panel.
    """
    dp = dedollar_fn(start=start, end=end, fetch_fn=fetch_fn)
    base = dp.data  # type: ignore[attr-defined]
    keep = [c for c in ("gold_nominal", "custody_share",
                        "foreign_official_custody", "total_public_debt")
            if c in base.columns]
    df = base[keep].copy()
    idx = df.index

    # CPI (real purchasing-power leg). Monthly index → ME last, reindexed.
    cpi = _to_monthly_last(fetch_fn(cpi_id, start)).reindex(idx)
    df["cpi"] = cpi

    # cumulative excess CB gold buying (the de-dollarization "bid" stock).
    if wgc_fn is None:
        wgc_fn = make_wgc_fn(signal="cum_excess")
    end_str = end if end is not None else (
        idx.max().strftime("%Y-%m") if len(idx) else None)
    cb = wgc_fn(start, end_str)
    df["cb_cum_excess"] = cb.reindex(idx) if cb is not None else np.nan

    if include_dxy:
        dxy = _to_monthly_mean(fetch_fn(dxy_id, start)).reindex(idx)
        df["dxy"] = dxy

    df["ln_gold"] = np.log(df["gold_nominal"].where(df["gold_nominal"] > 0))
    real = df["gold_nominal"] / df["cpi"].where(df["cpi"] > 0)
    df["ln_gold_real"] = np.log(real.where(real > 0))

    def _cov(s: pd.Series) -> str:
        sv = s.dropna()
        return (f"{sv.index.min():%Y-%m}..{sv.index.max():%Y-%m} (n={len(sv)})"
                if len(sv) else "no observations (n=0)")

    notes: Dict[str, str] = {
        "frequency": "month-end (ME). gold/custody from build_dedollar_panel; "
                     "CPI→ME last; DXY→ME mean.",
        "components_source": "cb_cum_excess = WGC cumulative EXCESS official "
                             "purchases (annual→monthly均摊, PR #10 make_wgc_fn); "
                             "custody_share = WMTSECL1/GFDEBTN (PR #8); "
                             f"cpi = {cpi_id}"
                             + (f"; dxy = {dxy_id}" if include_dxy else ""),
        "coverage": "; ".join(f"{c}:{_cov(df[c])}" for c in df.columns
                              if c in ("gold_nominal", "custody_share",
                                       "cb_cum_excess", "cpi")
                              or c == "dxy"),
        "caveat": "cb_cum_excess starts ~2010 (WGC), custody ~2002-12 → a DI using "
                  "both is a post-2010 object; fundamentals are low-frequency so DI "
                  "moves slowly and sub-annual deviation shape is gold-driven.",
    }
    for k, v in getattr(dp, "notes", {}).items():
        notes.setdefault(f"dedollar.{k}", v)
    return GapPanel(data=df, notes=notes)


# ── DI (de-dollarization index) ─────────────────────────────────────────────
@dataclass
class DIResult:
    di: pd.Series
    components: pd.DataFrame          # signed, z-scored, per-component (pre-weight)
    weights: Dict[str, float]         # effective (renormalized) weights actually used
    dropped: List[str]                # components dropped (entirely unavailable)
    notes: Dict[str, str] = field(default_factory=dict)


def build_di(
    panel: pd.DataFrame,
    components: Sequence[ComponentSpec] = DEFAULT_COMPONENTS,
    *,
    weights: Optional[Dict[str, float]] = None,
    min_present: Optional[int] = None,
) -> DIResult:
    """Build the de-dollarization index = weighted sum of signed z-scored
    components.

    * Each component column is z-scored over its own coverage and multiplied by
      its sign (so larger = more de-dollarization).
    * ``weights`` (col→w) default to equal weight; they are renormalized to sum to
      1 over the components actually present (the missing-component fallback).
    * A component that is **entirely NaN** (or absent from the panel) is dropped
      and recorded in ``dropped``; remaining weights renormalize.
    * DI at a row is the weighted mean over the components **non-NaN at that row**,
      requiring at least ``min_present`` of them (default: all remaining
      components → the strict common window). Per-row renormalization keeps DI on
      a comparable scale; rows with fewer than ``min_present`` present → NaN.

    Returns a ``DIResult`` (di series, signed component frame, effective weights,
    dropped list, notes)."""
    if not components:
        raise ValueError("components must be non-empty")
    signed: Dict[str, pd.Series] = {}
    dropped: List[str] = []
    for col, sign in components:
        if col not in panel.columns or panel[col].dropna().empty:
            dropped.append(col)
            continue
        signed[col] = full_zscore(panel[col]) * float(sign)
    if not signed:
        raise ValueError(
            f"no usable components (all of {[c for c, _ in components]} are "
            "absent or all-NaN)")

    present = list(signed.keys())
    if weights is None:
        w = {c: 1.0 for c in present}
    else:
        w = {c: float(weights.get(c, 0.0)) for c in present}
        if sum(w.values()) <= 0:
            raise ValueError(
                f"weights over present components {present} sum to <= 0")
    wsum = sum(w.values())
    w = {c: v / wsum for c, v in w.items()}  # renormalize over present

    comp_df = pd.DataFrame(signed)
    if min_present is None:
        min_present = len(present)
    min_present = max(1, min(min_present, len(present)))

    # per-row weighted mean over present components, renormalizing the weights of
    # the components that are non-NaN at that row.
    wvec = pd.Series(w)
    mask = comp_df.notna()
    n_present = mask.sum(axis=1)
    wmat = mask.mul(wvec, axis=1)               # weight where present, 0 where NaN
    row_wsum = wmat.sum(axis=1)
    weighted = (comp_df.fillna(0.0) * wmat).sum(axis=1)
    di = (weighted / row_wsum.where(row_wsum > 0)).where(n_present >= min_present)

    notes = {
        "definition": "DI = Σ w_i · sign_i · z(component_i); per-row weighted mean "
                      "over present components (renormalized); larger = more "
                      "de-dollarization.",
        "weights": ", ".join(f"{c}={w[c]:.3f}" for c in present),
        "min_present": str(min_present),
    }
    if dropped:
        notes["dropped"] = ("entirely-unavailable components dropped & weights "
                            f"renormalized: {dropped}")
    return DIResult(di=di.rename("DI"), components=comp_df, weights=w,
                    dropped=dropped, notes=notes)


# ── deviation (gold vs DI) ──────────────────────────────────────────────────
def rolling_ols_resid(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Residual of a *rolling* OLS ``y ~ a + b·x`` over the trailing `window`,
    evaluated at the window's last point: resid_t = y_t − (â + b̂·x_t), with
    (â, b̂) fit on [t-window+1, t]. Uses only data ≤ t → ex-ante. NaN in warm-up,
    where either input is NaN, or where the window's x is constant (b unidentified).

    A LOCAL residual (not a single full-sample levels fit) is the deliberate guard
    against the spurious-trend regression the repo's PR #11 placebo flagged: two
    co-trending series fit globally manufacture a high-R² relationship; the
    rolling residual instead measures deviation from the *prevailing local*
    relationship."""
    if window < 3:
        raise ValueError(f"window must be >= 3 (need slope + intercept), got {window}")
    yv = y.astype(float)
    xv = x.astype(float)
    out = pd.Series(np.nan, index=y.index, dtype="float64")
    yvals = yv.to_numpy()
    xvals = xv.to_numpy()
    n = len(yvals)
    for t in range(window - 1, n):
        sl = slice(t - window + 1, t + 1)
        yi = yvals[sl]
        xi = xvals[sl]
        ok = np.isfinite(yi) & np.isfinite(xi)
        if ok.sum() < 3:
            continue
        yo = yi[ok]
        xo = xi[ok]
        if not (np.isfinite(yvals[t]) and np.isfinite(xvals[t])):
            continue
        xvar = float(np.var(xo))
        if xvar <= 1e-12:  # constant x in window → slope unidentified
            continue
        A = np.column_stack([np.ones_like(xo), xo])
        beta, *_ = np.linalg.lstsq(A, yo, rcond=None)
        pred_t = beta[0] + beta[1] * xvals[t]
        out.iloc[t] = yvals[t] - pred_t
    return out.rename("resid")


@dataclass
class DeviationResult:
    resid: pd.Series          # rolling-OLS residual (ln gold − DI-implied)
    gap_z_roll: pd.Series     # trailing rolling z-score of the residual
    gap_z_full: pd.Series     # full-sample z-score of the residual (descriptive)
    window: int
    notes: Dict[str, str] = field(default_factory=dict)


def compute_deviation(
    y: pd.Series,
    di: pd.Series,
    *,
    window: int = DEFAULT_REG_WINDOW,
) -> DeviationResult:
    """Gold-vs-DI deviation for a given log-price leg `y` (ln_gold or ln_gold_real):
    rolling-OLS residual, plus its trailing rolling z-score (ex-ante) and its
    full-sample z-score (descriptive 'vs whole history'). Positive = gold above
    its DI-implied level (running ahead of the fundamentals)."""
    common = y.dropna().index.intersection(di.dropna().index)
    yc = y.reindex(common)
    dc = di.reindex(common)
    resid = rolling_ols_resid(yc, dc, window).reindex(y.index)
    return DeviationResult(
        resid=resid,
        gap_z_roll=rolling_zscore(resid, window).reindex(y.index),
        gap_z_full=full_zscore(resid),
        window=window,
        notes={
            "definition": "resid = ln(gold) − rolling-OLS(ln gold ~ DI) prediction; "
                          "gap_z = z-score of that residual (+ = gold above its "
                          "DI-implied level). Rolling fit guards vs spurious trend.",
            "window": str(window),
            "n_resid": str(int(resid.notna().sum())),
        },
    )


# ── historical extreme → forward returns (conditional DESCRIPTION) ──────────
def forward_log_return(price: pd.Series, horizon_months: int) -> pd.Series:
    """Forward log return ln(price[t+h] / price[t]). Genuinely forward → NaN in
    the last `h` months (unobservable), so the return leg itself has no
    look-ahead; only the conditioning threshold below is in-sample."""
    if horizon_months <= 0:
        raise ValueError(f"horizon_months must be positive, got {horizon_months}")
    fwd = np.log(price.shift(-horizon_months) / price)
    return fwd.rename(f"fwd_{horizon_months}m")


def _summ(x: pd.Series) -> Dict[str, float]:
    xv = x.dropna()
    if xv.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "p25": np.nan, "p75": np.nan, "hit": np.nan}
    return {
        "n": int(len(xv)),
        "mean": float(xv.mean()),
        "median": float(xv.median()),
        "p25": float(xv.quantile(0.25)),
        "p75": float(xv.quantile(0.75)),
        "hit": float((xv > 0).mean()),
    }


def conditional_forward_table(
    gap: pd.Series,
    price: pd.Series,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_q: float = DEFAULT_TOP_Q,
) -> pd.DataFrame:
    """Descriptive conditional table: for each horizon, the forward-return
    distribution when the deviation was in its **top `top_q` tail** (gold running
    ahead of the fundamentals) vs the **rest** of history.

    *** This is a conditional DESCRIPTION, NOT a prediction or a causal claim. ***
    The threshold ``quantile(gap, top_q)`` is full-sample (in-sample); a high
    deviation has historically often preceded weaker forward returns, but that is
    a description of past co-movement, not a guarantee. N is small (post-2010
    fundamentals) — reported in the table.

    Returns a long DataFrame: columns [horizon, regime, n, mean, median, p25,
    p75, hit]."""
    if not (0.0 < top_q < 1.0):
        raise ValueError(f"top_q must be in (0,1), got {top_q}")
    g = gap.dropna()
    if g.empty:
        return pd.DataFrame(
            columns=["horizon", "regime", "n", "mean", "median", "p25", "p75", "hit"])
    thr = float(g.quantile(top_q))
    extreme_idx = g[g >= thr].index
    rest_idx = g[g < thr].index
    rows = []
    for h in horizons:
        fwd = forward_log_return(price, h)
        for regime, idx in (("extreme_high", extreme_idx), ("rest", rest_idx)):
            s = _summ(fwd.reindex(idx))
            rows.append({"horizon": h, "regime": regime, **s})
    return pd.DataFrame(rows)


# ── current positioning + verdict ───────────────────────────────────────────
@dataclass
class CurrentReading:
    asof: Optional[pd.Timestamp]
    gap_z_full: float          # full-sample z of the latest residual
    gap_pct_full: float        # full-sample percentile of the latest residual ∈[0,1]
    gap_pct_roll: float        # latest leak-free trailing percentile ∈[0,1]
    di_pct_full: float         # where DI itself sits in its history ∈[0,1]


def current_reading(
    dev: DeviationResult,
    di: pd.Series,
    *,
    roll_window: int = DEFAULT_REG_WINDOW,
) -> CurrentReading:
    """Latest deviation positioning: full-sample z + percentile of the latest
    residual, the leak-free trailing percentile, and where DI itself sits."""
    resid = dev.resid.dropna()
    asof = resid.index.max() if not resid.empty else None
    latest = float(resid.iloc[-1]) if not resid.empty else np.nan
    pct_roll = rolling_percentile(dev.resid, roll_window)
    pct_roll_latest = float(pct_roll.dropna().iloc[-1]) if pct_roll.notna().any() else np.nan
    return CurrentReading(
        asof=asof,
        gap_z_full=float(dev.gap_z_full.reindex([asof]).iloc[0]) if asof is not None else np.nan,
        gap_pct_full=full_percentile(resid, latest),
        gap_pct_roll=pct_roll_latest,
        di_pct_full=full_percentile(di.dropna()),
    )


def adjudicate(
    reading: CurrentReading,
    *,
    extreme_pct: float = EXTREME_PCT,
    elevated_pct: float = ELEVATED_PCT,
    high_z: float = HIGH_Z,
) -> Tuple[str, str]:
    """Headline verdict on whether gold is running ahead of the de-dollarization
    fundamentals RIGHT NOW. Returns (label, text).

    Labels: EXTREME (price far ahead — top decile / high z), ELEVATED (above
    normal), NORMAL (within historical range), UNKNOWN (no reading). Honest by
    design: 'extreme' describes valuation richness vs history, NOT a forecast that
    a drawdown must follow."""
    pct = reading.gap_pct_full
    z = reading.gap_z_full
    if not np.isfinite(pct):
        return "UNKNOWN", "No deviation reading available (insufficient overlap)."
    if pct >= extreme_pct or (np.isfinite(z) and z >= high_z):
        label = "EXTREME"
        msg = (f"gold sits in the top {(1 - pct) * 100:.0f}% of its historical "
               f"deviation vs the de-dollarization index (z={z:+.2f}) — price is "
               "running FAR ahead of the fundamentals (most-expensive-narrative "
               "warning). Descriptive valuation read, not a drawdown forecast.")
    elif pct >= elevated_pct:
        label = "ELEVATED"
        msg = (f"deviation at the {pct * 100:.0f}th percentile (z={z:+.2f}) — gold "
               "is above its fundamentals-implied level but not at a historical "
               "extreme.")
    else:
        label = "NORMAL"
        msg = (f"deviation at the {pct * 100:.0f}th percentile (z={z:+.2f}) — gold "
               "is within its normal historical range vs the de-dollarization "
               "fundamentals.")
    return label, msg
