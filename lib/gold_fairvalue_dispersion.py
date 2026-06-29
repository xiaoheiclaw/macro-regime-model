"""Gold multi-model fair-value *dispersion* signal + dispersion-adjusted timing.

The question (layered on PR #5's S1 pure trend, the standard that already beat
buy-and-hold): can the **disagreement across independent fair-value estimators**
add value *on top of* trend — not by trading any single fair value (PR #1–#4
showed every anchor relation is unstable / drifts, so no one implied price is a
tradeable fair value), but by reading the **dispersion** between them?

Key insight (the whole reason this module exists):
  * each single estimator's implied price is *drifty and unreliable* on its own —
    NEVER trade the gap to any one of them (that was PR #1–#4's null result);
  * but the *cross-sectional dispersion* of the implied prices may itself carry
    information — HIGH dispersion = the valuation lenses disagree = no pricing
    consensus / a regime-turn warning; LOW dispersion = consensus / trend
    continuation. We scale S1's trend exposure DOWN when dispersion is high.

Six independent, deliberately non-overlapping lenses (each emits an implied gold
price, calibrated on a TRAILING rolling window only → ex-ante, no future data):
  (a) debt/GDP      — rolling OLS of ln(gold) on ln(debt/GDP)
  (b) real-rate lvl — rolling OLS of ln(gold) on the real-rate LEVEL (the classic
                      −0.9 level relation; rate is absolute, not logged)
  (c) M2/GDP        — rolling OLS of ln(gold) on ln(M2/GDP)
  (d) CPI (Jastram) — rolling OLS of ln(gold) on ln(CPI) (gold should track the
                      CPI multiple — purchasing-power anchor)
  (e) gold/oil      — ratio mean-reversion: implied = trailing-mean ln(gold/oil)
                      × today's oil. Oil = WTI spot (MCOILWTICO), 1986+ only.
  (f) gold/copper   — ratio mean-reversion, copper = BLS PPI copper & products
                      (WPUSI019011), 1967+. Spot copper (PCOPPUSDM) only starts
                      1992 so PPI is used for sample length (documented).

Dispersion is measured on the **ln(implied/market) gaps** (each lens's view of
how mispriced gold is, in %), NOT on the raw implied levels — different anchors
imply structurally different gold *levels* (debt-lens ~$5k vs CPI-lens ~$800),
so a raw-level CV would just measure that structural offset, not the time-varying
"disagreement about the turn". The cross-sectional std of the gaps is shift-
invariant (a constant offset added to every gap leaves it unchanged), so it
isolates genuine cross-lens disagreement.

The dispersion series is turned into a leak-free ∈[0,1] **rolling percentile
rank** (trailing window: 0 = consensus, 1 = max disagreement) which gates S1 via
a decreasing weight f(rank): low dispersion → keep full trend, high → exit.

Positions are decided at t and held through t+1 (the shared `run_backtest`
engine applies `.shift(1)`). Long-only 0–100%, vol-targeted, net of cost —
identical machinery and panel to S1, so the head-to-head is same-track.

This module reuses `build_anchor_panel` (gold + debt_gdp + m2_gdp + real_rate,
PR #1/#2) and `s1_trend` / sizing (PR #5). It adds three FRED pulls (CPI, oil,
copper) via the same `fetch_fred_series` seam, and touches no PR #1–#6 code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from lib.gold_anchor import build_anchor_panel, fetch_fred_series
from lib.gold_trend_timing import (  # noqa: F401  (re-exported for convenience)
    DEFAULT_LOOKBACKS,
    DEFAULT_TARGET_VOL,
    DEFAULT_VOL_WINDOW,
    s1_trend,
)

# ── Standard parameters (NOT tuned — conventional values; a sensitivity band
#    over the dispersion-rank window {60, 120} is reported by the runner) ──
DEFAULT_CALIB_WINDOW = 120    # months: trailing rolling window each estimator
                              # is calibrated on (and the ratio means use)
DEFAULT_DISP_WINDOW = 120     # months: trailing window for the dispersion rank
DEFAULT_MIN_ESTIMATORS = 2    # min non-NaN gaps required to define a dispersion
# Hard-tier weights on the dispersion rank terciles (low/mid/high disagreement).
# Low dispersion → full S1 trend; high → exit. Mid is a half-position.
DEFAULT_TIER_LOW = 1.0
DEFAULT_TIER_MID = 0.5
DEFAULT_TIER_HIGH = 0.0

DEFAULT_CPI_FRED_ID = "CPIAUCSL"        # CPI-U index, monthly, 1949+
DEFAULT_OIL_FRED_ID = "MCOILWTICO"      # WTI spot, monthly, 1986-01+ (pre-1986 NaN)
DEFAULT_COPPER_FRED_ID = "WPUSI019011"  # BLS PPI copper & products, monthly, 1967+


def _to_monthly_mean(s: pd.Series) -> pd.Series:
    """Resample a FRED series (any native freq) to a month-end (ME) mean. Local
    helper for the three pulls this module owns (CPI, oil, copper) — mirrors
    `gold_trend_timing._to_monthly_mean` rather than importing that private name.
    All three are natively monthly, so this collapses duplicates and snaps to the
    gold panel's month-end grid."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    return s.resample("ME").mean()


@dataclass
class DispersionPanel:
    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)


def build_dispersion_panel(
    start: str = "1968-01-01",
    end: Optional[str] = None,
    *,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    anchor_fn: Callable[..., object] = build_anchor_panel,
    cpi_id: str = DEFAULT_CPI_FRED_ID,
    oil_id: str = DEFAULT_OIL_FRED_ID,
    copper_id: str = DEFAULT_COPPER_FRED_ID,
) -> DispersionPanel:
    """Assemble the monthly panel the dispersion estimators read.

    Reuses `build_anchor_panel` for gold_nominal + debt_gdp + m2_gdp +
    real_rate_10y (PR #1/#2, already on main — not re-derived), then adds the
    three extra FRED pulls this module owns (CPI, oil, copper) on the same
    month-end grid. Injection: `fetch_fn` covers all FRED pulls; `anchor_fn` is
    the panel builder (stub it in tests with an object exposing `.data`)."""
    base = anchor_fn(start=start, end=end, fetch_fn=fetch_fn).data  # type: ignore[attr-defined]
    df = base[["gold_nominal", "debt_gdp", "m2_gdp", "real_rate_10y"]].copy()
    idx = df.index

    cpi = _to_monthly_mean(fetch_fn(cpi_id, start)).reindex(idx)
    oil = _to_monthly_mean(fetch_fn(oil_id, start)).reindex(idx)
    copper = _to_monthly_mean(fetch_fn(copper_id, start)).reindex(idx)
    df["cpi"] = cpi
    df["oil"] = oil
    df["copper"] = copper

    def _cov(s: pd.Series) -> str:
        sv = s.dropna()
        if len(sv) == 0:
            return "no observations (n=0)"
        return f"{sv.index.min():%Y-%m}..{sv.index.max():%Y-%m} (n={len(sv)})"

    notes: Dict[str, str] = {
        "frequency": "month-end (ME); CPI/oil/copper resampled to ME mean",
        "gold_source": "build_anchor_panel.gold_nominal (datasets.io LBMA)",
        "debt_gdp_source": "build_anchor_panel (GFDEBTN / GDP, quarterly→ffill)",
        "m2_gdp_source": "build_anchor_panel (M2SL / GDP)",
        "real_rate_source": "build_anchor_panel.real_rate_10y (DFII10 2003+; "
                            "pre-2003 GS10 − trailing-12m CPI YoY splice)",
        "cpi_source": f"{cpi_id} (CPI-U index, monthly) — coverage {_cov(cpi)}",
        "oil_source": f"{oil_id} (WTI spot, monthly) — coverage {_cov(oil)}; "
                      "pre-1986 → oil estimator NaN (spot only from 1986)",
        "copper_source": f"{copper_id} (BLS PPI copper & products, monthly) — "
                         f"coverage {_cov(copper)}; spot (PCOPPUSDM) only 1992+ "
                         "so PPI used for length",
        "calib_window": f"{DEFAULT_CALIB_WINDOW}m trailing per estimator "
                        "(standard, not tuned)",
    }
    return DispersionPanel(data=df, notes=notes)


# ── Estimators (all ex-ante: a value at t uses data ≤ t only) ───────────
def _rolling_ols_implied(ln_y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Implied ln(y) from a trailing-window OLS of ln_y on x (with intercept),
    evaluated at t using x[t] and coefficients fit on [t-window+1, t]. Returns
    exp(·) → the implied *level* of y. NaN until the window fills with non-NaN.

    A single-regressor rolling regression is just rolling means/cov/var:
      b = cov(ln_y, x) / var(x),  a = mean(ln_y) − b·mean(x),  fitted = a + b·x.
    All four rolling pieces use ``min_periods=window`` so the fit is defined only
    on a FULL window of paired observations (no silent partial-window fits that
    would pretend to a fair value on too little data). No future data is read."""
    if window <= 0:
        # window<=0 makes rolling read current/future data → look-ahead
        raise ValueError(f"window must be a positive integer, got {window}")
    m_y = ln_y.rolling(window, min_periods=window).mean()
    m_x = x.rolling(window, min_periods=window).mean()
    cov = ln_y.rolling(window, min_periods=window).cov(x)
    var = x.rolling(window, min_periods=window).var()  # ddof=1
    with np.errstate(invalid="ignore", divide="ignore"):
        b = cov / var                     # NaN where var==0 (flat x over window)
    a = m_y - b * m_x
    fitted = a + b * x                    # predict at t from x[t], coeffs ≤ t
    return np.exp(fitted)


def _ratio_meanrev_implied(ln_y: pd.Series, ln_x: pd.Series, window: int) -> pd.Series:
    """Implied y from gold/commodity *ratio* mean-reversion: the fair gold is
    the trailing-average ratio times today's commodity, i.e.
      ln_implied = mean(ln_y − ln_x)[trailing window] + ln_x[t].
    NaN until the window fills with non-NaN. No future data (trailing mean ≤ t,
    today's commodity known at t)."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    ratio = ln_y - ln_x
    mean_ratio = ratio.rolling(window, min_periods=window).mean()
    return np.exp(mean_ratio + ln_x)


def implied_debt_gdp(
    gold: pd.Series, debt_gdp: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (a): implied gold from a rolling debt/GDP relation (ln–ln)."""
    return _rolling_ols_implied(np.log(gold), np.log(debt_gdp), window)


def implied_real_rate(
    gold: pd.Series, real_rate: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (b): implied gold from the classic real-rate LEVEL relation
    (ln-gold on the real-rate level — the conventional ≈−0.9 slope; rates are
    absolute and can be negative, so NOT logged)."""
    return _rolling_ols_implied(np.log(gold), real_rate, window)


def implied_m2_gdp(
    gold: pd.Series, m2_gdp: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (c): implied gold from a rolling M2/GDP monetary-stock relation."""
    return _rolling_ols_implied(np.log(gold), np.log(m2_gdp), window)


def implied_cpi(
    gold: pd.Series, cpi: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (d): implied gold from a rolling CPI purchasing-power (Jastrow) relation
    — gold should track a multiple of CPI (ln–ln)."""
    return _rolling_ols_implied(np.log(gold), np.log(cpi), window)


def implied_gold_oil(
    gold: pd.Series, oil: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (e): implied gold from gold/oil ratio mean-reversion (WTI spot)."""
    return _ratio_meanrev_implied(np.log(gold), np.log(oil), window)


def implied_gold_copper(
    gold: pd.Series, copper: pd.Series, window: int = DEFAULT_CALIB_WINDOW
) -> pd.Series:
    """Lens (f): implied gold from gold/copper ratio mean-reversion."""
    return _ratio_meanrev_implied(np.log(gold), np.log(copper), window)


# ── Dispersion (cross-lens disagreement at each month) ──────────────────
def estimator_gaps(
    implieds: Dict[str, pd.Series], market: pd.Series
) -> pd.DataFrame:
    """The ln(implied_i / market) gap for each estimator — lens i's view of how
    mispriced gold is, in %, at each month. Columns are estimators, rows months;
    NaN where an estimator is undefined (warm-up / no oil pre-1986 / …).

    Using gaps (not raw implied levels) is deliberate: a raw-level dispersion
    would be dominated by the structural level differences between anchors
    (debt-lens ~$5k vs CPI-lens ~$800) rather than time-varying disagreement."""
    ln_mkt = np.log(market)
    return pd.DataFrame({name: np.log(imp) - ln_mkt for name, imp in implieds.items()})


def dispersion(
    gaps: pd.DataFrame, min_estimators: int = DEFAULT_MIN_ESTIMATORS
) -> pd.Series:
    """Cross-sectional std (ddof=1) of the ln-gaps at each month — the valuation
    DISAGREEMENT. Requires ≥ `min_estimators` non-NaN gaps; else NaN (no honest
    consensus measure on fewer than two lenses).

    Computed by hand (count → mean → sum of squared devs → /(count−1)) rather
    than ``gaps.std(axis=1)`` so the count gate is explicit and auditable, and so
    shift-invariance is obvious: a constant added to every gap moves the mean by
    the same constant and leaves the deviations — hence the std — unchanged.
    That is exactly the property we want: dispersion tracks disagreement, not any
    lens's level bias."""
    if min_estimators < 2:
        # <2 makes a "std" of one point meaningless (ddof=1 → div-by-0); a caller
        # asking for min_estimators<2 is really asking to fabricate a dispersion.
        raise ValueError(f"min_estimators must be >= 2, got {min_estimators}")
    count = gaps.notna().sum(axis=1)
    mean = gaps.mean(axis=1, skipna=True)
    dev = gaps.sub(mean, axis=0)
    ssq = dev.pow(2).sum(axis=1, skipna=True)  # NaN devs drop out via skipna
    with np.errstate(invalid="ignore", divide="ignore"):
        var = ssq / (count - 1)
    std = np.sqrt(var)
    std[count < min_estimators] = np.nan
    std.name = "dispersion"
    return std


def dispersion_rank(
    disp: pd.Series, window: int = DEFAULT_DISP_WINDOW
) -> pd.Series:
    """Leak-free ∈[0,1] rolling percentile rank of dispersion (trailing window):
    0 = dispersion at its trailing-window MIN (consensus), 1 = at its MAX (max
    disagreement). NaN until the window fills.

    The rank (not the raw dispersion) is what gates the strategy, so the gate is
    adaptive to the dispersion distribution and never depends on a tuned absolute
    threshold. `.rolling(window).rank(pct=True)` uses data ≤ t only → ex-ante."""
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    return disp.rolling(window, min_periods=window).rank(pct=True)


def estimator_count(gaps: pd.DataFrame) -> pd.Series:
    """Non-NaN estimator count per month — the (time-varying) lens set, for the
    coverage audit. Rises as oil (1986+) / copper calibration fill in."""
    return gaps.notna().sum(axis=1).rename("n_estimators")


def estimator_coverage(implieds: Dict[str, pd.Series]) -> pd.DataFrame:
    """First/last/count of non-NaN implied prices per estimator — the coverage
    table that documents each lens's sample (oil pre-1986 missing, etc.)."""
    rows = []
    for name, s in implieds.items():
        sv = s.dropna()
        rows.append({
            "estimator": name,
            "first": (sv.index.min() if len(sv) else pd.NaT),
            "last": (sv.index.max() if len(sv) else pd.NaT),
            "n_months": int(len(sv)),
        })
    return pd.DataFrame(rows).set_index("estimator")


# ── Strategy positions (weight to *hold next month*, decided at t) ──────
def _dispersion_weight(
    rank: pd.Series,
    mode: str,
    tiers: tuple[float, float, float],
) -> pd.Series:
    """Decreasing weight f(rank) ∈ [0,1]: low dispersion rank → keep trend, high
    → cut it. NaN where `rank` is NaN (warm-up).

    hard: tercile steps — rank<1/3 → tiers[0] (full), [1/3,2/3) → tiers[1],
          ≥2/3 → tiers[2] (exit).
    soft: f = 1 − rank (linear: 0→1, 0.5→0.5, 1→0)."""
    if mode == "hard":
        low, mid, high = tiers
        w = pd.Series(np.nan, index=rank.index, dtype="float64")
        r = rank
        w[r < 1.0 / 3.0] = low
        w[(r >= 1.0 / 3.0) & (r < 2.0 / 3.0)] = mid
        w[r >= 2.0 / 3.0] = high
        return w
    if mode == "soft":
        return (1.0 - rank).clip(lower=0.0, upper=1.0)
    raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")


def s4_dispersion(
    panel: pd.DataFrame,
    disp_rank: pd.Series,
    *,
    mode: str = "hard",
    lookbacks=DEFAULT_LOOKBACKS,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    tiers: tuple[float, float, float] = (DEFAULT_TIER_LOW, DEFAULT_TIER_MID, DEFAULT_TIER_HIGH),
) -> pd.Series:
    """S4: S1 pure-trend exposure × a dispersion weight, clipped to [0,1].

    position = s1_trend(panel) × f(disp_rank)
    Low dispersion (consensus) → f≈1 → S4 ≈ S1 (ride the trend). High dispersion
    (disagreement / turn warning) → f≈0 → cut trend exposure. When trend is
    already off (S1=0) S4 is 0 regardless — dispersion only modulates how much of
    a *trend-on* position to keep.

    Warm-up stays NaN until BOTH S1's trend/vol window and the dispersion-rank
    window have filled, so `run_backtest` trims those months (no cash stub)."""
    base = s1_trend(panel, lookbacks, target_vol, vol_window)  # [0,1], NaN warm-up
    f = _dispersion_weight(disp_rank.reindex(panel.index), mode, tiers)  # [0,1]
    return (base * f).clip(lower=0.0, upper=1.0)
