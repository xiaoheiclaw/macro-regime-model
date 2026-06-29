"""Gold de-dollarization *leading-indicator* size modulator (PR #8 — the last cut).

The question (layered on PR #5's S1 pure trend, the standard that already beat
buy-and-hold, and that PR #5-S2 / PR #6 / PR #7 all FAILED to beat): can a
**forward / high-frequency proxy for de-dollarization** add value *on top of* S1
trend — used NOT as an entry signal but as a **size / persistence modulator**?

The user's thesis: when foreign officials (central banks) are pulling back from
USD reserve assets — buying gold instead — a gold up-trend should *persist
longer*, so we should hold a *larger* slice of the S1 trend position; when they
are re-accumulating USD, fade the trend down. The bet is structural: a de-
dollarization proxy might *lead* price (a slow institutional flow) where the
relationship signals of S1-S2 / PR #6-#7 only *coincide* with it.

The honest prior (three prior cuts all lost to S1): any signal built from an
*external relationship* tends to LAG price, because price already discounts the
flow. This module is the cleanest, fastest de-dollarization proxy we can get, so
it is the fairest single test of that intuition — expected to lose, worth one
clean shot to falsify or (unlikely) flip.

Leading proxy — availability, in the task's priority order
----------------------------------------------------------
1. **Foreign official holdings of US Treasuries (preferred, fastest clean).**
   The Federal Reserve's H.4.1 custody series — *Securities Held in Custody for
   Foreign Official and International Accounts: Marketable U.S. Treasury
   Securities* (FRED ``WMTSECL1``, WEEKLY) — is the single cleanest, highest-
   frequency public fingerprint of foreign *central-bank* USD Treasury exposure.
   It is the proxy this module drives off. (The TIC "Major Foreign Holders"
   monthly table and Treasury-auction indirect-bid aggregates are NOT exposed as
   clean single FRED series — documented, not silently skipped.)
2. Treasury-auction indirect-bidder share (monthly aggregate) — no clean single
   series; noted, not used.
3. **(slow fallback) IMF COFER USD reserve share (quarterly)** — the most direct
   de-dollarization measure but lagged/low-frequency; not on FRED as a simple
   series, so noted as the slow alternative, not used.

De-dollarization *strength* is NOT the raw custody level (which grows with the
debt stock). It is the falling **share of total US public debt** held in foreign
official custody: ``share = custody / total_public_debt``. A *declining* share =
foreign officials retreating from USD = de-dollarization. We read its trailing
12m CHANGE (negated, so falling share → positive strength) and turn it into a
leak-free rolling percentile rank ∈[0,1] (0 = weakest, 1 = strongest de-
dollarization in the trailing window) — adaptive, no tuned absolute threshold.

Size modulation (NOT entry)
---------------------------
position = clip( S1_base × f(dedollar_rank), 0, 1 )

f is an *increasing* function (unlike PR #7 dispersion, which cut): weak de-
dollarization → f<1 (trim the trend), neutral → f=1 (= S1), strong → f>1
(amplify, but the 0–100% cap means we never lever — amplification only bites
where vol-targeting already left headroom). Where the rank is undefined (warm-up
OR the proxy series is unavailable) f falls back to NEUTRAL (1.0) → S5 ≡ S1: with
no de-dollarization view we simply hold the base trend (this is the "size
modulation, not entry" philosophy, and the graceful missing-data fallback).

Positions are decided at t and held through t+1 (the shared `run_backtest`
engine applies `.shift(1)`). Long-only 0–100%, vol-targeted, net of cost —
identical machinery and panel legs to S1, so the head-to-head is same-track.

This module reuses `build_anchor_panel` (gold_nominal, PR #1/#2) and
`s1_trend` / sizing (PR #5). It adds two FRED pulls (custody, total debt) via the
same `fetch_fred_series` seam, and touches no PR #1–#7 code.
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
#    over the rank window {36, 60} is reported by the runner) ──
DEFAULT_CHANGE_WINDOW = 12   # months: trailing change of the custody share (YoY)
DEFAULT_RANK_WINDOW = 48     # months: trailing window for the leak-free rank (4y)

# Size factor tiers. f<1 trims the S1 trend, f=1 keeps it (neutral), f>1 amplifies
# (capped by the 0–100% position cap → never leverage). Symmetric around neutral.
DEFAULT_F_MIN = 0.5      # weakest de-dollarization → cut S1 size to 50%
DEFAULT_F_NEUTRAL = 1.0  # mid / no signal → keep S1 unchanged
DEFAULT_F_MAX = 1.5      # strongest de-dollarization → amplify 1.5× (then capped)

# FRED ids this module owns.
DEFAULT_CUSTODY_FRED_ID = "WMTSECL1"  # H.4.1 foreign official custody of marketable
                                      # UST, WEEKLY, $M (start ~2002-12)
DEFAULT_DEBT_FRED_ID = "GFDEBTN"      # total federal public debt, quarterly, $M


def _to_monthly_mean(s: pd.Series) -> pd.Series:
    """Resample a (weekly/daily/monthly) FRED series to a month-end (ME) mean.
    Used for the custody series this module owns — mirrors
    `gold_trend_timing._to_monthly_mean` rather than importing that private name."""
    s = s.sort_index()
    if s.dropna().empty:
        return pd.Series(dtype="float64")
    return s.resample("ME").mean()


DEFAULT_DEBT_PUBLISH_LAG_M = 1  # months after quarter-end the level is treated as known


def _to_monthly_ffill(
    s: pd.Series, publish_lag_months: int = DEFAULT_DEBT_PUBLISH_LAG_M
) -> pd.Series:
    """Resample a quarterly stock (total public debt) to a strictly ex-ante monthly
    series. GFDEBTN is dated at the quarter START (e.g. 2018-01-01) but its value is
    the END-of-quarter level (≈2018-03-31), published shortly after. A naive
    ``resample("ME").last().ffill()`` would stamp that end-of-Q1 level on JANUARY —
    a within-quarter look-ahead (using March-31 data in January) that would taint
    the no-look-ahead backtest.

    Fix: map each observation to its quarter-END month, add a conservative
    `publish_lag_months` publication lag (the level is only *used* once observable),
    then put it on the month-end grid and forward-fill. ffill carries the last KNOWN
    level forward only, and the calendar-based ME grid is independent of where the
    sample ends, so truncating the future leaves every past month unchanged."""
    s = s.sort_index().dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    # quarter-start stamp → quarter-END (the date the level refers to), then lag for
    # publication, then snap to month-end.
    avail = (s.index + pd.offsets.QuarterEnd(0)
             + pd.offsets.DateOffset(months=publish_lag_months)
             + pd.offsets.MonthEnd(0))
    q = pd.Series(s.to_numpy(), index=avail).sort_index()
    q = q[~q.index.duplicated(keep="last")]
    return q.resample("ME").last().ffill()


@dataclass
class DedollarPanel:
    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)


def build_dedollar_panel(
    start: str = "1968-01-01",
    end: Optional[str] = None,
    *,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    anchor_fn: Callable[..., object] = build_anchor_panel,
    custody_id: str = DEFAULT_CUSTODY_FRED_ID,
    debt_id: str = DEFAULT_DEBT_FRED_ID,
) -> DedollarPanel:
    """Assemble the monthly panel the de-dollarization signal reads.

    Reuses `build_anchor_panel` for gold_nominal (PR #1/#2, not re-derived), then
    adds the two FRED pulls this module owns: foreign official custody of UST
    (weekly→ME mean) and total public debt (quarterly→ME ffill). Injection:
    `fetch_fn` covers ONLY this module's two FRED pulls (custody, debt); it is NOT
    forwarded to the anchor, which owns its own (GDP/M2/CPI/TIPS/…) fetchers — so a
    caller stubbing just custody/debt cannot corrupt the anchor's base panel. To
    inject anchor data (e.g. in tests), pass `anchor_fn` (an object exposing `.data`).

    The custody series only starts ~2002-12, so this is structurally a post-2002
    backtest — documented honestly in `notes`. If the custody series is entirely
    unavailable, `custody_share` is all-NaN and the strategy gracefully degrades to
    S1 (see `dedollar_factor`'s NaN→neutral fallback)."""
    base = anchor_fn(start=start, end=end).data  # type: ignore[attr-defined]
    df = base[["gold_nominal"]].copy()
    idx = df.index

    custody = _to_monthly_mean(fetch_fn(custody_id, start)).reindex(idx)
    debt = _to_monthly_ffill(fetch_fn(debt_id, start)).reindex(idx)
    df["foreign_official_custody"] = custody
    df["total_public_debt"] = debt
    # share of the US debt market held in foreign official custody. Both legs are
    # $M so the ratio is dimensionless. NaN where either leg is missing (no fake 0).
    df["custody_share"] = custody / debt

    def _cov(s: pd.Series) -> str:
        sv = s.dropna()
        if len(sv) == 0:
            return "no observations (n=0)"
        return f"{sv.index.min():%Y-%m}..{sv.index.max():%Y-%m} (n={len(sv)})"

    notes: Dict[str, str] = {
        "frequency": "month-end (ME); custody weekly→ME mean, debt quarterly→ME ffill",
        "gold_source": "build_anchor_panel.gold_nominal (datasets.io LBMA)",
        "custody_source": f"{custody_id} (Fed H.4.1 foreign official custody of "
                          f"marketable UST, weekly) — coverage {_cov(custody)}",
        "debt_source": f"{debt_id} (total federal public debt, quarterly $M) — "
                       f"coverage {_cov(debt)}",
        "signal_def": "dedollar strength = −Δ(custody_share) over the trailing "
                      f"{DEFAULT_CHANGE_WINDOW}m (falling foreign-official share of "
                      "the UST market = de-dollarization), ranked over a trailing "
                      f"{DEFAULT_RANK_WINDOW}m window into [0,1] (standard, not tuned)",
        "availability": "primary = foreign official UST custody (fastest clean public "
                        "proxy). TIC 'Major Foreign Holders' monthly & auction "
                        "indirect-bid aggregates are not clean single FRED series; "
                        "IMF COFER USD reserve share is the slow quarterly fallback "
                        "(not on FRED as a simple series). Custody used for speed.",
        "data_constraint": "custody series starts ~2002-12 → this is a post-2002 "
                           "backtest by construction; pre-2003 there is NO leading "
                           "de-dollarization proxy and S5 degrades to S1 (neutral).",
    }
    return DedollarPanel(data=df, notes=notes)


# ── Signal construction (all ex-ante: a value at t uses data ≤ t only) ──────
def custody_share(custody: pd.Series, debt: pd.Series) -> pd.Series:
    """Foreign official custody as a share of total public debt (both $M → ratio).
    NaN where either leg is missing."""
    return custody / debt


def dedollar_strength(
    share: pd.Series, window: int = DEFAULT_CHANGE_WINDOW
) -> pd.Series:
    """De-dollarization *strength* = the NEGATED trailing `window`-month change of
    the custody share: a falling foreign-official share (retreat from USD) → a
    POSITIVE strength. Uses share[t] − share[t-window] (trailing only → ex-ante);
    NaN until `window` months of history exist."""
    if window <= 0:
        # window<=0 makes shift() read current/future data → look-ahead
        raise ValueError(f"window must be a positive integer, got {window}")
    chg = share - share.shift(window)
    return (-chg).rename("dedollar_strength")


def dedollar_rank(
    strength: pd.Series, window: int = DEFAULT_RANK_WINDOW
) -> pd.Series:
    """Leak-free ∈[0,1] rolling rank of de-dollarization strength (trailing
    window): 0 = the trailing-window MIN (weakest de-dollarization), 1 = the MAX
    (strongest). NaN until the window fills, and NaN-preserving on an all-NaN input
    (the missing-data fallback path).

    Ranking (not the raw strength) drives the size factor, so the modulation is
    adaptive to the strength distribution and never depends on a tuned absolute
    threshold. Implementation mirrors PR #7's leak-free rank: a bare
    ``.rolling().rank(pct=True)`` floors at 1/window (not 0), so we rank with
    ``method="min"`` and rescale to a true [0,1]: (raw−1)/(window−1). Trailing
    window → ex-ante."""
    if window < 2:
        # window=1 makes (raw-1)/(window-1) a div-by-0 and is a degenerate rank
        # (the lone point is both min and max → no information).
        raise ValueError(f"window must be >= 2, got {window}")
    roll = strength.rolling(window, min_periods=window)
    raw = roll.rank(method="min")
    rank = ((raw - 1.0) / (window - 1.0)).clip(lower=0.0, upper=1.0)
    # A degenerate window (all values tied → max == min, e.g. a flat custody share)
    # has NO information, yet method="min" would force rank → 0.0 and the factor
    # would read that as the WEAKEST de-dollarization and trim size. Blank it to NaN
    # so `dedollar_factor` falls back to NEUTRAL (no signal → don't modulate).
    rmin = roll.min()
    rmax = roll.max()
    rank[rmax == rmin] = np.nan
    return rank


def signal_available(rank: pd.Series) -> bool:
    """True iff the de-dollarization rank has at least one defined month — i.e. the
    proxy series exists and its windows filled. The runner uses this to switch to an
    explicit S1-only fallback report when the proxy is unavailable."""
    return bool(rank.notna().any())


# ── Size factor + strategy positions (decided at t, held t+1) ───────────────
def dedollar_factor(
    rank: pd.Series,
    *,
    mode: str = "soft",
    f_min: float = DEFAULT_F_MIN,
    f_neutral: float = DEFAULT_F_NEUTRAL,
    f_max: float = DEFAULT_F_MAX,
) -> pd.Series:
    """INCREASING size factor f(rank): weak de-dollarization (rank→0) → f_min
    (trim the trend), neutral → f_neutral (= S1), strong (rank→1) → f_max
    (amplify, later capped by the position's 0–100% cap → never leverage).

    Where `rank` is NaN (warm-up OR the proxy series is unavailable) f falls back
    to **f_neutral** → S5 ≡ S1: with no de-dollarization view we hold the base
    trend unchanged. This single rule serves both warm-up (S5/S1 share an identical
    investable window — fair same-track comparison) and the missing-data fallback.

    soft: PIECEWISE-linear through the neutral knee, so rank=0.5 → f_neutral ALWAYS
          (even for an asymmetric tier set): rank∈[0,0.5] interpolates f_min→f_neutral,
          rank∈[0.5,1] interpolates f_neutral→f_max. Reduces to a plain line when
          f_neutral is the midpoint of [f_min, f_max].
    hard: tercile steps — rank<1/3 → f_min, [1/3,2/3) → f_neutral, ≥2/3 → f_max."""
    if not (f_min <= f_neutral <= f_max):
        # a non-monotone tier set would make "stronger de-dollarization" sometimes
        # cut size — silently inverting the whole hypothesis.
        raise ValueError(
            f"require f_min <= f_neutral <= f_max, got "
            f"({f_min}, {f_neutral}, {f_max})"
        )
    if mode == "soft":
        # two linear segments meeting at (0.5, f_neutral) so the documented
        # "rank 0.5 = neutral" contract holds for any (f_min, f_neutral, f_max).
        lower = f_min + (f_neutral - f_min) * (rank / 0.5)
        upper = f_neutral + (f_max - f_neutral) * ((rank - 0.5) / 0.5)
        f = lower.where(rank <= 0.5, upper)  # NaN rank → NaN here → neutral below
    elif mode == "hard":
        f = pd.Series(np.nan, index=rank.index, dtype="float64")
        r = rank
        f[r < 1.0 / 3.0] = f_min
        f[(r >= 1.0 / 3.0) & (r < 2.0 / 3.0)] = f_neutral
        f[r >= 2.0 / 3.0] = f_max
    else:
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
    # NaN rank (warm-up / missing proxy) → neutral: hold the base trend, don't modulate.
    return f.fillna(f_neutral)


def s5_dedollar(
    panel: pd.DataFrame,
    rank: pd.Series,
    *,
    mode: str = "soft",
    lookbacks=DEFAULT_LOOKBACKS,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    f_min: float = DEFAULT_F_MIN,
    f_neutral: float = DEFAULT_F_NEUTRAL,
    f_max: float = DEFAULT_F_MAX,
) -> pd.Series:
    """S5: S1 pure-trend exposure × an INCREASING de-dollarization size factor,
    clipped to [0,1].

    position = clip( s1_trend(panel) × f(dedollar_rank), 0, 1 )
    Strong de-dollarization → f>1 → size UP (ride the trend longer, but the cap
    means never above 100% — amplification only acts where vol-targeting left
    headroom). Weak → f<1 → size DOWN. No view (NaN rank) → f=neutral → S5 = S1.
    When trend is off (S1=0) S5 is 0 regardless — the factor only modulates the
    SIZE of a trend-on position, never creates a position (size modulation, not
    entry).

    The factor is neutral (not NaN) where the rank is undefined, so S5's only NaN
    is S1's own warm-up → S5 and S1 share an identical investable window (the
    `run_backtest` engine trims the same months for both)."""
    base = s1_trend(panel, lookbacks, target_vol, vol_window)  # [0,1], NaN warm-up
    f = dedollar_factor(
        rank.reindex(panel.index), mode=mode,
        f_min=f_min, f_neutral=f_neutral, f_max=f_max,
    )
    return (base * f).clip(lower=0.0, upper=1.0)
