"""Walk-forward (expanding-window) re-calibration of the gold-vs-de-dollarization
deviation — PR #15, a *cheap veto-style honesty check* on PR #14.

PR #14 (``lib.gold_dedollar_gap``) builds a de-dollarization index (DI), regresses
ln(gold) on it with a rolling OLS, and ranks the **latest** residual against its
own history with a **full-sample** z-score / percentile (``full_zscore`` /
``full_percentile``). That headline read — "gold sits at the ~88th percentile of
its deviation vs the fundamentals" — is **in-sample by construction**: every
historical month is ranked using the *entire* 2014→2026 distribution, i.e. data a
contemporaneous observer did not yet have.

This module changes ONLY the calibration口径. It re-ranks the **same** PR #14
residual series with an **expanding window**: at month *t* the mean/std (z) and the
cumulative rank (percentile) use only ``[start, t]`` — the history actually
available at *t*. It touches neither the DI construction nor the rolling-OLS fit
(those are imported / consumed as given). It is an **ex-post descriptive** honesty
audit, NOT a forecast and NOT a new signal.

What an expanding window can and cannot fix
-------------------------------------------
* FIXES — *calibration look-ahead*: a 2018 reading no longer "knows" the larger
  2020-2024 residuals when it is ranked. The historical "extreme" labels and the
  "extreme → forward return" table are re-derived ex-ante, so the conclusion
  "history shows high deviation mean-reverts" can be checked for hindsight.
* DOES NOT FIX — the DI's own hard-data limits (post-2010 / short sample /
  trending inputs). Those are data constraints PR #14 already flags; walk-forward
  calibration is orthogonal to them and this PR does not claim to resolve them.

A structural fact to read before quoting the "current" number
-------------------------------------------------------------
The *latest* month is the **end** of the expanding window, so "expanding up to now"
== "full sample" there. With the default ``exclude_current=False`` (include-current)
the current percentile/z are therefore **identical** to PR #14's by construction.
Dropping the single newest point under ``exclude_current=True`` moves the
**percentile** only by O(1/N) — but **NOT necessarily the z-score**: the newest
point is part of the mean/variance it is being standardized against, so removing it
can shift ``z_wf_excl`` materially (or to NaN if the prior window is near-constant)
when the latest residual is itself extreme. Report ``z_wf_excl − z_full`` directly
rather than assuming it is ~O(1/N). **The headline current reading is still NOT
where look-ahead can hide** — it is the *historical* percentile assignments (used
to call past episodes "extreme") that an expanding window actually corrects. The
verdict is adjudicated on those, not on a manufactured move in today's number.

Two calibration conventions (both reported)
-------------------------------------------
* ``exclude_current=False`` (default, "include-current") mirrors PR #14's
  descriptive ``full_percentile`` (value ranked **within** its own history,
  ``<=`` including itself) — the apples-to-apples expanding analogue. At the final
  month it coincides with the full-sample read. **The report's §3/§4 historical
  reread use this口径.**
* ``exclude_current=True`` ("strict ex-ante") ranks today only against history
  **strictly before** today (the baseline that already existed). **The report's §1
  current-verdict number shows this口径** as the more conservative read. Both
  conventions are leak-free (neither uses the future); strict is just stricter.

Everything is computed with explicit loops over only data ``<= t`` so the
leak-free property is transparent (and unit-tested): truncating the future leaves
every past value unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Reused from PR #14 — NOT re-derived. forward_log_return enforces a calendar-month
# (gap-safe) forward shift; full_* are the in-sample baselines we compare against.
from lib.gold_dedollar_gap import (
    DEFAULT_HORIZONS,
    DEFAULT_TOP_Q,
    forward_log_return,
    full_percentile,
    full_zscore,
)

# Warm-up band: an expanding window is unstable when the history is short, so a
# calibration only emits once at least this many observations have accrued.
DEFAULT_WARMUP = 24
WARMUP_BAND = (12, 24, 36)


# ── expanding-window calibrators (ex-ante by construction) ───────────────────
def _validate_warmup(min_periods: int) -> None:
    if min_periods < 2:
        raise ValueError(f"min_periods (warm-up) must be >= 2, got {min_periods}")


def _require_time_sorted(s: pd.Series) -> None:
    """The expanding calibrators iterate in positional order and treat earlier
    positions as 'history'. A non-time-sorted index would let a future month sit
    in an earlier position and leak into a past calibration — silently breaking the
    no-look-ahead contract (codex PR#15 P2). A duplicate timestamp is just as bad:
    ``full_percentile_series`` would dict-collapse same-index rows and
    ``current_walk_forward_reading`` would crash on a non-scalar ``.loc[asof]``
    (codex PR#15 R2 P2). Reject both loudly instead."""
    if not s.index.is_monotonic_increasing:
        raise ValueError(
            "series index must be monotonic increasing for walk-forward "
            "calibration (sort_index() first); positional iteration treats earlier "
            "rows as history, so an out-of-order index would leak the future.")
    if not s.index.is_unique:
        raise ValueError(
            "series index must be unique for walk-forward calibration; duplicate "
            "timestamps collapse the percentile baseline and break the asof read.")


def expanding_zscore(
    s: pd.Series, *, min_periods: int = DEFAULT_WARMUP, exclude_current: bool = False
) -> pd.Series:
    """Expanding-window z-score: ``z_t = (s_t - mean_{<=t}) / std_{<=t}`` where the
    mean/std use **only** the non-NaN observations in ``[start, t]`` (or
    ``[start, t)`` when ``exclude_current``). Ex-ante by construction — no value at
    *t* depends on any observation after *t*.

    NaN until ``min_periods`` (warm-up) non-NaN observations have accrued in the
    baseline; NaN where ``s_t`` itself is NaN (gaps preserved); NaN where the
    baseline is degenerate (std == 0, no information / no div-by-0).

    ``exclude_current=True`` ranks today against history **strictly before** today
    (the stricter ex-ante read); the default includes today in its own baseline,
    mirroring PR #14's descriptive convention so the two口径 are directly
    comparable (and coincide at the final month)."""
    _validate_warmup(min_periods)
    _require_time_sorted(s)
    vals = s.astype(float).to_numpy()
    out = np.full(len(vals), np.nan, dtype="float64")
    seen: List[float] = []
    for i, v in enumerate(vals):
        base_before = seen  # history strictly before t (live reference is fine: we
        #                     snapshot length/array below before/after the append)
        finite = np.isfinite(v)
        if exclude_current:
            base = list(base_before)
            if finite:
                seen.append(v)
        else:
            if finite:
                seen.append(v)
            base = list(seen)
        if finite and len(base) >= min_periods:
            arr = np.asarray(base, dtype="float64")
            sd = float(arr.std(ddof=0))
            if sd > 0:
                out[i] = (v - float(arr.mean())) / sd
    return pd.Series(out, index=s.index, name="z_wf")


def expanding_percentile(
    s: pd.Series, *, min_periods: int = DEFAULT_WARMUP, exclude_current: bool = False
) -> pd.Series:
    """Expanding-window percentile rank ∈[0,1]: fraction of the non-NaN history in
    ``[start, t]`` (or ``[start, t)`` when ``exclude_current``) that is ``<= s_t``.
    Ex-ante by construction — uses only data ``<= t``.

    NaN until ``min_periods`` non-NaN observations have accrued; NaN where ``s_t``
    is NaN. ``include`` convention (default) mirrors PR #14's ``full_percentile``
    (``<=`` counts the point itself) so at the final month it equals the
    full-sample percentile; ``exclude_current=True`` ranks today only against the
    prior history (strict ex-ante 'was today unprecedented?')."""
    _validate_warmup(min_periods)
    _require_time_sorted(s)
    vals = s.astype(float).to_numpy()
    out = np.full(len(vals), np.nan, dtype="float64")
    seen: List[float] = []
    for i, v in enumerate(vals):
        finite = np.isfinite(v)
        if exclude_current:
            base = list(seen)
            if finite:
                seen.append(v)
        else:
            if finite:
                seen.append(v)
            base = list(seen)
        if finite and len(base) >= min_periods:
            arr = np.asarray(base, dtype="float64")
            out[i] = float((arr <= v).mean())
    return pd.Series(out, index=s.index, name="pct_wf")


# ── full-sample (in-sample) baseline trajectory ──────────────────────────────
def full_percentile_series(s: pd.Series) -> pd.Series:
    """Full-sample percentile of *every* point within the WHOLE history
    (``pct_full_t = mean(all_history <= s_t)``). This is PR #14's descriptive,
    in-sample read applied across the trajectory — it 'peeks' at the entire sample
    (including each point's future) and is the line the expanding calibration is
    measured against. Non-finite values (NaN AND ±inf) are excluded so this口径
    matches the expanding calibrators' ``np.isfinite`` filter — otherwise an inf in
    the residual would be ranked by full-sample but dropped by walk-forward, making
    the two口径 inconsistent (codex PR#15 R3 P3)."""
    _require_time_sorted(s)
    sv = s[np.isfinite(s.astype(float))]
    if sv.empty:
        return pd.Series(np.nan, index=s.index, name="pct_full")
    arr = sv.to_numpy()
    pct = {idx: float((arr <= val).mean()) for idx, val in sv.items()}
    return pd.Series(pct, name="pct_full").reindex(s.index)


# ── side-by-side trajectory frame ────────────────────────────────────────────
def walk_forward_calibration(
    resid: pd.Series,
    *,
    warmup: int = DEFAULT_WARMUP,
    exclude_current: bool = False,
) -> pd.DataFrame:
    """Full-sample vs walk-forward calibration of the SAME residual series, as a
    side-by-side time-series frame (one row per month on ``resid``'s index).

    Columns:
      * ``resid``      the PR #14 rolling-OLS residual (unchanged input)
      * ``z_full``     full-sample z-score (in-sample; PR #14's headline口径)
      * ``pct_full``   full-sample percentile of each point vs the whole history
      * ``z_wf``       expanding-window z-score (ex-ante)
      * ``pct_wf``     expanding-window percentile (ex-ante)
      * ``pct_gap``    pct_wf − pct_full (how much hindsight moved the rank; +ve =
                       the point looked *more* extreme in real time than in-sample)

    The frame is the data behind the report's "对照线" and the CSV for Show Page."""
    resid = resid.rename("resid")
    z_full = full_zscore(resid).rename("z_full")
    pct_full = full_percentile_series(resid)
    z_wf = expanding_zscore(resid, min_periods=warmup, exclude_current=exclude_current)
    pct_wf = expanding_percentile(
        resid, min_periods=warmup, exclude_current=exclude_current
    )
    frame = pd.concat([resid, z_full, pct_full, z_wf, pct_wf], axis=1)
    frame["pct_gap"] = frame["pct_wf"] - frame["pct_full"]
    return frame


# ── current-reading comparison ───────────────────────────────────────────────
@dataclass
class WalkForwardReading:
    asof: Optional[pd.Timestamp]
    # full-sample (PR #14 headline口径)
    z_full: float
    pct_full: float
    # walk-forward, include-current (apples-to-apples; == full at the final month)
    z_wf_incl: float
    pct_wf_incl: float
    # walk-forward, exclude-current (strict ex-ante: today vs prior history only)
    z_wf_excl: float
    pct_wf_excl: float
    n_resid: int = 0          # defined residuals in history (sample depth)
    n_wf: int = 0             # defined residuals available through asof (the
    #                           include-current baseline size; the strict ex-ante /
    #                           exclude-current baseline is one fewer, n_wf - 1)
    warmup: int = DEFAULT_WARMUP


def current_walk_forward_reading(
    resid: pd.Series, *, warmup: int = DEFAULT_WARMUP
) -> WalkForwardReading:
    """The headline comparison: at the latest month with a defined residual, the
    deviation's rank under (a) PR #14's full-sample口径, (b) expanding-window
    include-current, (c) expanding-window exclude-current.

    Honest structural note baked into the dataclass docstring/usage: (a) and (b)
    coincide at the final month by construction (the window ends *now*); (c) drops
    only the single newest point, so its **percentile** moves by ~1/N — but its
    **z-score** can move more (the dropped point was in the mean/variance it is
    standardized against), so ``z_wf_excl`` is reported directly, not assumed small.
    A *large* percentile gap would be surprising — the look-ahead this PR really
    corrects lives in the historical episode reread, not in today's number."""
    _require_time_sorted(resid)
    rv = resid.dropna()
    if rv.empty:
        return WalkForwardReading(
            asof=None, z_full=np.nan, pct_full=np.nan, z_wf_incl=np.nan,
            pct_wf_incl=np.nan, z_wf_excl=np.nan, pct_wf_excl=np.nan,
            n_resid=0, n_wf=0, warmup=warmup,
        )
    asof = rv.index.max()
    if float(rv.std(ddof=0)) == 0.0 or rv.nunique() < 2:
        # degenerate residual carries no deviation information (mirrors PR #14's
        # flat-residual guard) → blank the rank fields.
        return WalkForwardReading(
            asof=asof, z_full=np.nan, pct_full=np.nan, z_wf_incl=np.nan,
            pct_wf_incl=np.nan, z_wf_excl=np.nan, pct_wf_excl=np.nan,
            n_resid=int(len(rv)), n_wf=int(len(rv)), warmup=warmup,
        )
    latest = float(rv.loc[asof])
    z_incl = expanding_zscore(resid, min_periods=warmup, exclude_current=False)
    p_incl = expanding_percentile(resid, min_periods=warmup, exclude_current=False)
    z_excl = expanding_zscore(resid, min_periods=warmup, exclude_current=True)
    p_excl = expanding_percentile(resid, min_periods=warmup, exclude_current=True)
    return WalkForwardReading(
        asof=asof,
        z_full=float(full_zscore(resid).reindex([asof]).iloc[0]),
        pct_full=full_percentile(rv, latest),
        z_wf_incl=float(z_incl.reindex([asof]).iloc[0]),
        pct_wf_incl=float(p_incl.reindex([asof]).iloc[0]),
        z_wf_excl=float(z_excl.reindex([asof]).iloc[0]),
        pct_wf_excl=float(p_excl.reindex([asof]).iloc[0]),
        n_resid=int(len(rv)),
        n_wf=int(len(rv)),
        warmup=warmup,
    )


def warmup_sensitivity(
    resid: pd.Series, *, warmups: Sequence[int] = WARMUP_BAND
) -> pd.DataFrame:
    """How the walk-forward *current* read responds to the warm-up gate {12,24,36}.

    Since the current month sits at the end of the expanding window, the
    include-current read is warm-up-invariant once the gate is cleared (the full
    history is used either way); the exclude-current read likewise only drops the
    newest point. The table therefore mostly exposes how far back the ex-ante
    trajectory can *start* (``first_wf_date``) and how many observations it spans —
    the sensitivity that matters for the early-history episode reread."""
    rows = []
    for w in warmups:
        rd = current_walk_forward_reading(resid, warmup=w)
        pct_wf = expanding_percentile(resid, min_periods=w, exclude_current=False)
        first = pct_wf.dropna().index.min()
        rows.append({
            "warmup": w,
            "pct_wf_incl": rd.pct_wf_incl,
            "z_wf_incl": rd.z_wf_incl,
            "pct_wf_excl": rd.pct_wf_excl,
            "z_wf_excl": rd.z_wf_excl,
            "first_wf_date": first,
            "n_wf_defined": int(pct_wf.notna().sum()),
        })
    return pd.DataFrame(rows)


# ── historical-extreme reread (the core ex-ante leakage test) ────────────────
@dataclass
class ExtremeReclassification:
    summary: Dict[str, float]
    episodes: pd.DataFrame      # per full-sample-extreme month: pct_full, pct_wf, agree
    notes: Dict[str, str] = field(default_factory=dict)


def extreme_reclassification(
    resid: pd.Series,
    *,
    top_q: float = DEFAULT_TOP_Q,
    warmup: int = DEFAULT_WARMUP,
    exclude_current: bool = False,
) -> ExtremeReclassification:
    """Re-judge PR #14's "historical extreme-high" months ex-ante.

    PR #14 labels a month *extreme* when its **full-sample** percentile ≥ ``top_q``
    (it ranks against the whole 2014→2026 distribution, i.e. data later than the
    month itself). Here we ask the contemporaneous question: at each such month,
    with only the history available **then** (expanding window, warm-up gated),
    would it ALSO have ranked ≥ ``top_q``?

    A high agreement rate ⇒ the "extreme" episodes were extreme in real time too ⇒
    PR #14's extreme narrative is not a hindsight artifact. A low rate ⇒ many
    "extremes" are only extreme with the benefit of the full distribution ⇒ the
    extremity is partly in-sample.

    Returns the per-episode frame (pct_full, pct_wf, ex-ante flag, agreement) and a
    summary (counts + agreement rate). Months whose expanding rank is still in
    warm-up (NaN) are reported separately as ``n_warmup`` — not silently counted as
    disagreements."""
    if not (0.0 < top_q < 1.0):
        raise ValueError(f"top_q must be in (0,1), got {top_q}")
    pct_full = full_percentile_series(resid)
    pct_wf = expanding_percentile(
        resid, min_periods=warmup, exclude_current=exclude_current
    )
    full_extreme = pct_full[pct_full >= top_q].index
    rows = []
    for ts in full_extreme:
        pw = float(pct_wf.reindex([ts]).iloc[0])
        in_warmup = not np.isfinite(pw)
        rows.append({
            "date": ts,
            "pct_full": float(pct_full.loc[ts]),
            "pct_wf": pw,
            "wf_extreme": (np.isfinite(pw) and pw >= top_q),
            "in_warmup": in_warmup,
        })
    episodes = pd.DataFrame(rows)
    n_full = len(full_extreme)
    if n_full == 0:
        summary = {"n_full_extreme": 0, "n_wf_extreme": 0, "n_warmup": 0,
                   "n_evaluable": 0, "agreement_rate": float("nan")}
    else:
        n_warmup = int(episodes["in_warmup"].sum())
        evaluable = episodes[~episodes["in_warmup"]]
        n_eval = int(len(evaluable))
        n_agree = int(evaluable["wf_extreme"].sum())
        summary = {
            "n_full_extreme": n_full,
            "n_wf_extreme": n_agree,
            "n_warmup": n_warmup,
            "n_evaluable": n_eval,
            "agreement_rate": (n_agree / n_eval) if n_eval else float("nan"),
        }
    notes = {
        "top_q": f"{top_q:.2f}",
        "warmup": str(warmup),
        "convention": ("exclude_current (strict ex-ante)" if exclude_current
                       else "include_current (today ranked within its own history)"),
        "reading": ("agreement_rate = share of full-sample-extreme months that an "
                    "expanding (ex-ante) rank also calls extreme; high = robust, "
                    "low = in-sample artifact. warm-up months excluded from the rate."),
    }
    return ExtremeReclassification(summary=summary, episodes=episodes, notes=notes)


def _summarize(x: pd.Series) -> Dict[str, float]:
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


def walk_forward_conditional_table(
    resid: pd.Series,
    price: pd.Series,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_q: float = DEFAULT_TOP_Q,
    warmup: int = DEFAULT_WARMUP,
    exclude_current: bool = False,
) -> pd.DataFrame:
    """PR #14's "extreme-high deviation → forward return" table, but with the
    extreme flag computed **ex-ante** (expanding percentile ≥ ``top_q``, warm-up
    gated) instead of a full-sample quantile cut.

    *** Still a conditional DESCRIPTION, NOT a prediction. *** It answers: if a
    real-time observer had flagged "gold extreme vs fundamentals" using only the
    history available then, did weaker forward returns actually follow? This is the
    look-ahead-free version of PR #14's §3 — the result that survives (or not) here
    is the honest one.

    The extreme/rest split per horizon is taken only over months whose forward
    return is observable (mirrors PR #14: the unobservable tail must not bias the
    groups). Returns a long frame [horizon, regime, n, mean, median, p25, p75, hit]
    with regime ∈ {extreme_high_wf, rest}."""
    if not (0.0 < top_q < 1.0):
        raise ValueError(f"top_q must be in (0,1), got {top_q}")
    pct_wf = expanding_percentile(
        resid, min_periods=warmup, exclude_current=exclude_current
    )
    flagged = pct_wf.dropna()
    rows = []
    for h in horizons:
        fwd = forward_log_return(price, h)
        valid = flagged.index.intersection(fwd.dropna().index)
        fv = pct_wf.reindex(valid)
        ext_idx = fv[fv >= top_q].index
        rest_idx = fv[fv < top_q].index
        for regime, idx in (("extreme_high_wf", ext_idx), ("rest", rest_idx)):
            rows.append({"horizon": h, "regime": regime,
                         **_summarize(fwd.reindex(idx))})
    return pd.DataFrame(rows)
