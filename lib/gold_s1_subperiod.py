"""S1 trend-timing sub-period decomposition — is the edge *post-2000* still real?

This is a one-vote-kill check on PR#5's S1 ("gold_trend_timing": vol-targeted
pure trend, blend of 3/6/12-month momentum, long-only 0↔100%). S1's full-sample
Sharpe of ~0.63 is widely *attributed to dodging the 1968–2000 bear*; if that is
all it is, then for a trader operating **today** S1 is close to worthless.

The question this module answers, ex-post and descriptively:

    Does S1 still beat buy-and-hold gold (S0) **after 2000** — net of trading
    cost — or has the edge decayed to historical bear-avoidance?

It deliberately *reuses* the leak-free engine from `lib.gold_trend_timing`
(`build_timing_panel`, `s1_trend`, `run_backtest`, `compute_metrics`,
`slice_segment`) and adds only the things PR#5 did not report:

  • finer reporting segments (1968-1980 / 1980-2000 / 2000-2011 / 2011-2015 /
    2016-2026, plus the 2000-2026 combined post-2000 window);
  • *lived-experience* drawdown caliber — longest underwater run (months) and
    longest consecutive-loss streak — to expose whipsaw pain a month-end
    snapshot MaxDD understates;
  • a discrete **trade count** (signal flips, not the continuous vol-target
    rebalances) alongside the annualised turnover;
  • a parameter-robustness panel (blend 2/6/12, single windows L2/L6/L12,
    equal-weight majority vote) so the post-2000 verdict does not hinge on the
    one 3/6/12 blend;
  • a paired in-sample significance read on the monthly net-return difference
    (S1 − S0) over the post-2000 window.

Everything is **in-sample and ex-post** — there is no walk-forward, no parameter
search holdout. A favourable verdict means "worth hardening with walk-forward",
not "proven out-of-sample". An unfavourable verdict is taken at face value.

The module is pure/functional; the network only enters through
`build_timing_panel` (inherited, fetch-injectable) so tests run offline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lib.gold_trend_timing import (
    ANNUAL,
    DEFAULT_TARGET_VOL,
    DEFAULT_VOL_WINDOW,
    compute_metrics,
    s0_buy_hold,
    s1_trend,
    slice_segment,
    trend_exposure,
    vol_scale,
)

# ── Reporting segments (inclusive year bounds) ─────────────────────────────
# Descriptive windows, NOT additive — and DELIBERATELY OVERLAPPING. With
# inclusive bounds the shared boundary years 1980, 2000 and 2011 each fall in
# *both* adjacent windows (e.g. all of 2000 is in 1980-2000 AND 2000-2011).
# That is intentional: these are narrative eras, not a partition, so a month may
# be counted under more than one era. The 2000-2026 row is the only one used for
# the verdict; the rest are for attribution colour.
SUBPERIOD_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("1968-1980", "1968-01-01", "1980-12-31"),  # great 70s bull
    ("1980-2000", "1980-01-01", "2000-12-31"),  # the dead decades (where timing earns its keep)
    ("2000-2011", "2000-01-01", "2011-12-31"),  # the modern bull
    ("2011-2015", "2011-01-01", "2015-12-31"),  # the give-back / whipsaw era
    ("2016-2026", "2016-01-01", "2026-12-31"),  # recent decade
    ("2000-2026", "2000-01-01", "2026-12-31"),  # COMBINED post-2000 (the verdict window)
)

# The combined post-2000 window the kill-condition is adjudicated on.
POST2000_SEGMENT = "2000-2026"

# Cost grid (bps per rebalance) for the sensitivity sweep.
COST_GRID: Tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0)

# The "realistic" cost at which the paired significance is read/displayed, and
# the "punitive" cost the verdict also requires a win at. Both must be grid pts.
PAIRED_DISPLAY_COST: float = 10.0
PUNITIVE_COST: float = 25.0

# S1 variants for the robustness panel: (label, kind, lookbacks).
#   kind "blend" → equal-weight average of the per-lookback on/off signals
#                  (a single-element tuple reproduces a pure single-window S1);
#   kind "vote"  → vol-targeted strict-majority vote of the same signals.
# The 3/6/12 blend is PR#5's headline and the primary variant for the verdict.
S1_VARIANTS: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("blend_3_6_12", "blend", (3, 6, 12)),  # PR#5 default — headline
    ("blend_2_6_12", "blend", (2, 6, 12)),
    ("single_L2", "blend", (2,)),
    ("single_L6", "blend", (6,)),
    ("single_L12", "blend", (12,)),
    ("vote_3_6_12", "vote", (3, 6, 12)),
)
PRIMARY_VARIANT = "blend_3_6_12"
S0_LABEL = "S0_buyhold"
PRIMARY_LABEL = f"S1_{PRIMARY_VARIANT}"


def segment_window(name: str) -> Tuple[str, str]:
    """Look up a segment's (start, end) bounds by name from SUBPERIOD_SEGMENTS,
    so callers (report tables AND the paired significance window) share one
    source of truth — editing the segment constant can't silently desync the
    paired CI sample from the verdict window."""
    for n, s, e in SUBPERIOD_SEGMENTS:
        if n == name:
            return s, e
    raise KeyError(f"unknown segment {name!r}; known: "
                   f"{[n for n, _, _ in SUBPERIOD_SEGMENTS]}")


# ── Majority-vote S1 (the only new signal; blends reuse s1_trend) ──────────
def s1_majority_vote(
    panel: pd.DataFrame,
    lookbacks: Sequence[int] = (3, 6, 12),
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Vol-targeted *strict-majority* trend: hold (sized by the vol-target
    multiplier) when **more than half** of the lookback momentum signals are
    positive, else cash. A tie (exactly half long, only possible with an even
    count) is conservatively treated as *no* majority → cash.

    Warm-up stays NaN (inherited from `trend_exposure`'s ``skipna=False`` blend)
    so `run_backtest` trims it rather than crediting a cash stub — identical
    contract to `s1_trend`, so the two are directly comparable."""
    te = trend_exposure(panel["gold_nominal"], lookbacks)  # mean of {0,1}, NaN warm-up
    vote = (te > 0.5).astype(float)
    vote[te.isna()] = np.nan  # preserve the warm-up NaN (te>0.5 would be False)
    vs = vol_scale(panel["gold_ret"], target_vol, vol_window)
    return (vote * vs).clip(lower=0.0, upper=1.0)


def variant_position(
    panel: pd.DataFrame,
    kind: str,
    lookbacks: Sequence[int],
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Dispatch a variant spec to its position series. ``blend`` reuses PR#5's
    `s1_trend` verbatim (single-element lookbacks = a pure single window);
    ``vote`` uses the majority-vote signal above."""
    if kind == "blend":
        return s1_trend(panel, lookbacks, target_vol, vol_window)
    if kind == "vote":
        return s1_majority_vote(panel, lookbacks, target_vol, vol_window)
    raise ValueError(f"unknown variant kind {kind!r} (expected 'blend' or 'vote')")


def build_positions(panel: pd.DataFrame) -> Dict[str, pd.Series]:
    """S0 plus every S1 variant, keyed by display label."""
    pos: Dict[str, pd.Series] = {S0_LABEL: s0_buy_hold(panel.index)}
    for label, kind, lbs in S1_VARIANTS:
        pos[f"S1_{label}"] = variant_position(panel, kind, lbs)
    return pos


# ── Lived-experience drawdown caliber ──────────────────────────────────────
def longest_underwater(net_ret: pd.Series) -> int:
    """Longest run, in months, that the growth-of-$1 curve sits **strictly
    below** its prior peak — the time an investor spends waiting to get back to
    even. Measured on the monthly net-return series with an implicit starting
    wealth of 1.0 prepended, so a drawdown that opens in month 1 counts. The
    recovery month (curve back at the peak) is *not* underwater.

    This is the whipsaw-pain caliber a month-end snapshot MaxDD hides: a -10%
    drawdown that takes five years to recover hurts more than a -18% one that
    snaps back in three months."""
    net = net_ret.dropna()
    if len(net) == 0:
        return 0
    wealth = pd.concat([pd.Series([1.0]), (1.0 + net).cumprod().reset_index(drop=True)],
                       ignore_index=True)
    peak = wealth.cummax()
    underwater = (wealth < peak).to_numpy()
    longest = cur = 0
    for uw in underwater:
        cur = cur + 1 if uw else 0
        if cur > longest:
            longest = cur
    return int(longest)


def max_consecutive_loss_months(net_ret: pd.Series) -> int:
    """Longest streak of consecutive months with net return < 0. Distinct from
    the underwater run (a single +0.01% month breaks the loss streak but may
    leave the curve underwater) — both are reported because both are felt."""
    net = net_ret.dropna()
    longest = cur = 0
    for r in net.to_numpy():
        cur = cur + 1 if r < 0 else 0
        if cur > longest:
            longest = cur
    return int(longest)


def trade_count(bt: pd.DataFrame, prev_held: Optional[float] = None,
                threshold: float = 1e-9) -> int:
    """Discrete **round-trip legs** that occur *inside* this frame: the number
    of times the invested state (``held > threshold``) changes, month to month.

    The boundary is handled with ``prev_held`` — the held weight the month
    *before* this frame's first row:
      • ``prev_held=None`` (start of the whole backtest, i.e. coming from cash)
        → a frame that opens invested counts its initial entry as one trade;
      • ``prev_held`` given (a mid-sample segment slice) → a state change *at*
        the first row vs the prior month counts, but merely *continuing* a
        position already held before the slice does NOT. So buy-and-hold sliced
        to a mid-sample segment correctly reports 0 trades (it never traded in
        that window), not a spurious 1.

    Distinct from annualised turnover, which a vol-targeted strategy racks up
    every month from small size adjustments even while staying continuously
    long. S0 over the full sample returns exactly 1 (the single entry)."""
    if len(bt) == 0 or "held" not in bt:
        return 0
    invested = (bt["held"] > threshold).to_numpy()
    flips = int(np.sum(invested[1:] != invested[:-1]))
    prev_invested = bool(
        prev_held is not None and not pd.isna(prev_held) and prev_held > threshold
    )
    if bool(invested[0]) != prev_invested:  # boundary entry/exit relative to prior month
        flips += 1
    return flips


def extended_metrics(bt: pd.DataFrame, prev_held: Optional[float] = None) -> Dict[str, float]:
    """`compute_metrics` (Sharpe/Calmar/CAGR/vol/MaxDD/hit/turnover/n_months)
    plus the three lived-experience fields. Same caliber for S0 and every S1.
    ``prev_held`` (held weight the month before this frame) lets `trade_count`
    distinguish a fresh entry from a position carried in across a segment edge."""
    m = dict(compute_metrics(bt))
    net = bt["net_ret"] if "net_ret" in bt else pd.Series(dtype="float64")
    m["longest_underwater_m"] = longest_underwater(net)
    m["max_consec_loss_m"] = max_consecutive_loss_months(net)
    m["n_trades"] = trade_count(bt, prev_held=prev_held)
    return m


# ── Fair common window across strategies ───────────────────────────────────
def common_window(
    backtests: Dict[str, pd.DataFrame],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """The window over which *every* strategy is investable (latest start,
    earliest end). S1 starts after its trend+vol warm-up while S0 is invested
    from month 1, so a fair head-to-head must trim S0 to this shared window
    (else S0 is credited extra early months S1 never traded).

    Returns ``(None, None)`` if any strategy is empty or the window is void —
    callers treat that as 'cannot adjudicate', not a crash."""
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for bt in backtests.values():
        if len(bt) == 0:
            return None, None
        spans.append((bt.index.min(), bt.index.max()))
    if not spans:
        return None, None
    cstart = max(s for s, _ in spans)
    cend = min(e for _, e in spans)
    if cstart > cend:
        return None, None
    return cstart, cend


METRIC_COLS: Tuple[str, ...] = (
    "sharpe", "calmar", "cagr", "max_dd",
    "longest_underwater_m", "max_consec_loss_m",
    "ann_turnover", "n_trades", "hit_rate", "n_months",
)


def segment_metrics(
    backtests: Dict[str, pd.DataFrame],
    seg_start: str,
    seg_end: str,
    common: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None,
) -> pd.DataFrame:
    """Extended metrics for every strategy on ``[seg_start, seg_end]`` clipped
    to the fair common window, so S0 and all S1 variants share the exact same
    months. Strategies with no overlap get an all-NaN/zero row (never a crash).
    Rows are ordered S0 first, then the variants in `S1_VARIANTS` order."""
    cstart, cend = common if common is not None else common_window(backtests)
    rows: Dict[str, Dict[str, float]] = {}
    for label, bt in backtests.items():
        if cstart is None or cend is None:
            rows[label] = extended_metrics(bt.iloc[0:0])
            continue
        lo = max(cstart, pd.Timestamp(seg_start))
        hi = min(cend, pd.Timestamp(seg_end))
        sl = slice_segment(bt, lo, hi) if lo <= hi else bt.iloc[0:0]
        # held weight the month before the slice → lets trade_count tell a fresh
        # entry from a position already carried into the segment. The common-window
        # start `cstart` is the evaluation origin: at lo == cstart everyone is
        # treated as flat (prev_held=None) so the window-opening segment shows S0's
        # single genuine entry as 1; a later segment carries the position in → 0.
        if lo > hi or lo == cstart:
            prev_held = None
        else:
            prior = bt["held"][bt.index < lo] if "held" in bt else None
            prev_held = float(prior.iloc[-1]) if prior is not None and len(prior) else None
        rows[label] = extended_metrics(sl, prev_held=prev_held)
    order = [S0_LABEL] + [f"S1_{lbl}" for lbl, _, _ in S1_VARIANTS]
    order = [o for o in order if o in rows] + [o for o in rows if o not in order]
    return pd.DataFrame({k: rows[k] for k in order}).T[list(METRIC_COLS)]


# ── Paired in-sample significance on the net-return difference ─────────────
def _bartlett_hac_se_mean(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett-kernel) HAC standard error of the *sample mean* of a
    serially-correlated series. Var(mean) = (1/n)[γ₀ + 2Σ_{l=1}^{L}(1−l/(L+1))γ_l],
    se = √Var(mean). Monthly strategy return differences show autocorrelation /
    volatility clustering, so an IID se understates uncertainty — this widens it."""
    n = len(x)
    if n < 2:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc) / n)
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma_l = float(np.dot(xc[l:], xc[:-l]) / n)
        var += 2.0 * w * gamma_l
    var = max(var, 0.0)  # HAC variance can go slightly negative numerically
    return float(np.sqrt(var / n))


def paired_net_diff_stats(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
    block_len: Optional[int] = None,
    hac_lag: Optional[int] = None,
) -> Dict[str, float]:
    """Paired monthly net-return difference (a − b) on the months both trade,
    treated as the **time series it is** (not IID):

      • t-stat uses a Newey-West / Bartlett **HAC** standard error of the mean
        (lag ≈ n^(1/3) by default) to account for autocorrelation;
      • the 95% CI on the annualised mean diff comes from a **moving-block
        bootstrap** (block length ≈ √n by default) that preserves short-range
        serial dependence — an IID resample would understate the CI width.

    ``a`` is the strategy under test (S1), ``b`` the benchmark (S0).

    DESCRIPTIVE, IN-SAMPLE: a CI excluding zero / a large |t| means the realised
    path shows a reliable gap on *this* sample — NOT an out-of-sample or
    walk-forward significance claim."""
    a = bt_a["net_ret"].dropna()
    b = bt_b["net_ret"].dropna()
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).dropna()
    n = int(len(d))
    # Validate params BEFORE the short-sample early return, so the input
    # contract is identical for long and short samples (a bad n_boot/block_len/
    # hac_lag is a caller error regardless of how many months survived).
    if n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
    if hac_lag is not None and hac_lag < 1:
        raise ValueError(f"hac_lag must be ≥ 1 when given, got {hac_lag}")
    if block_len is not None and block_len < 1:
        raise ValueError(f"block_len must be ≥ 1 when given, got {block_len}")

    nan = float("nan")
    if n < 2:
        return {"n": n, "mean_monthly": nan, "ann_mean": nan, "t_stat": nan,
                "ci_lo": nan, "ci_hi": nan, "ci_excludes_zero": False,
                "block_len": 0, "hac_lag": 0}
    arr = d.to_numpy(dtype="float64")
    mean_m = float(arr.mean())

    # clamp the (possibly user-supplied) HAC lag to [1, n-1] — a lag ≥ n has no
    # usable autocovariance — and report the ACTUAL lag used, not the raw input.
    L = hac_lag if hac_lag is not None else max(1, int(round(n ** (1.0 / 3.0))))
    L = max(1, min(L, n - 1))
    se = _bartlett_hac_se_mean(arr, L)
    t_stat = mean_m / se if (se == se and se > 0) else nan

    b_default = max(1, int(round(np.sqrt(n))))
    # clamp the (possibly user-supplied) block length to [1, n] so max_start ≥ 0
    b_len = min(max(1, block_len if block_len is not None else b_default), n)
    n_blocks = int(np.ceil(n / b_len))
    max_start = n - b_len  # ≥ 0 after the clamp; inclusive upper bound for a block start
    rng = np.random.default_rng(seed)
    # moving-block bootstrap: glue n_blocks contiguous length-b_len blocks, trim to n
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    offsets = np.arange(b_len)
    # gather[i] = concatenated blocks for replicate i, shape (n_boot, n_blocks*b_len)
    gather_idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    boot_means = arr[gather_idx].mean(axis=1) * ANNUAL
    ci_lo, ci_hi = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))
    return {
        "n": n,
        "mean_monthly": mean_m,
        "ann_mean": mean_m * ANNUAL,
        "t_stat": float(t_stat) if t_stat == t_stat else nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "block_len": int(b_len),
        "hac_lag": int(L),
    }


# ── Verdict (pure adjudication; the script only renders it) ────────────────
def verdict(
    seg_by_cost: Dict[float, Dict[str, pd.DataFrame]],
    paired_by_cost: Dict[float, Dict[str, float]],
) -> str:
    """Adjudicate the post-2000 kill condition on the PRIMARY (3/6/12 blend)
    variant — along BOTH axes, because they disagree and the honest answer is
    two-dimensional:

      • risk-adjusted (Sharpe AND Calmar, the same caliber PR#5 used): does S1
        beat S0 net of cost at the realistic 10bps and punitive 25bps?
      • raw return (CAGR + the paired monthly net-return excess and its CI): is
        the return excess positive and distinguishable from zero?

    A timing overlay can win the first while losing the second — that *is* the
    'stress insurance, not uniform alpha' pattern. Reporting only one axis would
    be the cherry-pick. Robustness (how many variants beat S0 on Sharpe
    post-2000 @10bps) is reported but does not flip the headline.

    Pure: takes the already-computed metric tables + paired stats and returns a
    markdown string. Lives in lib (not the script) so it is unit-testable
    without importing the I/O-heavy runner."""
    lines = ["## Verdict — is S1 still alive after 2000? (in-sample, ex-post)\n"]

    post = POST2000_SEGMENT

    def row(cost, label):
        return seg_by_cost[cost][post].loc[label]

    # Guard: need valid metrics on the post-2000 window. The verdict compares
    # Sharpe & Calmar AND CAGR at BOTH 10 and 25bps, so all of those must be
    # non-NaN — otherwise a missing 25bps row would make comparisons silently
    # False and masquerade as a "DECAYED" kill rather than "cannot adjudicate".
    try:
        s1_10 = row(PAIRED_DISPLAY_COST, PRIMARY_LABEL)
        s0_10 = row(PAIRED_DISPLAY_COST, S0_LABEL)
        s1_25 = row(PUNITIVE_COST, PRIMARY_LABEL)
        s0_25 = row(PUNITIVE_COST, S0_LABEL)
        pj = paired_by_cost[PAIRED_DISPLAY_COST]  # also required — guard in same try
    except KeyError:
        return ("## Verdict\n\n**Cannot adjudicate: post-2000 window or paired "
                "10bps stats missing from results.**")
    needed = ("sharpe", "calmar", "cagr")
    pj_keys = ("n", "ann_mean", "ci_lo", "ci_hi", "ci_excludes_zero")
    if not all(pd.notna(r[k]) for r in (s1_10, s0_10, s1_25, s0_25) for k in needed):
        return ("## Verdict\n\n**Insufficient sample on the post-2000 window "
                "(NaN Sharpe/Calmar/CAGR at 10 or 25bps) — cannot adjudicate. "
                "Widen the data.**")
    if not all(k in pj for k in pj_keys):
        return ("## Verdict\n\n**Cannot adjudicate: paired 10bps stats are missing "
                f"required keys {pj_keys}.**")
    # The paired excess drives the significance branch — it must be a real,
    # non-degenerate sample (n≥2) with finite mean/CI, else the "leans positive /
    # mixed" wording would be built on NaN comparisons silently resolving False.
    if (pj["n"] < 2 or not pd.notna(pj["ann_mean"])
            or not (pd.notna(pj["ci_lo"]) and pd.notna(pj["ci_hi"]))):
        return ("## Verdict\n\n**Cannot adjudicate: paired post-2000 net-diff sample "
                "is too short / NaN (n<2 or undefined mean/CI). Widen the data.**")

    def risk_adj_beats(s1, s0):  # PR#5 caliber: Sharpe AND Calmar
        return (s1["sharpe"] > s0["sharpe"]) and (s1["calmar"] > s0["calmar"])

    ra_10 = risk_adj_beats(s1_10, s0_10)
    ra_25 = risk_adj_beats(s1_25, s0_25)
    ret_10 = s1_10["cagr"] > s0_10["cagr"]   # raw-return win @10bps
    ret_25 = s1_25["cagr"] > s0_25["cagr"]

    ci_positive = bool(pj["ci_excludes_zero"]) and pj["ann_mean"] > 0
    ci_negative = bool(pj["ci_excludes_zero"]) and pj["ann_mean"] < 0

    # robustness across variants @10bps (risk-adjusted)
    tbl10 = seg_by_cost[PAIRED_DISPLAY_COST][post]
    variant_labels = [f"S1_{lbl}" for lbl, _, _ in S1_VARIANTS]
    n_beat = sum(
        1 for v in variant_labels
        if v in tbl10.index and pd.notna(tbl10.loc[v, "sharpe"])
        and tbl10.loc[v, "sharpe"] > tbl10.loc[S0_LABEL, "sharpe"]
    )
    n_var = len(variant_labels)

    lines.append(f"Post-2000 window **{post}**, primary variant **{PRIMARY_VARIANT}** "
                 "(PR#5's 3/6/12 blend), net of cost. **Two axes, read both:**\n")
    lines.append(f"- Risk-adjusted @10bps: S1 Sharpe {s1_10['sharpe']:.2f} vs S0 "
                 f"{s0_10['sharpe']:.2f}, Calmar {s1_10['calmar']:.2f} vs {s0_10['calmar']:.2f}, "
                 f"MaxDD {s1_10['max_dd']*100:.1f}% vs {s0_10['max_dd']*100:.1f}% → "
                 f"{'S1 WINS' if ra_10 else 'S1 does not win'}")
    lines.append(f"- Risk-adjusted @25bps: S1 Sharpe {s1_25['sharpe']:.2f} vs S0 "
                 f"{s0_25['sharpe']:.2f}, Calmar {s1_25['calmar']:.2f} vs {s0_25['calmar']:.2f} → "
                 f"{'S1 WINS' if ra_25 else 'S1 does not win'}")
    lines.append(f"- Raw return @10bps: S1 CAGR {s1_10['cagr']*100:.1f}% vs S0 "
                 f"{s0_10['cagr']*100:.1f}% → "
                 f"{'S1 higher' if ret_10 else 'S1 GIVES UP return'}")
    lines.append(f"- Paired monthly net excess (S1−S0) @10bps: ann {pj['ann_mean']*100:.1f}%, "
                 f"t={pj['t_stat']:.2f}, 95% CI [{pj['ci_lo']*100:.1f}%, {pj['ci_hi']*100:.1f}%] "
                 f"→ {'excludes zero' if pj['ci_excludes_zero'] else 'includes zero (NOT distinguishable from luck)'}")
    lines.append(f"- Robustness: {n_beat}/{n_var} S1 variants beat S0 on Sharpe over {post} @10bps")
    lines.append("")

    ra = ra_10 and ra_25         # risk-adjusted win at BOTH realistic & punitive cost
    ret = ret_10 and ret_25       # raw-return win at BOTH costs
    sig_phrase = ("distinguishable from zero" if ci_positive
                  else "significantly NEGATIVE" if ci_negative
                  else "statistically indistinguishable from zero")

    if not ra:
        lines.append(
            "**② S1 edge has DECAYED.** After 2000 it does not even win risk-adjusted "
            "net of cost — its full-sample Sharpe is largely the 1968–2000 bear it "
            "sidestepped. For a trader operating today, 'use S1 to trade gold' is "
            "close to void: just hold GLD/physical, or don't single-bet gold. "
            "Honest kill.")
    elif ra and ret and ci_positive:
        lines.append(
            "**① S1 STILL HAS EDGE post-2000 — on both axes.** It beats buy-and-hold "
            "risk-adjusted *and* on raw return at realistic and punitive costs, with "
            "the paired excess distinguishable from zero in-sample. The S1 story is "
            "not *only* 1968–2000 bear-avoidance. Worth hardening with a proper "
            "walk-forward / out-of-sample test before trading.")
    elif ra and ret:  # both axes nominally, but the return excess is not significant
        lines.append(
            "**①a S1 LEANS POSITIVE on both axes post-2000, but the return edge is "
            f"not significant.** It wins risk-adjusted (higher Sharpe & Calmar) net of "
            "cost at 10 and 25bps and is nominally ahead on raw CAGR too — yet the "
            f"paired monthly excess is {sig_phrase}, so the return advantage is not "
            "reliably separable from luck on this single path. Promising, not proven: "
            "the risk-adjusted edge is the solid part; a walk-forward test is needed "
            "before leaning on the return premium.")
    else:  # ra and not ret → risk-reducer, gives up raw return
        lines.append(
            "**①′ MIXED — S1 is a risk-reducer post-2000, NOT a return-enhancer.** "
            "It still wins risk-adjusted (higher Sharpe & Calmar, roughly half the "
            "drawdown) net of cost at both 10 and 25bps, but it GIVES UP raw CAGR to "
            f"buy-and-hold and the paired return excess is {sig_phrase}. This is the "
            "same 'stress insurance, not uniform alpha' character the v2 SP-CVaR layer "
            "shows. Read-through: the full-sample Sharpe 0.63 *return* edge was indeed "
            "largely 1968–2000 bear-avoidance and has decayed — but the "
            "*drawdown-control* edge persists. **For a drawdown-averse holder S1 is "
            "still worth a walk-forward test; for a total-return maximiser, just "
            "hold the metal.**")
    return "\n".join(lines)
