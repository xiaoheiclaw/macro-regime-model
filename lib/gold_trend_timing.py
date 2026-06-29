"""Gold long-only trend-timing backtest (trading-oriented, not a fair-value anchor).

Question: can a *long-only* spot-gold strategy that goes 0↔100% on (trend) and
(trend + macro-regime gate) beat buy-and-hold gold on a risk-adjusted basis,
net of costs, across the full 1968–2026 sample — including the 1980–2000 dead
decades where market-timing earns its keep by *avoiding the bear*?

Three comparison lines:
  S0  buy-and-hold gold (the yardstick)
  S1  pure trend: time-series momentum>0 → hold, else cash (T-bill); vol-targeted
      0–100% on trailing 6m realized vol. Lookback is a *scanned* parameter
      {3, 6, 12, and the equal-weight 3/6/12 blend} — all reported, blend favoured.
  S2  trend + regime gate: a *fast exit* that can cut exposure before price
      momentum turns. Regime favourable = (real-rate trailing change ≤ 0, i.e.
      not rising) AND (dollar trailing momentum ≤ 0, i.e. not strengthening).

All signals are ex-ante, monthly, no look-ahead: the position for month t+1 is
decided from data available at month t (a single `.shift(1)` at the return step).

This module is pure/functional and fetch-injectable so tests run without network.
Reuses `build_anchor_panel` from `lib.gold_anchor` for gold + real_rate_10y
(PR #1/#2, already on main) and adds a trade-weighted USD series (spliced
TWEXBMTH → DTWEXBGS) plus a T-bill cash leg (TB3MS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from lib.gold_anchor import _to_monthly, build_anchor_panel, fetch_fred_series

ANNUAL = 12  # months per year

# Default scanned trend lookbacks (months); "blend" = equal-weight of these.
DEFAULT_LOOKBACKS: tuple[int, ...] = (3, 6, 12)
DEFAULT_TARGET_VOL = 0.10  # annualised vol target for sizing (standard value)
DEFAULT_VOL_WINDOW = 6     # months of trailing realised vol
DEFAULT_REGIME_WINDOW = 12  # months for regime trailing change/momentum
DEFAULT_COST_BPS = 10.0    # per-rebalance cost on traded notional

# Reporting segments (inclusive year bounds). 2011 intentionally appears in two
# windows per the task spec — these are descriptive windows, not additive.
DEFAULT_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("1968-2000", "1968-01-01", "2000-12-31"),
    ("2001-2011", "2001-01-01", "2011-12-31"),
    ("2011-2018", "2011-01-01", "2018-12-31"),
    ("2019-2026", "2019-01-01", "2026-12-31"),
)


# ── Panel construction ─────────────────────────────────────────────────
def _splice_dollar(twex_monthly: pd.Series, dtwex_monthly: pd.Series) -> pd.Series:
    """Splice the discontinued monthly broad TWI (TWEXBMTH, 1973–~2020) with the
    daily broad goods+services index (DTWEXBGS, 2006→), rebasing the newer
    series to the older at the first overlapping month so the *level* is
    continuous. We only ever read the *direction* (momentum sign), so a clean
    join avoids a spurious multi-month momentum reading at the seam."""
    if twex_monthly.dropna().empty:
        return dtwex_monthly
    if dtwex_monthly.dropna().empty:
        return twex_monthly
    overlap = twex_monthly.dropna().index.intersection(dtwex_monthly.dropna().index)
    if len(overlap) == 0:
        # Disjoint indices: a level rebase is impossible (no common month), so
        # just take the union. The newer (DTWEXBGS) series takes precedence on
        # any month that somehow appears in both.
        return dtwex_monthly.combine_first(twex_monthly)
    join = overlap.min()
    scale = twex_monthly.loc[join] / dtwex_monthly.loc[join]
    dtwex_scaled = dtwex_monthly * scale
    # Old series up to (but not including) the join, scaled-new from the join on.
    old = twex_monthly[twex_monthly.index < join]
    return pd.concat([old, dtwex_scaled[dtwex_scaled.index >= join]]).sort_index()


@dataclass
class TimingPanel:
    data: pd.DataFrame
    notes: Dict[str, str] = field(default_factory=dict)


def build_timing_panel(
    start: str = "1968-01-01",
    end: Optional[str] = None,
    *,
    fetch_fn: Callable[[str, str], pd.Series] = fetch_fred_series,
    anchor_fn: Callable[..., object] = build_anchor_panel,
) -> TimingPanel:
    """Assemble the monthly month-end panel for the backtest.

    Columns: gold_nominal, gold_ret, real_rate_10y, usd_broad, tbill_yield,
    tbill_ret. Gold + real_rate_10y come straight from `build_anchor_panel`
    (reused, not re-derived); USD is the spliced trade-weighted index; the cash
    leg is the 3-month T-bill (TB3MS) converted to a monthly return.

    Injection: `fetch_fn` covers the FRED pulls owned by *this* function (USD,
    T-bill). Gold + real_rate_10y are owned by `anchor_fn` and use its own
    fetchers — to inject them (e.g. in tests) replace `anchor_fn` with a stub
    returning an object exposing `.data` (and optionally `.notes`).
    """
    panel = anchor_fn(start=start, end=end)
    base = panel.data  # type: ignore[attr-defined]
    df = base[["gold_nominal", "real_rate_10y"]].copy()
    idx = df.index

    twex = _to_monthly(fetch_fn("TWEXBMTH", start), "mean")
    dtwex = _to_monthly(fetch_fn("DTWEXBGS", start), "mean")
    usd = _splice_dollar(twex, dtwex)
    df["usd_broad"] = usd.reindex(idx)

    tb = _to_monthly(fetch_fn("TB3MS", start), "mean").reindex(idx)
    df["tbill_yield"] = tb
    # annual-percent yield → monthly simple return
    df["tbill_ret"] = (1.0 + tb / 100.0) ** (1.0 / ANNUAL) - 1.0

    df["gold_ret"] = df["gold_nominal"].pct_change()

    notes = {
        "frequency": "month-end (ME)",
        "gold_source": "build_anchor_panel.gold_nominal (datasets.io LBMA)",
        "real_rate_source": "build_anchor_panel.real_rate_10y "
        "(DFII10 TIPS 2003+; pre-2003 GS10 − trailing-12m CPI splice)",
        "usd_source": "TWEXBMTH (1973–) spliced→ DTWEXBGS (2006–), level-rebased at overlap",
        "usd_pre_1973": "no trade-weighted USD before 1973 (Bretton Woods); "
        "regime gate falls back to real-rate-only there",
        "cash_leg": "TB3MS 3m T-bill, annual% → monthly compounded return",
    }
    if hasattr(panel, "notes"):
        notes["real_rate_splice_detail"] = str(panel.notes.get("real_rate_splice", ""))  # type: ignore[attr-defined]
    return TimingPanel(data=df, notes=notes)


# ── Signals (all ex-ante; a value at index t uses data ≤ t only) ───────
def momentum_signal(price: pd.Series, lookback: int) -> pd.Series:
    """1.0 when trailing `lookback`-month log return > 0, else 0.0. NaN until
    enough history. Uses price[t] vs price[t-lookback] — no future data."""
    logp = np.log(price)
    mom = logp - logp.shift(lookback)
    sig = (mom > 0).astype(float)
    sig[mom.isna()] = np.nan
    return sig


def realized_vol(ret: pd.Series, window: int = DEFAULT_VOL_WINDOW) -> pd.Series:
    """Annualised trailing realised vol of monthly returns."""
    return ret.rolling(window).std() * np.sqrt(ANNUAL)


def vol_scale(
    ret: pd.Series,
    target_vol: float = DEFAULT_TARGET_VOL,
    window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Long-only vol-targeting multiplier in [0, 1]: target/realised, capped at
    1 (no leverage). NaN where realised vol is undefined."""
    rv = realized_vol(ret, window)
    scale = target_vol / rv
    return scale.clip(lower=0.0, upper=1.0)


def trend_exposure(price: pd.Series, lookbacks: Sequence[int]) -> pd.Series:
    """Equal-weight blend of the per-lookback on/off momentum signals.
    A single-element `lookbacks` reproduces a pure single-horizon trend.
    Returns a value in [0, 1] (e.g. {0, 1/3, 2/3, 1} for the 3/6/12 blend).

    Uses ``skipna=False`` so the blend stays NaN until *every* lookback has
    enough history — otherwise during warm-up the longest-history (shortest)
    lookback would drive the blend to a full 1.0 prematurely, which `s1_trend`
    would then take as full exposure on too little evidence."""
    if len(lookbacks) == 0:
        raise ValueError("lookbacks must be non-empty")
    sigs = [momentum_signal(price, L) for L in lookbacks]
    return pd.concat(sigs, axis=1).mean(axis=1, skipna=False)


def regime_gate(
    real_rate: pd.Series,
    usd: pd.Series,
    window: int = DEFAULT_REGIME_WINDOW,
) -> pd.Series:
    """Fast-exit gate. Favourable (1.0) when BOTH:
      • real rate is not rising:  real_rate[t] − real_rate[t-window] ≤ 0
      • the dollar is not strengthening: usd[t] − usd[t-window] ≤ 0
    Unfavourable → 0.0. A missing/undefined leg is treated as favourable (the
    gate is an overlay that only *removes* exposure on a confirmed adverse
    signal — notably USD pre-1973, where the gate degrades to real-rate-only)."""
    rr_chg = real_rate - real_rate.shift(window)
    usd_mom = usd - usd.shift(window)
    rr_ok = (rr_chg <= 0) | rr_chg.isna()
    usd_ok = (usd_mom <= 0) | usd_mom.isna()
    return (rr_ok & usd_ok).astype(float)


# ── Strategy positions (weight to *hold next month*, indexed at decision t) ──
def s0_buy_hold(index: pd.Index) -> pd.Series:
    """S0: always 100% gold."""
    return pd.Series(1.0, index=index)


def s1_trend(
    panel: pd.DataFrame,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """S1: trend exposure × vol-target multiplier, clipped to [0, 1]."""
    te = trend_exposure(panel["gold_nominal"], lookbacks)
    vs = vol_scale(panel["gold_ret"], target_vol, vol_window)
    pos = (te * vs).clip(lower=0.0, upper=1.0)
    return pos.fillna(0.0)


def s2_trend_regime(
    panel: pd.DataFrame,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
    regime_window: int = DEFAULT_REGIME_WINDOW,
    gate_off_exposure: float = 0.0,
) -> pd.Series:
    """S2: S1 with the regime gate multiplied in. When the gate is unfavourable,
    exposure is scaled by `gate_off_exposure` (default 0 = full exit to cash)."""
    base = s1_trend(panel, lookbacks, target_vol, vol_window)
    gate = regime_gate(panel["real_rate_10y"], panel["usd_broad"], regime_window)
    mult = gate.where(gate == 1.0, other=gate_off_exposure).fillna(1.0)
    return (base * mult).clip(lower=0.0, upper=1.0)


# ── Backtest engine ────────────────────────────────────────────────────
def run_backtest(
    positions: pd.Series,
    gold_ret: pd.Series,
    tbill_ret: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Apply `positions` (decided at t, held through t+1) with a cash leg and
    per-rebalance trading cost. Returns a frame with held weight, gross/net
    monthly return, turnover and cost — one row per *traded* month, with no NaN.

    No look-ahead: month-m return uses positions[m-1] (`positions.shift(1)`).
    Cost in month m is charged on |w_held[m] − w_held[m-1]| (the trade that set
    up month-m's weight), the first entry trading up from cash.

    Contract: the traded span is trimmed to the first…last month with complete
    (held, gold, tbill) data. Leading warm-up NaNs (before the first position)
    and a trailing incomplete month (e.g. this month's gold price not in yet)
    are trimmed away. But a missing return *strictly inside* that span is a data
    hole through which a backtest cannot honestly model holding or trading, so
    it raises ``ValueError`` rather than silently dropping the month (which
    would also drop that month's turnover/cost and under-count it).
    """
    pos = positions.reindex(gold_ret.index)
    held = pos.shift(1)
    valid = held.notna() & gold_ret.notna() & tbill_ret.notna()
    cols = ["held", "gold_ret", "tbill_ret", "turnover", "cost", "gross_ret", "net_ret"]
    if not valid.any():  # never investable
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name=gold_ret.index.name))

    first = valid.idxmax()              # first valid month
    last = valid[::-1].idxmax()         # last valid month
    core = valid.loc[first:last]
    if not core.all():
        where = list(core.index[~core][:5])
        raise ValueError(
            f"missing held/gold/tbill value inside the traded span "
            f"[{first:%Y-%m}…{last:%Y-%m}] (first {len(where)} of "
            f"{int((~core).sum())}): {where}. Supply gap-free returns."
        )

    held_s = held.loc[first:last]
    gold_s = gold_ret.loc[first:last]
    tbill_s = tbill_ret.loc[first:last]

    turnover = held_s.diff().abs()
    turnover.iloc[0] = abs(held_s.iloc[0])  # first entry: trade up from cash
    cost = turnover * (cost_bps / 1e4)
    gross = held_s * gold_s + (1.0 - held_s) * tbill_s
    net = gross - cost

    return pd.DataFrame(
        {
            "held": held_s,
            "gold_ret": gold_s,
            "tbill_ret": tbill_s,
            "turnover": turnover,
            "cost": cost,
            "gross_ret": gross,
            "net_ret": net,
        }
    )


# ── Metrics ────────────────────────────────────────────────────────────
def _max_drawdown(cum: pd.Series) -> float:
    """Max drawdown of a growth-of-$1 curve. `cum` starts *after* the first
    month's return, so the implicit starting wealth of 1.0 must be prepended —
    otherwise a drawdown from the opening month (peak still 1.0) is missed and
    the figure is understated (or 0)."""
    if len(cum) == 0:
        return float("nan")
    wealth = pd.concat([pd.Series([1.0]), pd.Series(cum.to_numpy())], ignore_index=True)
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def compute_metrics(
    bt: pd.DataFrame,
    rf: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Risk/return metrics from a backtest frame's `net_ret`.

    Sharpe is on *excess* return (net − T-bill) so cash-heavy strategies are
    not flattered by the risk-free leg. Hit rate = share of months with
    net_ret > 0. Turnover is annualised (sum of monthly turnover / years)."""
    net = bt["net_ret"].dropna()
    n = len(net)
    if n == 0:
        return {k: float("nan") for k in
                ("cagr", "ann_vol", "sharpe", "max_dd", "calmar",
                 "hit_rate", "ann_turnover", "n_months")}
    cum = (1.0 + net).cumprod()
    years = n / ANNUAL
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = net.std(ddof=1) * np.sqrt(ANNUAL)
    rf_series = bt["tbill_ret"] if rf is None else rf
    excess = (net - rf_series.reindex(net.index)).dropna()
    sharpe = (excess.mean() * ANNUAL) / (excess.std(ddof=1) * np.sqrt(ANNUAL)) \
        if excess.std(ddof=1) > 0 else float("nan")
    max_dd = _max_drawdown(cum)
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    hit = float((net > 0).mean())
    ann_turnover = float(bt["turnover"].sum() / years) if years > 0 else float("nan")
    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": hit,
        "ann_turnover": ann_turnover,
        "n_months": int(n),
    }


def equity_curve(bt: pd.DataFrame) -> pd.Series:
    """Cumulative net-of-cost growth of $1."""
    return (1.0 + bt["net_ret"].dropna()).cumprod()


def slice_segment(
    bt: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Inclusive date slice. Either bound may be None (open-ended on that side)."""
    mask = pd.Series(True, index=bt.index)
    if start is not None:
        mask &= bt.index >= pd.Timestamp(start)
    if end is not None:
        mask &= bt.index <= pd.Timestamp(end)
    return bt.loc[mask]
