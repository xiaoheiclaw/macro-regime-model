"""WGC central-bank gold-flow loader + signal builder (PR #10).

Purpose
-------
PR #9 decomposed gold's 2022→2026 surge into five additive layers and found the
②real-rate layer contributes **−53%** (the old "gold ≈ −real-rate" anchor has
*inverted*) while **+127%** of the move lands in the ε_flow residual — because
layer ⑤ (central-bank net purchases) had **no data wired in** (WGC has no FRED
feed; PR #9 left a `wgc_fn` injection seam and folded ⑤ into the residual).

This module supplies that missing layer. It carries the **World Gold Council
annual net official-sector purchase series** (tonnes), materializes it to
`data/wgc_cb_purchases.csv`, interpolates it to a monthly grid, and builds a
*level-like* flow **signal** suitable for the PR #9 log-linear Δ-attribution
(`contribution_k = coef_k · Δproxy_k`). Injecting `make_wgc_fn()` into
`build_attribution_panel(wgc_fn=...)` turns ⑤ from a residual into an explicit
regressor, so we can measure how much of the +127% residual it *claims*.

Data honesty (read before quoting any number)
---------------------------------------------
* The annual tonnage figures are **WGC Gold Demand Trends published estimates**.
  WGC **revises** them across releases (±tens of tonnes is routine), so treat
  every value as an estimate with revision noise, not a measured constant.
* WGC publishes quarterly detail, but a clean public long quarterly series is
  not readily machine-pullable here; we use the **annual** series and
  **interpolate to monthly** (annual flow spread evenly across the 12 months —
  "均摊"). Monthly shape is therefore an *artifact of interpolation*, not data.
* 2026 is incomplete at run time. By default the latest known annual pace is
  **carried forward** to cover the partial year so the cumulative signal reaches
  the decomposition endpoint; this is a flagged assumption (see `make_wgc_fn`),
  and because the 2022→2026 cumulative Δ is dominated by 2022-2025 it barely
  moves the result.
* Central-bank buying and the gold price are plausibly **endogenous** (price↑ →
  banks chase, or banks buy → price↑). This module does **no causal inference**;
  the PR #9 attribution is *variance attribution* (which layer co-moved), not a
  causal claim, and the coefficients are regime-dependent / non-extrapolable.

The authoritative numbers live in `WGC_ANNUAL_TONNES` below (committed); the
`data/` CSV is a regenerable mirror (the repo gitignores `data/`).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from lib.paths import DATA_DIR

# ── authoritative source-of-truth series (WGC Gold Demand Trends) ──────────
# Annual net official-sector gold purchases, tonnes. ESTIMATES w/ revision
# noise — see module docstring. (2010 is the first year WGC reports the modern
# net-purchase series.)
WGC_ANNUAL_TONNES: Dict[int, float] = {
    2010: 79.0,
    2011: 481.0,
    2012: 569.0,
    2013: 629.0,
    2014: 584.0,
    2015: 580.0,
    2016: 395.0,
    2017: 379.0,
    2018: 656.0,
    2019: 605.0,
    2020: 255.0,
    2021: 463.0,
    2022: 1082.0,
    2023: 1037.0,
    2024: 1045.0,
    2025: 863.0,
}

WGC_SOURCE = (
    "World Gold Council, Gold Demand Trends — annual net official-sector "
    "purchases (tonnes). ESTIMATES; revised across releases (±tens of tonnes)."
)

# "Normal regime" baseline = mean annual purchase over 2010-2021 (the decade of
# steady official buying *before* the 2022 step-up). Used to isolate the
# post-2022 EXCESS flow. = 472.9 t/yr ≈ 473 (matches the PR #10 brief).
BASELINE_YEARS = tuple(range(2010, 2022))
BASELINE_TONNES = float(np.mean([WGC_ANNUAL_TONNES[y] for y in BASELINE_YEARS]))

DEFAULT_WGC_CSV = os.path.join(DATA_DIR, "wgc_cb_purchases.csv")

VALID_SIGNALS = ("cum_excess", "cum_stock", "excess_flow", "flow")
DEFAULT_SIGNAL = "cum_excess"


# ── CSV materialization / load ─────────────────────────────────────────────
def write_wgc_csv(path: str = DEFAULT_WGC_CSV) -> str:
    """Write the authoritative annual series to `path` (with provenance header).
    Returns the path. `data/` is gitignored, so this is a regenerable mirror of
    `WGC_ANNUAL_TONNES`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"# {WGC_SOURCE}",
        "# ESTIMATES with revision noise; quarterly detail annualized→ here annual only.",
        f"# baseline (2010-2021 mean) = {BASELINE_TONNES:.1f} t/yr.",
        "year,net_purchases_tonnes",
    ]
    for y in sorted(WGC_ANNUAL_TONNES):
        lines.append(f"{y},{WGC_ANNUAL_TONNES[y]:g}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def load_wgc_annual(path: str = DEFAULT_WGC_CSV, *, write_if_missing: bool = True) -> pd.Series:
    """Load the annual WGC series as a Series indexed by year-end Timestamp
    (tonnes). Reads `path` if present; otherwise falls back to the embedded
    `WGC_ANNUAL_TONNES` and (by default) materializes the CSV so the `data/`
    artifact exists. Raises if the CSV is present but malformed/empty."""
    if os.path.exists(path):
        df = pd.read_csv(path, comment="#")
        if "year" not in df.columns or "net_purchases_tonnes" not in df.columns:
            raise ValueError(
                f"{path} missing required columns year,net_purchases_tonnes "
                f"(got {list(df.columns)})"
            )
        s = pd.Series(
            df["net_purchases_tonnes"].astype(float).values,
            index=pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31"),
        ).sort_index()
        if s.dropna().empty:
            raise ValueError(f"{path} has no usable rows")
        return s
    if write_if_missing:
        write_wgc_csv(path)
    return pd.Series(
        {pd.Timestamp(f"{y}-12-31"): v for y, v in sorted(WGC_ANNUAL_TONNES.items())}
    )


# ── monthly interpolation + signal construction ────────────────────────────
def _annual_to_monthly_flow(
    annual: pd.Series,
    *,
    end: Optional[pd.Timestamp] = None,
    carry_forward_partial: bool = True,
) -> pd.Series:
    """Spread each annual flow evenly across its 12 calendar months ("均摊"),
    returning a month-end Series of tonnes/month.

    The latest year in `annual` may be a partial/incomplete year (e.g. 2026 at
    run time). If `end` extends past the last annual label and
    `carry_forward_partial=True`, the most recent annual pace is carried forward
    to fill months up to `end` (a flagged assumption — see module docstring);
    otherwise the signal simply stops at the last full year."""
    annual = annual.dropna().sort_index()
    if annual.empty:
        return pd.Series(dtype="float64")
    years = sorted(annual.index.year.unique())
    last_year = years[-1]
    last_month = pd.Timestamp(f"{last_year}-12-31")
    grid_end = last_month
    if carry_forward_partial:
        # carry the latest annual pace forward to cover the partial/future year.
        # When an explicit `end` is given, extend to it; otherwise buffer one
        # extra year (the current incomplete year) so a downstream panel whose
        # gold index runs past the last full WGC year still gets a flow value.
        if end is not None:
            target = pd.Timestamp(end).to_period("M").to_timestamp("M")
        else:
            target = pd.Timestamp(f"{last_year + 1}-12-31")
        grid_end = max(last_month, target)
    idx = pd.date_range(f"{years[0]}-01-31", grid_end, freq="ME")
    monthly = pd.Series(index=idx, dtype="float64")
    for ts in idx:
        y = ts.year
        if y in annual.index.year:
            monthly.loc[ts] = float(annual.loc[annual.index.year == y].iloc[0]) / 12.0
        elif carry_forward_partial and y > last_year:
            # partial future year: carry the latest annual pace forward
            monthly.loc[ts] = float(annual.loc[last_month]) / 12.0
    return monthly.dropna()


def build_flow_signal(
    annual: pd.Series,
    *,
    signal: str = DEFAULT_SIGNAL,
    baseline: float = BASELINE_TONNES,
    end: Optional[pd.Timestamp] = None,
    carry_forward_partial: bool = True,
) -> pd.Series:
    """Build a monthly level-like flow **proxy** from the annual WGC series.

    Signals (all monthly, month-end):
      * ``cum_excess`` (default): cumulative tonnes bought *above the baseline
        path* = Σ(monthly_flow − baseline/12). Isolates the post-2022 abnormal
        accumulation; monotone-ish and its Δ over a window = cumulative EXCESS
        official buying — the natural level for additive Δ-attribution.
      * ``cum_stock``: cumulative tonnes since 2010 = Σ monthly_flow (no baseline
        subtraction).
      * ``excess_flow``: annual (flow − baseline), held flat within each year
        (step). The *rate* of abnormal buying, not a stock.
      * ``flow``: annual flow held flat within each year (step). Raw rate.

    Why ``cum_excess`` is the default (and reported as primary in PR #10): a flow
    *rate* level (``flow``/``excess_flow``) barely changes across 2022→2026
    (2022≈1082 → 2025≈863) so its Δ understates a *sustained* regime; the
    cumulative-excess *stock* captures "four years of buying ~2× the prior norm",
    which is the structural bid the de-dollarisation thesis posits.
    """
    if signal not in VALID_SIGNALS:
        raise ValueError(f"signal must be one of {VALID_SIGNALS}, got {signal!r}")
    monthly_flow = _annual_to_monthly_flow(
        annual, end=end, carry_forward_partial=carry_forward_partial
    )
    if monthly_flow.empty:
        return monthly_flow
    if signal == "flow":
        return monthly_flow * 12.0  # back to annualized rate (flat within year)
    if signal == "excess_flow":
        return monthly_flow * 12.0 - baseline
    if signal == "cum_stock":
        return monthly_flow.cumsum()
    # cum_excess
    return (monthly_flow - baseline / 12.0).cumsum()


def make_wgc_fn(
    *,
    signal: str = DEFAULT_SIGNAL,
    baseline: float = BASELINE_TONNES,
    path: str = DEFAULT_WGC_CSV,
    carry_forward_partial: bool = True,
) -> Callable[[str, Optional[str]], pd.Series]:
    """Build a `wgc_fn(start, end)` for `build_attribution_panel(wgc_fn=...)`.

    Returns a closure that loads the annual WGC series and emits the chosen
    monthly `signal`. `end` (the panel's requested end) drives the partial-year
    carry-forward so the signal reaches the decomposition endpoint."""

    def wgc_fn(start: str, end: Optional[str] = None) -> pd.Series:
        annual = load_wgc_annual(path)
        s = build_flow_signal(
            annual,
            signal=signal,
            baseline=baseline,
            end=pd.Timestamp(end) if end else None,
            carry_forward_partial=carry_forward_partial,
        )
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        return s

    return wgc_fn
