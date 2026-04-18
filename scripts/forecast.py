#!/usr/bin/env python3
"""
Phase 3: Kernel Analog Forecasting (baseline).

For the query state at time t, find N historical dates with similar macro
state and pull their realized forward paths. Emit as *joint* scenarios:
every asset in scenario_id=k uses the same analog date → the cross-asset
correlation observed historically is preserved (required by BL/SP-CVaR).

Baseline:
- State vector: standardized STATE_FEATURES (19 features; rates,
  credit, growth, inflation, vix, bcom_yoy, shiller_earnings_yield).
  Scaler is fit expanding-window on data ≤ asof (no look-ahead).
- Distance: combined state + regime
    d_total = d_state + α · d_global_regime + β · d_asset_regime
  where d_state is standardized Euclidean on STATE_FEATURES, and the
  regime terms are L2 on probability simplices (global_templates and
  asset_regime_probs respectively). α=2.0, β=0.5 as baseline.
- Selection: top-N_SCN by d_total; analogs must have complete forward
  H-month history (and lie strictly before asof).
- Weights: softmax(-d_total / auto_temp) normalized over selected.
- Horizons: 1, 3, 6, 12 months (18m excluded — too few independent samples)
- Fallback chain (per codex review):
  1) Regime-conditional KAF ← THIS IS THE BASELINE (regime-filter inline)
  2) Unconditional rolling-120 empirical quantiles ← < MIN_ANALOGS

Upgrade path (Phase 3b):
- Diffusion map / dynamics-adapted kernel (Alexander & Giannakis)
- Delay embedding (q-lag concatenated state vectors)
- Tethering: trajectory similarity over past 24m weighted 1, 1/2, 1/3

Contract (schema v2.1):
  Input:
    data/state_features.parquet
    data/v2/global_template_centroids.npz (scaler)
  Output:
    data/v2/forecast_scenarios.parquet
      asof_date, scenario_id, asset, horizon, log_return, weight
    data/v2/forecast_summary.parquet
      asof_date, asset, horizon, p10, p50, p90, mean, std, n_scn
    analysis/v2/forecast_<asof>.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore
from lib.schema import (
    SCHEMA_VERSION, FEATURE_SET_VERSION, base_meta,
)  # type: ignore

HORIZONS = [1, 3, 6, 12]
N_SCN = 200
MIN_ANALOGS = 50
ROLLING_FALLBACK_WINDOW = 120

# Regime-conditional analog filtering (Phase 3 closure per codex Critical #1).
# Combined distance: d = d_state + ALPHA_GLOBAL * d_global_regime + BETA_ASSET * d_asset_regime
# Weights chosen so the regime terms are roughly 20-30% of d_state at typical magnitudes
# (d_state ~ 5-7 on 19-D standardized; d_regime on K-simplex max √2 ~ 1.4). Tunable in Stage B.
ALPHA_GLOBAL_REGIME = 2.0
BETA_ASSET_REGIME = 0.5

# State (what defines the analog — levels/spreads/macro)
STATE_FEATURES = [
    "y3m", "y2y", "y5y", "y10y", "y30y",
    "y10y_diff",
    "curve_pc1_level", "curve_pc2_slope", "curve_pc3_curv",
    "yc_2s10s",
    "vix_level",
    "moody_baa_aaa", "moody_baa_10y",
    "ip_yoy", "payroll_yoy", "unrate",
    "cpi_yoy",
    "bcom_yoy",
    "shiller_earnings_yield",
]

# Assets whose forward paths we emit
ASSETS = [
    "spx_ret", "hsi_ret", "btc_ret",
    "oil_ret", "natgas_ret", "gold_ret", "silver_ret", "copper_ret", "bcom_ret",
    "dxy_ret",
    "bond_ret",  # synthetic 10Y bond return (duration proxy)
]


@dataclass
class AnalogSelection:
    dates: pd.DatetimeIndex
    distances: np.ndarray         # combined (state + regime)
    state_distances: np.ndarray   # state-only component
    regime_distances: np.ndarray  # regime-only component (global + asset combined)
    weights: np.ndarray
    source: str                   # "kaf" | "rolling_fallback"


def _load_regime_probs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns (global_probs_by_date, {asset: asset_probs_by_date}).
    Each DataFrame indexed by date, columns = prob_0..prob_{K-1}.
    Missing inputs → empty return; caller falls back to state-only distance.
    """
    g_path = Path(DATA_DIR) / "v2" / "global_templates.parquet"
    a_path = Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet"
    if not g_path.exists() or not a_path.exists():
        return pd.DataFrame(), {}
    g = pd.read_parquet(g_path)
    a = pd.read_parquet(a_path)
    a["date"] = pd.to_datetime(a["date"])
    g_cols = sorted([c for c in g.columns if c.startswith("prob_")])
    g = g[g_cols].copy()
    a_cols = sorted([c for c in a.columns if c.startswith("prob_")])
    asset_lookup: dict[str, pd.DataFrame] = {}
    for asset, grp in a.groupby("asset"):
        asset_lookup[str(asset)] = grp.set_index("date")[a_cols]
    return g, asset_lookup


def _regime_distance_components(
    query_date: pd.Timestamp,
    candidate_dates: pd.DatetimeIndex,
    global_probs: pd.DataFrame,
    asset_probs: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (d_global, d_asset_mean). d_global is L2 on global regime probs,
    d_asset_mean is mean L2 across per-asset regime probs (nan-ignored).
    Shape: (len(candidate_dates),). Zeros returned if probs unavailable.
    """
    n = len(candidate_dates)
    if global_probs.empty or query_date not in global_probs.index:
        return np.zeros(n), np.zeros(n)

    q_g = global_probs.loc[query_date].values
    cand_in_g = candidate_dates.isin(global_probs.index)
    d_g = np.zeros(n)
    if cand_in_g.any():
        mat = global_probs.loc[candidate_dates[cand_in_g]].values
        d_g_vals = np.linalg.norm(mat - q_g, axis=1)
        d_g[cand_in_g] = d_g_vals

    per_asset: list[np.ndarray] = []
    for asset, df in asset_probs.items():
        if query_date not in df.index:
            continue
        q_a = df.loc[query_date].values
        cand_in_a = candidate_dates.isin(df.index)
        if not cand_in_a.any():
            continue
        mat = df.loc[candidate_dates[cand_in_a]].values
        d_vals = np.full(n, np.nan)
        d_vals[cand_in_a] = np.linalg.norm(mat - q_a, axis=1)
        per_asset.append(d_vals)

    if per_asset:
        stacked = np.vstack(per_asset)
        with np.errstate(all="ignore"):
            d_a = np.nanmean(stacked, axis=0)
        mean_fill = np.nanmean(d_a) if np.isfinite(np.nanmean(d_a)) else 0.0
        d_a = np.where(np.isnan(d_a), mean_fill, d_a)
    else:
        d_a = np.zeros(n)
    return d_g, d_a


MIN_SCALER_TRAIN = 60  # minimum rows to fit the expanding scaler at each asof


def _forward_cum_return(ret_series: pd.Series, h: int) -> pd.Series:
    """Cumulative log return from t+1 to t+h, indexed at t."""
    return ret_series.shift(-1).rolling(h).sum().shift(-(h - 1))


def _build_state_matrix(
    state: pd.DataFrame,
    asof: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Returns (state_subset, standardized_matrix, scaler_meta).

    Scaler is refit on data strictly ≤ asof at each call (expanding window),
    so backtesting at historical asof dates sees exactly the scaling an
    analyst at that date would compute. No dependence on the global_template
    training scaler — that one is fit once on the full panel and would
    contaminate backtests.
    """
    missing = [f for f in STATE_FEATURES if f not in state.columns]
    if missing:
        sys.exit(f"STATE_FEATURES missing from state: {missing}")
    s = state[STATE_FEATURES].dropna()

    train_mask = s.index <= asof
    train = s.loc[train_mask]
    if len(train) < MIN_SCALER_TRAIN:
        sys.exit(f"Insufficient history ≤ {asof.date()} for scaler fit "
                 f"({len(train)} < {MIN_SCALER_TRAIN})")

    scaler = StandardScaler()
    scaler.fit(train.values)
    standardized = scaler.transform(s.values)
    meta = {
        "scaler_fit_start": str(train.index.min().date()),
        "scaler_fit_end":   str(train.index.max().date()),
        "scaler_fit_rows":  int(len(train)),
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
    }
    return s, standardized, meta


def _select_analogs(
    asof: pd.Timestamp,
    state_subset: pd.DataFrame,
    X: np.ndarray,
    h_max: int,
    n_scn: int,
) -> AnalogSelection:
    """
    Pick N nearest analogs with complete H-month forward history.

    Distance is combined: state (standardized Euclidean on 19 features) +
    α * global regime L2 + β * mean per-asset regime L2. This implements
    the Phase 3 regime-conditional analog pool the v2.1 design required.
    """
    if asof not in state_subset.index:
        sys.exit(f"asof {asof} not in state panel")
    query_idx = state_subset.index.get_loc(asof)
    query = X[query_idx]

    last_valid_analog_date = state_subset.index[-1] - pd.offsets.MonthEnd(h_max + 1)
    candidates_mask = (
        (state_subset.index <= last_valid_analog_date)
        & (state_subset.index < asof)
    )
    cand_idx = np.where(candidates_mask)[0]
    if len(cand_idx) < MIN_ANALOGS:
        return AnalogSelection(
            dates=pd.DatetimeIndex([]),
            distances=np.array([]),
            state_distances=np.array([]),
            regime_distances=np.array([]),
            weights=np.array([]),
            source="rolling_fallback",
        )

    X_cand = X[cand_idx]
    cand_dates = state_subset.index[cand_idx]
    d_state = np.linalg.norm(X_cand - query, axis=1)

    global_probs, asset_probs = _load_regime_probs()
    d_g, d_a = _regime_distance_components(asof, cand_dates, global_probs, asset_probs)
    d_regime = ALPHA_GLOBAL_REGIME * d_g + BETA_ASSET_REGIME * d_a
    d_combined = d_state + d_regime

    top_n = min(n_scn, len(cand_idx))
    top_ranks = np.argsort(d_combined)[:top_n]
    sel_cand_idx = cand_idx[top_ranks]
    sel_dates = state_subset.index[sel_cand_idx]
    sel_dist = d_combined[top_ranks]
    sel_state_dist = d_state[top_ranks]
    sel_regime_dist = d_regime[top_ranks]

    temperature = max(np.median(sel_dist), 1e-6)
    logits = -sel_dist / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    return AnalogSelection(
        dates=sel_dates,
        distances=sel_dist,
        state_distances=sel_state_dist,
        regime_distances=sel_regime_dist,
        weights=weights,
        source="kaf",
    )


def _fallback_rolling(
    asof: pd.Timestamp,
    state: pd.DataFrame,
    assets: list[str],
    horizons: list[int],
) -> list[dict]:
    """
    Joint-preserving unconditional rolling fallback. Each scenario_id
    corresponds to ONE historical date in the trailing
    ROLLING_FALLBACK_WINDOW window; all (asset, horizon) pairs with valid
    forward returns on that date share the same scenario_id. This honors
    the cross-asset joint contract required by BL/SP-CVaR.

    Assets that lack history on an analog date get dropped (no row),
    not remapped to a different date — so scenario_id K always means
    "if history from window[K] repeated forward".
    """
    window_idx = state.loc[:asof].tail(ROLLING_FALLBACK_WINDOW).index
    asset_cols: list[tuple[str, str]] = []
    for a in assets:
        col = a if a in state.columns else f"{a}_ret"
        if col in state.columns:
            asset_cols.append((a, col))

    # Pre-compute forward cumulative returns once per (asset, horizon)
    fwd_tables: dict[tuple[str, int], pd.Series] = {}
    for asset, ret_col in asset_cols:
        for h in horizons:
            fwd_tables[(asset, h)] = _forward_cum_return(state[ret_col], h)

    rows: list[dict] = []
    valid_sids: set[int] = set()
    for sid, dt in enumerate(window_idx):
        for (asset, h), fwd in fwd_tables.items():
            if dt not in fwd.index:
                continue
            v = fwd.loc[dt]
            if not np.isfinite(v):
                continue
            rows.append({
                "asof_date": asof,
                "scenario_id": int(sid),
                "analog_date": dt,
                "asset": asset,
                "horizon": int(h),
                "log_return": float(v),
                "weight": 0.0,  # filled below
            })
            valid_sids.add(int(sid))

    if valid_sids:
        w = 1.0 / len(valid_sids)
        for r in rows:
            r["weight"] = w
    return rows


def run(asof: str | None = None) -> dict:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}")
    state = pd.read_parquet(state_path)

    # Resolve asof first; scaler must be fit on ≤ asof only (no look-ahead)
    all_idx = state[STATE_FEATURES].dropna().index
    asof_ts = pd.Timestamp(asof) if asof else all_idx.max()
    if asof_ts not in all_idx:
        prev = all_idx[all_idx <= asof_ts]
        if len(prev) == 0:
            sys.exit(f"no state data <= {asof_ts}")
        asof_ts = prev[-1]
    asof_str = asof_ts.strftime("%Y-%m-%d")

    state_subset, X, scaler_meta = _build_state_matrix(state, asof_ts)

    h_max = max(HORIZONS)
    sel = _select_analogs(asof_ts, state_subset, X, h_max=h_max, n_scn=N_SCN)
    print(f"Analog selection: {sel.source} | n={len(sel.dates)}")

    scenarios: list[dict] = []
    if sel.source == "rolling_fallback":
        print(f"  ⚠ insufficient analogs ({len(sel.dates)} < {MIN_ANALOGS}); "
              f"falling back to rolling-{ROLLING_FALLBACK_WINDOW} quantiles")
        scenarios = _fallback_rolling(asof_ts, state, ASSETS, HORIZONS)
    else:
        for scenario_id, (analog_dt, analog_w) in enumerate(zip(sel.dates, sel.weights)):
            for asset in ASSETS:
                if asset not in state.columns:
                    continue
                for h in HORIZONS:
                    fwd = _forward_cum_return(state[asset], h)
                    v = fwd.loc[analog_dt] if analog_dt in fwd.index else np.nan
                    if not np.isfinite(v):
                        continue
                    scenarios.append({
                        "asof_date": asof_ts,
                        "scenario_id": int(scenario_id),
                        "analog_date": analog_dt,
                        "asset": asset,
                        "horizon": int(h),
                        "log_return": float(v),
                        "weight": float(analog_w),
                    })

    scn_df = pd.DataFrame(scenarios)

    # Summary statistics
    summary_rows = []
    if not scn_df.empty:
        for (asset, h), grp in scn_df.groupby(["asset", "horizon"]):
            w = grp["weight"].values
            r = grp["log_return"].values
            # Weighted quantiles via cumulative distribution
            order = np.argsort(r)
            r_sorted = r[order]
            w_sorted = w[order]
            cum = np.cumsum(w_sorted) / w_sorted.sum()
            def q(p):
                idx = np.searchsorted(cum, p)
                idx = min(idx, len(r_sorted) - 1)
                return float(r_sorted[idx])
            mean_v = float(np.average(r, weights=w))
            std_v = float(np.sqrt(np.average((r - mean_v) ** 2, weights=w)))
            summary_rows.append({
                "asof_date": asof_ts,
                "asset": asset,
                "horizon": int(h),
                "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
                "mean": mean_v, "std": std_v,
                "n_scn": int(len(grp)),
            })
    sum_df = pd.DataFrame(summary_rows)

    # Persist
    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn_df.to_parquet(out_dir / "forecast_scenarios.parquet", index=False)
    sum_df.to_parquet(out_dir / "forecast_summary.parquet", index=False)

    # Doc
    doc_path = Path(ANALYSIS_DIR) / "v2" / f"forecast_{asof_str}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(_render_doc(sel, scn_df, sum_df, asof_str))

    print(f"\n✓ scenarios → {out_dir / 'forecast_scenarios.parquet'}  ({len(scn_df)} rows)")
    print(f"✓ summary   → {out_dir / 'forecast_summary.parquet'}  ({len(sum_df)} rows)")
    print(f"✓ doc       → {doc_path}")

    # Terminal glance at 6m forecast
    if not sum_df.empty:
        six = sum_df[sum_df["horizon"] == 6].sort_values("asset")
        print("\n6-month forecast summary (log return):")
        for r in six.itertuples():
            print(f"  {r.asset:<14} p10={r.p10:+.3f} p50={r.p50:+.3f} "
                  f"p90={r.p90:+.3f} mean={r.mean:+.3f}")

    meta = base_meta(
        layer="forecast",
        data_asof=asof_str,
        model_version=f"kaf_baseline_alpha{ALPHA_GLOBAL_REGIME}_beta{BETA_ASSET_REGIME}",
        extra={
            "method": "kaf_baseline" if sel.source == "kaf" else "rolling_fallback",
            "joint_scenario": True,   # contract: same scenario_id → same analog_date
            "n_scenarios": int(scn_df["scenario_id"].nunique()) if not scn_df.empty else 0,
            "n_assets": int(scn_df["asset"].nunique()) if not scn_df.empty else 0,
            "horizons": HORIZONS,
            "state_features": STATE_FEATURES,
            "assets": ASSETS,
            "upgrade_path": "diffusion-map kernel + delay embedding + tethering",
            "regime_filter": {
                "alpha_global": ALPHA_GLOBAL_REGIME,
                "beta_asset": BETA_ASSET_REGIME,
                "enabled": (Path(DATA_DIR) / "v2" / "global_templates.parquet").exists()
                           and (Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet").exists(),
            },
            "scaler": scaler_meta,
        },
    )
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _render_doc(
    sel: AnalogSelection,
    scn_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    asof: str,
) -> str:
    lines = [
        f"# KAF Forecast — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · method: {sel.source} · "
        f"n_scenarios: {scn_df['scenario_id'].nunique() if not scn_df.empty else 0}",
        "",
        "> **Baseline**: combined state + regime distance "
        f"(α={ALPHA_GLOBAL_REGIME}, β={BETA_ASSET_REGIME}), softmax weighting, "
        "joint scenarios (same scenario_id = same historical analog month). "
        "Upgrade path: diffusion kernel, delay embedding, trajectory tethering.",
        "",
    ]
    if sel.source == "kaf" and len(sel.dates):
        lines.append("## Top-10 analog months (combined state + regime distance)")
        lines.append("")
        lines.append("| rank | analog month | d_total | d_state | d_regime | weight |")
        lines.append("|---|---|---|---|---|---|")
        for i in range(min(10, len(sel.dates))):
            lines.append(
                f"| {i+1} | {sel.dates[i].date()} | "
                f"{sel.distances[i]:.2f} | {sel.state_distances[i]:.2f} | "
                f"{sel.regime_distances[i]:.2f} | {sel.weights[i]:.3f} |"
            )
        lines.append("")

    if not sum_df.empty:
        lines.append("## Forecast summary (log return)")
        lines.append("")
        for h in sorted(sum_df["horizon"].unique()):
            lines.append(f"### horizon = {h}m")
            lines.append("")
            lines.append("| asset | p10 | p50 | p90 | mean | std | n_scn |")
            lines.append("|---|---|---|---|---|---|---|")
            sub = sum_df[sum_df["horizon"] == h].sort_values("asset")
            for r in sub.itertuples():
                lines.append(
                    f"| {r.asset} | {r.p10:+.3f} | {r.p50:+.3f} | "
                    f"{r.p90:+.3f} | {r.mean:+.3f} | {r.std:.3f} | {r.n_scn} |"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
