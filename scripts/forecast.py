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
    data/v2/global_templates.parquet            (regime time series)
    data/v2/asset_regime_probs.parquet          (regime time series)
    data/v2/template_asset_joint_current.parquet (query-time joint)
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
# Combined distance: d = d_state + ALPHA_REGIME * d_joint_regime
# where d_joint_regime is the mean-over-assets Frobenius distance between the
# current per-asset (K_g × K_a) joint matrix (from template_asset_joint_current)
# and each candidate's independence-product joint.
# ALPHA tuned so regime term is ~20% of d_state at typical magnitudes
# (d_state ~ 5-7; Frobenius on independence-product simplex ~ 0.5-1.2). Stage B
# should calibrate against CRPS sensitivity.
ALPHA_REGIME = 3.0

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


def _load_regime_artifacts() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Returns (global_probs_by_date, {asset: asset_probs_by_date}, joint_current).

    - global_probs, asset_probs: time-series of regime probabilities
    - joint_current: the template_map.py output at latest asof, long format
      (asset, global_template_id, asset_regime_id, prob_joint) — used as the
      query-time joint distribution. This is how forecast consumes Phase 2c.

    Missing inputs → empty returns; caller falls back gracefully.
    """
    v2 = Path(DATA_DIR) / "v2"
    g_path = v2 / "global_templates.parquet"
    a_path = v2 / "asset_regime_probs.parquet"
    j_path = v2 / "template_asset_joint_current.parquet"
    if not g_path.exists() or not a_path.exists():
        return pd.DataFrame(), {}, pd.DataFrame()
    g = pd.read_parquet(g_path)
    a = pd.read_parquet(a_path)
    a["date"] = pd.to_datetime(a["date"])
    g_cols = sorted([c for c in g.columns if c.startswith("prob_")])
    g = g[g_cols].copy()
    a_cols = sorted([c for c in a.columns if c.startswith("prob_")])
    asset_lookup: dict[str, pd.DataFrame] = {}
    for asset, grp in a.groupby("asset"):
        asset_lookup[str(asset)] = grp.set_index("date")[a_cols]
    joint_current = pd.read_parquet(j_path) if j_path.exists() else pd.DataFrame()
    return g, asset_lookup, joint_current


def _joint_regime_distance(
    query_date: pd.Timestamp,
    candidate_dates: pd.DatetimeIndex,
    global_probs: pd.DataFrame,
    asset_probs: dict[str, pd.DataFrame],
    joint_current: pd.DataFrame,
) -> np.ndarray:
    """
    For each candidate s, compute mean-over-assets Frobenius distance between
    current joint matrix Q[a] (K_g × K_a, from template_map.joint_current) and
    candidate joint matrix C[a, s] = p_g(s) ⊗ p_a(s, a).

    Returns a vector of length len(candidate_dates). Uses zeros where probs
    unavailable — caller combines with state distance.
    """
    n = len(candidate_dates)
    if global_probs.empty:
        return np.zeros(n)

    g_cols = global_probs.columns.tolist()
    K_g = len(g_cols)

    # Build query-side Q[a] from joint_current (per-asset K_g × K_a joint matrix).
    # If joint_current is missing, fall back to independence at query_date.
    q_joint_by_asset: dict[str, np.ndarray] = {}
    if not joint_current.empty and (
        query_date in pd.to_datetime(joint_current["asof"]).values
    ):
        sub = joint_current[pd.to_datetime(joint_current["asof"]) == query_date]
        for asset, grp in sub.groupby("asset"):
            K_a = int(grp["asset_regime_id"].max()) + 1
            mat = np.zeros((K_g, K_a))
            for r in grp.itertuples():
                mat[int(r.global_template_id), int(r.asset_regime_id)] = float(r.prob_joint)
            q_joint_by_asset[str(asset)] = mat
    else:
        # Independence fallback
        if query_date in global_probs.index:
            q_g = global_probs.loc[query_date].values
            for asset, df in asset_probs.items():
                if query_date in df.index:
                    q_a = df.loc[query_date].values
                    q_joint_by_asset[asset] = np.outer(q_g, q_a)

    if not q_joint_by_asset:
        return np.zeros(n)

    # Candidate joint: independence product (we don't have history-wide joint_current)
    per_asset_d: list[np.ndarray] = []
    for asset, q_mat in q_joint_by_asset.items():
        df = asset_probs.get(asset)
        if df is None:
            continue
        cand_has_g = candidate_dates.isin(global_probs.index)
        cand_has_a = candidate_dates.isin(df.index)
        both = cand_has_g & cand_has_a
        if not both.any():
            continue
        d_vec = np.full(n, np.nan)
        for i, date in enumerate(candidate_dates):
            if not both[i]:
                continue
            p_g = global_probs.loc[date].values
            p_a = df.loc[date].values
            c_mat = np.outer(p_g, p_a)
            # Frobenius
            d_vec[i] = np.linalg.norm(q_mat - c_mat)
        per_asset_d.append(d_vec)

    if not per_asset_d:
        return np.zeros(n)
    stacked = np.vstack(per_asset_d)
    with np.errstate(all="ignore"):
        d = np.nanmean(stacked, axis=0)
    fill = np.nanmean(d) if np.isfinite(np.nanmean(d)) else 0.0
    return np.where(np.isnan(d), fill, d)


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

    # Forward-outcome leakage fix: analog s must have s + h_max ≤ asof so its
    # realized forward path is strictly ≤ asof (an analyst at asof actually
    # saw those outcomes). Anchor to asof, NOT to state panel end (which
    # would let forward paths peek past asof during backtest).
    last_valid_analog_date = asof - pd.offsets.MonthEnd(h_max)
    candidates_mask = (state_subset.index <= last_valid_analog_date)
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

    # Regime distance = Frobenius on per-asset joint (g × r) matrices, comparing
    # current Q[a] (from template_map.joint_current) vs candidate C[a,s] = p_g(s)⊗p_a(s,a).
    # This explicitly consumes Phase 2c output (template_asset_joint_current).
    global_probs, asset_probs, joint_current = _load_regime_artifacts()
    d_regime_raw = _joint_regime_distance(asof, cand_dates, global_probs, asset_probs, joint_current)
    d_regime = ALPHA_REGIME * d_regime_raw
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
    # Forward-outcome leakage fix: window end is asof - h_max so forward paths
    # of every scenario date are strictly ≤ asof (no future information).
    h_max = max(horizons)
    window_end = asof - pd.offsets.MonthEnd(h_max)
    window_idx = state.loc[:window_end].tail(ROLLING_FALLBACK_WINDOW).index
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
        model_version=f"kaf_baseline_alpha{ALPHA_REGIME}_jointFrob",
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
                "alpha_regime": ALPHA_REGIME,
                "distance": "frobenius_on_per_asset_joint_(g,r)",
                "query_joint_source": "template_asset_joint_current.parquet",
                "enabled": (Path(DATA_DIR) / "v2" / "global_templates.parquet").exists()
                           and (Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet").exists()
                           and (Path(DATA_DIR) / "v2" / "template_asset_joint_current.parquet").exists(),
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
        "> **Baseline**: combined state + joint-regime Frobenius distance "
        f"(α={ALPHA_REGIME}), softmax weighting, joint scenarios (same "
        "scenario_id = same historical analog month). Query-time joint "
        "comes from template_asset_joint_current (Phase 2c output).",
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
