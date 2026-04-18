#!/usr/bin/env python3
"""
Phase 3: Kernel Analog Forecasting (baseline).

For the query state at time t, find N historical dates with similar macro
state and pull their realized forward paths. Emit as *joint* scenarios:
every asset in scenario_id=k uses the same analog date → the cross-asset
correlation observed historically is preserved (required by BL/SP-CVaR).

Baseline:
- State vector: standardized CLUSTERING_FEATURES (19 features; rates,
  credit, growth, inflation, vix, bcom_yoy, shiller_earnings_yield)
- Distance: Euclidean on standardized state
- Selection: top-N_SCN nearest analogs that have complete forward H-month
  history (and are not in the forward window of the query)
- Weights: softmax(-distance / auto_temp) normalized over selected
- Horizons: 1, 3, 6, 12 months (18m excluded — too few independent samples)
- Fallback chain (per codex review):
  1) Regime-conditional historical distribution (not yet implemented)
  2) VAR baseline (not yet)
  3) This KAF baseline ← we're here
  4) Unconditional rolling-120 empirical quantiles ← triggered if < MIN_ANALOGS

Upgrade path (Phase 3b):
- Diffusion map / dynamics-adapted kernel (Alexander & Giannakis)
- Delay embedding (q-lag concatenated state vectors)
- Tethering: trajectory similarity over past 24m weighted 1, 1/2, 1/3
- Regime-conditional filter using template_asset_joint_current

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore

SCHEMA_VERSION = "v2.1"
HORIZONS = [1, 3, 6, 12]
N_SCN = 200
MIN_ANALOGS = 50
ROLLING_FALLBACK_WINDOW = 120

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
    distances: np.ndarray
    weights: np.ndarray
    source: str             # "kaf" | "rolling_fallback"


def _load_scaler() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (mean, scale, feature_names) from global_template centroids."""
    npz = np.load(Path(DATA_DIR) / "v2" / "global_template_centroids.npz",
                  allow_pickle=True)
    feature_names = [str(s) for s in npz["feature_names"]]
    return npz["scaler_mean"], npz["scaler_scale"], feature_names


def _forward_cum_return(ret_series: pd.Series, h: int) -> pd.Series:
    """Cumulative log return from t+1 to t+h, indexed at t."""
    return ret_series.shift(-1).rolling(h).sum().shift(-(h - 1))


def _build_state_matrix(state: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (state[STATE_FEATURES] dropna, standardized ndarray)."""
    missing = [f for f in STATE_FEATURES if f not in state.columns]
    if missing:
        sys.exit(f"STATE_FEATURES missing from state: {missing}")
    s = state[STATE_FEATURES].dropna()
    mean, scale, fnames = _load_scaler()
    # Scaler was trained on a subset (clustering features); re-standardize in
    # place using available training stats where feature names align.
    scaler_map = {n: (mean[i], scale[i]) for i, n in enumerate(fnames)}
    standardized = np.empty_like(s.values, dtype="float64")
    for j, f in enumerate(STATE_FEATURES):
        if f in scaler_map:
            m, sc = scaler_map[f]
        else:
            # Feature not in original scaler (bcom_yoy, shiller_earnings_yield
            # were added post-global_template training); standardize on the
            # series itself. Self-standardization is fine — global_template
            # captures the rate/credit/macro subspace; added features get
            # their own z-score.
            m = s[f].mean()
            sc = s[f].std() or 1.0
        standardized[:, j] = (s[f].values - m) / (sc if sc > 0 else 1.0)
    return s, standardized


def _select_analogs(
    asof: pd.Timestamp,
    state_subset: pd.DataFrame,
    X: np.ndarray,
    h_max: int,
    n_scn: int,
) -> AnalogSelection:
    """Pick N nearest analogs with complete H-month forward history."""
    if asof not in state_subset.index:
        sys.exit(f"asof {asof} not in state panel")
    query_idx = state_subset.index.get_loc(asof)
    query = X[query_idx]

    # Candidate pool: exclude the last h_max+1 months (forward incomplete)
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
            weights=np.array([]),
            source="rolling_fallback",
        )

    X_cand = X[cand_idx]
    dist = np.linalg.norm(X_cand - query, axis=1)
    top_n = min(n_scn, len(cand_idx))
    top_ranks = np.argsort(dist)[:top_n]
    sel_cand_idx = cand_idx[top_ranks]
    sel_dates = state_subset.index[sel_cand_idx]
    sel_dist = dist[top_ranks]

    # Softmax weighting with auto-temperature = median distance (among selected)
    temperature = max(np.median(sel_dist), 1e-6)
    logits = -sel_dist / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    return AnalogSelection(dates=sel_dates, distances=sel_dist,
                           weights=weights, source="kaf")


def _fallback_rolling(
    asof: pd.Timestamp,
    state: pd.DataFrame,
    assets: list[str],
    horizons: list[int],
) -> list[dict]:
    """Unconditional rolling-120 empirical quantiles as synthetic scenarios."""
    rows = []
    window = state.loc[:asof].tail(ROLLING_FALLBACK_WINDOW)
    for a in assets:
        if f"{a}_ret" not in state.columns and a not in state.columns:
            continue
        ret_col = a if a in state.columns else f"{a}_ret"
        for h in horizons:
            fwd = _forward_cum_return(state[ret_col], h).loc[window.index].dropna()
            if len(fwd) < 12:
                continue
            # Emit each historical forward return as a weak scenario
            for sid, (dt, v) in enumerate(fwd.items()):
                rows.append({
                    "asof_date": asof,
                    "scenario_id": sid,
                    "asset": a,
                    "horizon": h,
                    "log_return": float(v),
                    "weight": 1.0 / len(fwd),
                })
    return rows


def run(asof: str | None = None) -> dict:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}")
    state = pd.read_parquet(state_path)

    state_subset, X = _build_state_matrix(state)
    asof_ts = pd.Timestamp(asof) if asof else state_subset.index.max()
    if asof_ts not in state_subset.index:
        # Snap to nearest <= asof
        prev = state_subset.index[state_subset.index <= asof_ts]
        if len(prev) == 0:
            sys.exit(f"no state data <= {asof_ts}")
        asof_ts = prev[-1]
    asof_str = asof_ts.strftime("%Y-%m-%d")

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

    meta = {
        "schema_version": SCHEMA_VERSION,
        "asof": asof_str,
        "method": "kaf_baseline" if sel.source == "kaf" else "rolling_fallback",
        "n_scenarios": int(scn_df["scenario_id"].nunique()) if not scn_df.empty else 0,
        "n_assets": int(scn_df["asset"].nunique()) if not scn_df.empty else 0,
        "horizons": HORIZONS,
        "state_features": STATE_FEATURES,
        "assets": ASSETS,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "upgrade_path": "diffusion-map kernel + delay embedding + tethering + regime filter",
    }
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
        "> **Baseline**: Euclidean state distance, softmax weighting, joint "
        "scenarios (same scenario_id = same historical analog month = "
        "consistent cross-asset paths). Upgrade path: diffusion kernel, "
        "delay embedding, trajectory tethering, regime-conditional pool.",
        "",
    ]
    if sel.source == "kaf" and len(sel.dates):
        lines.append("## Top-10 analog months (by state similarity)")
        lines.append("")
        lines.append("| rank | analog month | distance | weight |")
        lines.append("|---|---|---|---|")
        for i in range(min(10, len(sel.dates))):
            lines.append(
                f"| {i+1} | {sel.dates[i].date()} | "
                f"{sel.distances[i]:.2f} | {sel.weights[i]:.3f} |"
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
