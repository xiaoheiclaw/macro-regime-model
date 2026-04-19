#!/usr/bin/env python3
"""
v3 proof-of-concept: simplified regime-conditional joint forecast.

Observation from Phase 3b findings: the RegimeCond benchmark (sample
forward paths from past months in same regime) outperforms v2 KAF at
long horizons on most assets. The Phase 1 mask + Phase 3 state-distance
analog ranking machinery is over-engineered for allocation-relevant
horizons.

v3 architecture:
  1. Reuse Phase 2 global_template layer (K=4 macro regimes)
  2. Skip Phase 1 (mask) and the KAF state-distance ranking
  3. At asof, sample N_SCN analog dates uniformly from past months with
     same global regime argmax (no state distance, no regime joint
     Frobenius, no tether)
  4. Emit joint scenarios with same scenario_id → same analog date
     (preserves cross-asset correlation)
  5. Parametric fallback for BTC/Bond unchanged (short history or
     macro-independent assets)

Output schema: identical to forecast.py → data/v2/forecast_scenarios.parquet
So v2_allocation.run() consumes either v2 or v3 scenarios
transparently.

Contract:
  Input:
    data/state_features.parquet
    data/v2/global_templates.parquet  (current-asof regime assignments)
  Output: overwrites data/v2/forecast_scenarios.parquet with v3 scenarios
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore
from lib.schema import SCHEMA_VERSION, base_meta  # type: ignore

# Reuse v2 constants
from forecast import (
    HORIZONS, N_SCN, MIN_ANALOGS, ROLLING_FALLBACK_WINDOW,
    PARAMETRIC_FALLBACK_ASSETS, PARAMETRIC_WINDOW_M, ASSETS,
    _forward_cum_return, _fallback_rolling,
)


def _regime_eligible_analogs(
    tmpl_series: pd.Series,
    asof: pd.Timestamp,
    h_max: int,
) -> pd.DatetimeIndex:
    """Past asofs with same global regime argmax AND ≥ h_max months
    forward history (so realized forward returns exist)."""
    if asof not in tmpl_series.index:
        return pd.DatetimeIndex([])
    cur = int(tmpl_series.loc[asof])
    last_valid = asof - pd.offsets.MonthEnd(h_max)
    past = tmpl_series.loc[:last_valid]
    return past[past == cur].index


def run(asof: str | None = None) -> dict:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}")
    state = pd.read_parquet(state_path)

    g_path = Path(DATA_DIR) / "v2" / "global_templates.parquet"
    if not g_path.exists():
        sys.exit(f"missing {g_path} — run global_template first")
    g_df = pd.read_parquet(g_path)
    tmpl_series = g_df["template_id"]

    all_idx = state.index
    asof_ts = pd.Timestamp(asof) if asof else all_idx.max()
    if asof_ts not in all_idx:
        prev = all_idx[all_idx <= asof_ts]
        asof_ts = prev[-1] if len(prev) else asof_ts
    asof_str = asof_ts.strftime("%Y-%m-%d")

    h_max = max(HORIZONS)
    eligible = _regime_eligible_analogs(tmpl_series, asof_ts, h_max)
    print(f"v3 forecast @ {asof_str}: current regime = "
          f"{int(tmpl_series.loc[asof_ts]) if asof_ts in tmpl_series.index else 'NA'}, "
          f"{len(eligible)} eligible analogs")

    scenarios: list[dict] = []
    method = "v3_regime_conditional"
    if len(eligible) < MIN_ANALOGS:
        print(f"  < {MIN_ANALOGS} analogs → rolling fallback")
        scenarios = _fallback_rolling(asof_ts, state, ASSETS, HORIZONS)
        method = "rolling_fallback"
    else:
        rng = np.random.default_rng(asof_ts.toordinal())
        sampled = rng.choice(eligible.values, size=N_SCN, replace=True)
        analog_w = 1.0 / N_SCN

        # Parametric fallback stats for short-history / macro-indep assets
        parametric_stats: dict[str, tuple[float, float]] = {}
        for asset in PARAMETRIC_FALLBACK_ASSETS:
            if asset not in state.columns:
                continue
            hist = state[asset].loc[:asof_ts].dropna().tail(PARAMETRIC_WINDOW_M)
            if len(hist) >= 24:
                parametric_stats[asset] = (float(hist.mean()), float(hist.std()))

        for scenario_id, analog_dt in enumerate(sampled):
            analog_dt = pd.Timestamp(analog_dt)
            for asset in ASSETS:
                if asset not in state.columns:
                    continue
                for h in HORIZONS:
                    if asset in parametric_stats:
                        mu_m, sigma_m = parametric_stats[asset]
                        r = float(rng.normal(mu_m * h, sigma_m * np.sqrt(h)))
                        scenarios.append({
                            "asof_date": asof_ts,
                            "scenario_id": int(scenario_id),
                            "analog_date": analog_dt,
                            "asset": asset,
                            "horizon": int(h),
                            "log_return": r,
                            "weight": float(analog_w),
                        })
                        continue
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

    # Summary stats
    sum_rows = []
    if not scn_df.empty:
        for (asset, h), grp in scn_df.groupby(["asset", "horizon"]):
            r = grp["log_return"].values
            w = grp["weight"].values
            mu = float(np.average(r, weights=w))
            std = float(np.sqrt(np.average((r - mu) ** 2, weights=w)))
            order = np.argsort(r)
            cum = np.cumsum(w[order]) / w.sum()
            def q(p):
                idx = min(np.searchsorted(cum, p), len(r) - 1)
                return float(r[order][idx])
            sum_rows.append({
                "asof_date": asof_ts, "asset": asset, "horizon": int(h),
                "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
                "mean": mu, "std": std, "n_scn": len(grp),
            })
    sum_df = pd.DataFrame(sum_rows)

    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn_df.to_parquet(out_dir / "forecast_scenarios.parquet", index=False)
    sum_df.to_parquet(out_dir / "forecast_summary.parquet", index=False)

    meta = base_meta(
        layer="forecast_v3",
        data_asof=asof_str,
        model_version="v3_regime_conditional",
        extra={
            "method": method,
            "joint_scenario": True,
            "n_scenarios": int(scn_df["scenario_id"].nunique()) if not scn_df.empty else 0,
            "n_assets": int(scn_df["asset"].nunique()) if not scn_df.empty else 0,
            "n_eligible_analogs": len(eligible),
            "horizons": HORIZONS,
            "assets": ASSETS,
            "note": "v3 proof-of-concept: regime-conditional sampling, drops KAF state-distance ranking",
        },
    )
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"✓ v3 scenarios → {len(scn_df)} rows, {scn_df['scenario_id'].nunique() if not scn_df.empty else 0} unique")
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default=None)
    a = ap.parse_args()
    run(asof=a.asof)
