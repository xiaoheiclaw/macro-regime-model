#!/usr/bin/env python3
"""
Phase 2b: per-asset state classification (Gaussian Mixture baseline).

For each asset independently, fits K=3 Gaussian Mixture on
(monthly_return, rolling_vol_12m, return_zscore_36m) features, then applies
temporal smoothing to reduce one-month label flips. States are ordered by
posterior-weighted mean return so regime_id=0 is always "bear" and
regime_id=K-1 is "bull" across assets.

Baseline. Upgrade path: Shu/Mulvey statistical jump model with explicit
temporal sparsity penalty on state transitions.

Contract (schema v2.1):
  Input:  data/state_features.parquet (needs columns <asset>_ret)
  Output: data/v2/asset_regime_probs.parquet (wide long-by-asset)
            columns: date, asset, regime_id (argmax, int),
                     prob_0..prob_{K-1}
          data/v2/asset_regime_meta.json
          analysis/v2/asset_regimes_{asof}.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore

SCHEMA_VERSION = "v2.1"
K = 3                        # bear / sideways / bull
SMOOTHING_LAMBDA = 0.2
SEED = 42
VOL_WINDOW = 12
ZSCORE_WINDOW = 36
MIN_OBS = 48                 # minimum rows required to fit

ASSETS = [
    "spx_ret", "hsi_ret", "btc_ret",
    "oil_ret", "natgas_ret", "gold_ret",
    "silver_ret", "copper_ret", "bcom_ret",
    "dxy_ret",
]


@dataclass
class AssetFit:
    asset: str
    assignments: pd.DataFrame   # index=date, cols=regime_id, prob_0..prob_{K-1}
    state_stats: list[dict]     # per regime: mean_ret, mean_vol, weight
    bic: float
    aic: float


def _build_features(ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=ret.index)
    df["ret"] = ret
    df["vol"] = ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    roll_mean = ret.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW // 3).mean()
    roll_std = ret.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW // 3).std()
    df["ret_z"] = (ret - roll_mean) / roll_std
    return df


def _temporal_smooth(probs: np.ndarray, lam: float) -> np.ndarray:
    out = probs.copy()
    for i in range(1, len(out)):
        out[i] = (1 - lam) * probs[i] + lam * out[i - 1]
    out /= out.sum(axis=1, keepdims=True)
    return out


def fit_asset(
    asset: str,
    ret: pd.Series,
    k: int = K,
    asof: pd.Timestamp | None = None,
) -> AssetFit | None:
    """
    Fit per-asset GMM. If asof is provided, training is restricted to
    observations ≤ asof (expanding-window backtest safe).
    """
    feat = _build_features(ret)
    if asof is not None:
        feat = feat.loc[:asof]
    feat = feat.dropna()
    if len(feat) < MIN_OBS:
        print(f"  ⚠ {asset}: {len(feat)} obs < {MIN_OBS}, skipped")
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(feat[["ret", "vol", "ret_z"]].values)

    gm = GaussianMixture(
        n_components=k,
        random_state=SEED,
        n_init=10,
        covariance_type="full",
        reg_covar=1e-4,
    )
    gm.fit(X)

    # Reorder states by posterior-weighted mean return (ascending)
    proba = gm.predict_proba(X)
    ret_vals = feat["ret"].values
    vol_vals = feat["vol"].values
    state_mean_ret = []
    for j in range(k):
        w = proba[:, j]
        wsum = w.sum()
        if wsum > 0:
            state_mean_ret.append(np.average(ret_vals, weights=w))
        else:
            state_mean_ret.append(np.nan)
    order = np.argsort(state_mean_ret)
    proba_reordered = proba[:, order]
    proba_smoothed = _temporal_smooth(proba_reordered, SMOOTHING_LAMBDA)
    regime_id = proba_smoothed.argmax(axis=1).astype(int)

    # State-level summary (post-reorder)
    state_stats = []
    for new_j in range(k):
        old_j = order[new_j]
        w = proba[:, old_j]
        wsum = w.sum()
        state_stats.append({
            "regime_id": int(new_j),
            "mean_ret_monthly": float(np.average(ret_vals, weights=w)) if wsum > 0 else float("nan"),
            "mean_vol_monthly": float(np.average(vol_vals, weights=w)) if wsum > 0 else float("nan"),
            "weight": float(gm.weights_[old_j]),
        })

    assignments = pd.DataFrame(
        {
            "regime_id": regime_id,
            **{f"prob_{j}": proba_smoothed[:, j] for j in range(k)},
        },
        index=feat.index,
    )
    assignments.index.name = "date"
    return AssetFit(
        asset=asset,
        assignments=assignments,
        state_stats=state_stats,
        bic=float(gm.bic(X)),
        aic=float(gm.aic(X)),
    )


def render_doc(fits: list[AssetFit], asof: str) -> str:
    k = K
    lines = [
        f"# Per-Asset Regimes — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · method: GMM baseline · K={k} · "
        f"features: (return, vol_{VOL_WINDOW}m, ret_z_{ZSCORE_WINDOW}m) · "
        f"smoothing λ={SMOOTHING_LAMBDA}",
        "",
        "> **Note**: baseline. Upgrade path: Shu/Mulvey statistical jump model "
        "with explicit temporal sparsity penalty on state transitions. States "
        "are reordered per asset so regime_id=0 is always the lowest-return "
        "(bear) state and regime_id=K-1 is the highest-return (bull) state.",
        "",
        "## State statistics per asset",
        "",
        "| asset | regime | mean ret (monthly) | mean vol (monthly) | weight |",
        "|---|---|---|---|---|",
    ]
    for f in fits:
        for stat in f.state_stats:
            rid = stat["regime_id"]
            label = {0: "bear", K - 1: "bull"}.get(rid, "neutral")
            lines.append(
                f"| {f.asset} | T{rid} ({label}) | "
                f"{stat['mean_ret_monthly']:+.3%} | "
                f"{stat['mean_vol_monthly']:.3%} | "
                f"{stat['weight']:.2f} |"
            )
        lines.append("| — | — | — | — | — |")
    lines.append("")

    lines.append("## Current regime per asset (last observation)")
    lines.append("")
    lines.append("| asset | latest date | argmax | P(bear) | P(neutral) | P(bull) |")
    lines.append("|---|---|---|---|---|---|")
    for f in fits:
        last_dt = f.assignments.index[-1]
        last_row = f.assignments.iloc[-1]
        lines.append(
            f"| {f.asset} | {last_dt.date()} | T{int(last_row['regime_id'])} | "
            f"{last_row['prob_0']:.2f} | {last_row['prob_1']:.2f} | {last_row['prob_2']:.2f} |"
        )
    lines.append("")

    # Model fit stats
    lines.append("## Model fit diagnostics")
    lines.append("")
    lines.append("| asset | n_obs | BIC | AIC |")
    lines.append("|---|---|---|---|")
    for f in fits:
        lines.append(f"| {f.asset} | {len(f.assignments)} | {f.bic:.1f} | {f.aic:.1f} |")
    lines.append("")
    return "\n".join(lines)


def run(asof: str | None = None) -> dict:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}. Run build_state_features.py first.")
    state = pd.read_parquet(state_path)
    print(f"Loaded state_features: {state.shape}")

    asof_ts = pd.Timestamp(asof) if asof is not None else None
    fits: list[AssetFit] = []
    for asset in ASSETS:
        if asset not in state.columns:
            print(f"  ⚠ {asset}: not in state, skipped")
            continue
        fit = fit_asset(asset, state[asset], asof=asof_ts)
        if fit is None:
            continue
        fits.append(fit)
        last_row = fit.assignments.iloc[-1]
        print(f"  {asset:<14} n={len(fit.assignments):>3} "
              f"latest T{int(last_row['regime_id'])} "
              f"(bear={last_row['prob_0']:.2f}, bull={last_row[f'prob_{K-1}']:.2f}) "
              f"BIC={fit.bic:.1f}")

    # Long-format consolidated output
    rows = []
    for f in fits:
        df = f.assignments.reset_index()
        df["asset"] = f.asset
        rows.append(df[["date", "asset", "regime_id"] + [f"prob_{j}" for j in range(K)]])
    out = pd.concat(rows, ignore_index=True)

    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "asset_regime_probs.parquet", index=False)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "method": "gmm_baseline",
        "K": K,
        "n_assets": len(fits),
        "assets": [f.asset for f in fits],
        "features": ["ret", f"vol_{VOL_WINDOW}m", f"ret_z_{ZSCORE_WINDOW}m"],
        "smoothing_lambda": SMOOTHING_LAMBDA,
        "min_obs": MIN_OBS,
        "seed": SEED,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "upgrade_path": "Shu/Mulvey statistical jump model",
        "per_asset": {
            f.asset: {
                "n_obs": len(f.assignments),
                "state_stats": f.state_stats,
                "bic": f.bic,
                "aic": f.aic,
            }
            for f in fits
        },
    }
    (out_dir / "asset_regime_meta.json").write_text(json.dumps(meta, indent=2))

    asof_str = asof or max(f.assignments.index.max() for f in fits).strftime("%Y-%m-%d")
    doc_path = Path(ANALYSIS_DIR) / "v2" / f"asset_regimes_{asof_str}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(render_doc(fits, asof_str))

    print(f"\n✓ asset_regime_probs → {out_dir / 'asset_regime_probs.parquet'}  ({len(out)} rows)")
    print(f"✓ meta              → {out_dir / 'asset_regime_meta.json'}")
    print(f"✓ doc               → {doc_path}")
    return meta


if __name__ == "__main__":
    run()
