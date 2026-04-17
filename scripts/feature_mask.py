#!/usr/bin/env python3
"""
Phase 1: asset-aware mutual-information feature mask.

For each (asset, forward-return horizon, state feature) compute MI between
the feature at time t and the cumulative log return from t+1 to t+h.
This is the baseline before upgrading to NN attention masks (Phase 5).

Contract (schema v2.1):
  Input:  data/state_features.parquet (monthly, T×F)
  Output: data/v2/feature_mask.parquet (long format)
            columns: asset, feature, horizon, mi_raw, mi_norm, n_pairs
          analysis/v2/mask_heatmap_{asof}.md (doc + markdown heatmap)

Normalization: per (asset, horizon), mi_raw sum-scaled to 1 across features.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore

SCHEMA_VERSION = "v2.1"
HORIZONS = [1, 3, 6, 12]  # months; 18m excluded (too few independent samples)

# Return features treated as target assets.
ASSET_FEATURES = [
    "spx_ret", "hsi_ret", "btc_ret",
    "oil_ret", "natgas_ret", "gold_ret", "silver_ret", "copper_ret", "bcom_ret",
    "dxy_ret",
]


@dataclass
class MaskResult:
    table: pd.DataFrame          # long format
    meta: dict


def _forward_log_return(asset_ret: pd.Series, h: int) -> pd.Series:
    """
    Cumulative log return from t+1 to t+h, indexed at t (no look-ahead).
    asset_ret is already a monthly log return series.
    """
    # Shift −1 so the next-month return lands at index t, then rolling-sum forward.
    future = asset_ret.shift(-1).rolling(h).sum().shift(-(h - 1))
    return future


def compute_mask(
    state: pd.DataFrame,
    asset_features: list[str],
    horizons: list[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Returns long-format DataFrame with columns:
      asset, feature, horizon, mi_raw, mi_norm, n_pairs
    """
    present_assets = [a for a in asset_features if a in state.columns]
    missing = set(asset_features) - set(present_assets)
    if missing:
        print(f"  ⚠ assets not in state: {sorted(missing)}")

    # Predictors: all non-asset columns (and not y*_diff derivative artefacts).
    predictors = [c for c in state.columns if c not in asset_features]

    rows: list[dict] = []
    for asset in present_assets:
        for h in horizons:
            y = _forward_log_return(state[asset], h)
            mi_per_feat: dict[str, tuple[float, int]] = {}
            for feat in predictors:
                x = state[feat]
                aligned = pd.concat([x, y], axis=1).dropna()
                n = len(aligned)
                if n < 36:
                    mi_per_feat[feat] = (np.nan, n)
                    continue
                xv = aligned.iloc[:, 0].values.reshape(-1, 1)
                yv = aligned.iloc[:, 1].values
                try:
                    mi = mutual_info_regression(
                        xv, yv, discrete_features=False, random_state=random_state
                    )[0]
                except Exception as e:
                    print(f"    MI failed for ({asset}, {feat}, h={h}): {e}")
                    mi = np.nan
                mi_per_feat[feat] = (float(mi) if np.isfinite(mi) else np.nan, n)

            raw_series = pd.Series({f: v[0] for f, v in mi_per_feat.items()})
            total = np.nansum(raw_series.values)
            norm_series = raw_series / total if total and np.isfinite(total) else raw_series * np.nan

            for feat, (mi, n) in mi_per_feat.items():
                rows.append({
                    "asset": asset,
                    "feature": feat,
                    "horizon": h,
                    "mi_raw": mi,
                    "mi_norm": float(norm_series[feat]) if np.isfinite(norm_series[feat]) else np.nan,
                    "n_pairs": n,
                })

    return pd.DataFrame(rows)


def render_heatmap_md(table: pd.DataFrame, asof: str) -> str:
    min_n = int(table["n_pairs"].min()) if len(table) else 0
    max_n = int(table["n_pairs"].max()) if len(table) else 0
    lines = [
        f"# Feature Mask Heatmap — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · metric: MI (mutual information) · normalized per (asset, horizon)",
        "",
        "> **Caveat**: MI is sensitive to sample size. Sample counts vary widely "
        f"across feature pairs (min={min_n}, max={max_n}). Short-history series "
        "(e.g. SOFR from 2018, BEI from 2003) tend to score artificially high. "
        "See `n_pairs` column in `feature_mask.parquet` before interpreting ranks. "
        "Baseline MI; upgrade path: NN attention mask (Phase 5) or pair-aligned "
        "rolling MI on common window.",
        "",
        "## Top-5 features per (asset, horizon) with n_pairs",
        "",
    ]
    for asset, grp in table.groupby("asset"):
        lines.append(f"### {asset}")
        lines.append("")
        lines.append("| horizon | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |")
        lines.append("|---|---|---|---|---|---|")
        for h, sub in grp.groupby("horizon"):
            top = sub.sort_values("mi_raw", ascending=False).head(5)
            cells = [
                f"{r.feature} ({r.mi_norm:.2f}, n={r.n_pairs})"
                if np.isfinite(r.mi_norm)
                else f"{r.feature} (–, n={r.n_pairs})"
                for r in top.itertuples()
            ]
            lines.append(f"| {h}m | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Full MI (horizon=6m, normalized)")
    lines.append("")
    wide = (
        table[table["horizon"] == 6]
        .pivot(index="feature", columns="asset", values="mi_norm")
        .round(3)
    )
    header = "| feature | " + " | ".join(str(c) for c in wide.columns) + " |"
    sep = "|" + "|".join(["---"] * (len(wide.columns) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for feat, row in wide.iterrows():
        cells = [f"{v:.3f}" if np.isfinite(v) else "–" for v in row.values]
        lines.append(f"| {feat} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def run(asof: str | None = None) -> MaskResult:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}. Run build_state_features.py first.")

    state = pd.read_parquet(state_path)
    print(f"Loaded state_features: {state.shape[0]} rows × {state.shape[1]} cols")

    asof_str = asof or state.index.max().strftime("%Y-%m-%d")

    table = compute_mask(state, ASSET_FEATURES, HORIZONS)
    if table.empty:
        sys.exit("empty mask — no assets matched")

    # Save
    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_mask.parquet"
    table.to_parquet(out_path, index=False)

    # Heatmap doc
    analysis_v2 = Path(ANALYSIS_DIR) / "v2"
    analysis_v2.mkdir(parents=True, exist_ok=True)
    doc_path = analysis_v2 / f"mask_heatmap_{asof_str}.md"
    doc_path.write_text(render_heatmap_md(table, asof_str))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "asof": asof_str,
        "assets": [a for a in ASSET_FEATURES if a in state.columns],
        "horizons": HORIZONS,
        "n_features_scored": table["feature"].nunique(),
        "n_rows": len(table),
    }
    (out_dir / "feature_mask_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓ feature_mask → {out_path}  ({len(table)} rows)")
    print(f"✓ heatmap doc → {doc_path}")

    # Brief stdout report
    print("\nTop-3 features per asset (horizon=6m):")
    for asset, grp in table[table["horizon"] == 6].groupby("asset"):
        top = grp.sort_values("mi_raw", ascending=False).head(3)
        feats = ", ".join(f"{r.feature}({r.mi_norm:.2f})" for r in top.itertuples())
        print(f"  {asset:<14} {feats}")

    return MaskResult(table=table, meta=meta)


if __name__ == "__main__":
    run()
