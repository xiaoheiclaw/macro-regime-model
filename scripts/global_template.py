#!/usr/bin/env python3
"""
Phase 2a: Global macro template layer (K-Means baseline).

Partitions history into K macro templates based on state features (rates,
credit, vol). Each month is soft-assigned to templates; assignments are
temporally smoothed to avoid one-month flips.

Baseline: K-Means on standardized state vectors.
Upgrade path: Wasserstein-HMM on rolling windows (Boukardagha et al.) —
see [[paper-wasserstein-hmm-regime]].

Contract (schema v2.1):
  Input:  data/state_features.parquet (monthly, T×F)
  Output: data/v2/global_templates.parquet
            columns: template_id (argmax, int), prob_0..prob_{K-1}, n_features_used
          data/v2/global_template_centroids.npz
            centroids_standardized (K, F), centroids_raw (K, F), feature_names,
            K, common_start, scaler_mean, scaler_scale
          analysis/v2/global_templates_{asof}.md
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore

SCHEMA_VERSION = "v2.1"
K = 4                    # default template count (tunable)
SEED = 42
SMOOTHING_LAMBDA = 0.2   # prob_t = (1-λ)*raw + λ*prob_{t-1}
COMMON_START = "1990-02-01"  # first month with y10y_diff + VIX + Moody's spreads

# State features used for clustering. Exclude *_ret (flow variables dominate
# variance; regime is about state levels/spreads). Include growth+inflation
# via ALFRED vintage features (cpi_yoy, ip_yoy, payroll_yoy, unrate) so
# templates reflect the 4-quadrant macro framework, not just policy+risk.
# Use Moody's BAA-AAA for credit (long history, from 1990); ICE HY/IG OAS
# excluded because FRED history was truncated by ICE to 2023+.
CLUSTERING_FEATURES = [
    # Rates / policy
    "y3m", "y2y", "y5y", "y10y", "y30y",
    "y10y_diff",
    "curve_pc1_level", "curve_pc2_slope", "curve_pc3_curv",
    "yc_2s10s",
    # Risk / credit
    "vix_level",
    "moody_baa_aaa", "moody_baa_10y",
    # Growth (ALFRED vintage)
    "ip_yoy", "payroll_yoy", "unrate",
    # Inflation (ALFRED vintage)
    "cpi_yoy",
    # Commodities (Phase 0c). BCOM composite has longest yfinance history;
    # oil/copper YoY start 2001 — excluded from clustering to preserve
    # pre-2001 regimes (dot-com, Asian crisis). They remain in state for MI.
    "bcom_yoy",
    # Equity valuation (Phase 0c)
    "shiller_earnings_yield",
]


@dataclass
class TemplateResult:
    assignments: pd.DataFrame      # columns: template_id, prob_0..prob_{K-1}, n_features_used
    centroids_standardized: np.ndarray  # (K, F)
    centroids_raw: np.ndarray           # (K, F)
    features: list[str]
    scaler: StandardScaler
    inertia: float
    silhouette: float
    meta: dict


def soft_assign(
    X: np.ndarray, centroids: np.ndarray, temperature: float | None = None
) -> np.ndarray:
    """Softmax(-distance/temperature) soft assignment. Temperature auto = median pairwise d."""
    d = cdist(X, centroids, metric="euclidean")
    if temperature is None:
        temperature = max(np.median(d), 1e-6)
    logits = -d / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def temporal_smooth(probs: np.ndarray, lam: float) -> np.ndarray:
    smoothed = probs.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = (1 - lam) * probs[i] + lam * smoothed[i - 1]
    # renormalize for numeric drift
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def fit(
    state: pd.DataFrame,
    k: int = K,
    asof: pd.Timestamp | None = None,
) -> TemplateResult:
    """
    Fit K-Means on state vectors. If asof is provided, training is restricted
    to rows ≤ asof (expanding-window backtest safe). With asof=None, uses all
    data (original live-mode behavior).
    """
    avail = [f for f in CLUSTERING_FEATURES if f in state.columns]
    missing = set(CLUSTERING_FEATURES) - set(avail)
    if missing:
        print(f"  ⚠ clustering features missing from state: {sorted(missing)}")

    X_df = state[avail].loc[COMMON_START:].dropna()
    if asof is not None:
        X_df = X_df.loc[:asof]
    print(f"Common window: {X_df.index.min().date()} → {X_df.index.max().date()}"
          f" ({len(X_df)} rows, {len(avail)} features)")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    km.fit(X)

    # Order templates by cluster size so template_id is reproducible across runs
    labels_raw = km.labels_
    _, counts = np.unique(labels_raw, return_counts=True)
    order = np.argsort(-counts)  # largest first
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels_raw])
    centroids_std = km.cluster_centers_[order]
    centroids_raw = scaler.inverse_transform(centroids_std)

    # Soft assign using the reordered centroids
    probs_raw = soft_assign(X, centroids_std)
    probs = temporal_smooth(probs_raw, SMOOTHING_LAMBDA)
    template_id = probs.argmax(axis=1)

    sizes = np.bincount(template_id, minlength=k)
    sil = silhouette_score(X, labels) if k > 1 else float("nan")

    assignments = pd.DataFrame(
        {
            "template_id": template_id.astype(int),
            **{f"prob_{i}": probs[:, i] for i in range(k)},
            "n_features_used": len(avail),
        },
        index=X_df.index,
    )
    assignments.index.name = "date"

    meta = {
        "schema_version": SCHEMA_VERSION,
        "method": "kmeans_baseline",
        "K": k,
        "features_used": avail,
        "common_start": COMMON_START,
        "n_rows": len(assignments),
        "template_sizes": {int(i): int(s) for i, s in enumerate(sizes)},
        "inertia": float(km.inertia_),
        "silhouette": float(sil),
        "smoothing_lambda": SMOOTHING_LAMBDA,
        "seed": SEED,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "upgrade_path": "Wasserstein-HMM on rolling windows (Boukardagha et al.)",
    }

    return TemplateResult(
        assignments=assignments,
        centroids_standardized=centroids_std,
        centroids_raw=centroids_raw,
        features=avail,
        scaler=scaler,
        inertia=float(km.inertia_),
        silhouette=float(sil),
        meta=meta,
    )


def render_doc(res: TemplateResult, asof: str) -> str:
    k = res.meta["K"]
    lines = [
        f"# Global Macro Templates — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · method: K-Means baseline · K={k} · "
        f"common window from {COMMON_START} · silhouette={res.silhouette:.3f}",
        "",
        "> **Note**: baseline implementation. Full Wasserstein-HMM "
        "(Boukardagha) upgrade pending. K-Means on standardized state vectors "
        "(rates/credit/vol, no returns). Temperature auto (median pairwise d). "
        f"Smoothing λ={SMOOTHING_LAMBDA}.",
        "",
        "## Template sizes (argmax assignment)",
        "",
        "| template_id | count | fraction |",
        "|---|---|---|",
    ]
    sizes = res.assignments["template_id"].value_counts().sort_index()
    N = len(res.assignments)
    for tid in range(k):
        cnt = int(sizes.get(tid, 0))
        lines.append(f"| T{tid} | {cnt} | {cnt / N:.1%} |")
    lines.append("")

    # Centroid loadings
    lines.append("## Template centroids (raw scale)")
    lines.append("")
    header = "| feature | " + " | ".join(f"T{i}" for i in range(k)) + " |"
    sep = "|" + "|".join(["---"] * (k + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for fi, fname in enumerate(res.features):
        vals = res.centroids_raw[:, fi]
        cells = [f"{v:.2f}" for v in vals]
        lines.append(f"| {fname} | " + " | ".join(cells) + " |")
    lines.append("")

    # Recent assignments
    lines.append("## Recent template history (last 24 months)")
    lines.append("")
    recent = res.assignments.tail(24)
    lines.append("| date | template | top prob | 2nd prob |")
    lines.append("|---|---|---|---|")
    for dt, row in recent.iterrows():
        ranked = sorted(
            [(row[f"prob_{i}"], i) for i in range(k)],
            reverse=True,
        )
        lines.append(
            f"| {dt.date()} | T{int(row['template_id'])} | "
            f"T{ranked[0][1]}:{ranked[0][0]:.2f} | T{ranked[1][1]}:{ranked[1][0]:.2f} |"
        )
    lines.append("")

    # Flip rate
    flips = (res.assignments["template_id"].diff() != 0).sum() - 1
    lines.append(f"## Stability")
    lines.append("")
    lines.append(f"- Template flips (argmax changes): **{flips} / {N-1}** months ({flips / (N-1):.1%})")
    lines.append(f"- K-Means inertia: {res.inertia:.1f}")
    lines.append(f"- Silhouette: {res.silhouette:.3f}")
    lines.append("")
    return "\n".join(lines)


def run(asof: str | None = None) -> dict:
    state_path = Path(DATA_DIR) / "state_features.parquet"
    if not state_path.exists():
        sys.exit(f"missing {state_path}. Run build_state_features.py first.")
    state = pd.read_parquet(state_path)
    print(f"Loaded state_features: {state.shape}")

    asof_ts = pd.Timestamp(asof) if asof is not None else None
    res = fit(state, k=K, asof=asof_ts)
    print(f"K-Means inertia: {res.inertia:.2f} | silhouette: {res.silhouette:.3f}")
    print(f"Template sizes: {res.meta['template_sizes']}")

    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.assignments.to_parquet(out_dir / "global_templates.parquet")
    np.savez(
        out_dir / "global_template_centroids.npz",
        centroids_standardized=res.centroids_standardized,
        centroids_raw=res.centroids_raw,
        feature_names=np.array(res.features),
        K=np.array([K]),
        common_start=np.array([COMMON_START]),
        scaler_mean=res.scaler.mean_,
        scaler_scale=res.scaler.scale_,
    )

    asof_str = asof or res.assignments.index.max().strftime("%Y-%m-%d")
    doc_path = Path(ANALYSIS_DIR) / "v2" / f"global_templates_{asof_str}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(render_doc(res, asof_str))

    print(f"\n✓ global_templates → {out_dir / 'global_templates.parquet'}")
    print(f"✓ centroids        → {out_dir / 'global_template_centroids.npz'}")
    print(f"✓ doc              → {doc_path}")

    recent = res.assignments.tail(1).iloc[0]
    k = K
    rp = {i: round(float(recent[f"prob_{i}"]), 3) for i in range(k)}
    print(
        f"\nLatest ({res.assignments.index[-1].date()}): "
        f"template=T{int(recent['template_id'])} probs={rp}"
    )
    return res.meta


if __name__ == "__main__":
    run()
