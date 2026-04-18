#!/usr/bin/env python3
"""
Phase 2c: joint conditional distribution over (global macro template,
asset-specific regime). Addresses codex review concern that the two regime
systems need an explicit semantic map, not implicit mixing.

Three outputs:

1. **Empirical historical joint (argmax counts)**
   For each asset, count (global_template_id, asset_regime_id) pairs across
   history using argmax labels. Gives the conditional distribution
     P(asset_regime | global_template)
   which is the sanity-check table: "when macro is in crisis (T3), is SPX
   usually bear?"

2. **Empirical joint (probability-weighted)**
   Softer version using expected counts from the full posteriors rather than
   argmax.

3. **Current joint state (at asof)**
   P(g, r | t=asof, asset) ≈ p_global(g | t) · p_asset(r | t, asset)
   Assumes independence at point-in-time; Phase 3 analog matcher will use
   this as the query vector against historical joint state sequences.

Contract (schema v2.1):
  Input:
    data/v2/global_templates.parquet
    data/v2/asset_regime_probs.parquet
  Output:
    data/v2/template_asset_empirical.parquet        (long; counts + conditionals)
    data/v2/template_asset_joint_current.parquet    (long; one row per (asset,g,r))
    analysis/v2/template_map_<asof>.md
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

SCHEMA_VERSION = "v2.1"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    g_path = Path(DATA_DIR) / "v2" / "global_templates.parquet"
    a_path = Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet"
    if not g_path.exists() or not a_path.exists():
        sys.exit(f"missing inputs: run global_template + asset_regime first")
    g = pd.read_parquet(g_path)
    a = pd.read_parquet(a_path)
    a["date"] = pd.to_datetime(a["date"])
    return g, a


def _global_prob_columns(g: pd.DataFrame) -> list[str]:
    return sorted([c for c in g.columns if c.startswith("prob_")])


def _asset_prob_columns(a: pd.DataFrame) -> list[str]:
    return sorted([c for c in a.columns if c.startswith("prob_")])


def build_empirical(
    g: pd.DataFrame,
    a: pd.DataFrame,
) -> pd.DataFrame:
    g_cols = _global_prob_columns(g)
    a_cols = _asset_prob_columns(a)
    K_g = len(g_cols)
    K_a = len(a_cols)

    rows = []
    for asset, grp in a.groupby("asset"):
        # Align to dates present in both panels
        grp = grp.set_index("date")
        dates = grp.index.intersection(g.index)
        if len(dates) == 0:
            continue
        p_g = g.loc[dates, g_cols].values            # (T, K_g)
        p_a = grp.loc[dates, a_cols].values          # (T, K_a)

        # Argmax-based counts (discrete labels)
        g_lab = p_g.argmax(axis=1)
        a_lab = p_a.argmax(axis=1)

        # Probability-weighted expected counts: E[1(g=i, r=j)] = Σ_t p_g(i,t) * p_a(j,t)
        expected = p_g.T @ p_a  # (K_g, K_a)

        for gi in range(K_g):
            g_total_counts = int((g_lab == gi).sum())
            g_expected_total = float(p_g[:, gi].sum())
            for aj in range(K_a):
                count = int(((g_lab == gi) & (a_lab == aj)).sum())
                expected_count = float(expected[gi, aj])
                rows.append({
                    "asset": asset,
                    "global_template_id": gi,
                    "asset_regime_id": aj,
                    "count_argmax": count,
                    "expected_count": expected_count,
                    "prob_joint_argmax": count / len(dates) if dates.size else 0.0,
                    "prob_joint_expected": expected_count / len(dates) if dates.size else 0.0,
                    "prob_cond_argmax": (count / g_total_counts) if g_total_counts else 0.0,
                    "prob_cond_expected": (expected_count / g_expected_total) if g_expected_total > 0 else 0.0,
                    "n_obs_total": len(dates),
                    "n_obs_global": g_total_counts,
                })
    return pd.DataFrame(rows)


def build_current_joint(
    g: pd.DataFrame,
    a: pd.DataFrame,
    asof: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Independence-product joint at the latest date ≤ asof that is present in
    both panels. If asof is None, uses the latest observed date.

    Per-asset latest-available date (≤ asof) may differ — we record each
    asset's `asof` separately in the output. `asof_used` (2nd return value)
    is the max across assets (for reporting).
    """
    g_cols = _global_prob_columns(g)
    a_cols = _asset_prob_columns(a)
    K_g = len(g_cols)
    K_a = len(a_cols)

    asof_ts = pd.Timestamp(asof) if asof is not None else None
    rows = []
    asof_used: pd.Timestamp | None = None
    for asset, grp in a.groupby("asset"):
        grp = grp.set_index("date")
        common = grp.index.intersection(g.index)
        if asof_ts is not None:
            common = common[common <= asof_ts]
        if len(common) == 0:
            continue
        latest = common.max()
        asof_used = max(asof_used, latest) if asof_used is not None else latest
        pg = g.loc[latest, g_cols].values
        pa = grp.loc[latest, a_cols].values
        joint = np.outer(pg, pa)
        for gi in range(K_g):
            for aj in range(K_a):
                rows.append({
                    "asset": asset,
                    "asof": latest,
                    "global_template_id": gi,
                    "asset_regime_id": aj,
                    "prob_global": float(pg[gi]),
                    "prob_asset": float(pa[aj]),
                    "prob_joint": float(joint[gi, aj]),
                })
    return pd.DataFrame(rows), asof_used


def render_doc(empirical: pd.DataFrame, current: pd.DataFrame, asof: str) -> str:
    K_g = int(empirical["global_template_id"].max()) + 1
    K_a = int(empirical["asset_regime_id"].max()) + 1
    a_label = {0: "bear", K_a - 1: "bull"}

    lines = [
        f"# Template × Asset-Regime Joint Map — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · K_global={K_g} · K_asset={K_a}",
        "",
        "> Empirical P(asset_regime | global_template) shows how each asset "
        "typically behaves in each macro regime. Used by Phase 3 to filter "
        "analog pools: \"given current macro=Tg, asset=X regime=Tr, pull "
        "historical dates with similar joint state.\"",
        "",
        "## Empirical P(asset_regime | global_template) — prob-weighted",
        "",
    ]
    for asset, grp in empirical.groupby("asset"):
        lines.append(f"### {asset}")
        lines.append("")
        header = "| macro | " + " | ".join(
            f"T{j} ({a_label.get(j, 'neutral')})" for j in range(K_a)
        ) + " | n_obs |"
        sep = "|" + "|".join(["---"] * (K_a + 2)) + "|"
        lines.append(header)
        lines.append(sep)
        for gi in range(K_g):
            row = grp[grp["global_template_id"] == gi].sort_values("asset_regime_id")
            cells = [f"{r.prob_cond_expected:.2f}" for r in row.itertuples()]
            n_obs_global = int(row.iloc[0]["n_obs_global"]) if len(row) else 0
            lines.append(f"| T{gi} | " + " | ".join(cells) + f" | {n_obs_global} |")
        lines.append("")

    # Current snapshot section: show per asset which (g, r) pair has highest joint prob
    lines.append("## Current joint state snapshot (independence product)")
    lines.append("")
    lines.append("| asset | top joint (g,r) | prob_joint | P(global=best_g) | P(asset=best_r) |")
    lines.append("|---|---|---|---|---|")
    for asset, grp in current.groupby("asset"):
        top = grp.sort_values("prob_joint", ascending=False).iloc[0]
        lines.append(
            f"| {asset} | (T{int(top['global_template_id'])},"
            f"T{int(top['asset_regime_id'])}) | "
            f"{top['prob_joint']:.3f} | "
            f"{top['prob_global']:.2f} | "
            f"{top['prob_asset']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(asof: str | None = None) -> dict:
    g, a = _load_inputs()
    print(f"Loaded global_templates {g.shape} · asset_regime_probs {a.shape}")

    empirical = build_empirical(g, a)
    current, asof_ts = build_current_joint(g, a, asof=asof)
    asof_str = asof_ts.strftime("%Y-%m-%d") if asof_ts is not None else (asof or "latest")

    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    empirical.to_parquet(out_dir / "template_asset_empirical.parquet", index=False)
    current.to_parquet(out_dir / "template_asset_joint_current.parquet", index=False)

    doc_path = Path(ANALYSIS_DIR) / "v2" / f"template_map_{asof_str}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(render_doc(empirical, current, asof_str))

    print(f"\n✓ empirical  → {out_dir / 'template_asset_empirical.parquet'} ({len(empirical)} rows)")
    print(f"✓ current    → {out_dir / 'template_asset_joint_current.parquet'} ({len(current)} rows)")
    print(f"✓ doc        → {doc_path}")

    # Quick terminal summary of conditional for each asset in T3 (crisis)
    print("\nP(asset_regime | crisis=T3):")
    k_g = int(empirical['global_template_id'].max())
    crisis = empirical[empirical["global_template_id"] == k_g]
    for asset, grp in crisis.groupby("asset"):
        grp = grp.sort_values("asset_regime_id")
        probs = " / ".join(f"T{r.asset_regime_id}:{r.prob_cond_expected:.2f}"
                           for r in grp.itertuples())
        print(f"  {asset:<14} {probs}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "asof": asof_str,
        "K_global": int(empirical["global_template_id"].max()) + 1,
        "K_asset": int(empirical["asset_regime_id"].max()) + 1,
        "n_assets": empirical["asset"].nunique(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "template_map_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    run()
