#!/usr/bin/env python3
"""
Phase 4: v2 allocation — consumes forecast_scenarios.parquet.

Two methods on the same joint scenarios (cross-asset correlation preserved):

1. Mean-Variance (BL-style)
   Uses 12m scenario mean and covariance as BL posterior; solves
   constrained MV utility. No explicit equilibrium blending (pure
   scenario-based). Risk aversion δ = 2.5.

2. SP-CVaR
   Min CVaR(α=5%) of portfolio 6m arithmetic return over scenarios.
   Joint scenarios preserve tail correlation (BL-CVaR gap vs v1).

Common constraints: Σw=1, 0≤w_i≤0.5 (per-asset cap).

Universe: 6 assets matched to v1 (SPX, Bond, Gold, Oil, BTC, DXY) so
v1 vs v2 side-by-side makes sense. State features hold bond_ret as
duration proxy for the bond leg.

Contract (schema v2.1):
  Input:  data/v2/forecast_scenarios.parquet
  Output: data/v2/weights_v2.parquet  — long format with (asof_date,
            method, asset, weight)
          data/v2/allocation_v2_meta.json
          analysis/v2/allocation_<asof>.md — v2 weights + v1 comparison
            if v1 outputs exist
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore
from lib.schema import SCHEMA_VERSION, base_meta  # type: ignore
RISK_AVERSION = 2.5
CVAR_ALPHA = 0.05
MAX_WEIGHT = 0.50
BL_HORIZON_M = 12
CVAR_HORIZON_M = 6

# v1-aligned universe (for direct comparison)
V1_UNIVERSE = [
    ("SPX",         "spx_ret"),
    ("US10Y_yield", "bond_ret"),   # bond_ret is a duration-proxy log return
    ("Gold",        "gold_ret"),
    ("WTI_crude",   "oil_ret"),
    ("BTC",         "btc_ret"),
    ("DXY",         "dxy_ret"),
]
V1_LABEL_TO_RET = {lbl: ret for lbl, ret in V1_UNIVERSE}
LABELS = [lbl for lbl, _ in V1_UNIVERSE]


def _load_scenarios() -> pd.DataFrame:
    path = Path(DATA_DIR) / "v2" / "forecast_scenarios.parquet"
    if not path.exists():
        sys.exit(f"missing {path} — run forecast.py first")
    df = pd.read_parquet(path)
    return df


def _scenario_matrix(
    scn_df: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (R_log, scenario_weights, labels).
      R_log: (N_scn, N_assets) log returns (aligned, dropping scenarios with
             any missing asset).
      scenario_weights: (N_scn,) normalized.
      labels: list of v1 labels in column order.
    """
    sub = scn_df[scn_df["horizon"] == horizon].copy()
    wide = sub.pivot_table(
        index="scenario_id",
        columns="asset",
        values="log_return",
        aggfunc="first",
    )
    # Rename asset cols from v2 _ret form to v1 labels
    col_map = {ret: lbl for lbl, ret in V1_UNIVERSE}
    wide = wide.rename(columns=col_map)
    present = [lbl for lbl in LABELS if lbl in wide.columns]
    missing = set(LABELS) - set(present)
    if missing:
        print(f"  ⚠ v1 labels missing from scenarios (horizon={horizon}): "
              f"{sorted(missing)}")
    wide = wide[present].dropna(how="any")

    # Scenario weights: take per-scenario weight from any asset row (same for
    # all assets in a scenario by construction)
    w_per_scn = (
        sub.drop_duplicates("scenario_id").set_index("scenario_id")["weight"]
    )
    w_per_scn = w_per_scn.loc[wide.index]
    w = w_per_scn.values
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return wide.values, w, present


def mean_variance_weights(
    R_log: np.ndarray,
    scn_weights: np.ndarray,
    risk_aversion: float = RISK_AVERSION,
    max_w: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Constrained MV on log returns. Using log returns directly is a simplification
    (ignores Jensen's inequality) but fine as a baseline for the 6-asset, 12m case.
    """
    n = R_log.shape[1]
    mu = np.average(R_log, axis=0, weights=scn_weights)
    centered = R_log - mu
    Sigma = (centered.T * scn_weights) @ centered

    def neg_util(w):
        return -(w @ mu) + 0.5 * risk_aversion * w @ Sigma @ w

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_w)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_util, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 500})
    if not res.success:
        print(f"  ⚠ MV optimization did not converge: {res.message}")
    w = np.clip(res.x, 0.0, max_w)
    if w.sum() > 0:
        w /= w.sum()
    return w


def cvar_weights(
    R_log: np.ndarray,
    scn_weights: np.ndarray,
    alpha: float = CVAR_ALPHA,
    max_w: float = MAX_WEIGHT,
) -> tuple[np.ndarray, float, float]:
    """
    Min CVaR(α) on arithmetic returns derived from log returns per scenario.
    Returns (weights, expected_return, realized_cvar).
    """
    R_arith = np.exp(R_log) - 1.0
    n = R_arith.shape[1]

    def neg_cvar(w):
        port = R_arith @ w
        losses = -port
        # Weighted tail of losses
        order = np.argsort(-losses)
        sorted_loss = losses[order]
        sorted_w = scn_weights[order]
        cum = np.cumsum(sorted_w)
        mask = cum <= alpha
        if not mask.any():
            mask[0] = True
        tl = sorted_loss[mask]
        tw = sorted_w[mask]
        return float(np.average(tl, weights=tw))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_w)] * n
    w0 = np.full(n, 1.0 / n)

    # COBYLA handles non-smooth objective better than SLSQP here
    # Codex Round-6 Important #3: COBYLA on non-smooth CVaR with only
    # ~10 tail scenarios can be unstable. Multiple restarts + keep best.
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        *[{"type": "ineq", "fun": (lambda w, i=i: w[i])} for i in range(n)],
        *[{"type": "ineq", "fun": (lambda w, i=i: max_w - w[i])} for i in range(n)],
    ]
    rng_opt = np.random.default_rng(42)
    best_x = None
    best_obj = np.inf
    n_restarts = 4
    for trial in range(n_restarts):
        if trial == 0:
            start = w0
        else:
            start = rng_opt.dirichlet(np.ones(n))
            start = np.minimum(start, max_w)
            start = start / start.sum() if start.sum() > 0 else w0
        res_t = minimize(neg_cvar, start, method="COBYLA", constraints=constraints,
                         options={"rhobeg": 0.05, "maxiter": 1000, "catol": 1e-6})
        x_t = np.clip(res_t.x, 0.0, max_w)
        x_t = x_t / x_t.sum() if x_t.sum() > 0 else start
        try:
            obj_t = neg_cvar(x_t)
            if obj_t < best_obj:
                best_obj, best_x = obj_t, x_t
        except Exception:
            continue
    w = best_x if best_x is not None else np.full(n, 1.0 / n)
    port = R_arith @ w
    exp_ret = float(np.average(port, weights=scn_weights))
    realized_cvar = float(neg_cvar(w))
    return w, exp_ret, realized_cvar


def _render_doc(
    asof: str,
    bl_weights: np.ndarray,
    bl_labels: list[str],
    bl_mu_yr: np.ndarray,
    bl_vol_yr: float,
    cvar_wts: np.ndarray,
    cvar_labels: list[str],
    cvar_exp: float,
    cvar_realized: float,
    v1_comparison: dict | None,
) -> str:
    lines = [
        f"# v2 Allocation — {asof}",
        "",
        f"schema_version: `{SCHEMA_VERSION}` · BL horizon={BL_HORIZON_M}m · "
        f"SP-CVaR horizon={CVAR_HORIZON_M}m · α={CVAR_ALPHA} · "
        f"max_weight={MAX_WEIGHT:.0%}",
        "",
        "## Weights",
        "",
        "| asset | Market | BL (MV) | SP-CVaR | v1 BL | v1 SP |",
        "|---|---|---|---|---|---|",
    ]
    v1_bl = (v1_comparison or {}).get("bl", {})
    v1_sp = (v1_comparison or {}).get("sp", {})
    market_weights = {"SPX": 0.40, "US10Y_yield": 0.25, "Gold": 0.10,
                      "WTI_crude": 0.10, "BTC": 0.05, "DXY": 0.10}
    for lbl in LABELS:
        mkt = market_weights.get(lbl, 0.0)
        bl_w = bl_weights[bl_labels.index(lbl)] if lbl in bl_labels else 0.0
        cv_w = cvar_wts[cvar_labels.index(lbl)] if lbl in cvar_labels else 0.0
        v1_bl_w = v1_bl.get(lbl, "—")
        v1_sp_w = v1_sp.get(lbl, "—")
        v1_bl_s = f"{v1_bl_w:.0%}" if isinstance(v1_bl_w, float) else v1_bl_w
        v1_sp_s = f"{v1_sp_w:.0%}" if isinstance(v1_sp_w, float) else v1_sp_w
        lines.append(f"| {lbl} | {mkt:.0%} | {bl_w:.0%} | {cv_w:.0%} | {v1_bl_s} | {v1_sp_s} |")
    lines.append("")

    lines.append("## BL (MV) diagnostics")
    lines.append("")
    lines.append(
        f"- Annualised expected log-return (weighted scenario mean): "
        f"{bl_weights @ bl_mu_yr:+.2%}"
    )
    lines.append(f"- Portfolio std (scenario): {bl_vol_yr:.2%}")
    lines.append("")
    lines.append("## SP-CVaR diagnostics")
    lines.append("")
    lines.append(f"- Expected {CVAR_HORIZON_M}m arithmetic return: {cvar_exp:+.2%}")
    lines.append(f"- Realised CVaR({CVAR_ALPHA:.0%}): {cvar_realized:.2%}")
    lines.append("")
    return "\n".join(lines)


def _load_v1_comparison() -> dict | None:
    """Best-effort load of v1 BL and SP weights for side-by-side."""
    out = {"bl": {}, "sp": {}}
    # v1 BL writes efficient frontier + weights; the "current_weights" are in
    # bl_efficient_frontier (approx) — we just grab last SP from
    # stochastic_prog_weights
    sp_path = Path(DATA_DIR) / "stochastic_prog_weights.csv"
    if sp_path.exists():
        try:
            df = pd.read_csv(sp_path)
            # Columns likely: current_regime + weight columns; take the last
            # "current" row if structure matches
            if "asset" in df.columns and "weight" in df.columns:
                for r in df.itertuples():
                    if hasattr(r, "asset") and hasattr(r, "weight"):
                        out["sp"][str(r.asset)] = float(r.weight)
        except Exception as e:
            print(f"  ⚠ v1 SP parse failed: {e}")
    return out if out["bl"] or out["sp"] else None


def run(asof: str | None = None, exclude_assets: set[str] | None = None) -> dict:
    scn = _load_scenarios()
    if scn.empty:
        sys.exit("empty scenarios file")
    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp(scn["asof_date"].max())
    if exclude_assets:
        scn = scn[~scn["asset"].isin(exclude_assets)].copy()

    # MV (BL-style) on 12m horizon
    R12, w12, labels12 = _scenario_matrix(scn, BL_HORIZON_M)
    bl_w = mean_variance_weights(R12, w12)
    mu12 = np.average(R12, axis=0, weights=w12)
    centered = R12 - mu12
    Sigma12 = (centered.T * w12) @ centered
    bl_vol = float(np.sqrt(bl_w @ Sigma12 @ bl_w))

    # SP-CVaR on 6m horizon
    R6, w6, labels6 = _scenario_matrix(scn, CVAR_HORIZON_M)
    cvar_w, cvar_exp, cvar_realized = cvar_weights(R6, w6)

    # Persist
    rows = []
    for w, labels, method in [(bl_w, labels12, "mv_bl_12m"),
                               (cvar_w, labels6, "sp_cvar_6m")]:
        for lbl, wt in zip(labels, w):
            rows.append({
                "asof_date": asof_ts,
                "method": method,
                "asset": lbl,
                "weight": float(wt),
            })
    weights_df = pd.DataFrame(rows)
    out_dir = Path(DATA_DIR) / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_df.to_parquet(out_dir / "weights_v2.parquet", index=False)

    meta = base_meta(
        layer="allocation",
        data_asof=asof_ts.strftime("%Y-%m-%d"),
        model_version=f"mv_bl_d{RISK_AVERSION}_h{BL_HORIZON_M}m+cvar_a{CVAR_ALPHA}_h{CVAR_HORIZON_M}m_cap{int(MAX_WEIGHT*100)}",
        extra={
            "bl_horizon_m": BL_HORIZON_M,
            "cvar_horizon_m": CVAR_HORIZON_M,
            "cvar_alpha": CVAR_ALPHA,
            "max_weight": MAX_WEIGHT,
            "risk_aversion": RISK_AVERSION,
            "bl_weights": {labels12[i]: float(bl_w[i]) for i in range(len(labels12))},
            "cvar_weights": {labels6[i]: float(cvar_w[i]) for i in range(len(labels6))},
            "bl_portfolio_vol": bl_vol,
            "cvar_expected_return": cvar_exp,
            "cvar_realized_cvar": cvar_realized,
            "universe": [lbl for lbl, _ in V1_UNIVERSE],
        },
    )
    (out_dir / "allocation_v2_meta.json").write_text(json.dumps(meta, indent=2))

    # Doc with v1 comparison if available
    v1 = _load_v1_comparison()
    asof_str = asof_ts.strftime("%Y-%m-%d")
    doc_path = Path(ANALYSIS_DIR) / "v2" / f"allocation_{asof_str}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(_render_doc(
        asof_str, bl_w, labels12, mu12, bl_vol,
        cvar_w, labels6, cvar_exp, cvar_realized, v1,
    ))

    print(f"\n✓ weights   → {out_dir / 'weights_v2.parquet'}  ({len(weights_df)} rows)")
    print(f"✓ doc       → {doc_path}")
    print("\nBL (MV) weights:")
    for lbl, w in zip(labels12, bl_w):
        print(f"  {lbl:<14} {w:.1%}")
    print(f"  → annualised expected log-return: {bl_w @ mu12:+.2%}, vol: {bl_vol:.2%}")
    print("\nSP-CVaR weights:")
    for lbl, w in zip(labels6, cvar_w):
        print(f"  {lbl:<14} {w:.1%}")
    print(f"  → E[r_6m]: {cvar_exp:+.2%}, realised CVaR(5%): {cvar_realized:.2%}")

    return meta


if __name__ == "__main__":
    run()
