#!/usr/bin/env python3
"""
Stage B.0: backtest harness (leakage-light).

Iterates asof monthly over a user-specified range, reruns template_map +
forecast at each asof (both asof-aware), computes CRPS and PIT of the
v2 scenarios vs realized forward returns, and compares against a Gaussian
rolling-mean benchmark (Normal(μ_{roll,120m}, σ_{roll,120m}·√h)).

Leakage profile:
- forecast analog-eligibility respects asof (no forward-outcome crossing)
- template_map current_joint is asof-aware
- state_features: ALFRED vintage-resolved features are asof-correct;
  other FRED features are as-is (codex Important #5, deferred)
- global_template + asset_regime are fit ONCE on full 1990-2026 history
  (leakage on regime centroids/GMM params). Stage B.1 target: refit
  per asof via expanding/rolling window.

Contract (schema v2.1):
  Input: state_features.parquet + all v2/* artifacts
  Output:
    data/v2/backtest/results_<start>_<end>.parquet
      columns: asof, asset, horizon, realized, crps_v2, crps_bench,
               pit_v2, pit_bench, n_scn
    data/v2/backtest/backtest_meta_<start>_<end>.json
    analysis/v2/backtest_<start>_<end>.md — per (asset, horizon) summary

Usage:
  uv run python scripts/backtest.py --start 2015-01 --end 2024-12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore
from lib.schema import base_meta  # type: ignore
from lib.metrics import (  # type: ignore
    crps_sample, crps_normal, pit_sample, pit_normal,
    energy_score_sample, energy_score_mvn, ar1_forecast,
    moving_block_bootstrap_mean, moving_block_bootstrap_sharpe_diff,
)


ROLLING_WINDOW = 120
SCENARIOS_PATH = Path(DATA_DIR) / "v2" / "forecast_scenarios.parquet"


def _forward_cum_return(ret: pd.Series, h: int) -> pd.Series:
    return ret.shift(-1).rolling(h).sum().shift(-(h - 1))


def _realized(state: pd.DataFrame, asset: str, asof: pd.Timestamp, h: int) -> float | None:
    if asset not in state.columns:
        return None
    fwd = _forward_cum_return(state[asset], h)
    if asof not in fwd.index:
        return None
    v = fwd.loc[asof]
    return float(v) if np.isfinite(v) else None


def _gaussian_benchmark(
    state: pd.DataFrame, asset: str, asof: pd.Timestamp, h: int, window: int = ROLLING_WINDOW
) -> tuple[float | None, float | None]:
    if asset not in state.columns:
        return None, None
    hist = state[asset].loc[:asof].dropna().tail(window)
    if len(hist) < 24:
        return None, None
    mu = float(hist.mean()) * h
    sigma = float(hist.std()) * np.sqrt(h)
    return mu, sigma


# Asset label → state column mapping (v1 allocation universe)
V1_LABEL_TO_RET = {
    "SPX":         "spx_ret",
    "US10Y_yield": "bond_ret",
    "Gold":        "gold_ret",
    "WTI_crude":   "oil_ret",
    "BTC":         "btc_ret",
    "DXY":         "dxy_ret",
}


def _portfolio_realized(
    state: pd.DataFrame,
    weights: dict[str, float],
    asof: pd.Timestamp,
    h: int,
    missing_tol: float = 0.02,
) -> float | None:
    """
    Realized portfolio log-return over t+1 .. t+h for given weights.
    Converts per-asset cumulative log returns to arithmetic then weights,
    returns log of 1+arith.

    Codex Round-6 Important #6: if any weighted asset has missing realized
    data, the "truncated portfolio" would misrepresent returns. Reject
    scenarios with > missing_tol fraction of weight on missing assets.
    """
    total = 0.0
    missing_w = 0.0
    covered_w = 0.0
    for label, w in weights.items():
        if w == 0:
            continue
        col = V1_LABEL_TO_RET.get(label, label)
        if col not in state.columns:
            missing_w += w
            continue
        fwd = _forward_cum_return(state[col], h)
        if asof not in fwd.index:
            missing_w += w
            continue
        v = fwd.loc[asof]
        if not np.isfinite(v):
            missing_w += w
            continue
        total += w * (np.exp(float(v)) - 1.0)
        covered_w += w
    if covered_w == 0:
        return None
    if missing_w / (missing_w + covered_w) > missing_tol:
        return None    # too much portfolio weight on missing assets — reject
    return float(np.log1p(total))


def _mvn_allocate(
    state: pd.DataFrame,
    asof: pd.Timestamp,
    h: int = 6,
    n_scn: int = 200,
    alpha: float = 0.05,
    max_weight: float = 0.50,
) -> dict[str, float] | None:
    """
    Fair benchmark for v2 SP-CVaR: generate n_scn scenarios from a
    multivariate Normal fit to rolling 120m of asset returns (not KAF
    analog ranking), then run the SAME SP-CVaR optimizer on them.
    If v2 SP-CVaR realized Sharpe > MVN SP-CVaR realized Sharpe,
    joint-structure information is real at the allocation layer.
    """
    import v2_allocation
    labels = [lbl for lbl, _ in v2_allocation.V1_UNIVERSE]
    ret_cols = [v2_allocation.V1_LABEL_TO_RET[lbl] for lbl in labels]
    present = [(lbl, col) for lbl, col in zip(labels, ret_cols) if col in state.columns]
    if not present:
        return None
    cols = [col for _, col in present]
    labs = [lbl for lbl, _ in present]
    hist = state[cols].loc[:asof].dropna(how="any").tail(120)
    if len(hist) < max(24, len(cols) + 2):
        return None
    mu = hist.mean().values * h
    Sigma = hist.cov().values * h + 1e-8 * np.eye(len(cols))
    rng = np.random.default_rng(pd.Timestamp(asof).toordinal() + 9999)
    try:
        R_log = rng.multivariate_normal(mu, Sigma, size=n_scn)
    except np.linalg.LinAlgError:
        return None
    w_eq = np.full(n_scn, 1.0 / n_scn)
    w_opt, _, _ = v2_allocation.cvar_weights(R_log, w_eq, alpha=alpha, max_w=max_weight)
    return {lab: float(w_opt[i]) for i, lab in enumerate(labs)}


def _joint_gaussian_benchmark(
    state: pd.DataFrame,
    assets: list[str],
    asof: pd.Timestamp,
    h: int,
    window: int = ROLLING_WINDOW,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Multivariate Normal with rolling (μ, Σ) scaled to horizon h."""
    cols = [a for a in assets if a in state.columns]
    if not cols:
        return None
    hist = state[cols].loc[:asof].dropna(how="any").tail(window)
    if len(hist) < max(24, len(cols) + 2):
        return None
    mu = hist.mean().values * h
    Sigma = hist.cov().values * h
    return mu, Sigma


def _refit_regime_layers(asof_str: str) -> None:
    import global_template, asset_regime, template_map
    global_template.run(asof=asof_str)
    asset_regime.run(asof=asof_str)
    template_map.run(asof=asof_str)


def _run_forecast_at(
    asof_str: str,
    expanding_regimes: bool,
    alpha: float | None = None,
    gamma: float | None = None,
    kernel: str | None = None,
    regime_switch_vix_pct: float | None = None,
    skip_regime_refit: bool = False,
) -> bool:
    """
    Emit forecast at asof. Refits regime layers if needed.
    skip_regime_refit: used when we've already refit at this asof in a prior
    alpha/gamma-grid iteration (regime layers don't depend on α or γ).
    """
    import forecast, template_map
    try:
        if expanding_regimes and not skip_regime_refit:
            _refit_regime_layers(asof_str)
        elif not expanding_regimes:
            template_map.run(asof=asof_str)
        forecast.run(
            asof=asof_str,
            alpha=alpha,
            gamma=gamma,
            kernel=kernel,
            regime_switch_vix_pct=regime_switch_vix_pct,
        )
        return True
    except SystemExit as e:
        print(f"  skip {asof_str}: {e}")
        return False
    except Exception as e:
        print(f"  error {asof_str}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM")
    ap.add_argument("--step", type=int, default=1, help="month step")
    ap.add_argument("--no-expanding-regimes", action="store_true",
                    help="skip global_template + asset_regime refit per asof "
                         "(faster, but leaks regime training on full history)")
    ap.add_argument("--alphas", type=str, default=None,
                    help="comma-separated alpha values to grid-search "
                         "(default: single forecast with forecast.ALPHA_REGIME_DEFAULT)")
    ap.add_argument("--gammas", type=str, default=None,
                    help="comma-separated gamma (tether weight) values to grid-search")
    ap.add_argument("--kernels", type=str, default=None,
                    help="comma-separated state-distance kernels "
                         "(euclidean,rbf,mahalanobis)")
    ap.add_argument("--vix-switch", type=float, default=None,
                    help="regime-switch VIX percentile (e.g., 0.50); "
                         "below → all assets Gaussian, above → KAF")
    ap.add_argument("--allocate", action="store_true",
                    help="also run v2 allocation (MV + SP-CVaR) per asof and "
                         "track realized portfolio forward returns")
    ap.add_argument("--exclude-assets", type=str, default=None,
                    help="comma-separated forecast asset names to exclude "
                         "from allocation (e.g. 'btc_ret' to test no-BTC)")
    args = ap.parse_args()

    state = pd.read_parquet(Path(DATA_DIR) / "state_features.parquet")
    asofs = pd.date_range(args.start, args.end, freq=f"{args.step}ME")
    print(f"Backtest: {len(asofs)} asof steps, {args.start} → {args.end}")

    alpha_grid = [float(x) for x in args.alphas.split(",")] if args.alphas else None
    gamma_grid = [float(x) for x in args.gammas.split(",")] if args.gammas else None
    kernel_grid = [k.strip() for k in args.kernels.split(",")] if args.kernels else None
    if alpha_grid:
        print(f"α grid: {alpha_grid}")
    if gamma_grid:
        print(f"γ grid: {gamma_grid}")
    if kernel_grid:
        print(f"kernel grid: {kernel_grid}")

    t0 = time.time()
    rows: list[dict] = []
    energy_rows: list[dict] = []   # per-(asof, horizon): joint energy score v2 vs MVN
    alloc_rows: list[dict] = []    # per-(asof, method, horizon): realized portfolio return
    expanding = not args.no_expanding_regimes
    alphas_to_run = alpha_grid if alpha_grid is not None else [None]
    gammas_to_run = gamma_grid if gamma_grid is not None else [None]
    kernels_to_run = kernel_grid if kernel_grid is not None else [None]

    for i, asof in enumerate(asofs, 1):
        asof_str = asof.strftime("%Y-%m-%d")
        elapsed = time.time() - t0
        eta = elapsed / i * (len(asofs) - i) if i > 0 else 0
        print(f"[{i}/{len(asofs)}] {asof_str} (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        if expanding:
            _refit_regime_layers(asof_str)
        else:
            import template_map
            template_map.run(asof=asof_str)

        for alpha in alphas_to_run:
            for gamma in gammas_to_run:
                for kernel in kernels_to_run:
                    ok = _run_forecast_at(
                        asof_str,
                        expanding_regimes=expanding,
                        alpha=alpha,
                        gamma=gamma,
                        kernel=kernel,
                        regime_switch_vix_pct=args.vix_switch,
                        skip_regime_refit=True,
                    )
                    if not ok:
                        continue
                    scn = pd.read_parquet(SCENARIOS_PATH)
                    if scn.empty:
                        continue
                    alpha_val = float(alpha) if alpha is not None else float("nan")
                    gamma_val = float(gamma) if gamma is not None else float("nan")
                    kernel_val = kernel if kernel is not None else ""
                    for (asset, h), grp in scn.groupby(["asset", "horizon"]):
                        samples = grp["log_return"].values
                        weights = grp["weight"].values
                        realized = _realized(state, asset, asof, int(h))
                        if realized is None:
                            continue
                        crps_v2 = crps_sample(samples, realized, weights)
                        pit_v2 = pit_sample(samples, realized, weights)
                        bench_mu, bench_sigma = _gaussian_benchmark(state, asset, asof, int(h))
                        if bench_mu is None:
                            crps_bench, pit_bench = float("nan"), float("nan")
                        else:
                            crps_bench = crps_normal(bench_mu, bench_sigma, realized)
                            pit_bench = pit_normal(bench_mu, bench_sigma, realized)

                        # AR(1) per-asset benchmark (alternative baseline)
                        ar_res = ar1_forecast(state[asset], int(h))
                        if ar_res is None:
                            crps_ar, pit_ar = float("nan"), float("nan")
                        else:
                            ar_mu, ar_sigma = ar_res
                            crps_ar = crps_normal(ar_mu, ar_sigma, realized)
                            pit_ar = pit_normal(ar_mu, ar_sigma, realized)

                        rows.append({
                            "asof": asof,
                            "alpha": alpha_val,
                            "gamma": gamma_val,
                            "kernel": kernel_val,
                            "asset": asset,
                            "horizon": int(h),
                            "realized": realized,
                            "crps_v2": crps_v2,
                            "crps_bench": crps_bench,
                            "crps_ar1": crps_ar,
                            "pit_v2": pit_v2,
                            "pit_bench": pit_bench,
                            "pit_ar1": pit_ar,
                            "n_scn": len(samples),
                        })

                    # Energy Score per (asof, horizon) — tests joint dependence.
                    # Pin to a fixed asset set across asofs (codex Round-6
                    # Important #8): ES dimensionality matters for comparability.
                    ES_ASSETS = ["spx_ret", "bond_ret", "gold_ret", "oil_ret", "dxy_ret"]
                    for h_int in scn["horizon"].unique():
                        sub_h = scn[(scn["horizon"] == int(h_int)) &
                                    (scn["asset"].isin(ES_ASSETS))]
                        if sub_h.empty:
                            continue
                        wide = sub_h.pivot_table(
                            index="scenario_id",
                            columns="asset",
                            values="log_return",
                            aggfunc="first",
                        )
                        # Require all pinned assets present; drop incomplete scenarios
                        if not set(ES_ASSETS).issubset(set(wide.columns)):
                            continue
                        wide = wide[ES_ASSETS].dropna(how="any")
                        if wide.empty:
                            continue
                        # Realized vector for these assets at asof
                        realized_vec = np.array([
                            _realized(state, a, asof, int(h_int)) for a in wide.columns
                        ])
                        if any(r is None for r in realized_vec):
                            continue
                        realized_vec = realized_vec.astype(float)
                        # Scenario weights (one per scenario_id)
                        w_map = sub_h.drop_duplicates("scenario_id").set_index("scenario_id")["weight"]
                        w_vec = w_map.loc[wide.index].values
                        w_vec = w_vec / w_vec.sum() if w_vec.sum() > 0 else np.full(len(w_vec), 1.0/len(w_vec))
                        es_v2 = energy_score_sample(wide.values, realized_vec, w_vec)
                        # MVN benchmark
                        mvn = _joint_gaussian_benchmark(state, list(wide.columns), asof, int(h_int))
                        es_mvn = float("nan")
                        if mvn is not None:
                            # Codex Round-7: multi-seed averaging to reduce MC noise
                            es_mvn = energy_score_mvn(
                                mvn[0], mvn[1], realized_vec,
                                n_samples=5000, n_seeds=3,
                                base_seed=pd.Timestamp(asof).toordinal() + int(h_int),
                            )
                        energy_rows.append({
                            "asof": asof,
                            "alpha": alpha_val,
                            "gamma": gamma_val,
                            "kernel": kernel_val,
                            "horizon": int(h_int),
                            "n_assets": int(wide.shape[1]),
                            "es_v2": es_v2,
                            "es_mvn": es_mvn,
                        })

                    if args.allocate:
                        try:
                            import v2_allocation
                            excl = set(args.exclude_assets.split(",")) if args.exclude_assets else None
                            alloc_meta = v2_allocation.run(asof=asof_str, exclude_assets=excl)
                            for method, weights in [
                                ("mv_bl_12m", alloc_meta.get("bl_weights", {})),
                                ("sp_cvar_6m", alloc_meta.get("cvar_weights", {})),
                            ]:
                                for h_alloc in (1, 3, 6, 12):
                                    pr = _portfolio_realized(state, weights, asof, h_alloc)
                                    if pr is None:
                                        continue
                                    alloc_rows.append({
                                        "asof": asof,
                                        "method": method,
                                        "horizon": h_alloc,
                                        "realized_log_return": pr,
                                        "weights": json.dumps(weights),
                                    })
                            # 60/40 benchmark (SPX 60, Bond 40)
                            for h_alloc in (1, 3, 6, 12):
                                pr = _portfolio_realized(
                                    state, {"SPX": 0.6, "US10Y_yield": 0.4}, asof, h_alloc
                                )
                                if pr is None:
                                    continue
                                alloc_rows.append({
                                    "asof": asof,
                                    "method": "bench_60_40",
                                    "horizon": h_alloc,
                                    "realized_log_return": pr,
                                    "weights": json.dumps({"SPX": 0.6, "US10Y_yield": 0.4}),
                                })
                            # MVN-scenario allocation: fair isolation of
                            # "does joint structure help optimizer?" vs
                            # "does optimizer help at all?"
                            mvn_weights = _mvn_allocate(state, asof, h=6, n_scn=200)
                            if mvn_weights is not None:
                                for h_alloc in (1, 3, 6, 12):
                                    pr = _portfolio_realized(state, mvn_weights, asof, h_alloc)
                                    if pr is None:
                                        continue
                                    alloc_rows.append({
                                        "asof": asof,
                                        "method": "mvn_sp_cvar_6m",
                                        "horizon": h_alloc,
                                        "realized_log_return": pr,
                                        "weights": json.dumps(mvn_weights),
                                    })
                        except Exception as e:
                            print(f"  ⚠ allocation at {asof_str} failed: {e}")

    if not rows:
        sys.exit("no rows collected — check asof range and data")

    df = pd.DataFrame(rows)

    out_dir = Path(DATA_DIR) / "v2" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start}_{args.end}"
    out_path = out_dir / f"results_{tag}.parquet"
    df.to_parquet(out_path, index=False)

    meta = base_meta(
        layer="backtest",
        data_asof=str(df["asof"].max().date()),
        model_version="stage_b0_gaussian_bench",
        extra={
            "start": args.start,
            "end": args.end,
            "step_m": args.step,
            "n_asofs": int(df["asof"].nunique()),
            "n_rows": len(df),
            "rolling_benchmark_window_m": ROLLING_WINDOW,
            "expanding_regimes": not args.no_expanding_regimes,
            "leakage_notes": [
                "forecast analog-eligibility respects asof (no fwd crossing)",
                "template_map asof-aware",
                ("global_template & asset_regime refit per asof (expanding window)"
                 if not args.no_expanding_regimes else
                 "global_template & asset_regime fit on full history (leakage; --no-expanding-regimes)"),
                "non-vintage FRED series as-is — Important (deferred)",
            ],
        },
    )
    meta_path = out_dir / f"backtest_meta_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # Summary grouped by (α, asset, horizon) if α grid was run; else (asset, horizon)
    has_alpha = df["alpha"].notna().any() and alpha_grid is not None
    group_keys = ["alpha", "asset", "horizon"] if has_alpha else ["asset", "horizon"]
    summary = df.groupby(group_keys).agg(
        n=("crps_v2", "count"),
        crps_v2_mean=("crps_v2", "mean"),
        crps_bench_mean=("crps_bench", "mean"),
        skill=("crps_v2", lambda s: 1.0 - s.mean() / df.loc[s.index, "crps_bench"].mean()
               if df.loc[s.index, "crps_bench"].mean() > 0 else float("nan")),
        pit_v2_mean=("pit_v2", "mean"),
        pit_bench_mean=("pit_bench", "mean"),
    ).round(4)

    # Render markdown
    lines = [
        f"# Stage B Backtest — {args.start} → {args.end}",
        "",
        f"schema_version: v2.1 · n_asofs={df['asof'].nunique()} · n_rows={len(df)}"
        + (f" · α grid: {alpha_grid}" if has_alpha else ""),
        "",
        "Benchmark: Gaussian Normal(μ, σ·√h) with (μ, σ) from rolling "
        f"{ROLLING_WINDOW}-month window of asset monthly log returns.",
        "",
        "Skill = 1 − CRPS_v2 / CRPS_bench (positive ⇒ v2 beats Gaussian).",
        "",
    ]
    if has_alpha:
        lines.append("## CRPS by (α, asset, horizon) — optimal α per (asset, h)")
        lines.append("")
        lines.append("| asset | h | best α | skill@best | skill@α=3.0 | PIT v2 @best |")
        lines.append("|---|---|---|---|---|---|")
        # For each (asset, h), find α that maximizes skill
        for (asset, h), sub in df.groupby(["asset", "horizon"]):
            per_alpha = sub.groupby("alpha").agg(
                crps_v2=("crps_v2", "mean"),
                crps_bench=("crps_bench", "mean"),
                pit=("pit_v2", "mean"),
            )
            per_alpha["skill"] = 1.0 - per_alpha["crps_v2"] / per_alpha["crps_bench"]
            best_alpha = per_alpha["skill"].idxmax()
            best = per_alpha.loc[best_alpha]
            alpha3 = per_alpha.loc[3.0] if 3.0 in per_alpha.index else None
            skill3 = f"{alpha3['skill']:+.1%}" if alpha3 is not None else "—"
            lines.append(
                f"| {asset} | {h}m | **{best_alpha:.1f}** | {best['skill']:+.1%} | "
                f"{skill3} | {best['pit']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("## CRPS by (asset, horizon)")
        lines.append("")
        lines.append("| asset | h | n | CRPS v2 | CRPS bench | skill | PIT v2 | PIT bench |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for idx, r in summary.iterrows():
            asset, h = idx  # (asset, horizon) only
            skill_str = f"{r['skill']:+.1%}" if np.isfinite(r["skill"]) else "—"
            lines.append(
                f"| {asset} | {h}m | {int(r['n'])} | {r['crps_v2_mean']:.4f} | "
                f"{r['crps_bench_mean']:.4f} | {skill_str} | "
                f"{r['pit_v2_mean']:.3f} | {r['pit_bench_mean']:.3f} |"
            )
        lines.append("")
    # Stress-period breakdown — reveals when v2 actually wins.
    # Overall skill aggregates calm + turbulent; separating them shows
    # the regime-conditional KAF edge is not uniform over time.
    stress_periods = [
        ("GFC-era",    "2007-07", "2009-12"),
        ("Calm expansion", "2013-01", "2019-12"),
        ("COVID",      "2020-01", "2021-06"),
        ("Inflation shock", "2022-01", "2023-06"),
        ("Recent",     "2023-07", "2024-12"),
    ]
    lines.append("## Skill by period (stress vs calm)")
    lines.append("")
    lines.append("| window | dates | n | CRPS v2 | CRPS bench | skill |")
    lines.append("|---|---|---|---|---|---|")
    for name, s, e in stress_periods:
        sub = df[(df["asof"] >= pd.Timestamp(s)) & (df["asof"] <= pd.Timestamp(e))]
        if sub.empty:
            continue
        v = sub["crps_v2"].mean()
        b = sub["crps_bench"].mean()
        sk = 1.0 - v / b if b > 0 else float("nan")
        lines.append(f"| {name} | {s}→{e} | {len(sub)} | {v:.4f} | {b:.4f} | {sk:+.1%} |")
    lines.append("")

    lines.append("## Headline averages")
    lines.append("")
    if has_alpha:
        for alpha_val, sub in df.groupby("alpha"):
            skill = 1.0 - sub["crps_v2"].mean() / sub["crps_bench"].mean() \
                if sub["crps_bench"].mean() > 0 else float("nan")
            lines.append(
                f"- α={alpha_val}: mean CRPS v2 {sub['crps_v2'].mean():.4f} "
                f"vs bench {sub['crps_bench'].mean():.4f} → skill **{skill:+.1%}**"
            )
    else:
        overall_skill = 1.0 - df["crps_v2"].mean() / df["crps_bench"].mean() \
            if df["crps_bench"].mean() > 0 else float("nan")
        lines.append(f"- Mean CRPS v2: **{df['crps_v2'].mean():.4f}**")
        lines.append(f"- Mean CRPS bench: {df['crps_bench'].mean():.4f}")
        lines.append(f"- Overall skill: **{overall_skill:+.1%}**")
        lines.append(f"- PIT v2 mean: {df['pit_v2'].mean():.3f} (uniform target 0.5)")
    lines.append("")

    doc_dir = Path(ANALYSIS_DIR) / "v2"
    doc_dir.mkdir(parents=True, exist_ok=True)
    # Energy Score — PAIRED analysis with moving-block bootstrap CI.
    # Each (asof, horizon) has both ES_v2 and ES_MVN (same scenarios, same
    # realized), so the clean test is paired d_t = ES_MVN - ES_v2 > 0
    # (positive = v2 wins). Overlapping h-month forwards require block
    # length = h.
    if energy_rows:
        es_df = pd.DataFrame(energy_rows)
        es_df.to_parquet(out_dir / f"energy_{tag}.parquet", index=False)
        lines.append("## Multivariate Energy Score — paired v2 vs MVN")
        lines.append("")
        lines.append(
            "> MVN benchmark uses 5000 MC samples averaged over 3 seeds "
            "(~15k effective; Round-7 fix to reduce MC noise from ~0.045 to "
            "~0.008). d = ES_MVN − ES_v2 at each (asof, horizon); positive = "
            "v2 wins. 95% CI via moving-block bootstrap with block=horizon."
        )
        lines.append("")
        lines.append("| horizon | n | mean ES v2 | mean ES MVN | mean diff | 95% CI (blk-bootstrap) | skill % |")
        lines.append("|---|---|---|---|---|---|---|")
        for h, sub in es_df.groupby("horizon"):
            sub = sub.dropna(subset=["es_v2", "es_mvn"]).sort_values("asof")
            if sub.empty:
                continue
            diff = (sub["es_mvn"] - sub["es_v2"]).values
            mean_d, lo, hi = moving_block_bootstrap_mean(diff, block_length=int(h))
            mean_v2 = float(sub["es_v2"].mean())
            mean_mvn = float(sub["es_mvn"].mean())
            skill_pct = 1.0 - mean_v2 / mean_mvn if mean_mvn > 0 else float("nan")
            sig = "✓" if (lo > 0 or hi < 0) else "—"
            lines.append(
                f"| {h}m | {len(sub)} | {mean_v2:.4f} | {mean_mvn:.4f} | "
                f"{mean_d:+.4f} | [{lo:+.4f}, {hi:+.4f}] {sig} | {skill_pct:+.1%} |"
            )
        lines.append("")

    if alloc_rows:
        alloc_df = pd.DataFrame(alloc_rows)
        alloc_df.to_parquet(out_dir / f"allocation_{tag}.parquet", index=False)
        lines.append("## Downstream allocation — overall outcomes")
        lines.append("")
        lines.append("| method | h | n | mean log-ret | ann log-ret | ann vol | ann Sharpe |")
        lines.append("|---|---|---|---|---|---|---|")
        for (method, h), sub in alloc_df.groupby(["method", "horizon"]):
            r = sub["realized_log_return"]
            mean_r = float(r.mean())
            vol_r = float(r.std())
            ann_r = mean_r * (12 / h)
            ann_vol = vol_r * np.sqrt(12 / h)
            sharpe = ann_r / ann_vol if ann_vol > 0 else float("nan")
            lines.append(
                f"| {method} | {h}m | {len(sub)} | {mean_r:+.4f} | "
                f"{ann_r:+.2%} | {ann_vol:.2%} | {sharpe:+.2f} |"
            )
        lines.append("")

        # Paired comparison on common asofs — the honest isolation
        for (a, b, label) in [
            ("sp_cvar_6m", "mvn_sp_cvar_6m", "v2 SP-CVaR vs MVN SP-CVaR"),
            ("sp_cvar_6m", "bench_60_40",   "v2 SP-CVaR vs 60/40"),
        ]:
            lines.append(f"## Paired comparison: {label}")
            lines.append("")
            lines.append(
                "> Inner-joined on common asofs. Paired mean-diff + Sharpe-diff "
                "via moving-block bootstrap (block=horizon) 95% CIs. Monthly "
                "asofs with h-month forward are overlapping, so block bootstrap "
                "rather than iid resampling."
            )
            lines.append("")
            lines.append("| horizon | n | paired Δmean (CI) | sig? | paired ΔSharpe (CI) | sig? |")
            lines.append("|---|---|---|---|---|---|")
            for h in sorted(alloc_df["horizon"].unique()):
                a_sub = alloc_df[(alloc_df.method == a) & (alloc_df.horizon == h)]
                b_sub = alloc_df[(alloc_df.method == b) & (alloc_df.horizon == h)]
                if a_sub.empty or b_sub.empty:
                    continue
                paired = a_sub.merge(b_sub, on="asof", suffixes=("_a", "_b")).sort_values("asof")
                if paired.empty:
                    continue
                r_a = paired["realized_log_return_a"].values
                r_b = paired["realized_log_return_b"].values
                d = r_a - r_b
                mean_d, lo_m, hi_m = moving_block_bootstrap_mean(d, block_length=int(h))
                sig_m = "✓" if (lo_m > 0 or hi_m < 0) else "—"
                ds, ds_lo, ds_hi = moving_block_bootstrap_sharpe_diff(r_a, r_b, block_length=int(h))
                sig_s = "✓" if (ds_lo > 0 or ds_hi < 0) else "—"
                lines.append(
                    f"| {h}m | {len(paired)} | {mean_d:+.4f} "
                    f"[{lo_m:+.4f}, {hi_m:+.4f}] | {sig_m} | "
                    f"{ds:+.2f} [{ds_lo:+.2f}, {ds_hi:+.2f}] | {sig_s} |"
                )
            lines.append("")

        # True monthly-rebalance strategy backtest (net-of-tcost sequential PnL)
        # Codex Round-7 asked for this: the paired h=6 CRPS-style test isn't
        # the same as a real strategy's month-by-month net return stream.
        lines.append("## True monthly-rebalance strategy backtest (net-of-tcost)")
        lines.append("")
        lines.append(
            "> Sequential strategy: at each month-end t, rebalance to w_t "
            "(paying 10 bps round-trip × L1 turnover), hold for 1 month. "
            "Realized return at t = realized_log_return at horizon=1m for w_t. "
            "Net return stream is GROSS − tcost drag per step. Metrics: "
            "ann net Sharpe, max drawdown on the actual realized path."
        )
        lines.append("")
        tcost_rt = 0.0010   # 10 bps round-trip per unit turnover
        strat_rows: list[dict] = []
        per_method_series: dict[str, np.ndarray] = {}
        for method in sorted(alloc_df["method"].unique()):
            sub_s = alloc_df[(alloc_df.method == method) & (alloc_df.horizon == 1)].sort_values("asof").reset_index(drop=True)
            if len(sub_s) < 12:
                continue
            wlist = [json.loads(w) for w in sub_s["weights"].values]
            keys = sorted({k for wd in wlist for k in wd.keys()})
            vecs = np.array([[wd.get(k, 0.0) for k in keys] for wd in wlist])
            # turnover[0] = 0 (no prior weights); turnover[t] = L1 diff t-1→t
            turnover = np.concatenate([[0.0], np.sum(np.abs(np.diff(vecs, axis=0)), axis=1)])
            drag = turnover * tcost_rt
            r_gross = sub_s["realized_log_return"].values
            r_net = r_gross - drag
            ann_ret = float(r_net.mean() * 12)
            ann_vol = float(r_net.std(ddof=1) * np.sqrt(12))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
            # Max DD on cum log returns
            cum = np.cumsum(r_net)
            run_max = np.maximum.accumulate(cum)
            max_dd_log = float((cum - run_max).min())
            max_dd = float(np.exp(max_dd_log) - 1)
            total_net = float(np.exp(r_net.sum()) - 1)
            strat_rows.append({
                "method": method,
                "n_months": len(r_net),
                "ann_net_return": ann_ret,
                "ann_net_vol": ann_vol,
                "sharpe": sharpe,
                "total_net": total_net,
                "max_dd": max_dd,
                "mean_drag_bps": float(drag.mean() * 10000),
            })
            per_method_series[method] = r_net
        lines.append("| method | n_m | ann net ret | ann vol | Sharpe | total net | max DD | mean drag (bps/m) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in strat_rows:
            lines.append(
                f"| {r['method']} | {r['n_months']} | {r['ann_net_return']:+.2%} | "
                f"{r['ann_net_vol']:.2%} | {r['sharpe']:+.2f} | "
                f"{r['total_net']:+.1%} | {r['max_dd']:+.1%} | {r['mean_drag_bps']:.1f} |"
            )
        lines.append("")

        # Paired net-return test for v2 vs MVN sequential strategies
        if "sp_cvar_6m" in per_method_series and "mvn_sp_cvar_6m" in per_method_series:
            r_v2 = per_method_series["sp_cvar_6m"]
            r_mvn = per_method_series["mvn_sp_cvar_6m"]
            n_align = min(len(r_v2), len(r_mvn))
            # Align from end (recent asofs are common)
            r_v2, r_mvn = r_v2[-n_align:], r_mvn[-n_align:]
            d = r_v2 - r_mvn
            mean_d, lo_m, hi_m = moving_block_bootstrap_mean(d, block_length=1)
            ds, ds_lo, ds_hi = moving_block_bootstrap_sharpe_diff(r_v2, r_mvn, block_length=1)
            sig_m = "✓" if (lo_m > 0 or hi_m < 0) else "—"
            sig_s = "✓" if (ds_lo > 0 or ds_hi < 0) else "—"
            lines.append("### Paired NET v2 SP-CVaR vs MVN SP-CVaR (monthly sequential)")
            lines.append("")
            lines.append(f"- n_months aligned: {n_align}")
            lines.append(f"- paired Δmean (monthly log): {mean_d:+.5f} "
                         f"CI [{lo_m:+.5f}, {hi_m:+.5f}] {sig_m}  →  ann {mean_d*12:+.2%}")
            lines.append(f"- paired ΔSharpe (monthly): {ds:+.3f} "
                         f"CI [{ds_lo:+.3f}, {ds_hi:+.3f}] {sig_s}")
            lines.append("")

        # Turnover analysis — how much tradeable edge survives transaction cost
        lines.append("## Turnover & transaction-cost drag")
        lines.append("")
        lines.append(
            "> L1 monthly turnover = Σ|w_t − w_{t-1}|; weights emitted at each "
            "asof. Cost drag assumes 10 bps round-trip per unit turnover, "
            "applied at the monthly rebalance cadence (realistic for h=1, "
            "upper-bound for longer holds). Only the v2 methods have meaningful "
            "turnover — 60/40 is static."
        )
        lines.append("")
        lines.append("| method | mean L1 turnover / month | ann drag (10 bps rt) |")
        lines.append("|---|---|---|")
        for method in sorted(alloc_df["method"].unique()):
            if method == "bench_60_40":
                continue
            sub_m = alloc_df[(alloc_df.method == method) & (alloc_df.horizon == 6)].sort_values("asof")
            if len(sub_m) < 2:
                continue
            weights_list = [json.loads(w) for w in sub_m["weights"].values]
            all_keys = sorted({k for wd in weights_list for k in wd.keys()})
            vecs = np.array([[wd.get(k, 0.0) for k in all_keys] for wd in weights_list])
            turn = np.sum(np.abs(np.diff(vecs, axis=0)), axis=1)
            mean_turn = float(turn.mean())
            ann_drag = mean_turn * 0.0010 * 12   # 10bps round-trip × 12 months
            lines.append(f"| {method} | {mean_turn:.3f} | {ann_drag:.2%} |")
        lines.append("")

    if "crps_ar1" in df.columns and df["crps_ar1"].notna().any():
        lines.append("## AR(1) benchmark (stronger than unconditional Gaussian)")
        lines.append("")
        lines.append("| asset | h | skill vs AR(1) | skill vs Gaussian |")
        lines.append("|---|---|---|---|")
        for (asset, h), sub in df.groupby(["asset", "horizon"]):
            s_ar = sub["crps_ar1"].mean()
            s_g = sub["crps_bench"].mean()
            s_v = sub["crps_v2"].mean()
            skill_ar = 1 - s_v / s_ar if s_ar > 0 else float("nan")
            skill_g = 1 - s_v / s_g if s_g > 0 else float("nan")
            lines.append(f"| {asset} | {h}m | {skill_ar:+.1%} | {skill_g:+.1%} |")
        lines.append("")

    doc_path = doc_dir / f"backtest_{tag}.md"
    doc_path.write_text("\n".join(lines))

    print(f"\n✓ results  → {out_path}")
    print(f"✓ meta     → {meta_path}")
    print(f"✓ doc      → {doc_path}")
    print()
    print(summary.to_string())
    print()
    has_gamma = df["gamma"].notna().any() and gamma_grid is not None
    has_kernel = ("kernel" in df.columns) and kernel_grid is not None
    grp_cols = []
    if has_alpha: grp_cols.append("alpha")
    if has_gamma: grp_cols.append("gamma")
    if has_kernel: grp_cols.append("kernel")
    if grp_cols:
        print(f"\n{'+'.join(grp_cols)} sweep overall skill:")
        for key, sub in df.groupby(grp_cols):
            s = 1.0 - sub["crps_v2"].mean() / sub["crps_bench"].mean()
            label = ", ".join(f"{c}={k}" for c, k in zip(grp_cols, key if isinstance(key, tuple) else (key,)))
            print(f"  {label}: skill {s:+.1%}")
    else:
        s = 1.0 - df["crps_v2"].mean() / df["crps_bench"].mean()
        print(f"Overall skill: {s:+.1%} | "
              f"mean CRPS v2={df['crps_v2'].mean():.4f} vs bench {df['crps_bench'].mean():.4f}")


if __name__ == "__main__":
    main()
