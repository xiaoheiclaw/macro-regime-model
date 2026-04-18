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
from lib.metrics import crps_sample, crps_normal, pit_sample, pit_normal  # type: ignore


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
        forecast.run(asof=asof_str, alpha=alpha, gamma=gamma)
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
    args = ap.parse_args()

    state = pd.read_parquet(Path(DATA_DIR) / "state_features.parquet")
    asofs = pd.date_range(args.start, args.end, freq=f"{args.step}ME")
    print(f"Backtest: {len(asofs)} asof steps, {args.start} → {args.end}")

    alpha_grid = [float(x) for x in args.alphas.split(",")] if args.alphas else None
    gamma_grid = [float(x) for x in args.gammas.split(",")] if args.gammas else None
    if alpha_grid:
        print(f"α grid: {alpha_grid}")
    if gamma_grid:
        print(f"γ grid: {gamma_grid}")

    t0 = time.time()
    rows: list[dict] = []
    expanding = not args.no_expanding_regimes
    alphas_to_run = alpha_grid if alpha_grid is not None else [None]
    gammas_to_run = gamma_grid if gamma_grid is not None else [None]

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
            if not _run_forecast_at(
                asof_str,
                expanding_regimes=expanding,
                alpha=alpha,
                gamma=gamma,
                skip_regime_refit=True,
            ):
                continue

            scn = pd.read_parquet(SCENARIOS_PATH)
            if scn.empty:
                continue

            alpha_val = float(alpha) if alpha is not None else float("nan")
            gamma_val = float(gamma) if gamma is not None else float("nan")
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

                rows.append({
                    "asof": asof,
                    "alpha": alpha_val,
                    "gamma": gamma_val,
                    "asset": asset,
                    "horizon": int(h),
                    "realized": realized,
                    "crps_v2": crps_v2,
                    "crps_bench": crps_bench,
                    "pit_v2": pit_v2,
                    "pit_bench": pit_bench,
                    "n_scn": len(samples),
                })

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
    doc_path = doc_dir / f"backtest_{tag}.md"
    doc_path.write_text("\n".join(lines))

    print(f"\n✓ results  → {out_path}")
    print(f"✓ meta     → {meta_path}")
    print(f"✓ doc      → {doc_path}")
    print()
    print(summary.to_string())
    print()
    has_gamma = df["gamma"].notna().any() and gamma_grid is not None
    grp_cols = [c for c in ["alpha", "gamma"] if (c == "alpha" and has_alpha) or (c == "gamma" and has_gamma)]
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
