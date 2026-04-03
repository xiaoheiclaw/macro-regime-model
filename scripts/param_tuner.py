#!/usr/bin/env python3
"""
Layer 3: Parameter Tuner — proposes parameter adjustments based on Layer 1 + 2 data.

Manual trigger only. Proposes changes with guardrails, requires confirmation.
Usage: python scripts/param_tuner.py
"""
import os, sys, json, csv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, TUNING_PARAMS
from lib.tuning import load_tuning_params

SCORECARD_PATH = os.path.join(DATA_DIR, "prediction_scorecard.csv")
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")
TUNING_HISTORY_PATH = os.path.join(DATA_DIR, "tuning_history.csv")

# Guardrails
MAX_CHANGE_PCT = 0.20        # max 20% change per review
OMEGA_BOUNDS = (0.5, 2.0)
ENSEMBLE_BOUNDS = (0.3, 0.7)
MIN_EVALS_OMEGA = 60         # need 60+ evaluations to touch omega
MIN_EVALS_ENSEMBLE = 90      # need 90+ evaluations for ensemble


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def limit_change(current, proposed, max_pct):
    """Limit change to max_pct of current value."""
    max_delta = abs(current) * max_pct
    if max_delta == 0:
        max_delta = max_pct  # handle zero case
    delta = proposed - current
    if abs(delta) > max_delta:
        delta = max_delta if delta > 0 else -max_delta
    return current + delta


def analyze_omega(scorecard: pd.DataFrame, current_scale: float) -> dict:
    """Analyze TimesFM accuracy and propose omega_scale adjustment."""
    if len(scorecard) < MIN_EVALS_OMEGA:
        return {
            "skip": True,
            "reason": f"Need {MIN_EVALS_OMEGA}+ evaluations, have {len(scorecard)}",
        }

    per_asset = {}
    for asset in scorecard["asset"].unique():
        a = scorecard[scorecard["asset"] == asset]
        dir_valid = a["direction_correct"].dropna()
        if len(dir_valid) < 10:
            continue
        da = dir_valid.mean()
        n = len(dir_valid)
        # 95% CI for directional accuracy
        se = np.sqrt(da * (1 - da) / n)
        ci_low = da - 1.96 * se
        ci_high = da + 1.96 * se
        per_asset[asset] = {
            "directional_accuracy": round(da, 3),
            "ci_95": (round(ci_low, 3), round(ci_high, 3)),
            "n": n,
            "mae": round(a["abs_error"].mean(), 4),
        }

    # Overall accuracy
    all_dir = scorecard["direction_correct"].dropna()
    overall_da = all_dir.mean()

    # Proposal logic:
    # If overall accuracy < 50% → inflate omega (trust AI less)
    # If overall accuracy > 65% → deflate omega (trust AI more)
    # Otherwise → no change
    proposed = current_scale
    rationale = "No change"

    if overall_da < 0.50:
        proposed = current_scale * 1.15  # +15%
        rationale = f"Overall directional accuracy {overall_da:.0%} < 50%, inflating omega to trust AI less"
    elif overall_da > 0.65:
        proposed = current_scale * 0.90  # -10%
        rationale = f"Overall directional accuracy {overall_da:.0%} > 65%, deflating omega to trust AI more"

    proposed = limit_change(current_scale, proposed, MAX_CHANGE_PCT)
    proposed = clamp(proposed, *OMEGA_BOUNDS)
    proposed = round(proposed, 3)

    return {
        "skip": False,
        "current": current_scale,
        "proposed": proposed,
        "changed": proposed != current_scale,
        "rationale": rationale,
        "overall_da": round(overall_da, 3),
        "per_asset": per_asset,
    }


def analyze_ensemble(scorecard: pd.DataFrame, current_weights: dict) -> dict:
    """Propose ensemble weight adjustments (placeholder — needs regime hit rate data)."""
    if len(scorecard) < MIN_EVALS_ENSEMBLE:
        return {
            "skip": True,
            "reason": f"Need {MIN_EVALS_ENSEMBLE}+ evaluations, have {len(scorecard)}",
        }

    # For now: no automatic proposal. This needs regime-conditional performance
    # data which requires ensemble_regime.csv cross-referenced with actual returns.
    return {
        "skip": False,
        "current": current_weights,
        "proposed": current_weights,
        "changed": False,
        "rationale": "Ensemble weight tuning requires regime-conditional backtest (future enhancement)",
    }


def main():
    print("=" * 60)
    print(f"Parameter Tuner - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    params = load_tuning_params()

    # Load data
    has_scorecard = os.path.exists(SCORECARD_PATH)
    has_backtest = os.path.exists(BACKTEST_SUMMARY_PATH)

    if not has_scorecard:
        print("⚠ No prediction scorecard found. Run daily pipeline for at least 7 days first.")
        return

    scorecard = pd.read_csv(SCORECARD_PATH)
    print(f"\nScorecard: {len(scorecard)} evaluations")

    if has_backtest:
        with open(BACKTEST_SUMMARY_PATH) as f:
            bt_summary = json.load(f)
        print(f"Backtest: {len(bt_summary.get('strategies', {}))} strategies tracked")

    # ── Omega analysis ──
    print("\n" + "-" * 40)
    print("1. OMEGA SCALE (TimesFM trust level)")
    print("-" * 40)
    omega_result = analyze_omega(scorecard, params.get("omega_scale", 1.0))

    if omega_result["skip"]:
        print(f"⏭ Skipped: {omega_result['reason']}")
    else:
        print(f"Overall directional accuracy: {omega_result['overall_da']:.0%}")
        print(f"\nPer-asset breakdown:")
        for asset, m in omega_result.get("per_asset", {}).items():
            ci = m["ci_95"]
            print(f"  {asset:>15s}: {m['directional_accuracy']:.0%} ({ci[0]:.0%}-{ci[1]:.0%}, n={m['n']}), MAE={m['mae']:.4f}")
        print(f"\nCurrent omega_scale: {omega_result['current']}")
        print(f"Proposed omega_scale: {omega_result['proposed']}")
        print(f"Rationale: {omega_result['rationale']}")

    # ── Ensemble weights ──
    print("\n" + "-" * 40)
    print("2. ENSEMBLE WEIGHTS (Markov vs Wasserstein)")
    print("-" * 40)
    ens_result = analyze_ensemble(scorecard, params.get("ensemble_weights", {"markov": 0.6, "wkm": 0.4}))

    if ens_result["skip"]:
        print(f"⏭ Skipped: {ens_result['reason']}")
    else:
        print(f"Current: Markov {ens_result['current']['markov']}, WKM {ens_result['current']['wkm']}")
        print(f"Proposed: Markov {ens_result['proposed']['markov']}, WKM {ens_result['proposed']['wkm']}")
        print(f"Rationale: {ens_result['rationale']}")

    # ── Correlation thresholds ──
    print("\n" + "-" * 40)
    print("3. CORRELATION THRESHOLDS")
    print("-" * 40)
    print("Current thresholds:")
    for k, v in params.get("correlation_thresholds", {}).items():
        print(f"  {k}: {v}")
    print("(Threshold tuning requires regime-conditional return analysis — future enhancement)")

    # ── Apply changes ──
    any_changes = (
        (not omega_result.get("skip", True) and omega_result.get("changed", False))
        or (not ens_result.get("skip", True) and ens_result.get("changed", False))
    )

    if not any_changes:
        print("\n✓ No parameter changes proposed.")
        return

    print("\n" + "=" * 60)
    print("PROPOSED CHANGES:")
    print("=" * 60)
    changes = []
    if not omega_result.get("skip") and omega_result.get("changed"):
        print(f"  omega_scale: {omega_result['current']} → {omega_result['proposed']}")
        changes.append(f"omega_scale: {omega_result['current']} → {omega_result['proposed']}")

    confirm = input("\nApply these changes? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Write updated params
    if not omega_result.get("skip") and omega_result.get("changed"):
        params["omega_scale"] = omega_result["proposed"]
    if not ens_result.get("skip") and ens_result.get("changed"):
        params["ensemble_weights"] = ens_result["proposed"]

    params["version"] = params.get("version", 0) + 1
    params["updated"] = datetime.now().strftime("%Y-%m-%d")
    params["updated_by"] = "param_tuner"

    with open(TUNING_PARAMS, "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"✓ Updated {TUNING_PARAMS}")

    # Log to history
    write_header = not os.path.exists(TUNING_HISTORY_PATH)
    with open(TUNING_HISTORY_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "version", "changes", "rationale"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            params["version"],
            "; ".join(changes),
            omega_result.get("rationale", "") + " | " + ens_result.get("rationale", ""),
        ])
    print(f"✓ Logged to {TUNING_HISTORY_PATH}")


if __name__ == "__main__":
    main()
