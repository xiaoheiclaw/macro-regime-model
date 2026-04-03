#!/usr/bin/env python3
"""
Layer 1: Prediction Scorecard — evaluate TimesFM predictions against actuals.

Runs daily. Compares 7-trading-day-old predictions to realized returns.
Outputs per-asset accuracy metrics (MAE, directional accuracy, bias).
"""
import os, sys, json, csv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR
from lib.data_loader import load_merged
from lib.universe import ASSETS

HISTORY_PATH = os.path.join(DATA_DIR, "timesfm_views_history.csv")
SCORECARD_PATH = os.path.join(DATA_DIR, "prediction_scorecard.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "prediction_scorecard_summary.json")
EVAL_LAG = 7  # trading days to look back
BOND_DURATION = 8  # match black_litterman.py convention
SUMMARY_WINDOW = 90  # rolling days for summary stats


def compute_actual_return(df: pd.DataFrame, asset: str, start_idx: int, end_idx: int) -> float:
    """Compute realized return between two index positions in merged_data."""
    if asset == "US10Y_yield":
        # Bond return proxy: -delta_yield * duration / 100
        y0 = df.iloc[start_idx][asset]
        y1 = df.iloc[end_idx][asset]
        return -(y1 - y0) * BOND_DURATION / 100
    else:
        p0 = df.iloc[start_idx][asset]
        p1 = df.iloc[end_idx][asset]
        if p0 == 0 or pd.isna(p0) or pd.isna(p1):
            return np.nan
        return (p1 / p0) - 1


def main():
    print("=" * 60)
    print(f"Prediction Scorecard - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Check if history exists
    if not os.path.exists(HISTORY_PATH):
        print("⚠ No prediction history yet. Skipping evaluation.")
        return

    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    df = load_merged()
    df_dates = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df["date"])

    # If df has date as column, set it as index for alignment
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    today = pd.Timestamp.now().normalize()

    # Find predictions that are at least EVAL_LAG trading days old
    new_evals = []
    already_evaluated = set()
    if os.path.exists(SCORECARD_PATH):
        existing = pd.read_csv(SCORECARD_PATH)
        for _, row in existing.iterrows():
            already_evaluated.add((row["prediction_date"], row["asset"]))

    prediction_dates = history["date"].dt.normalize().unique()

    for pred_date in prediction_dates:
        # Find the index position of pred_date in df
        if pred_date not in df.index:
            # Find nearest trading day
            mask = df.index <= pred_date
            if not mask.any():
                continue
            pred_idx = df.index[mask].max()
        else:
            pred_idx = pred_date

        pred_pos = df.index.get_loc(pred_idx)
        eval_pos = pred_pos + EVAL_LAG

        # Not enough trading days yet
        if eval_pos >= len(df):
            continue

        eval_date = df.index[eval_pos]

        # Get predictions for this date
        day_preds = history[history["date"].dt.normalize() == pred_date]

        for _, pred_row in day_preds.iterrows():
            asset = pred_row["asset"]
            pred_date_str = pred_date.strftime("%Y-%m-%d")

            if (pred_date_str, asset) in already_evaluated:
                continue
            if asset not in df.columns:
                continue

            predicted = pred_row["predicted_return"]
            actual = compute_actual_return(df, asset, pred_pos, eval_pos)

            if pd.isna(actual):
                continue

            error = predicted - actual
            direction_correct = int(np.sign(predicted) == np.sign(actual)) if predicted != 0 else np.nan

            new_evals.append({
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "prediction_date": pred_date_str,
                "asset": asset,
                "predicted_return": round(predicted, 6),
                "actual_return": round(actual, 6),
                "error": round(error, 6),
                "abs_error": round(abs(error), 6),
                "direction_correct": direction_correct,
            })

    # Append new evaluations
    if new_evals:
        write_header = not os.path.exists(SCORECARD_PATH)
        with open(SCORECARD_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_evals[0].keys())
            if write_header:
                w.writeheader()
            w.writerows(new_evals)
        print(f"✓ {len(new_evals)} new evaluations appended to scorecard")
    else:
        print("No new predictions ready for evaluation (need 7 trading days)")

    # Build summary from full scorecard
    if not os.path.exists(SCORECARD_PATH):
        print("⚠ No scorecard data yet.")
        return

    sc = pd.read_csv(SCORECARD_PATH, parse_dates=["eval_date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=SUMMARY_WINDOW)
    recent = sc[sc["eval_date"] >= cutoff]

    if len(recent) == 0:
        print("⚠ No recent evaluations within summary window.")
        return

    per_asset = {}
    for asset in recent["asset"].unique():
        a = recent[recent["asset"] == asset]
        dir_valid = a["direction_correct"].dropna()
        per_asset[asset] = {
            "mae": round(a["abs_error"].mean(), 4),
            "directional_accuracy": round(dir_valid.mean(), 2) if len(dir_valid) > 0 else None,
            "bias": round(a["error"].mean(), 4),
            "n_evals": len(a),
        }

    dir_all = recent["direction_correct"].dropna()
    summary = {
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "lookback_days": SUMMARY_WINDOW,
        "per_asset": per_asset,
        "overall": {
            "mae": round(recent["abs_error"].mean(), 4),
            "directional_accuracy": round(dir_all.mean(), 2) if len(dir_all) > 0 else None,
            "n_evals": len(recent),
        },
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n📊 Prediction Scorecard ({SUMMARY_WINDOW}d rolling):")
    print(f"{'Asset':>15s}  {'MAE':>6s}  {'Dir%':>5s}  {'Bias':>7s}  {'N':>3s}")
    print("-" * 45)
    for asset, m in per_asset.items():
        dir_str = f"{m['directional_accuracy']*100:.0f}%" if m["directional_accuracy"] is not None else "N/A"
        print(f"{asset:>15s}  {m['mae']:6.4f}  {dir_str:>5s}  {m['bias']:+7.4f}  {m['n_evals']:3d}")
    o = summary["overall"]
    dir_str = f"{o['directional_accuracy']*100:.0f}%" if o["directional_accuracy"] is not None else "N/A"
    print("-" * 45)
    print(f"{'Overall':>15s}  {o['mae']:6.4f}  {dir_str:>5s}  {'':>7s}  {o['n_evals']:3d}")


if __name__ == "__main__":
    main()
