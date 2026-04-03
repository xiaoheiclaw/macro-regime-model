#!/usr/bin/env python3
"""
Layer 2: Strategy Backtest — track portfolio PnL using historical allocation weights.

Runs daily. Records today's weights from BL/SP outputs, then computes
cumulative returns, rolling Sharpe, and max drawdown for each strategy.
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

WEIGHTS_HISTORY_PATH = os.path.join(DATA_DIR, "weights_history.csv")
SP_WEIGHTS_PATH = os.path.join(DATA_DIR, "stochastic_prog_weights.csv")
BACKTEST_DAILY_PATH = os.path.join(DATA_DIR, "backtest_daily.csv")
BACKTEST_SUMMARY_PATH = os.path.join(DATA_DIR, "backtest_summary.json")
BOND_DURATION = 8  # match black_litterman.py convention


def record_todays_weights():
    """Append today's weights from stochastic_prog_weights.csv to history."""
    if not os.path.exists(SP_WEIGHTS_PATH):
        print("⚠ No stochastic_prog_weights.csv found. Skipping weight recording.")
        return False

    # Check file freshness (only record if modified today)
    mtime = datetime.fromtimestamp(os.path.getmtime(SP_WEIGHTS_PATH))
    today = datetime.now().date()
    if mtime.date() != today:
        print(f"⚠ Weight file is from {mtime.date()}, not today. Skipping.")
        return False

    today_str = today.strftime("%Y-%m-%d")

    # Check if already recorded today
    if os.path.exists(WEIGHTS_HISTORY_PATH):
        existing = pd.read_csv(WEIGHTS_HISTORY_PATH)
        if today_str in existing["date"].values:
            print(f"✓ Weights already recorded for {today_str}")
            return True

    # Read current weights
    sp_weights = pd.read_csv(SP_WEIGHTS_PATH)

    write_header = not os.path.exists(WEIGHTS_HISTORY_PATH)
    with open(WEIGHTS_HISTORY_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "strategy"] + ASSETS)
        for _, row in sp_weights.iterrows():
            w.writerow([today_str, row["strategy"]] + [row.get(a, 0) for a in ASSETS])

    print(f"✓ Recorded {len(sp_weights)} strategies for {today_str}")
    return True


def compute_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for all assets, matching BL convention."""
    returns = pd.DataFrame(index=df.index)
    for a in ASSETS:
        if a not in df.columns:
            continue
        if a == "US10Y_yield":
            returns[a] = -df[a].diff() * BOND_DURATION / 100
        else:
            returns[a] = df[a].pct_change()
    return returns


def run_backtest():
    """Compute portfolio PnL from weight history."""
    if not os.path.exists(WEIGHTS_HISTORY_PATH):
        print("⚠ No weight history yet. Need at least 2 days of data.")
        return

    wh = pd.read_csv(WEIGHTS_HISTORY_PATH, parse_dates=["date"])
    df = load_merged()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    asset_returns = compute_asset_returns(df)

    strategies = wh["strategy"].unique()
    all_rows = []

    for strategy in strategies:
        sw = wh[wh["strategy"] == strategy].set_index("date").sort_index()
        cum_return = 0.0
        peak = 1.0
        equity = 1.0

        for i in range(len(sw)):
            weight_date = sw.index[i]
            # Find the next trading day's return
            future_dates = asset_returns.index[asset_returns.index > weight_date]
            if len(future_dates) == 0:
                continue

            # Use next trading day's return
            next_date = future_dates[0]
            day_returns = asset_returns.loc[next_date]

            weights = sw.iloc[i][ASSETS].astype(float)
            port_return = (weights * day_returns).sum()

            if pd.isna(port_return):
                continue

            equity *= (1 + port_return)
            cum_return = equity - 1
            peak = max(peak, equity)
            drawdown = (equity / peak) - 1

            all_rows.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "strategy": strategy,
                "daily_return": round(port_return, 6),
                "cumulative_return": round(cum_return, 6),
                "drawdown": round(drawdown, 6),
            })

    if not all_rows:
        print("⚠ Not enough data for backtest yet.")
        return

    bt = pd.DataFrame(all_rows)
    bt.to_csv(BACKTEST_DAILY_PATH, index=False)
    print(f"✓ Backtest updated: {len(bt)} data points")

    # Build summary
    summary = {"as_of": datetime.now().strftime("%Y-%m-%d"), "strategies": {}}

    for strategy in strategies:
        s = bt[bt["strategy"] == strategy]
        if len(s) < 2:
            continue

        rets = s["daily_return"].values
        ann_factor = 252

        sharpe_all = (rets.mean() / rets.std() * np.sqrt(ann_factor)) if rets.std() > 0 else 0

        # Rolling 60d Sharpe
        sharpe_60d = None
        if len(rets) >= 60:
            r60 = rets[-60:]
            sharpe_60d = round((r60.mean() / r60.std() * np.sqrt(ann_factor)) if r60.std() > 0 else 0, 2)

        summary["strategies"][strategy] = {
            "cumulative_return": round(s["cumulative_return"].iloc[-1], 4),
            "sharpe_all": round(sharpe_all, 2),
            "sharpe_60d": sharpe_60d,
            "max_drawdown": round(s["drawdown"].min(), 4),
            "n_days": len(s),
        }

    # Rank by Sharpe
    ranked = sorted(
        summary["strategies"].items(),
        key=lambda x: x[1].get("sharpe_60d") or x[1]["sharpe_all"],
        reverse=True,
    )
    summary["ranking"] = [r[0] for r in ranked]

    with open(BACKTEST_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print
    print(f"\n📊 Strategy Performance:")
    print(f"{'Strategy':>15s}  {'CumRet':>7s}  {'Sharpe':>7s}  {'60d':>5s}  {'MaxDD':>7s}  {'Days':>4s}")
    print("-" * 55)
    for name, m in summary["strategies"].items():
        s60 = f"{m['sharpe_60d']:.1f}" if m["sharpe_60d"] is not None else "N/A"
        print(f"{name:>15s}  {m['cumulative_return']:+7.2%}  {m['sharpe_all']:7.2f}  {s60:>5s}  {m['max_drawdown']:7.2%}  {m['n_days']:4d}")
    print(f"\nRanking: {' > '.join(summary['ranking'])}")


def main():
    print("=" * 60)
    print(f"Strategy Backtest - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    record_todays_weights()
    run_backtest()


if __name__ == "__main__":
    main()
