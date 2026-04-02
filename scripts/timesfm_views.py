#!/usr/bin/env python3
"""
TimesFM → Black-Litterman Views Bridge

Calls TimesFM CLI to predict multi-asset prices, then converts predictions
into Black-Litterman compatible views JSON at data/timesfm_views.json.
"""
import os, sys, json, subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, TIMESFM_PYTHON, TIMESFM_SCRIPT

# ── Config ──────────────────────────────────────────────
# Maps macro_regime_model asset names → yfinance tickers for TimesFM
ASSET_TICKERS = {
    "SPX":        "SPY",       # ETF proxy (more liquid than ^GSPC)
    "Gold":       "GC=F",
    "WTI_crude":  "CL=F",
    "BTC":        "BTC-USD",
    "DXY":        "DX-Y.NYB",
}

# Bond proxy: TLT (20+ year Treasury ETF) → infer yield direction
BOND_TICKER = "TLT"
BOND_DURATION = 17  # approximate effective duration of TLT

FORECAST_DAYS = 7
OUTPUT_PATH = os.path.join(DATA_DIR, "timesfm_views.json")


def call_timesfm(tickers: str, days: int) -> list[dict]:
    """Call TimesFM CLI and return parsed JSON results."""
    if not os.path.exists(TIMESFM_PYTHON):
        raise FileNotFoundError(f"TimesFM venv not found: {TIMESFM_PYTHON}")
    if not os.path.exists(TIMESFM_SCRIPT):
        raise FileNotFoundError(f"TimesFM script not found: {TIMESFM_SCRIPT}")

    env = os.environ.copy()
    # Ensure proxy is set for yfinance data fetch
    if "ALL_PROXY" not in env and "HTTP_PROXY" not in env:
        env["ALL_PROXY"] = "socks5://127.0.0.1:59526"

    result = subprocess.run(
        [TIMESFM_PYTHON, TIMESFM_SCRIPT,
         "--ticker", tickers,
         "--days", str(days),
         "--output", "json"],
        capture_output=True, text=True, env=env,
        timeout=600,  # 10 min max (model loading can be slow)
    )

    if result.returncode != 0:
        raise RuntimeError(f"TimesFM failed (rc={result.returncode}):\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}")


    # TimesFM mixes library warnings and progress bars into stdout;
    # extract JSON by finding the outermost [ ... ] array
    stdout = result.stdout
    json_start = stdout.find("\n[")
    if json_start == -1:
        json_start = stdout.find("[")
    if json_start == -1:
        raise RuntimeError(f"No JSON array in TimesFM output.\nSTDOUT: {stdout[-500:]}")

    # Find matching closing bracket
    json_str = stdout[json_start:]
    # Try to parse; if trailing garbage, find the last ]
    json_end = json_str.rfind("]")
    if json_end == -1:
        raise RuntimeError(f"No closing ] in TimesFM output.\nJSON start: {json_str[:200]}")

    json_str = json_str[:json_end + 1]
    return json.loads(json_str)


def build_views(predictions: list[dict], bond_pred: dict | None) -> dict:
    """Convert TimesFM predictions to B-L views format."""
    views = {}
    confidence = {}

    # Reverse map: ticker → asset name
    ticker_to_asset = {v: k for k, v in ASSET_TICKERS.items()}

    for pred in predictions:
        ticker = pred["ticker"]
        asset = ticker_to_asset.get(ticker)
        if asset is None:
            continue

        change_pct = pred["summary"]["predicted_change_pct"]
        views[asset] = round(change_pct / 100, 6)  # Convert % to decimal

        # Confidence from q10/q90 spread (wider = less confident)
        last_price = pred["last_close"]
        final_pred = pred["predictions"][-1]
        band_width = (final_pred["q90"] - final_pred["q10"]) / last_price
        confidence[asset] = round(band_width, 6)

    # Bond: infer yield view from TLT prediction
    if bond_pred is not None:
        tlt_change = bond_pred["summary"]["predicted_change_pct"] / 100
        # TLT return ≈ -duration × yield_change
        # So yield_change ≈ -TLT_return / duration
        yield_change_pct = -tlt_change / BOND_DURATION
        # B-L uses bond *return* (not yield change), which is ≈ -yield_change × duration
        # So the view for US10Y_yield in B-L return terms = tlt_change (same sign)
        views["US10Y_yield"] = round(tlt_change, 6)

        last_price = bond_pred["last_close"]
        final_pred = bond_pred["predictions"][-1]
        band_width = (final_pred["q90"] - final_pred["q10"]) / last_price
        confidence["US10Y_yield"] = round(band_width, 6)

    return {
        "generated": datetime.now().isoformat(),
        "forecast_days": FORECAST_DAYS,
        "scenario": {
            "name": f"TimesFM {FORECAST_DAYS}d Forecast",
            "prob": 1.0,
            "views": views,
            "confidence": confidence,
        },
    }


def main():
    print("=" * 60)
    print(f"TimesFM Views Generator - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Build ticker list
    all_tickers = list(ASSET_TICKERS.values()) + [BOND_TICKER]
    ticker_str = ",".join(all_tickers)
    print(f"\nTickers: {ticker_str}")
    print(f"Forecast: {FORECAST_DAYS} days")

    # Call TimesFM
    print("\nCalling TimesFM...")
    results = call_timesfm(ticker_str, FORECAST_DAYS)

    # Separate bond prediction from the rest
    bond_pred = None
    asset_preds = []
    for r in results:
        if r["ticker"] == BOND_TICKER:
            bond_pred = r
        else:
            asset_preds.append(r)

    # Build views
    output = build_views(asset_preds, bond_pred)

    # Summary
    print(f"\nViews generated:")
    for asset, view in output["scenario"]["views"].items():
        conf = output["scenario"]["confidence"].get(asset, 0)
        direction = "+" if view >= 0 else ""
        print(f"  {asset:15s}: {direction}{view*100:.2f}%  (band: {conf*100:.1f}%)")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
