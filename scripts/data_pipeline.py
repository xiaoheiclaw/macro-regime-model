#!/usr/bin/env python3
"""
Macro Regime Attribution Engine - Data Pipeline
Pulls multi-asset + macro factor data (2020-01-01 to present)
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR

# ── Config ──────────────────────────────────────────────
START = "2020-01-01"
END = datetime.now().strftime("%Y-%m-%d")

# ── 1. Asset Prices via yfinance ────────────────────────
import yfinance as yf

TICKERS = {
    # US rates (proxy via ETFs + direct)
    "^TNX":    "US10Y_yield",     # 10Y Treasury Yield
    "^FVX":    "US5Y_yield",      # 5Y Treasury Yield
    # Commodities — energy
    "CL=F":    "WTI_crude",       # WTI Oil
    "NG=F":    "NatGas",          # Natural Gas
    # Commodities — metals
    "GC=F":    "Gold",            # Gold futures
    "SI=F":    "Silver",          # Silver futures
    "HG=F":    "Copper",          # Copper futures (Dr. Copper)
    # Commodities — agriculture
    "ZC=F":    "Corn",            # Corn futures
    "ZS=F":    "Soybeans",        # Soybean futures (China trade sensitive)
    # Equities
    "^GSPC":   "SPX",             # S&P 500
    "^VIX":    "VIX",             # Volatility
    "^HSI":    "HSI",             # Hang Seng
    # FX
    "DX-Y.NYB": "DXY",           # Dollar Index
    # Crypto
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
}

print("=" * 60)
print(f"Macro Regime Data Pipeline | {START} → {END}")
print("=" * 60)

print("\n[1/3] Fetching asset prices via yfinance...")
frames = {}
for ticker, name in TICKERS.items():
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  ⚠ {name} ({ticker}): empty")
            continue
        # Use Close price; for yields ^TNX already gives yield level
        col = df["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        frames[name] = col
        print(f"  ✓ {name}: {len(col)} rows")
    except Exception as e:
        print(f"  ✗ {name} ({ticker}): {e}")

asset_df = pd.DataFrame(frames)
asset_df.index = pd.to_datetime(asset_df.index)
asset_df.index.name = "date"

# ── 2. FRED macro factors ──────────────────────────────
print("\n[2/3] Fetching macro factors via FRED...")

# Try using fredapi if API key available, otherwise use pandas_datareader or direct CSV
FRED_SERIES = {
    "DGS10":   "US10Y_FRED",       # 10Y yield (backup)
    "DGS2":    "US2Y_yield",       # 2Y yield
    "T5YIE":   "BEI_5Y",          # 5Y Breakeven Inflation
    "T10YIE":  "BEI_10Y",         # 10Y Breakeven Inflation
    "SOFR":    "SOFR_rate",        # SOFR
    "WALCL":   "Fed_balance_sheet", # Fed total assets
    "DTWEXBGS": "USD_broad",       # Trade-weighted USD (broad)
}

fred_frames = {}
try:
    from fredapi import Fred
    # Try common API key locations
    fred_key = os.environ.get("FRED_API_KEY", None)
    if not fred_key:
        # Try reading from file
        for p in ["~/.fred_api_key", "~/.config/fred/api_key"]:
            fp = os.path.expanduser(p)
            if os.path.exists(fp):
                fred_key = open(fp).read().strip()
                break

    if fred_key:
        fred = Fred(api_key=fred_key)
        for series_id, name in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=START)
                fred_frames[name] = s
                print(f"  ✓ {name}: {len(s)} rows")
            except Exception as e:
                print(f"  ✗ {name} ({series_id}): {e}")
    else:
        print("  ⚠ No FRED API key found. Trying public CSV endpoint...")
        raise ImportError("no key")

except (ImportError, Exception) as e:
    # Fallback: fetch FRED data via public CSV endpoint
    print("  Using FRED public CSV fallback...")
    for series_id, name in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
            s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
            s = s.replace(".", np.nan).astype(float)
            fred_frames[name] = s
            print(f"  ✓ {name}: {len(s)} rows")
        except Exception as ex:
            print(f"  ✗ {name} ({series_id}): {ex}")

fred_df = pd.DataFrame(fred_frames)
fred_df.index = pd.to_datetime(fred_df.index)
fred_df.index.name = "date"

# ── 3. Merge & Clean ───────────────────────────────────
print("\n[3/3] Merging and cleaning...")

merged = asset_df.join(fred_df, how="outer")

# Forward fill (max 5 days for weekends/holidays)
merged = merged.ffill(limit=5)

# Keep only trading days (weekdays)
merged = merged[merged.index.dayofweek < 5]

# Compute derived signals
if "US10Y_yield" in merged.columns and "US2Y_yield" in merged.columns:
    merged["yield_curve_2s10s"] = merged["US10Y_yield"] - merged["US2Y_yield"]

if "US10Y_yield" in merged.columns and "BEI_10Y" in merged.columns:
    merged["real_yield_10Y"] = merged["US10Y_yield"] - merged["BEI_10Y"]

if "WTI_crude" in merged.columns and "US10Y_yield" in merged.columns:
    # Rolling 20-day correlation: Oil vs 10Y yield
    merged["corr_oil_10Y_20d"] = (
        merged["WTI_crude"].pct_change()
        .rolling(20)
        .corr(merged["US10Y_yield"].diff())
    )

if "SPX" in merged.columns and "US10Y_yield" in merged.columns:
    # Rolling 20-day correlation: SPX vs 10Y yield
    merged["corr_spx_10Y_20d"] = (
        merged["SPX"].pct_change()
        .rolling(20)
        .corr(merged["US10Y_yield"].diff())
    )

if "Gold" in merged.columns and "BTC" in merged.columns:
    # Rolling 20-day correlation: Gold vs BTC
    merged["corr_gold_btc_20d"] = (
        merged["Gold"].pct_change()
        .rolling(20)
        .corr(merged["BTC"].pct_change())
    )

# Missing data report
missing = merged.isnull().sum()
missing_pct = (missing / len(merged) * 100).round(1)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

# Save
out_path = os.path.join(DATA_DIR, "merged_data.csv")
merged.to_csv(out_path)
print(f"\n✓ Saved {len(merged)} rows × {len(merged.columns)} cols → {out_path}")

# Save column descriptions
col_desc = {
    "US10Y_yield": "US 10Y Treasury Yield (%, from ^TNX)",
    "US5Y_yield": "US 5Y Treasury Yield (%, from ^FVX)",
    "US2Y_yield": "US 2Y Treasury Yield (%, from FRED DGS2)",
    "WTI_crude": "WTI Crude Oil Futures ($/bbl)",
    "Gold": "Gold Futures ($/oz)",
    "SPX": "S&P 500 Index",
    "VIX": "CBOE Volatility Index",
    "HSI": "Hang Seng Index",
    "DXY": "US Dollar Index",
    "BTC": "Bitcoin (USD)",
    "ETH": "Ethereum (USD)",
    "BEI_5Y": "5Y Breakeven Inflation Rate (%)",
    "BEI_10Y": "10Y Breakeven Inflation Rate (%)",
    "SOFR_rate": "Secured Overnight Financing Rate (%)",
    "Fed_balance_sheet": "Fed Total Assets ($M, WALCL)",
    "USD_broad": "Trade-Weighted USD Index (Broad)",
    "US10Y_FRED": "US 10Y Yield from FRED (backup)",
    "yield_curve_2s10s": "2s10s Yield Curve Spread (10Y - 2Y)",
    "real_yield_10Y": "10Y Real Yield (Nominal - BEI)",
    "corr_oil_10Y_20d": "20d Rolling Corr: Oil returns vs 10Y yield change",
    "corr_spx_10Y_20d": "20d Rolling Corr: SPX returns vs 10Y yield change",
    "corr_gold_btc_20d": "20d Rolling Corr: Gold returns vs BTC returns",
}

with open(os.path.join(DATA_DIR, "column_descriptions.json"), "w") as f:
    json.dump(col_desc, f, indent=2)

# Print summary
print(f"\n{'='*60}")
print("DATA SUMMARY")
print(f"{'='*60}")
print(f"Date range: {merged.index.min().date()} → {merged.index.max().date()}")
print(f"Columns: {len(merged.columns)}")
print(f"\nLatest values (last row):")
latest = merged.iloc[-1]
for col in ["US10Y_yield", "US2Y_yield", "WTI_crude", "Gold", "SPX", "VIX",
            "DXY", "BTC", "BEI_5Y", "BEI_10Y", "corr_oil_10Y_20d", "corr_spx_10Y_20d",
            "corr_gold_btc_20d", "real_yield_10Y", "yield_curve_2s10s"]:
    if col in latest.index and pd.notna(latest[col]):
        print(f"  {col}: {latest[col]:.4f}")

if len(missing_report) > 0:
    print(f"\nMissing data:")
    print(missing_report.to_string())

print("\n✓ Pipeline complete.")
