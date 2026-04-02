#!/usr/bin/env python3
"""
Macro Regime Attribution Engine - Phase 1 Analysis
Regime detection + correlation decomposition
"""
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR
from lib.data_loader import load_merged

# ── Load data ───────────────────────────────────────────
df = load_merged()
print(f"Loaded {len(df)} rows × {len(df.columns)} cols")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

# Try TIP for inflation proxy
try:
    tip = yf.download("TIP", start="2020-01-01", progress=False, auto_adjust=True)["Close"]
    if isinstance(tip, pd.DataFrame):
        tip = tip.iloc[:, 0]
    tip.index = pd.to_datetime(tip.index)
    df["TIP"] = tip.reindex(df.index, method="ffill")
    print(f"  ✓ Added TIP (inflation proxy): {df['TIP'].notna().sum()} rows")
except:
    pass

# ── Core Analysis ───────────────────────────────────────

report = []
report.append("# Macro Regime Attribution Engine - Phase 1 Report")
report.append(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"# Data: {df.index.min().date()} → {df.index.max().date()}")
report.append("")

# ── 1. Latest Market Snapshot ───────────────────────────
report.append("## 1. Market Snapshot (Latest)")
report.append("")
latest = df.iloc[-1]
prev_5d = df.iloc[-6] if len(df) > 5 else df.iloc[0]
prev_20d = df.iloc[-21] if len(df) > 20 else df.iloc[0]

snap_cols = {
    "US10Y_yield": ("10Y Yield", "%", 3),
    "US2Y_yield": ("2Y Yield", "%", 3),
    "yield_curve_2s10s": ("2s10s Spread", "bp×100", 3),
    "WTI_crude": ("WTI Crude", "$/bbl", 2),
    "Gold": ("Gold", "$/oz", 0),
    "SPX": ("S&P 500", "", 0),
    "VIX": ("VIX", "", 2),
    "DXY": ("DXY", "", 2),
    "BTC": ("Bitcoin", "$", 0),
    "SOFR_rate": ("SOFR", "%", 3),
}

report.append("| Indicator | Latest | 5d Chg | 20d Chg |")
report.append("|---|---|---|---|")
for col, (name, unit, dec) in snap_cols.items():
    if col in df.columns and pd.notna(latest.get(col)):
        val = latest[col]
        chg5 = val - prev_5d.get(col, val) if pd.notna(prev_5d.get(col)) else 0
        chg20 = val - prev_20d.get(col, val) if pd.notna(prev_20d.get(col)) else 0
        fmt = f"{{:.{dec}f}}"
        report.append(f"| {name} | {fmt.format(val)} {unit} | {chg5:+.2f} | {chg20:+.2f} |")

report.append("")

# ── 2. Correlation Regime Analysis ──────────────────────
report.append("## 2. Correlation Regime Analysis")
report.append("")

# Oil vs 10Y yield correlation over different windows
windows = [10, 20, 60, 120]
oil_ret = df["WTI_crude"].pct_change()
yield_chg = df["US10Y_yield"].diff()

report.append("### Oil vs 10Y Yield Rolling Correlation")
report.append("Positive = inflation regime (oil up → yields up)")
report.append("Negative = growth/fiscal regime (oil up → yields down = recession pricing)")
report.append("")
report.append("| Window | Current | 1M Ago | 3M Ago | Interpretation |")
report.append("|---|---|---|---|---|")

for w in windows:
    corr_series = oil_ret.rolling(w).corr(yield_chg)
    current = corr_series.iloc[-1] if pd.notna(corr_series.iloc[-1]) else np.nan
    m1 = corr_series.iloc[-22] if len(corr_series) > 22 and pd.notna(corr_series.iloc[-22]) else np.nan
    m3 = corr_series.iloc[-66] if len(corr_series) > 66 and pd.notna(corr_series.iloc[-66]) else np.nan

    if pd.notna(current):
        if current > 0.3:
            interp = "⚠️ Inflation regime"
        elif current < -0.3:
            interp = "🔄 Growth/Fiscal regime"
        else:
            interp = "⚡ Transitional"
    else:
        interp = "N/A"

    c_str = f"{current:.3f}" if pd.notna(current) else "N/A"
    m1_str = f"{m1:.3f}" if pd.notna(m1) else "N/A"
    m3_str = f"{m3:.3f}" if pd.notna(m3) else "N/A"
    report.append(f"| {w}d | {c_str} | {m1_str} | {m3_str} | {interp} |")

report.append("")

# SPX vs 10Y correlation
if "SPX" in df.columns and df["SPX"].notna().sum() > 60:
    spx_ret = df["SPX"].pct_change()
    report.append("### SPX vs 10Y Yield Rolling Correlation")
    report.append("Positive = growth regime (stocks & yields move together)")
    report.append("Negative = risk-off / liquidity regime")
    report.append("")
    report.append("| Window | Current | 1M Ago | Interpretation |")
    report.append("|---|---|---|---|")
    for w in [20, 60]:
        corr_s = spx_ret.rolling(w).corr(yield_chg)
        cur = corr_s.iloc[-1] if pd.notna(corr_s.iloc[-1]) else np.nan
        m1 = corr_s.iloc[-22] if len(corr_s) > 22 and pd.notna(corr_s.iloc[-22]) else np.nan
        interp = "Growth" if pd.notna(cur) and cur > 0.2 else "Risk-off" if pd.notna(cur) and cur < -0.2 else "Mixed"
        c_s = f"{cur:.3f}" if pd.notna(cur) else "N/A"
        m_s = f"{m1:.3f}" if pd.notna(m1) else "N/A"
        report.append(f"| {w}d | {c_s} | {m_s} | {interp} |")
    report.append("")

# Gold vs BTC correlation
if "Gold" in df.columns and "BTC" in df.columns:
    gold_ret = df["Gold"].pct_change()
    btc_ret = df["BTC"].pct_change()
    report.append("### Gold vs BTC Rolling Correlation")
    report.append("High positive = monetary debasement trade (both as fiat alternatives)")
    report.append("Low/negative = BTC trades as risk asset, not gold alternative")
    report.append("")
    report.append("| Window | Current | 1M Ago | 3M Ago |")
    report.append("|---|---|---|---|")
    for w in [20, 60]:
        corr_s = gold_ret.rolling(w).corr(btc_ret)
        cur = corr_s.iloc[-1] if pd.notna(corr_s.iloc[-1]) else np.nan
        m1 = corr_s.iloc[-22] if len(corr_s) > 22 and pd.notna(corr_s.iloc[-22]) else np.nan
        m3 = corr_s.iloc[-66] if len(corr_s) > 66 and pd.notna(corr_s.iloc[-66]) else np.nan
        c_s = f"{cur:.3f}" if pd.notna(cur) else "N/A"
        m1_s = f"{m1:.3f}" if pd.notna(m1) else "N/A"
        m3_s = f"{m3:.3f}" if pd.notna(m3) else "N/A"
        report.append(f"| {w}d | {c_s} | {m1_s} | {m3_s} |")
    report.append("")

# ── 3. Term Premium Proxy ──────────────────────────────
report.append("## 3. Term Premium Analysis")
report.append("")

if "US10Y_yield" in df.columns and "US2Y_yield" in df.columns:
    # Use 2s10s as crude term premium proxy
    spread = df["yield_curve_2s10s"]
    report.append("### Yield Curve 2s10s Spread (proxy for term premium direction)")
    report.append(f"- Current: {spread.iloc[-1]:.3f}%")
    report.append(f"- 1M ago: {spread.iloc[-22]:.3f}%" if len(spread) > 22 else "")
    report.append(f"- 3M ago: {spread.iloc[-66]:.3f}%" if len(spread) > 66 else "")
    report.append(f"- 6M ago: {spread.iloc[-132]:.3f}%" if len(spread) > 132 else "")
    report.append("")

    # Slok's framework: 10Y = Fed path (~3.9%) + excess premium
    fed_path_estimate = 3.9  # Slok's estimate
    current_10y = latest.get("US10Y_yield", np.nan)
    if pd.notna(current_10y):
        excess = current_10y - fed_path_estimate
        report.append(f"### Slok Decomposition (Apollo)")
        report.append(f"- Current 10Y: {current_10y:.3f}%")
        report.append(f"- Est. Fed-path fair value: {fed_path_estimate:.1f}%")
        report.append(f"- **Excess premium: {excess*100:.0f}bp**")
        report.append(f"- (Article cited ~55bp; current: {excess*100:.0f}bp)")
        report.append("")
        report.append("Excess premium sources (qualitative):")
        report.append("1. Fiscal supply concerns (Treasury-SOFR widening)")
        report.append("2. Gulf state selling ($58B outflow from Fed custody)")
        report.append("3. QT runoff effect")
        report.append("4. Fed independence concerns")
        report.append("")

# ── 4. Regime Classification ───────────────────────────
report.append("## 4. Regime Classification")
report.append("")

# Compute regime scores
scores = {}

# Factor 1: Oil-Bond correlation (negative = growth/fiscal regime)
if "corr_oil_10Y_20d" in df.columns:
    oil_bond_corr = df["corr_oil_10Y_20d"].iloc[-1]
    if pd.notna(oil_bond_corr):
        if oil_bond_corr < -0.2:
            scores["oil_bond"] = ("Growth/Fiscal", 1.0)
        elif oil_bond_corr > 0.3:
            scores["oil_bond"] = ("Inflation", -1.0)
        else:
            scores["oil_bond"] = ("Transitional", 0.0)

# Factor 2: VIX level (>25 = stress)
vix = latest.get("VIX", np.nan)
if pd.notna(vix):
    if vix > 30:
        scores["vix"] = ("High stress", 1.0)
    elif vix > 25:
        scores["vix"] = ("Elevated", 0.5)
    else:
        scores["vix"] = ("Calm", -0.5)

# Factor 3: Yield curve direction (steepening = fiscal/growth fears)
if "yield_curve_2s10s" in df.columns:
    curve_now = spread.iloc[-1]
    curve_1m = spread.iloc[-22] if len(spread) > 22 else curve_now
    curve_chg = curve_now - curve_1m
    if curve_chg > 0.1:
        scores["curve"] = ("Bear steepening → fiscal supply", 1.0)
    elif curve_chg < -0.1:
        scores["curve"] = ("Flattening → recession", 0.5)
    else:
        scores["curve"] = ("Stable", 0.0)

# Factor 4: Gold momentum
if "Gold" in df.columns:
    gold_ret_20d = (latest["Gold"] / df["Gold"].iloc[-22] - 1) * 100 if len(df) > 22 else 0
    if gold_ret_20d > 5:
        scores["gold"] = (f"Strong rally +{gold_ret_20d:.1f}% → debasement trade", 1.0)
    elif gold_ret_20d > 2:
        scores["gold"] = (f"Rally +{gold_ret_20d:.1f}%", 0.5)
    else:
        scores["gold"] = (f"Flat {gold_ret_20d:+.1f}%", 0.0)

# Factor 5: DXY direction
if "DXY" in df.columns:
    dxy_now = latest.get("DXY", np.nan)
    dxy_1m = df["DXY"].iloc[-22] if len(df) > 22 else dxy_now
    if pd.notna(dxy_now) and pd.notna(dxy_1m):
        dxy_chg = dxy_now - dxy_1m
        if dxy_chg < -2:
            scores["dxy"] = (f"USD weakening ({dxy_chg:+.1f}) → fiscal credibility erosion", 1.0)
        elif dxy_chg > 2:
            scores["dxy"] = (f"USD strengthening ({dxy_chg:+.1f}) → safe haven", -0.5)
        else:
            scores["dxy"] = (f"Stable ({dxy_chg:+.1f})", 0.0)

report.append("| Factor | Reading | Regime Signal |")
report.append("|---|---|---|")
for k, (desc, score) in scores.items():
    signal = "🔴 Fiscal/Growth" if score > 0.5 else "🟡 Mixed" if score >= 0 else "🟢 Inflation/Normal"
    report.append(f"| {k} | {desc} | {signal} |")

# Overall regime
avg_score = np.mean([v[1] for v in scores.values()]) if scores else 0
report.append("")
if avg_score > 0.5:
    regime = "D) 混合 → 偏向财政扩张/增长恐慌"
elif avg_score > 0:
    regime = "B/D) 增长恐慌与财政预期并存"
elif avg_score > -0.5:
    regime = "A) 通胀恐慌仍主导"
else:
    regime = "A) 纯通胀恐慌"

report.append(f"**综合 Regime 判断: {regime}**")
report.append(f"**Regime Score: {avg_score:.2f}** (>0.5=财政/增长, <0=通胀)")
report.append("")

# ── 5. Key Charts Data (for later visualization) ──────
report.append("## 5. Historical Context")
report.append("")

# Oil-Bond correlation regime history
if "corr_oil_10Y_20d" in df.columns:
    corr_hist = df["corr_oil_10Y_20d"].dropna()
    report.append("### Oil-Bond Correlation (20d) Key Periods:")
    # Find regime transitions
    for period_name, start, end in [
        ("COVID crash", "2020-02-01", "2020-04-30"),
        ("2021 reflation", "2021-01-01", "2021-06-30"),
        ("2022 rate hikes", "2022-01-01", "2022-12-31"),
        ("2023 normalization", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025 H2", "2025-07-01", "2025-12-31"),
        ("2026 YTD", "2026-01-01", "2026-03-31"),
        ("Last 20 trading days", df.index[-21].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")),
    ]:
        mask = (corr_hist.index >= start) & (corr_hist.index <= end)
        subset = corr_hist[mask]
        if len(subset) > 0:
            report.append(f"- **{period_name}**: mean={subset.mean():.3f}, range=[{subset.min():.3f}, {subset.max():.3f}]")

report.append("")

# ── 6. Article Validation ──────────────────────────────
report.append("## 6. Article Data Point Validation")
report.append("")
report.append("| Claim | Our Data | Match? |")
report.append("|---|---|---|")

# WTI > $100
wti = latest.get("WTI_crude", np.nan)
report.append(f"| WTI > $100 | ${wti:.2f} | {'✓' if pd.notna(wti) and wti > 100 else '✗'} |")

# 10Y at 4.4%
y10 = latest.get("US10Y_yield", np.nan)
report.append(f"| 10Y ~4.4% | {y10:.3f}% | {'✓' if pd.notna(y10) and abs(y10 - 4.4) < 0.15 else '~'} |")

# VIX elevated
report.append(f"| Elevated volatility | VIX={vix:.1f} | {'✓' if pd.notna(vix) and vix > 25 else '✗'} |")

# Gold surging
if "Gold" in df.columns and len(df) > 22:
    gold_1m_ret = (latest["Gold"] / df["Gold"].iloc[-22] - 1) * 100
    report.append(f"| Gold surging | +{gold_1m_ret:.1f}% (20d) | {'✓' if gold_1m_ret > 3 else '~'} |")

# BTC surging
if "BTC" in df.columns and len(df) > 22:
    btc_1m_ret = (latest["BTC"] / df["BTC"].iloc[-22] - 1) * 100
    report.append(f"| Crypto surging | BTC +{btc_1m_ret:.1f}% (20d) | {'✓' if btc_1m_ret > 3 else '~' if btc_1m_ret > 0 else '✗'} |")

# Slok 55bp premium
if pd.notna(current_10y):
    premium_bp = (current_10y - 3.9) * 100
    report.append(f"| 55bp excess premium | {premium_bp:.0f}bp | {'✓' if abs(premium_bp - 55) < 15 else '~'} |")

report.append("")

# ── Save ────────────────────────────────────────────────
output = "\n".join(report)
out_path = os.path.join(ANALYSIS_DIR, "regime_report_20260331.md")
with open(out_path, "w") as f:
    f.write(output)

print("\n" + "=" * 60)
print("REPORT OUTPUT")
print("=" * 60)
print(output)
print(f"\n✓ Saved to {out_path}")
