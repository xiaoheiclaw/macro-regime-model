#!/usr/bin/env python3
"""
Daily Dashboard Runner - runs all modules and outputs summary
Designed to be called by cron daily
"""
import os, sys, json, subprocess
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR, SCRIPTS_DIR
from lib.data_loader import load_merged

today = pd.Timestamp.now().strftime("%Y%m%d")

# ── Step 1: Run data pipeline ──────────────────────────
print("=" * 60)
print(f"DAILY REGIME DASHBOARD - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "data_pipeline.py")],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(f"Pipeline failed:\n{result.stderr}")
    sys.exit(1)

# ── Step 2: Quick regime snapshot ───────────────────────
df = load_merged()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# Key metrics
oil_ret = df["WTI_crude"].pct_change()
yield_chg = df["US10Y_yield"].diff()
corr_ob_20 = oil_ret.rolling(20).corr(yield_chg).iloc[-1]

spx_ret = df["SPX"].pct_change() if "SPX" in df.columns else pd.Series()
corr_sb_20 = spx_ret.rolling(20).corr(yield_chg).iloc[-1] if len(spx_ret) > 20 else np.nan

gold_ret = df["Gold"].pct_change()
btc_ret = df["BTC"].pct_change()
corr_gb_20 = gold_ret.rolling(20).corr(btc_ret).iloc[-1]

# Regime score (simple composite)
score = 0
signals = []

# Oil-Bond: positive = inflation, negative = growth/fiscal
if pd.notna(corr_ob_20):
    if corr_ob_20 < -0.2:
        score += 2; signals.append(f"Oil-Bond脱钩 ({corr_ob_20:.2f})")
    elif corr_ob_20 < 0.1:
        score += 1; signals.append(f"Oil-Bond弱化 ({corr_ob_20:.2f})")

# SPX-Bond: negative = risk-off regime
if pd.notna(corr_sb_20):
    if corr_sb_20 < -0.3:
        score += 2; signals.append(f"股债避险 ({corr_sb_20:.2f})")
    elif corr_sb_20 < 0:
        score += 1; signals.append(f"股债弱避险 ({corr_sb_20:.2f})")

# VIX
vix = latest.get("VIX", 20)
if pd.notna(vix):
    if vix > 30: score += 1; signals.append(f"VIX高压 ({vix:.0f})")

# Gold-BTC correlation (high = debasement trade)
if pd.notna(corr_gb_20):
    if corr_gb_20 > 0.4:
        score += 1; signals.append(f"Gold-BTC联动 ({corr_gb_20:.2f})")

# Regime label
if score >= 4: regime = "🔴 增长恐慌/财政定价"
elif score >= 2: regime = "🟡 过渡/混合"
else: regime = "🟢 通胀主导/正常"

# Daily change highlights
changes = []
for col, name, fmt in [
    ("US10Y_yield", "10Y", "{:+.1f}bp"),
    ("WTI_crude", "Oil", "{:+.1f}%"),
    ("Gold", "Gold", "{:+.1f}%"),
    ("SPX", "SPX", "{:+.1f}%"),
    ("BTC", "BTC", "{:+.1f}%"),
    ("DXY", "DXY", "{:+.2f}"),
    ("VIX", "VIX", "{:+.1f}"),
]:
    if col in df.columns:
        curr = latest.get(col, np.nan)
        prev_val = prev.get(col, np.nan)
        if pd.notna(curr) and pd.notna(prev_val):
            if col == "US10Y_yield":
                chg = (curr - prev_val) * 100  # in bp
            elif col in ["WTI_crude", "Gold", "SPX", "BTC"]:
                chg = (curr / prev_val - 1) * 100 if prev_val != 0 else 0
            else:
                chg = curr - prev_val
            changes.append(f"{name} {fmt.format(chg)}")

# Build summary
summary = f"""📊 **Regime Dashboard {pd.Timestamp.now().strftime('%Y-%m-%d')}**

**Regime: {regime}** (score: {score}/6)
信号: {', '.join(signals) if signals else 'None'}

**日变动:** {' | '.join(changes)}

**关键相关性 (20d):**
• Oil↔Bond: {corr_ob_20:.2f} {'⚡脱钩' if corr_ob_20 < 0 else '通胀传导'}
• SPX↔Bond: {f"{corr_sb_20:.2f}" if pd.notna(corr_sb_20) else "N/A"} {"⚡避险" if pd.notna(corr_sb_20) and corr_sb_20 < -0.2 else "增长"}
• Gold↔BTC: {corr_gb_20:.2f} {'💰贬值交易' if corr_gb_20 > 0.3 else '分化'}

**水位:**
• 10Y: {latest.get('US10Y_yield', 0):.2f}% | Oil: ${latest.get('WTI_crude', 0):.0f}
• Gold: ${latest.get('Gold', 0):.0f} | VIX: {latest.get('VIX', 0):.1f}
• BTC: ${latest.get('BTC', 0):,.0f} | DXY: {latest.get('DXY', 0):.1f}"""

print("\n" + summary)

# Save
with open(os.path.join(ANALYSIS_DIR, f"dashboard_{today}.md"), "w") as f:
    f.write(summary)

# Also save as latest
with open(os.path.join(ANALYSIS_DIR, "dashboard_latest.md"), "w") as f:
    f.write(summary)

# Output JSON for cron notification
output_json = {
    "regime": regime,
    "score": score,
    "signals": signals,
    "summary": summary,
}
with open(os.path.join(DATA_DIR, "dashboard_latest.json"), "w") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"\n✓ Dashboard saved.")
