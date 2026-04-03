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

# ── Step 1b: Run TimesFM views (optional, non-blocking) ──
print("\n" + "=" * 60)
print("Running TimesFM predictions...")
tfm_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "timesfm_views.py")],
    capture_output=True, text=True,
    timeout=660,
)
if tfm_result.returncode == 0:
    print(tfm_result.stdout)
else:
    print(f"⚠ TimesFM failed (non-critical):\n{tfm_result.stderr[-500:]}")

# ── Step 2: Quick regime snapshot ───────────────────────
df = load_merged()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# Key metrics — dual window (20d = early warning, 60d = confirmation)
oil_ret = df["WTI_crude"].pct_change()
yield_chg = df["US10Y_yield"].diff()
corr_ob_20 = oil_ret.rolling(20).corr(yield_chg).iloc[-1]
corr_ob_60 = oil_ret.rolling(60).corr(yield_chg).iloc[-1]

spx_ret = df["SPX"].pct_change() if "SPX" in df.columns else pd.Series()
corr_sb_20 = spx_ret.rolling(20).corr(yield_chg).iloc[-1] if len(spx_ret) > 20 else np.nan
corr_sb_60 = spx_ret.rolling(60).corr(yield_chg).iloc[-1] if len(spx_ret) > 60 else np.nan

gold_ret = df["Gold"].pct_change()
btc_ret = df["BTC"].pct_change()
corr_gb_20 = gold_ret.rolling(20).corr(btc_ret).iloc[-1]

# Dual regime scores: short-term (20d) and long-term (60d) independently
score_short = 0
score_long = 0
signals_short = []
signals_long = []

# Oil-Bond: positive = inflation, negative = growth/fiscal
if pd.notna(corr_ob_20):
    if corr_ob_20 < -0.2:
        score_short += 2; signals_short.append(f"Oil-Bond脱钩 ({corr_ob_20:.2f})")
    elif corr_ob_20 < 0.1:
        score_short += 1; signals_short.append(f"Oil-Bond弱化 ({corr_ob_20:.2f})")
if pd.notna(corr_ob_60):
    if corr_ob_60 < -0.2:
        score_long += 2; signals_long.append(f"Oil-Bond脱钩 ({corr_ob_60:.2f})")
    elif corr_ob_60 < 0.1:
        score_long += 1; signals_long.append(f"Oil-Bond弱化 ({corr_ob_60:.2f})")

# SPX-Bond: negative = risk-off regime
if pd.notna(corr_sb_20):
    if corr_sb_20 < -0.3:
        score_short += 2; signals_short.append(f"股债避险 ({corr_sb_20:.2f})")
    elif corr_sb_20 < 0:
        score_short += 1; signals_short.append(f"股债弱避险 ({corr_sb_20:.2f})")
if pd.notna(corr_sb_60):
    if corr_sb_60 < -0.3:
        score_long += 2; signals_long.append(f"股债避险 ({corr_sb_60:.2f})")
    elif corr_sb_60 < 0:
        score_long += 1; signals_long.append(f"股债弱避险 ({corr_sb_60:.2f})")

# VIX (shared, not window-dependent)
vix = latest.get("VIX", 20)
if pd.notna(vix) and vix > 30:
    score_short += 1; signals_short.append(f"VIX高压 ({vix:.0f})")
    score_long += 1; signals_long.append(f"VIX高压 ({vix:.0f})")

# Gold-BTC (shared)
if pd.notna(corr_gb_20):
    if corr_gb_20 > 0.4:
        score_short += 1; signals_short.append(f"Gold-BTC联动 ({corr_gb_20:.2f})")
        score_long += 1; signals_long.append(f"Gold-BTC联动 ({corr_gb_20:.2f})")

def _regime_label(s):
    if s >= 4: return "🔴 增长恐慌/财政定价"
    elif s >= 2: return "🟡 过渡/混合"
    else: return "🟢 通胀主导/正常"

regime_short = _regime_label(score_short)
regime_long = _regime_label(score_long)
# Combined score for backward compatibility (JSON output etc.)
score = score_short
regime = regime_short

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

**短期 (20d): {regime_short}** (score: {score_short}/6)
信号: {', '.join(signals_short) if signals_short else 'None'}

**长期 (60d): {regime_long}** (score: {score_long}/6)
信号: {', '.join(signals_long) if signals_long else 'None'}

**日变动:** {' | '.join(changes)}

**关键相关性 (20d → 60d):**
• Oil↔Bond: {corr_ob_20:.2f} → {corr_ob_60:.2f} {'⚡脱钩' if corr_ob_20 < 0 else '通胀传导'}
• SPX↔Bond: {f"{corr_sb_20:.2f}" if pd.notna(corr_sb_20) else "N/A"} → {f"{corr_sb_60:.2f}" if pd.notna(corr_sb_60) else "N/A"} {"⚡避险" if pd.notna(corr_sb_20) and corr_sb_20 < -0.2 else "增长"}
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
    "regime_short": regime_short,
    "score_short": score_short,
    "signals_short": signals_short,
    "regime_long": regime_long,
    "score_long": score_long,
    "signals_long": signals_long,
    "regime": regime,
    "score": score,
    "summary": summary,
}
with open(os.path.join(DATA_DIR, "dashboard_latest.json"), "w") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"\n✓ Dashboard saved.")
