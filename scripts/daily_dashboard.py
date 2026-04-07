#!/usr/bin/env python3
"""
Daily Dashboard Runner - Full Phase 1–4 Pipeline
Designed to be called by cron daily
"""
import os, sys, json, subprocess, re
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# yfinance and FRED need direct connections — strip proxy
for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(_proxy_key, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR, SCRIPTS_DIR
from lib.data_loader import load_merged
from lib.tuning import load_tuning_params

today = pd.Timestamp.now().strftime("%Y-%m-%d")
today_str = today

# ── Step 0: Data pipeline ──────────────────────────────
print("=" * 60)
print(f"DAILY REGIME DASHBOARD - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "data_pipeline.py")],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"Pipeline failed:\n{result.stderr}")
    sys.exit(1)

# ── Step 1: Load data ──────────────────────────────────
tp = load_tuning_params()
ct = tp["correlation_thresholds"]
df = load_merged()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ── Phase 1: Correlation snapshots ─────────────────────
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

# ── Phase 2: Markov Regime Switching (non-blocking) ────
print("\n" + "=" * 60)
print("Phase 2: Markov Regime Switching...")
markov_summary = {}
ms_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "regime_switching.py")],
    capture_output=True, text=True, timeout=300,
)
if ms_result.returncode == 0:
    print(ms_result.stdout[:500])
    # Save full report as separate file
    report_path = os.path.join(ANALYSIS_DIR, f"regime_switching_{today}.md")
    with open(report_path, "w") as f:
        f.write(ms_result.stdout)
    
    # Extract key signals from output
    output = ms_result.stdout
    # Parse "Current: Regime X (P=YY%)" for Model 2 (SPX→Bond)
    stock_bond_match = re.search(r'Model 2.*?Current: Regime (\d) \(P=([\d.]+)%\)', output, re.DOTALL)
    yield_level_match = re.search(r'Model 3.*?Current: Regime (\d) \(P=([\d.]+)%\)', output, re.DOTALL)
    
    if stock_bond_match:
        markov_summary["stock_bond_regime"] = int(stock_bond_match.group(1))
        markov_summary["stock_bond_prob"] = float(stock_bond_match.group(2))
    if yield_level_match:
        markov_summary["yield_regime"] = int(yield_level_match.group(1))
        markov_summary["yield_prob"] = float(yield_level_match.group(2))
else:
    print(f"⚠ Markov failed: {ms_result.stderr[-300:]}")

# ── Phase 3: Kalman Betas (non-blocking) ───────────────
print("\n" + "=" * 60)
print("Phase 3: Kalman Filter Time-Varying Betas...")
kalman_summary = {}
kf_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "kalman_betas.py")],
    capture_output=True, text=True, timeout=300,
)
if kf_result.returncode == 0:
    print(kf_result.stdout[:500])
    report_path = os.path.join(ANALYSIS_DIR, f"kalman_betas_{today}.md")
    with open(report_path, "w") as f:
        f.write(kf_result.stdout)
    
    # Extract latest betas
    output = kf_result.stdout
    oil_beta_match = re.search(r'最新 β \(oil→yield\):\s*([-\d.]+)', output)
    spxbeta_match = re.search(r'最新 β \(spx→yield\):\s*([-\d.]+)', output)
    gold_beta_match = re.search(r'最新 β \(gold→yield\):\s*([-\d.]+)', output)
    
    if oil_beta_match:
        kalman_summary["oil_beta"] = float(oil_beta_match.group(1))
    if spxbeta_match:
        kalman_summary["spx_beta"] = float(spxbeta_match.group(1))
    if gold_beta_match:
        kalman_summary["gold_beta"] = float(gold_beta_match.group(1))
else:
    print(f"⚠ Kalman failed: {kf_result.stderr[-300:]}")

# ── Step 2b: TimesFM predictions (non-blocking) ────────
print("\n" + "=" * 60)
print("Running TimesFM predictions...")
tfm_summary = {}
tfm_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "timesfm_views.py")],
    capture_output=True, text=True, timeout=660,
)
if tfm_result.returncode == 0:
    print(tfm_result.stdout[:500])
    # Load saved timesfm views
    tfm_path = os.path.join(DATA_DIR, "timesfm_views.json")
    if os.path.exists(tfm_path):
        with open(tfm_path) as f:
            tfm_summary = json.load(f)
else:
    print(f"⚠ TimesFM failed (non-critical):\n{tfm_result.stderr[-500:]}")

# ── Step 2c: Prediction scorecard (non-blocking) ───────
print("\n" + "=" * 60)
print("Running prediction scorecard...")
sc_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "prediction_scorecard.py")],
    capture_output=True, text=True, timeout=60,
)
if sc_result.returncode != 0:
    print(f"⚠ Scorecard: {sc_result.stderr[-300:]}")

# ── Phase 4a: Black-Litterman (non-blocking) ──────────
print("\n" + "=" * 60)
print("Phase 4a: Black-Litterman allocation...")
bl_summary = {}
bl_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "black_litterman.py")],
    capture_output=True, text=True, timeout=300,
)
if bl_result.returncode == 0:
    print(bl_result.stdout[:500])
    report_path = os.path.join(ANALYSIS_DIR, f"black_litterman_{today}.md")
    with open(report_path, "w") as f:
        f.write(bl_result.stdout)
    bl_output = bl_result.stdout
    bl_summary["sharpe_weights"] = {}
    bl_summary["cvar_weights"] = {}
    mode = None
    for line in bl_output.splitlines():
        if "**BL Max Sharpe 按权重排序：**" in line:
            mode = "sharpe"
            continue
        if "**BL Min-CVaR 按权重排序：**" in line:
            mode = "cvar"
            continue
        m = re.search(r"\*\*(.+?):\s*(\d+)%\*\*", line)
        if m and mode:
            asset = m.group(1).strip()
            weight = int(m.group(2))
            if mode == "sharpe":
                bl_summary["sharpe_weights"][asset] = weight
            else:
                bl_summary["cvar_weights"][asset] = weight
else:
    print(f"⚠ BL failed: {bl_result.stderr[-300:]}")

# ── Phase 4b: Stochastic Programming (non-blocking) ───
print("\n" + "=" * 60)
print("Phase 4b: Stochastic Programming...")
sp_summary = {}
sp_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "stochastic_programming.py")],
    capture_output=True, text=True, timeout=300,
)
if sp_result.returncode == 0:
    print(sp_result.stdout[:500])
    report_path = os.path.join(ANALYSIS_DIR, f"stochastic_programming_{today}.md")
    with open(report_path, "w") as f:
        f.write(sp_result.stdout)
    sp_output = sp_result.stdout
    current_regime_match = re.search(r"Current regime: R(\d)", sp_output)
    if current_regime_match:
        sp_summary["current_regime"] = int(current_regime_match.group(1))
else:
    print(f"⚠ Stochastic Programming failed: {sp_result.stderr[-300:]}")

# ── Build Regime Scores ────────────────────────────────
score_short = 0
score_long = 0
signals_short = []
signals_long = []

ob_dec = ct["oil_bond_decouple"]
ob_weak = ct["oil_bond_weak"]
if pd.notna(corr_ob_20):
    if corr_ob_20 < ob_dec:
        score_short += 2; signals_short.append(f"Oil-Bond脱钩 ({corr_ob_20:.2f})")
    elif corr_ob_20 < ob_weak:
        score_short += 1; signals_short.append(f"Oil-Bond弱化 ({corr_ob_20:.2f})")
if pd.notna(corr_ob_60):
    if corr_ob_60 < ob_dec:
        score_long += 2; signals_long.append(f"Oil-Bond脱钩 ({corr_ob_60:.2f})")
    elif corr_ob_60 < ob_weak:
        score_long += 1; signals_long.append(f"Oil-Bond弱化 ({corr_ob_60:.2f})")

sb_ro = ct["spx_bond_riskoff"]
if pd.notna(corr_sb_20):
    if corr_sb_20 < sb_ro:
        score_short += 2; signals_short.append(f"股债避险 ({corr_sb_20:.2f})")
    elif corr_sb_20 < 0:
        score_short += 1; signals_short.append(f"股债弱避险 ({corr_sb_20:.2f})")
if pd.notna(corr_sb_60):
    if corr_sb_60 < sb_ro:
        score_long += 2; signals_long.append(f"股债避险 ({corr_sb_60:.2f})")
    elif corr_sb_60 < 0:
        score_long += 1; signals_long.append(f"股债弱避险 ({corr_sb_60:.2f})")

vix = latest.get("VIX", 20)
if pd.notna(vix) and vix > 30:
    score_short += 1; signals_short.append(f"VIX高压 ({vix:.0f})")
    score_long += 1; signals_long.append(f"VIX高压 ({vix:.0f})")

gb_link = ct["gold_btc_linked"]
if pd.notna(corr_gb_20):
    if corr_gb_20 > gb_link:
        score_short += 1; signals_short.append(f"Gold-BTC联动 ({corr_gb_20:.2f})")
        score_long += 1; signals_long.append(f"Gold-BTC联动 ({corr_gb_20:.2f})")

def _regime_label(s):
    if s >= 4: return "🔴 增长恐慌/财政定价"
    elif s >= 2: return "🟡 过渡/混合"
    else: return "🟢 通胀主导/正常"

regime_short = _regime_label(score_short)
regime_long = _regime_label(score_long)

# ── Load SP weights from CSV ──────────────────────────
sp_weights_path = os.path.join(DATA_DIR, "stochastic_prog_weights.csv")
sp_weights = {}
if os.path.exists(sp_weights_path):
    sp_df = pd.read_csv(sp_weights_path, index_col="strategy")
    asset_cols = ["SPX", "US10Y_yield", "Gold", "WTI_crude", "BTC", "DXY"]
    perf_cols = ["ann_return", "ann_vol", "sharpe", "var_95", "cvar_95"]
    for strat in sp_df.index:
        sp_weights[strat] = {
            "weights": {c: sp_df.loc[strat, c] for c in asset_cols if c in sp_df.columns},
            "perf": {c: sp_df.loc[strat, c] for c in perf_cols if c in sp_df.columns},
        }

# ── Compute ensemble stress probability ──────────────
stress_prob = np.nan
if markov_summary:
    sb_prob = markov_summary.get("stock_bond_prob", 0)
    yl_prob = markov_summary.get("yield_prob", 0)
    stress_prob = 0.6 * sb_prob/100 + 0.4 * yl_prob/100

# ── Build Compact Report ──────────────────────────────
lines = []

# Header
lines.append(f"# 📊 Macro Regime — {today_str}")
lines.append("")

# ── 水位 ──
lines.append("## 水位")
lines.append("")
lines.append("| 指标 | 值 | 日变动 |")
lines.append("|------|-----|--------|")

def _fmt_chg(col, curr, prev_val):
    if col == "US10Y_yield":
        return f"{(curr - prev_val) * 100:+.1f}bp"
    elif col in ["WTI_crude", "Gold", "SPX", "BTC"]:
        return f"{(curr / prev_val - 1) * 100:+.1f}%" if prev_val != 0 else "—"
    elif col == "VIX":
        return f"{curr - prev_val:+.1f}"
    else:
        return f"{curr - prev_val:+.2f}"

water_items = [
    ("US10Y_yield", "10Y", lambda v: f"{v:.2f}%"),
    ("WTI_crude", "WTI", lambda v: f"${v:.0f}"),
    ("Gold", "Gold", lambda v: f"${v:,.0f}"),
    ("SPX", "SPX", lambda v: f"{v:,.0f}"),
    ("BTC", "BTC", lambda v: f"${v:,.0f}"),
    ("VIX", "VIX", lambda v: f"{v:.1f}"),
    ("DXY", "DXY", lambda v: f"{v:.1f}"),
]
for col, name, fmt_fn in water_items:
    if col in df.columns:
        curr = latest.get(col, np.nan)
        prev_val = prev.get(col, np.nan)
        if pd.notna(curr):
            val_str = fmt_fn(curr)
            chg_str = _fmt_chg(col, curr, prev_val) if pd.notna(prev_val) else "—"
            lines.append(f"| {name} | {val_str} | {chg_str} |")

# BEI if available
bei = latest.get("BEI_5Y", np.nan)
if pd.notna(bei):
    lines.append(f"| BEI 5Y | {bei:.2f}% | |")
# 2s10s
ycs = latest.get("yield_curve_2s10s", np.nan)
if pd.notna(ycs):
    lines.append(f"| 2s10s | {ycs:+.2f}% | {'正斜率' if ycs >= 0 else '⚠️ 倒挂'} |")
lines.append("")

# ── 原理 ──
lines.append("## 原理")
lines.append("")
lines.append('核心问题：**当前市场在按什么\u201c规则\u201d运行？** 同样油价上涨，通胀模式下收益率跟涨（做多风险），恐慌模式下收益率跟跌（避险）。配置应完全相反。')
lines.append("")
lines.append("四层递进检测：")
lines.append("1. **相关性快照** — 三对资产（油↔债、股↔债、金↔BTC）的 20d/60d 滚动相关，最直觉的体制信号")
lines.append('2. **Markov Switching** \u2014 隐状态模型，从\u201c油/股变了之后债怎么变\u201d反推当前体制概率；Wasserstein K-Means 补充分布形状维度，6:4 加权融合')
lines.append("3. **Kalman Filter** — 输出连续 β（遗忘因子 λ=0.97），追踪因子敞口的逐日变化，比 Markov 的开/关更精细")
lines.append("4. **配置优化** — Black-Litterman 融合市场均衡+TimesFM 观点（AI 自报置信度控制信任权重）；随机规划模拟 1000 条体制路径，找多路径鲁棒配置")
lines.append("")

# ── 体制信号 ──
lines.append("## 体制信号")
lines.append("")
lines.append("| 层 | 判断 | 关键数字 |")
lines.append("|----|------|----------|")

# Layer 1: Correlation
sb20_str = f"{corr_sb_20:.2f}" if pd.notna(corr_sb_20) else "N/A"
lines.append(f"| 相关性 | {regime_short} | 股债 {sb20_str} / 油债 {corr_ob_20:+.2f} / 金BTC {corr_gb_20:+.2f} |")

# Layer 2: Markov + Wasserstein
if markov_summary:
    sb_reg = markov_summary.get("stock_bond_regime", "?")
    sb_prob = markov_summary.get("stock_bond_prob", 0)
    yl_reg = markov_summary.get("yield_regime", "?")
    yl_prob = markov_summary.get("yield_prob", 0)
    # Load ensemble from CSV for oil model
    ens_path = os.path.join(DATA_DIR, "ensemble_regime.csv")
    oil_calm_str = ""
    if os.path.exists(ens_path):
        ens_df = pd.read_csv(ens_path)
        if len(ens_df) > 0:
            last_ens = ens_df.iloc[-1]
            oil_calm_str = f", 油模型 Calm({last_ens.get('markov_P_R0', 0)*100:.0f}%)"
    stress_label = "🔴 Stress" if stress_prob > 0.6 else ("🟡 过渡" if stress_prob > 0.4 else "🟢 Calm")
    lines.append(f"| Markov+Wass | {stress_label} {stress_prob:.0%} | 股模型 Stress({sb_prob:.0f}%), 收益率 Stress({yl_prob:.0f}%){oil_calm_str} |")
else:
    lines.append("| Markov+Wass | ⚠ 未运行 | — |")

# Layer 3: Kalman
if kalman_summary:
    oil_b = kalman_summary.get("oil_beta", 0)
    spx_b = kalman_summary.get("spx_beta", 0)
    gold_b = kalman_summary.get("gold_beta", 0)
    if spx_b < -0.5:
        k_label = "🔴 强避险"
    elif spx_b < 0:
        k_label = "🟡 弱避险"
    else:
        k_label = "🟢 增长主导"
    lines.append(f"| Kalman β | {k_label} | SPX→Yield **{spx_b:+.3f}** / Oil→Yield {oil_b:+.2f} / Gold→Yield {gold_b:+.2f} |")
else:
    lines.append("| Kalman β | ⚠ 未运行 | — |")
lines.append("")

# Synthesis
if pd.notna(stress_prob):
    if stress_prob > 0.6:
        synthesis = "**综合：Stress 体制确认。**"
    elif stress_prob > 0.4:
        synthesis = "**综合：过渡至 Stress。切换进行中但未完成。**"
    else:
        synthesis = "**综合：Calm/通胀体制主导。**"
    lines.append(synthesis)
    lines.append("")

# ── 配置建议 ──
lines.append("## 配置建议")
lines.append("")

# Build allocation table from available data
asset_display = [
    ("SPX", "SPX"),
    ("US10Y_yield", "10Y"),
    ("Gold", "Gold"),
    ("WTI_crude", "Oil"),
    ("BTC", "BTC"),
    ("DXY", "USD"),
]
has_bl = bool(bl_summary.get("sharpe_weights"))
has_sp = "SP-CVaR" in sp_weights

if has_bl or has_sp:
    # BL output uses display names like "S&P 500"; map to our column names
    _bl_name_map = {
        "S&P 500": "SPX", "10Y Treasury": "US10Y_yield", "Gold": "Gold",
        "Oil": "WTI_crude", "Bitcoin": "BTC", "USD": "DXY",
    }
    bl_sharpe_raw = bl_summary.get("sharpe_weights", {})
    bl_sharpe = {}
    for k, v in bl_sharpe_raw.items():
        mapped = _bl_name_map.get(k, k)
        bl_sharpe[mapped] = v

    header = "| 资产 | 市场 |"
    sep = "|------|------|"
    if has_bl:
        header += " BL-Sharpe |"
        sep += "-----------|"
    if has_sp:
        header += " SP-CVaR |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    market_weights = {"SPX": 40, "US10Y_yield": 25, "Gold": 10, "WTI_crude": 10, "BTC": 5, "DXY": 10}
    for col, display in asset_display:
        row = f"| {display} | {market_weights.get(col, 0)}% |"
        if has_bl:
            bl_w = bl_sharpe.get(col, 0)
            bold = "**" if bl_w >= 30 else ""
            row += f" {bold}{bl_w}%{bold} |"
        if has_sp:
            sp_w = sp_weights["SP-CVaR"]["weights"].get(col, 0)
            w_pct = int(round(sp_w * 100))
            bold = "**" if w_pct >= 30 else ""
            row += f" {bold}{w_pct}%{bold} |"
        lines.append(row)
    lines.append("")

    # Performance summary
    if has_bl:
        bl_perf = sp_weights.get("BL-Sharpe", {}).get("perf", {})
        if bl_perf:
            lines.append(f"- BL-Sharpe: Return {bl_perf.get('ann_return', 0)*100:+.1f}%, Vol {bl_perf.get('ann_vol', 0)*100:.1f}%, Sharpe {bl_perf.get('sharpe', 0):.2f}")
    if has_sp:
        sp_perf = sp_weights["SP-CVaR"]["perf"]
        lines.append(f"- SP-CVaR: Return {sp_perf.get('ann_return', 0)*100:+.1f}%, Vol {sp_perf.get('ann_vol', 0)*100:.1f}%, CVaR {sp_perf.get('cvar_95', 0)*100:.1f}%")
    lines.append("")

# Direction
if pd.notna(stress_prob):
    if stress_prob > 0.5:
        lines.append("**方向：防御优先。加 USD/债券/黄金，减 SPX/Oil，BTC 2-5%。**")
    else:
        lines.append("**方向：进攻优先。SPX/原油为主，黄金做通胀对冲，债券可适度降低。**")
    lines.append("")

# ── 关注催化 ──
lines.append("## 关注催化")
lines.append("")
lines.append("- 地缘政治进展")
lines.append("- Fed 政策信号")
lines.append("- 通胀数据 (CPI/PCE)")
lines.append("- 股债相关性是否继续恶化或修复")
lines.append("")

# ── 反馈闭环 ──
feedback_lines = []
sc_summary_path = os.path.join(DATA_DIR, "prediction_scorecard_summary.json")
if os.path.exists(sc_summary_path):
    with open(sc_summary_path) as f:
        sc_sum = json.load(f)
    parts = []
    for asset, m in sc_sum.get("per_asset", {}).items():
        da = m.get("directional_accuracy")
        if da is not None:
            parts.append(f"{asset} {da*100:.0f}%")
    if parts:
        feedback_lines.append(f"- TimesFM 方向准确率: {' | '.join(parts)}")

bt_summary_path = os.path.join(DATA_DIR, "backtest_summary.json")
if os.path.exists(bt_summary_path):
    with open(bt_summary_path) as f:
        bt_sum = json.load(f)
    ranking = bt_sum.get("ranking", [])
    if ranking:
        rank_parts = []
        for name in ranking[:3]:
            s = bt_sum["strategies"][name]
            sh = s.get("sharpe_60d") or s.get("sharpe_all", 0)
            rank_parts.append(f"{name}({sh:.1f})")
        feedback_lines.append(f"- 策略排名(Sharpe): {' > '.join(rank_parts)}")

if feedback_lines:
    lines.append("## 反馈闭环")
    lines.append("")
    for l in feedback_lines:
        lines.append(l)
    lines.append("")

# ── 局限 ──
lines.append("## 局限")
lines.append("")
lines.append("1. **滞后** — 统计模型需数据积累确认切换，真正变化第一天模型还没翻牌")
lines.append("2. **线性假设** — Kalman 假设 y=α+βx，极端非线性事件（如 2020 负油价）失效")
lines.append("3. **TimesFM 不懂因果** — 只看价格模式，不知道地缘事件升级；观点置信度部分弥补但不根治")
lines.append("4. **无样本外回测** — 全样本拟合，实际效果可能不如回看")
lines.append("5. **SP-Sharpe 集中度不实用** — 数学最优 ≠ 可持有，实际需额外约束")
lines.append("")
lines.append("---")
lines.append(f"*Markov+Wasserstein Ensemble / Kalman Filter / BL+SP | v2 | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")

summary = "\n".join(lines)

# ── Strategy backtest (non-blocking) ───────────────────
print("\n" + "=" * 60)
print("Running strategy backtest...")
bt_result = subprocess.run(
    [sys.executable, os.path.join(SCRIPTS_DIR, "strategy_backtest.py")],
    capture_output=True, text=True, timeout=120,
)
if bt_result.returncode == 0:
    print(bt_result.stdout[:500])

# ── Save ───────────────────────────────────────────────
with open(os.path.join(ANALYSIS_DIR, f"dashboard_{today}.md"), "w") as f:
    f.write(summary)

with open(os.path.join(ANALYSIS_DIR, "dashboard_latest.md"), "w") as f:
    f.write(summary)

# JSON for cron notification
output_json = {
    "regime_short": regime_short,
    "score_short": score_short,
    "signals_short": signals_short,
    "regime_long": regime_long,
    "score_long": score_long,
    "signals_long": signals_long,
    "markov_summary": markov_summary,
    "kalman_summary": kalman_summary,
    "summary_preview": summary[:2000],
}
with open(os.path.join(DATA_DIR, "dashboard_latest.json"), "w") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"\n✓ Dashboard saved → {ANALYSIS_DIR}/dashboard_{today}.md")
