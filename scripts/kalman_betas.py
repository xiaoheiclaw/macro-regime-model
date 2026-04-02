#!/usr/bin/env python3
"""
Phase 3a: BEI (Breakeven Inflation) supplement + Kalman Filter time-varying beta
"""
import os, sys
import pandas as pd
import numpy as np
import numba
from numba import njit
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR
from lib.data_loader import load_merged

df = load_merged()

# ── Try to get BEI via yfinance TIP/IEF ratio as proxy ─
import yfinance as yf
print("Fetching BEI proxies...")
try:
    tip = yf.download("TIP", start="2020-01-01", progress=False, auto_adjust=True)["Close"]
    ief = yf.download("IEF", start="2020-01-01", progress=False, auto_adjust=True)["Close"]
    if isinstance(tip, pd.DataFrame): tip = tip.iloc[:, 0]
    if isinstance(ief, pd.DataFrame): ief = ief.iloc[:, 0]
    # TIP/IEF ratio rises when inflation expectations rise
    bei_proxy = (tip / ief).dropna()
    bei_proxy.index = pd.to_datetime(bei_proxy.index)
    df["BEI_proxy"] = bei_proxy.reindex(df.index, method="ffill")
    print(f"  ✓ BEI proxy (TIP/IEF): {df['BEI_proxy'].notna().sum()} rows")
except Exception as e:
    print(f"  ✗ BEI proxy: {e}")

# Try FRED direct for BEI
try:
    import urllib.request
    url5 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T5YIE&cosd=2020-01-01"
    url10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE&cosd=2020-01-01"
    
    for url, name in [(url5, "BEI_5Y"), (url10, "BEI_10Y")]:
        try:
            s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
            s = s.replace(".", np.nan).astype(float)
            s.index = pd.to_datetime(s.index)
            df[name] = s.reindex(df.index, method="ffill")
            print(f"  ✓ {name}: {df[name].notna().sum()} rows")
        except Exception as ex:
            print(f"  ✗ {name}: {ex}")
except: pass

print(f"\nData: {len(df)} rows, cols: {list(df.columns)}")

# ── Kalman Filter: Time-Varying Beta ────────────────────
print("\n=== Kalman Filter: Time-Varying Factor Betas ===")

report = []
report.append("# Phase 3: Time-Varying Factor Attribution (Kalman Filter)")
report.append(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report.append("")

@njit(cache=True)
def _recursive_beta_core(y, x, lambda_decay, init_indices, start_from):
    """Numba-accelerated core of recursive least squares Kalman filter."""
    n = len(y)
    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    init = len(init_indices)

    # Build X_init and y_init
    X_init = np.empty((init, 2), dtype=np.float64)
    y_init = np.empty(init, dtype=np.float64)
    for k in range(init):
        idx = init_indices[k]
        X_init[k, 0] = 1.0
        X_init[k, 1] = x[idx]
        y_init[k] = y[idx]

    # XtX = X_init.T @ X_init
    XtX = np.zeros((2, 2), dtype=np.float64)
    for k in range(init):
        for i in range(2):
            for j in range(2):
                XtX[i, j] += X_init[k, i] * X_init[k, j]

    # inv(XtX) via 2x2 analytic formula
    det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
    if abs(det) < 1e-30:
        return betas, alphas
    inv_XtX = np.empty((2, 2), dtype=np.float64)
    inv_XtX[0, 0] = XtX[1, 1] / det
    inv_XtX[0, 1] = -XtX[0, 1] / det
    inv_XtX[1, 0] = -XtX[1, 0] / det
    inv_XtX[1, 1] = XtX[0, 0] / det

    P = np.empty((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            P[i, j] = inv_XtX[i, j] * 10.0

    # theta = inv(XtX) @ X_init.T @ y_init
    Xty = np.zeros(2, dtype=np.float64)
    for k in range(init):
        for i in range(2):
            Xty[i] += X_init[k, i] * y_init[k]

    theta = np.empty(2, dtype=np.float64)
    for i in range(2):
        theta[i] = inv_XtX[i, 0] * Xty[0] + inv_XtX[i, 1] * Xty[1]

    # Fill initial period
    for k in range(init):
        idx = init_indices[k]
        alphas[idx] = theta[0]
        betas[idx] = theta[1]

    # Main Kalman loop
    x_t = np.empty(2, dtype=np.float64)
    K = np.empty(2, dtype=np.float64)
    for t in range(start_from, n):
        x_t[0] = 1.0
        x_t[1] = x[t]

        if np.isnan(x_t[1]) or np.isnan(y[t]):
            if t > 0:
                betas[t] = betas[t - 1]
                alphas[t] = alphas[t - 1]
            continue

        # Prediction error
        e = y[t] - (x_t[0] * theta[0] + x_t[1] * theta[1])

        # P = P / lambda_decay
        for i in range(2):
            for j in range(2):
                P[i, j] /= lambda_decay

        # denom = x_t @ P @ x_t + 1
        denom = 1.0
        for i in range(2):
            for j in range(2):
                denom += x_t[i] * P[i, j] * x_t[j]

        # K = P @ x_t / denom
        for i in range(2):
            K[i] = (P[i, 0] * x_t[0] + P[i, 1] * x_t[1]) / denom

        # theta = theta + K * e
        for i in range(2):
            theta[i] += K[i] * e

        # P = (I - K * x_t.T) @ P
        # First compute KxT = outer(K, x_t)
        new_P = np.empty((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                val = 0.0
                for k in range(2):
                    ik = -K[i] * x_t[k]
                    if i == k:
                        ik += 1.0
                    val += ik * P[k, j]
                new_P[i, j] = val
        for i in range(2):
            for j in range(2):
                P[i, j] = new_P[i, j]

        alphas[t] = theta[0]
        betas[t] = theta[1]

    return betas, alphas

def recursive_beta(y, x, lambda_decay=0.97):
    """Exponentially weighted recursive least squares (numba-accelerated)"""
    n = len(y)
    valid = ~(np.isnan(y) | np.isnan(x))
    valid_idx = np.where(valid)[0]
    
    if len(valid_idx) < 40:
        print(f"  ⚠ Not enough valid obs: {len(valid_idx)}")
        return np.full(n, np.nan), np.full(n, np.nan)
    
    init_indices = valid_idx[:30].astype(np.int64)
    start_from = int(valid_idx[30])
    
    betas, alphas = _recursive_beta_core(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(x, dtype=np.float64),
        lambda_decay, init_indices, start_from)
    
    valid_count = np.sum(~np.isnan(betas))
    print(f"  ✓ {valid_count} valid beta estimates")
    return betas, alphas

# Warm up numba JIT
_wy = np.random.randn(100)
_wx = np.random.randn(100)
_ = recursive_beta(_wy, _wx, 0.97)

# ── Beta 1: Oil → 10Y Yield ────────────────────────────
report.append("## 1. Oil → 10Y Yield: Time-Varying Beta")
report.append("")

oil_ret = df["WTI_crude"].pct_change().values * 100
yield_chg = df["US10Y_yield"].diff().values * 100

beta_oil, alpha_oil = recursive_beta(yield_chg, oil_ret, lambda_decay=0.97)
df["beta_oil_yield"] = beta_oil

# Report key periods
report.append("Oil β 越高 = 通胀传导越强; β 接近0或负 = 脱钩/衰退定价")
report.append("")
report.append("| Period | Avg β | Interpretation |")
report.append("|---|---|---|")

for name, start, end in [
    ("2020 COVID", "2020-02-01", "2020-06-30"),
    ("2021 Reflation", "2021-01-01", "2021-12-31"),
    ("2022 Rate Hikes", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025 H1", "2025-01-01", "2025-06-30"),
    ("2025 H2", "2025-07-01", "2025-12-31"),
    ("2026 Jan", "2026-01-01", "2026-01-31"),
    ("2026 Feb", "2026-02-01", "2026-02-28"),
    ("2026 Mar 1-15", "2026-03-01", "2026-03-15"),
    ("2026 Mar 16-31", "2026-03-16", "2026-03-31"),
]:
    mask = (df.index >= start) & (df.index <= end)
    subset = df.loc[mask, "beta_oil_yield"].dropna()
    if len(subset) > 0:
        avg = subset.mean()
        if avg > 1.0: interp = "🔴 强通胀传导"
        elif avg > 0.3: interp = "🟠 温和通胀"
        elif avg > -0.3: interp = "🟡 弱/脱钩"
        else: interp = "🟢 反向（衰退定价）"
        report.append(f"| {name} | {avg:.3f} | {interp} |")

report.append("")
report.append(f"**最新 β (oil→yield): {df['beta_oil_yield'].dropna().iloc[-1]:.3f}**")
report.append("")

# ── Beta 2: SPX → 10Y Yield ────────────────────────────
report.append("## 2. SPX → 10Y Yield: Time-Varying Beta")
report.append("")

if "SPX" in df.columns:
    spx_ret = df["SPX"].pct_change().values * 100
    beta_spx, _ = recursive_beta(yield_chg, spx_ret, lambda_decay=0.97)
    df["beta_spx_yield"] = beta_spx

    report.append("SPX β > 0 = 增长主导（股涨→收益率涨）; β < 0 = 避险/流动性主导")
    report.append("")
    report.append("| Period | Avg β | Interpretation |")
    report.append("|---|---|---|")

    for name, start, end in [
        ("2020 COVID", "2020-02-01", "2020-06-30"),
        ("2021 Reflation", "2021-01-01", "2021-12-31"),
        ("2022 Rate Hikes", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025 H2", "2025-07-01", "2025-12-31"),
        ("2026 Jan", "2026-01-01", "2026-01-31"),
        ("2026 Feb", "2026-02-01", "2026-02-28"),
        ("2026 Mar 1-15", "2026-03-01", "2026-03-15"),
        ("2026 Mar 16-31", "2026-03-16", "2026-03-31"),
    ]:
        mask = (df.index >= start) & (df.index <= end)
        subset = df.loc[mask, "beta_spx_yield"].dropna()
        if len(subset) > 0:
            avg = subset.mean()
            if avg > 0.5: interp = "📈 增长主导"
            elif avg > 0: interp = "📊 弱增长"
            elif avg > -0.5: interp = "📉 弱避险"
            else: interp = "🔻 强避险/脱钩"
            report.append(f"| {name} | {avg:.3f} | {interp} |")

    report.append("")
    report.append(f"**最新 β (spx→yield): {df['beta_spx_yield'].dropna().iloc[-1]:.3f}**")
    report.append("")

# ── Beta 3: Gold → 10Y Yield ───────────────────────────
report.append("## 3. Gold → 10Y Yield: Time-Varying Beta")
report.append("")

gold_ret = df["Gold"].pct_change().values * 100
beta_gold, _ = recursive_beta(yield_chg, gold_ret, lambda_decay=0.97)
df["beta_gold_yield"] = beta_gold

report.append("Gold β < 0 = 传统避险（金涨→收益率跌）; β > 0 = 通胀对冲（金涨→收益率也涨）")
report.append("")
report.append("| Period | Avg β | Interpretation |")
report.append("|---|---|---|")

for name, start, end in [
    ("2020 COVID", "2020-02-01", "2020-06-30"),
    ("2022 Rate Hikes", "2022-01-01", "2022-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025 H2", "2025-07-01", "2025-12-31"),
    ("2026 Jan", "2026-01-01", "2026-01-31"),
    ("2026 Feb", "2026-02-01", "2026-02-28"),
    ("2026 Mar", "2026-03-01", "2026-03-31"),
]:
    mask = (df.index >= start) & (df.index <= end)
    subset = df.loc[mask, "beta_gold_yield"].dropna()
    if len(subset) > 0:
        avg = subset.mean()
        if avg < -0.3: interp = "🟢 避险（传统）"
        elif avg > 0.3: interp = "🔴 通胀对冲"
        else: interp = "🟡 中性"
        report.append(f"| {name} | {avg:.3f} | {interp} |")

report.append("")
report.append(f"**最新 β (gold→yield): {df['beta_gold_yield'].dropna().iloc[-1]:.3f}**")
report.append("")

# ── BEI Analysis ────────────────────────────────────────
report.append("## 4. Inflation Expectations (BEI)")
report.append("")

bei_col = None
for c in ["BEI_5Y", "BEI_10Y", "BEI_proxy"]:
    if c in df.columns and df[c].notna().sum() > 100:
        bei_col = c
        break

if bei_col:
    bei = df[bei_col].dropna()
    report.append(f"Using: {bei_col}")
    report.append("")

    if bei_col == "BEI_proxy":
        report.append("(TIP/IEF ratio proxy - direction meaningful, absolute level not)")
        report.append("")
    
    # Key levels
    jan_avg = bei.loc["2026-01"].mean() if "2026-01" in bei.index.strftime("%Y-%m").values else np.nan
    latest = bei.iloc[-1]
    
    if pd.notna(jan_avg):
        change = latest - jan_avg
        pct_change = change / jan_avg * 100
        report.append(f"- Jan 2026 avg: {jan_avg:.4f}")
        report.append(f"- Latest: {latest:.4f}")
        report.append(f"- Change: {change:.4f} ({pct_change:+.2f}%)")
        
        if change < 0:
            report.append(f"- ✓ **通胀预期在下行** — 与文章论点一致")
        else:
            report.append(f"- ✗ 通胀预期在上行")
    report.append("")
else:
    report.append("⚠ BEI data not available (FRED proxy failed, yfinance proxy failed)")
    report.append("")

# ── Summary ─────────────────────────────────────────────
report.append("## 🎯 Phase 3 综合结论")
report.append("")
report.append("### Factor Beta 全景")
report.append("")
report.append("| Factor | Latest β | 3月上半月 | 3月下半月 | 变化 |")
report.append("|---|---|---|---|---|")

for name, col in [("Oil→Yield", "beta_oil_yield"), ("SPX→Yield", "beta_spx_yield"), ("Gold→Yield", "beta_gold_yield")]:
    if col in df.columns:
        latest_b = df[col].dropna().iloc[-1] if df[col].notna().any() else np.nan
        m1 = df.loc["2026-03-01":"2026-03-15", col].dropna().mean() if col in df.columns else np.nan
        m2 = df.loc["2026-03-16":"2026-03-31", col].dropna().mean() if col in df.columns else np.nan
        chg = m2 - m1 if pd.notna(m1) and pd.notna(m2) else np.nan
        chg_s = f"{chg:+.3f}" if pd.notna(chg) else "N/A"
        m1_s = f"{m1:.3f}" if pd.notna(m1) else "N/A"
        m2_s = f"{m2:.3f}" if pd.notna(m2) else "N/A"
        report.append(f"| {name} | {latest_b:.3f} | {m1_s} | {m2_s} | {chg_s} |")

report.append("")

# Save updated data
# Save only the Kalman beta columns (not the whole merged dataset)
beta_cols = [c for c in df.columns if c.startswith("beta_")]
if beta_cols:
    df[beta_cols].to_csv(os.path.join(DATA_DIR, "kalman_betas.csv"))

output = "\n".join(report)
out_path = os.path.join(ANALYSIS_DIR, "kalman_betas_20260331.md")
with open(out_path, "w") as f:
    f.write(output)

print("\n" + "=" * 60)
print(output)
print(f"\n✓ Saved to {out_path}")
