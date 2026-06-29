#!/usr/bin/env python3
"""US equity leverage fragility monitor — data pipeline.

Pulls the authoritative raw series, computes the 3-layer signal snapshot
(fuel / amplifier / ignition), and emits two artifacts:
  1. data/leverage_signals.json   (persisted in this repo)
  2. ~/.avibe/show/<session>/src/signals.ts  (consumed by the Show Page)

Run:
  uv run --with pandas --with numpy --with xlrd python scripts/leverage_monitor.py

  (xlrd reads the legacy Shiller .xls; openpyxl — declared in pyproject and so
  pulled in by `uv run` — reads the FINRA .xlsx. Network access required.)

Data sources (all free, all first-party — no sell-side chart dependency):
  - FINRA "Customer Margin Balances" (gross margin debt, monthly)  [official, 1997+]
    https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx
  - Shiller S&P (Yale ie_data.xls)                                 [academic, 1871+]
    http://www.econ.yale.edu/~shiller/data/ie_data.xls
  - S&P 500 monthly close via Yahoo (^GSPC), used to chain Shiller forward
    (the FRED CSV endpoint was unreachable from this host)         [chained → 2026]

The research-grade L2 fields (AXW funding-cost futures spread, dealer equity
repo exposure, breadth) are sell-side-report constants entered as manual
snapshots BY DESIGN — they have no free first-party feed and are not automated.

Methodology notes baked into the output (anti-misuse):
  - The "debt ÷ S&P price-index proxy" ratio is reported but FLAGGED — note
    the denominator is the S&P PRICE INDEX, not true market cap. It is
    denominator-polluted (price is driven by the leverage itself) and lags at
    tops, peaks instead at panic bottoms (2008-10). Context only, NOT a
    primary risk signal.
  - Primary signals: absolute debt level + 2nd derivative (mom/yoy), and
    net/gross "hardness" (less cash buffer = more fragile).
"""
from __future__ import annotations
import io
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Repo root derived from this file's location — portable across CI / forks /
# arbitrary checkouts (no hardcoded $HOME path). DATA_DIR is created lazily in
# write_signals(), never at import time.
PROJ = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ / "data"
# Optional Show-Page sink: opt-in only, resolved at write time (see write_signals).
# Set env LEVERAGE_SHOW_SRC to a Show-Page `src/` dir to also emit signals.ts there.
SHOW_SRC_ENV = "LEVERAGE_SHOW_SRC"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    if retries < 1:
        raise ValueError("retries must be ≥1")
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:  # network hiccup / proxy jitter — retry
            last = e
            print(f"    (attempt {i+1}/{retries} failed {type(e).__name__}: {url[:60]})")
            if i < retries - 1:                # don't sleep after the final attempt
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def fetch_finra() -> pd.DataFrame:
    """FINRA Customer Margin Balances. Returns ym-indexed: debit, cash_credit, margin_credit (in $M)."""
    raw = _get("https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx")
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    df = df.dropna(how="all", axis=1)
    if df.shape[1] < 4:
        raise ValueError(f"Unexpected FINRA sheet shape {df.shape}: need ≥4 columns")
    # Prefer matching columns BY HEADER KEYWORD (robust to inserted columns);
    # fall back to first-4-positional if the headers aren't recognizable.
    df = _select_finra_columns(df)
    df = df.dropna(subset=["debit_M"]).copy()
    # Validate column SEMANTICS, not just count — guards against layout drift
    # silently feeding a wrong (e.g. text-blurb) column in as debit_M.
    if pd.to_datetime(df["ym"], errors="coerce").isna().mean() > 0.5:
        raise ValueError(f"FINRA col-0 does not parse as dates: {df['ym'].head(3).tolist()}")
    for c in ("debit_M", "cash_credit_M", "margin_credit_M"):
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() < 0.5:
            raise ValueError(f"FINRA column {c!r} is mostly non-numeric — layout drift?")
        df[c] = num
    # ym normalization (→ 'YYYY-MM') is the transform path's job (build_series);
    # fetch_finra only validates schema/types and returns the raw-ish frame.
    return df


def _select_finra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map FINRA columns to [ym, debit_M, cash_credit_M, margin_credit_M] by header
    keyword (robust to inserted columns); fall back to the first 4 columns."""
    low = {c: str(c).lower() for c in df.columns}

    def find(*keys, exclude=()):
        for c in df.columns:
            t = low[c]
            if all(k in t for k in keys) and not any(x in t for x in exclude):
                return c
        return None

    date_c = find("month") or find("date") or find("year")
    debit_c = find("debit")
    cash_c = find("cash")                                # "...in Cash Accounts"
    margin_c = find("margin", exclude=("debit",))        # "...in Margin Accounts"
    picked = [date_c, debit_c, cash_c, margin_c]
    if all(c is not None for c in picked) and len(set(picked)) == 4:
        out = df[picked].copy()
    else:
        out = df.iloc[:, :4].copy()                      # positional fallback
    out.columns = ["ym", "debit_M", "cash_credit_M", "margin_credit_M"]
    return out


def _to_ym(s: pd.Series) -> pd.Series:
    """Coerce a date-ish column (Timestamp, 'Jan-97', '1997-01-31', '1997-01', …)
    to canonical 'YYYY-MM' so FINRA aligns with Shiller/Yahoo. Fail fast if any
    row is unparseable — a silent NaT would drop rows and corrupt the series."""
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        bad = s[dt.isna()].astype(str).tolist()[:5]
        raise ValueError(f"FINRA date column has unparseable values: {bad}")
    return dt.dt.strftime("%Y-%m")


def fetch_shiller_sp() -> pd.Series:
    """Shiller S&P price (P), monthly, ym-indexed."""
    raw = _get("http://www.econ.yale.edu/~shiller/data/ie_data.xls")
    r = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=7)
    r = r[["Date", "P"]].dropna().copy()
    # Keep every derived column in the SAME frame so footnote / hole / bad-month
    # rows are dropped consistently (no parallel-Series misalignment).
    r["date_num"] = pd.to_numeric(r["Date"], errors="coerce")
    r = r.dropna(subset=["date_num"]).copy()
    r["year"] = r["date_num"].astype(int)
    r["month"] = np.round((r["date_num"] - r["year"]) * 100).astype(int)
    r = r[r["month"].between(1, 12)].copy()
    r["ym"] = pd.to_datetime(
        dict(year=r["year"], month=r["month"], day=1)
    ).dt.strftime("%Y-%m")
    r["P"] = pd.to_numeric(r["P"], errors="coerce")
    return r.dropna(subset=["P"]).sort_values("ym").set_index("ym")["P"]


def fetch_sp500_monthly() -> pd.Series:
    """S&P 500 monthly close, ym-indexed. Used to chain Shiller forward.
    Source: Yahoo query1 (FRED API/CSV endpoint was unreachable from this host)."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           "?interval=1mo&range=10y")
    d = json.loads(_get(url, 40))
    chart = d.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo ^GSPC error: {chart['error']}")
    result = chart.get("result") or []
    if not result:
        raise ValueError("Yahoo ^GSPC returned no result")
    r = result[0]
    ts = r.get("timestamp")
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    cl = quote.get("close")
    if not ts or not cl or len(ts) != len(cl):
        raise ValueError(f"Yahoo ^GSPC malformed series: len(ts)={len(ts or [])} len(close)={len(cl or [])}")
    df = pd.DataFrame({"ts": ts, "close": cl}).dropna()
    df["ym"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m")
    return df.groupby("ym")["close"].last().astype(float).sort_index()


def chain_sp(shiller: pd.Series, ext_sp: pd.Series) -> pd.Series:
    """Shiller base, extended forward with Yahoo ^GSPC month-over-month returns
    (keeps one price level). Anchored at the last month present in BOTH series,
    so it never assumes Yahoo contains Shiller's last month."""
    overlap = shiller.index.intersection(ext_sp.index)
    if overlap.empty:
        raise ValueError("Cannot chain S&P: no overlap between Shiller and Yahoo series")
    anchor = overlap.max()
    fmonths = sorted(m for m in ext_sp.index if m > anchor)
    ext = {}; prev = anchor; P = float(shiller.loc[anchor])
    for m in fmonths:
        P *= float(ext_sp.loc[m]) / float(ext_sp.loc[prev]); ext[m] = P; prev = m
    if not ext:
        return shiller.sort_index()
    out = pd.concat([shiller, pd.Series(ext)]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def build_series() -> pd.DataFrame:
    fin = fetch_finra()
    fin["ym"] = _to_ym(fin["ym"])   # normalize again here (idempotent) so the
    fin = fin.set_index("ym")       # transform path is robust to any caller/mock
    sp = chain_sp(fetch_shiller_sp(), fetch_sp500_monthly())
    fin["sp"] = sp
    fin = fin.dropna(subset=["sp"]).reset_index().sort_values("ym")
    fin["gross_B"] = fin["debit_M"] / 1000.0
    # Do NOT fillna(0) the credit legs: a missing source value is not zero credit
    # — that would systematically overstate net_B / net_gross (the "hardness"
    # signal). Missing → NaN, which we forbid on the current row below.
    fin["net_B"] = (fin["debit_M"] - fin["cash_credit_M"]
                    - fin["margin_credit_M"]) / 1000.0
    fin["net_gross"] = fin["net_B"] / fin["gross_B"]
    fin["ratio"] = fin["gross_B"] / fin["sp"]            # debt ÷ S&P price-index proxy (NOT market cap; denominator-polluted — flagged)
    fin["dt"] = pd.to_datetime(fin["ym"], format="%Y-%m")
    if len(fin) < 2:
        raise ValueError(
            f"build_series produced {len(fin)} rows after FINRA×SP merge — "
            "check source coverage overlap (need ≥2 months for mom/compute)")
    cur = fin.iloc[-1]
    missing = [c for c in ("debit_M", "cash_credit_M", "margin_credit_M", "sp")
               if pd.isna(cur[c])]
    if missing:
        raise ValueError(f"Latest month {cur['ym']} is missing {missing} — refusing "
                         "to emit NaN signals for the current snapshot")
    return fin.reset_index(drop=True)


def pct(df: pd.DataFrame, col: str, val: float) -> float:
    # "% of history at or below current" — use <= so an all-time high reads 100%,
    # matching the "percentile" label in the UI.
    return float((df[col] <= val).mean() * 100)


def _layer_status(items: list[dict]) -> str:
    """Aggregate a layer's status from its items so the summary can't drift from
    the (possibly stale-degraded) per-item statuses. Worst non-muted wins;
    all-muted → muted."""
    statuses = [it.get("status") for it in items]
    if "red" in statuses:
        return "red"
    if "amber" in statuses or "watch" in statuses:
        return "amber"
    if "green" in statuses:
        return "green"
    return "muted"


def _record_high_streak(s: pd.Series) -> int:
    """Number of most-recent consecutive months that each set a new all-time
    high (strictly above every prior month). 0 if the latest is not a record."""
    prev_max = s.cummax().shift(1)
    is_new_high = (s > prev_max) | prev_max.isna()  # first obs counts as a high
    streak = 0
    for v in reversed(is_new_high.tolist()):
        if v:
            streak += 1
        else:
            break
    return streak


# Manual / research-grade snapshots have no live feed; flag them stale so a
# refreshed `updated` date can't masquerade them as current auto data.
STALE_AFTER_DAYS = 21


def _mark_stale(item: dict, today: pd.Timestamp) -> dict:
    """If a manual/semi item carries an `asof`, annotate age and degrade status
    to muted once older than STALE_AFTER_DAYS."""
    asof = item.get("asof")
    if not asof:
        return item
    age = int((today - pd.to_datetime(asof)).days)
    item["age_days"] = age
    item["stale"] = age > STALE_AFTER_DAYS
    if item["stale"]:
        item["status"] = "muted"
        item["note"] = (f"⏳STALE {age}d(>{STALE_AFTER_DAYS}d)；需手工刷新 "
                        + item.get("note", "")).rstrip()
    return item


def compute(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        raise ValueError(f"compute needs ≥2 rows (mom/yoy), got {len(df)}")
    today = pd.Timestamp(datetime.now().date())
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    yoy_row = df[df["dt"] == (cur["dt"] - pd.DateOffset(years=1))]
    yoy = (cur["gross_B"] / yoy_row["gross_B"].values[0] - 1) * 100 if len(yoy_row) else None
    p21 = df[df["ym"] == "2021-10"]
    p21 = p21.iloc[0] if len(p21) else None
    rp = df.loc[df["ratio"].idxmax()]
    net_med = float(df["net_gross"].median())

    # L1 fuel signals
    debt_at_high = bool(df["gross_B"].iloc[-1] >= df["gross_B"].max() - 1e-6)
    streak = _record_high_streak(df["gross_B"])
    if debt_at_high and streak >= 2:
        debt_note = f"史上最高，连续 {streak} 月创新高"
    elif debt_at_high:
        debt_note = "史上最高（本月创新高）"
    else:
        debt_note = "高位"
    l1_items = [
        {"label": "Gross margin debt 绝对水平", "value": f"${cur['gross_B']:,.0f}B",
         "status": "red" if debt_at_high else "amber",
         "note": debt_note},
        {"label": "净新增 (mom)", "value": f"{(cur['gross_B']/prev['gross_B']-1)*100:+.1f}%",
         "status": "red" if (cur['gross_B'] > prev['gross_B']) else "green",
         "note": "二阶导：仍在加速"},
        {"label": "同比 (yoy)", "value": f"{yoy:+.0f}%" if yoy is not None else "n/a",
         "status": "red" if (yoy and yoy > 20) else "amber", "note": "vs 一年前"},
        {"label": "net/gross 硬度", "value": f"{cur['net_gross']:.2f}",
         "status": "red" if cur['net_gross'] > 0.6 else ("amber" if cur['net_gross'] > net_med else "green"),
         "note": f"中位 {net_med:.2f}；越高=现金缓冲越少=越脆"},
        {"label": "debt÷S&P价格指数(代理) ratio 百分位", "value": f"{pct(df,'ratio',cur['ratio']):.0f}%",
         "status": "muted", "note": "⚠ 分母是S&P价格指数(非真实市值)，分母污染、顶部失真，仅作背景——不据此判断风险"},
    ]
    l1 = {"name": "燃料 (Fuel) — 干柴多干", "items": l1_items, "status": _layer_status(l1_items)}

    # L2 amplifier (partly semi-auto / manual snapshot)
    l2_items = [
        _mark_stale({"label": "杠杆 ETF AUM (TQQQ/SOXL/UPRO 等)", "value": "~$247B",
         "status": "amber", "avail": "semi", "asof": "2026-06-26",
         "note": "JPM 6/26，集中科技；半自动（周度抓发行商）"}, today),
        _mark_stale({"label": "融资成本 AXW 期货利差", "value": "+140bp",
         "status": "red", "avail": "manual", "asof": "2026-06-16",
         "note": "大摩 6/16，除年末外史上最高；🔴 研报级，手工录入"}, today),
        {"label": "半导体 realized vol", "value": "待接入",
         "status": "muted", "avail": "todo", "note": "MVP 待加：SOXX/费半日价格→vol"},
        _mark_stale({"label": "交易商股权回购敞口", "value": "$2230B",
         "status": "red", "avail": "manual", "asof": "2026-06-16",
         "note": "大摩 6/16 史上最高；🔴 研报级，手工录入"}, today),
    ]
    l2 = {"name": "放大器 (Amplifier) — 弹药与传导", "items": l2_items, "status": _layer_status(l2_items)}

    # L3 ignition
    l3_items = [
        {"label": "三联启动信号", "value": "未触发",
         "status": "green",
         "note": "半导体vol跳 + 杠杆ETF尾盘流量激增 + 广度塌缩，三者同现才算点火"},
        {"label": "下一高脆弱窗口", "value": "NVDA FY27 Q2 ~8月底",
         "status": "amber", "note": "广度最窄+AI最拥挤，财报/指引 miss 风险最高"},
    ]
    l3 = {"name": "启动 (Ignition) — 火花+点火", "items": l3_items, "status": _layer_status(l3_items)}

    # FINRA publishes a month's balances ~end of the following month; the next
    # pending month is cur+1, expected ~end of cur+2. Derived from latest data.
    pend = cur["dt"] + pd.DateOffset(months=1)
    rel = cur["dt"] + pd.DateOffset(months=2)
    monitors = [
        {"name": f"FINRA gross margin debt ({pend.year}-{pend.month:02d} 值)",
         "current": f"待发布（约 {rel.year}-{rel.month:02d} 下旬）",
         "threshold": "续创新高=压力续增；掉头=拐点", "status": "watch", "source": "FINRA", "avail": "auto"},
        _mark_stale({"name": "AXW 期货利差", "current": "+140bp", "threshold": "回落=缓解；续升=临界", "status": "red", "source": "大摩研报", "avail": "manual", "note": "", "asof": "2026-06-16"}, today),
        _mark_stale({"name": "市场广度 (>5 行业跑赢)", "current": "仅信息技术 1 个", "threshold": "扩散=健康；持续仅科技=脆弱", "status": "red", "source": "大摩/JPM", "avail": "semi", "note": "", "asof": "2026-06-26"}, today),
        {"name": "半导体 realized vol + ETF 流量", "current": "待接入", "threshold": "vol跳+流量激增=点火", "status": "watch", "source": "价格估算", "avail": "todo"},
        {"name": "零售信用违约率", "current": "待接入", "threshold": "margin退潮+零售信用恶化=系统性", "status": "watch", "source": "FRED", "avail": "todo"},
    ]

    events = [
        {"date": "2026-07 下旬", "label": "FINRA 6月 margin debt 发布", "type": "data"},
        {"date": "2026-07", "label": "CPI / FOMC", "type": "macro"},
        {"date": "2026-08 下旬", "label": "NVDA FY27 Q2 财报（最高脆弱窗口）", "type": "earnings"},
        {"date": "2026-08", "label": "超大规模云厂商 capex 指引", "type": "earnings"},
    ]

    # chart history (sample to ~ monthly, all points fine — ~350)
    hist = df.tail(360)
    chart = {
        "dt": hist["ym"].tolist(),
        "gross": [round(x, 1) for x in hist["gross_B"].tolist()],
        "ratio": [round(float(x), 4) for x in hist["ratio"].tolist()],
    }

    return {
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "data_range": f"{df['ym'].iloc[0]} → {df['ym'].iloc[-1]}",
        "n_months": len(df),
        "latest": {
            "ym": cur["ym"], "gross_B": round(float(cur["gross_B"]), 0),
            "net_B": round(float(cur["net_B"]), 0), "net_gross": round(float(cur["net_gross"]), 3),
            "mom_pct": round((cur["gross_B"] / prev["gross_B"] - 1) * 100, 1),
            "yoy_pct": round(yoy, 0) if yoy is not None else None,
            "ratio": round(float(cur["ratio"]), 4),
            "ratio_percentile": round(pct(df, "ratio", cur["ratio"]), 0),
            "sp": round(float(cur["sp"]), 0),
        },
        "vs_2021_peak": {
            "gross_B": round(float(p21["gross_B"]), 0) if p21 is not None else None,
            "gross_delta_pct": round((cur["gross_B"] / p21["gross_B"] - 1) * 100, 0) if p21 is not None else None,
            "ratio": round(float(p21["ratio"]), 4) if p21 is not None else None,
        },
        "ratio_peak": {"ym": rp["ym"], "ratio": round(float(rp["ratio"]), 4),
                       "note": "GFC 恐慌底——分母污染反向证据：暴跌时比率反而冲顶"},
        "layers": [l1, l2, l3],
        "monitors": monitors,
        "events": events,
        "chart": chart,
        "methodology": (
            "跌幅 = 火花 × 放大器 × 燃料。本仪表盘监控三者状态，不预测火花。"
            "primary 信号：绝对债务水平 + 二阶导（mom/yoy）+ net/gross 硬度。"
            "debt÷S&P价格指数(代理) ratio 因分母污染（价格被杠杆本身推高，且分母是价格指数而非真实市值），顶部失真、底部虚高，仅作背景。"
        ),
        "sources": [
            "FINRA Customer Margin Balances (官方, 1997+)",
            "Shiller S&P (Yale ie_data.xls, 1871+)",
            "Yahoo Finance ^GSPC monthly close (chains Shiller forward → 2026)",
            "摩根士丹利 6/16、摩根大通 6/26 研报（研报级字段，手工录入）",
        ],
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same dir + os.replace, so a crash mid-write
    never leaves a half-written artifact for the Show Page to choke on."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_signals(sig: dict, show_src: Path | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # allow_nan=False → fail fast rather than emit non-standard JSON (NaN/Infinity)
    # that strict consumers reject.
    payload = json.dumps(sig, indent=2, ensure_ascii=False, allow_nan=False)
    _atomic_write(DATA_DIR / "leverage_signals.json", payload)
    # Show-Page sink is opt-in: explicit arg, else env LEVERAGE_SHOW_SRC, resolved
    # at call time (so importers can set it after import).
    if show_src is None:
        env = os.environ.get(SHOW_SRC_ENV)
        show_src = Path(env).expanduser() if env else None
    if show_src is None:
        return
    if show_src.is_dir():
        ts = "// AUTO-GENERATED by scripts/leverage_monitor.py — do not edit by hand.\n"
        ts += "export const SIGNALS = " + json.dumps(sig, ensure_ascii=False, allow_nan=False) + " as const;\n"
        _atomic_write(show_src / "signals.ts", ts)
        print(f"  → wrote show page: {show_src/'signals.ts'}")
    else:
        print(f"  ({SHOW_SRC_ENV}={show_src} is not a directory, skipped ts)")


def main() -> None:
    print("[1/3] Fetch FINRA + Shiller + Yahoo ^GSPC ...")
    df = build_series()
    print(f"      merged {len(df)} months: {df['ym'].iloc[0]} → {df['ym'].iloc[-1]}")
    print("[2/3] Compute 3-layer signals ...")
    sig = compute(df)
    print(f"      latest {sig['latest']['ym']}: gross ${sig['latest']['gross_B']:,.0f}B "
          f"({sig['latest']['mom_pct']:+.1f}% mom), net/gross {sig['latest']['net_gross']}")
    print("[3/3] Write artifacts ...")
    write_signals(sig)
    print(f"  → wrote {DATA_DIR/'leverage_signals.json'}")
    print("done.")


if __name__ == "__main__":
    main()
