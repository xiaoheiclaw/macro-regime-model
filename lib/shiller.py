"""
Robert Shiller's stock market data (ie_data.xls) — S&P 500 composite with
dividends, earnings, CPI, and CAPE (P/E10) from 1871 monthly.

Primary source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
Mirror:        https://shillerdata.com/ (.xlsx)

Cache: ~/.cache/shiller/ie_data.<ext> (30-day TTL)

Returns a monthly DataFrame indexed by month-end with columns:
  cape              — Cyclically Adjusted P/E ratio (10-year real earnings)
  earnings_yield    — 1 / cape (as a fraction, e.g. 0.04 = 4%)
  sp500_price       — nominal S&P composite level
  dividend          — nominal dividend
  earnings          — nominal earnings (CAPE denominator inputs)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/.cache/shiller"))
CACHE_TTL_DAYS = 30

# shillerdata.com is the current source of truth (updated monthly). The Yale
# personal page stopped updating in 2023. We still keep Yale as last-resort.
SOURCES = [
    ("https://shillerdata.com/ie_data.xlsx",               "ie_data.xlsx", "openpyxl"),
    ("https://shillerdata.com/ie_data.xls",                "ie_data.xls",  "xlrd"),
    ("http://www.econ.yale.edu/~shiller/data/ie_data.xls", "ie_data.xls",  "xlrd"),
]

HTTP_TIMEOUT = 60.0


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=CACHE_TTL_DAYS)


def _fetch(url: str, dst: Path) -> bool:
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            r = client.get(url)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  fetch {url} failed: {e}")
        return False


def download(force: bool = False) -> tuple[Path, str]:
    """Download Shiller xls/xlsx to cache. Returns (path, engine)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache first
    for _, fname, engine in SOURCES:
        p = CACHE_DIR / fname
        if p.exists() and _is_fresh(p) and not force:
            return p, engine

    # Fetch fresh
    for url, fname, engine in SOURCES:
        dst = CACHE_DIR / fname
        print(f"  trying {url}")
        if _fetch(url, dst) and dst.stat().st_size > 10_000:
            return dst, engine

    raise RuntimeError(
        f"All Shiller sources failed. Tried: {[u for u, _, _ in SOURCES]}. "
        f"Manually download ie_data.xls to {CACHE_DIR}"
    )


def _parse_shiller_date(x) -> Optional[pd.Timestamp]:
    """Parse 1871.01, 1871.10, 2026.04 formats to month-end timestamps."""
    if pd.isna(x):
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    year = int(x)
    # Fractional part: .01 = Jan, .10 = Oct, .11 = Nov, .12 = Dec
    frac = round((x - year) * 100)
    if frac == 0:
        month = 10  # Shiller uses e.g. 1871.1 for Oct but sometimes shown as 1871.10
    else:
        month = int(frac)
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def load(force: bool = False) -> pd.DataFrame:
    path, engine = download(force=force)
    # 'Data' sheet. Metadata rows at top; data begins ~row 7.
    raw = pd.read_excel(path, sheet_name="Data", skiprows=7, engine=engine)
    # Clean up: drop empty columns / rows
    raw = raw.dropna(axis=1, how="all")
    raw.columns = [str(c).strip() for c in raw.columns]

    # Robust column detection
    date_col = next((c for c in raw.columns if c.lower() == "date"), None)
    cape_col = next((c for c in raw.columns
                     if "CAPE" in c.upper() or "P/E10" in c.upper()
                     or "P/E10" in c.replace(" ", "").upper()), None)
    price_col = next((c for c in raw.columns
                      if c in ("P", "Price") or c.lower().startswith("s&p")), None)
    div_col = next((c for c in raw.columns if c in ("D", "Dividend")), None)
    earn_col = next((c for c in raw.columns if c in ("E", "Earnings")), None)

    if date_col is None or cape_col is None:
        raise RuntimeError(
            f"Couldn't find Date/CAPE columns in Shiller data. "
            f"Columns: {list(raw.columns)[:20]}"
        )

    df = pd.DataFrame({
        "date": raw[date_col].apply(_parse_shiller_date),
        "cape": pd.to_numeric(raw[cape_col], errors="coerce"),
    })
    if price_col:
        df["sp500_price"] = pd.to_numeric(raw[price_col], errors="coerce")
    if div_col:
        df["dividend"] = pd.to_numeric(raw[div_col], errors="coerce")
    if earn_col:
        df["earnings"] = pd.to_numeric(raw[earn_col], errors="coerce")

    df = df.dropna(subset=["date", "cape"])
    df["earnings_yield"] = 1.0 / df["cape"]
    df = df.set_index("date").sort_index()
    return df
