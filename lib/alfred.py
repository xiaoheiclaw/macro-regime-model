"""
ALFRED (Archival FRED) real-time vintage data client.

ALFRED stores every FRED data release. A single observation_date (e.g. CPI for
2024-08) can have many vintage rows — each reflecting what was published at
a specific realtime_start date. For honest backtests we MUST use the vintage
that was actually available at decision time.

API: https://alfred.stlouisfed.org/docs/api/
Output_type=1 (default): observations by real-time period.

Cache layout:
  ~/.cache/alfred_realtime/<series_id>.parquet
    columns: observation_date (datetime64), realtime_start (datetime64),
             realtime_end (datetime64), value (float64)
  ~/.cache/alfred_realtime/_manifest.json
    {series_id: {"fetched_at": ISO, "n_rows": int, "realtime_start": str}}

Public helpers:
  get_api_key()          → FRED/ALFRED API key from env or file
  fetch_realtime(...)    → full real-time history for a series (DataFrame)
  series_as_of(...)      → pd.Series of observations as known at a given date
  refresh_all(series_list, force=False)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

ALFRED_BASE = "https://api.stlouisfed.org/fred"
CACHE_DIR = Path(os.path.expanduser("~/.cache/alfred_realtime"))
CACHE_TTL_DAYS = 30
DEFAULT_REALTIME_START = "1990-01-01"
HTTP_TIMEOUT = 60.0
HTTP_RETRIES = 3


def get_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()
    for p in ("~/.fred_api_key", "~/.config/fred/api_key"):
        fp = os.path.expanduser(p)
        if os.path.exists(fp):
            return open(fp).read().strip()
    raise RuntimeError(
        "FRED API key not found. Set FRED_API_KEY env var or place key in ~/.fred_api_key"
    )


def _cache_path(series_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{series_id}.parquet"


def _manifest_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "_manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(json.dumps(m, indent=2))


def _is_cache_fresh(series_id: str) -> bool:
    manifest = _load_manifest()
    entry = manifest.get(series_id)
    if not entry:
        return False
    if not _cache_path(series_id).exists():
        return False
    fetched = datetime.fromisoformat(entry["fetched_at"])
    return (datetime.now() - fetched) < timedelta(days=CACHE_TTL_DAYS)


def fetch_realtime(
    series_id: str,
    realtime_start: str = DEFAULT_REALTIME_START,
    use_cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download the full real-time history of a FRED series.

    Returns a DataFrame with columns:
      observation_date, realtime_start, realtime_end, value
    Each row is one published value with its validity window [realtime_start,
    realtime_end]. realtime_end = "9999-12-31" means still current.
    """
    cache_path = _cache_path(series_id)
    if use_cache and not force and _is_cache_fresh(series_id):
        return pd.read_parquet(cache_path)

    api_key = get_api_key()
    url = f"{ALFRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start,
        "realtime_end": "9999-12-31",
    }

    last_err: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            break
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_err = e
            if attempt < HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"ALFRED fetch failed for {series_id}: {e}") from e

    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"No observations returned for {series_id}")

    df = pd.DataFrame(obs)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["observation_date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce")
    df = df[["observation_date", "realtime_start", "realtime_end", "value"]]
    df = df.sort_values(["observation_date", "realtime_start"]).reset_index(drop=True)

    # Persist
    df.to_parquet(cache_path)
    manifest = _load_manifest()
    manifest[series_id] = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(df),
        "realtime_start": realtime_start,
        "n_observation_dates": df["observation_date"].nunique(),
        "latest_observation": df["observation_date"].max().isoformat(),
    }
    _save_manifest(manifest)
    return df


def series_as_of(
    series_id: str,
    asof: str | pd.Timestamp,
    realtime_start: str = DEFAULT_REALTIME_START,
) -> pd.Series:
    """
    Return the series as it was known on `asof` (an analyst sitting at their
    desk on that date would see exactly these values).

    For each observation_date, takes the latest vintage whose realtime_start <=
    asof. Excludes observations not yet published at asof.
    """
    df = fetch_realtime(series_id, realtime_start=realtime_start)
    asof_ts = pd.Timestamp(asof)
    mask = df["realtime_start"] <= asof_ts
    if not mask.any():
        return pd.Series(dtype="float64", name=series_id)
    pub = df.loc[mask].copy()
    # Keep the last-published vintage per observation_date
    pub = pub.sort_values(["observation_date", "realtime_start"])
    snap = pub.groupby("observation_date", as_index=True).tail(1)
    snap = snap.set_index("observation_date").sort_index()
    return pd.Series(snap["value"].values, index=snap.index, name=series_id)


def series_panel_by_asof(
    series_id: str,
    asofs: pd.DatetimeIndex,
    realtime_start: str = DEFAULT_REALTIME_START,
) -> pd.DataFrame:
    """
    Build a panel where columns are asof dates and rows are observation dates.
    panel[obs_date, asof_date] = the value known at asof_date for obs_date.

    This is the right shape for computing vintage-aware rolling features
    without repeated lookups.
    """
    df = fetch_realtime(series_id, realtime_start=realtime_start)
    obs_dates = sorted(df["observation_date"].unique())
    panel = pd.DataFrame(index=pd.DatetimeIndex(obs_dates), columns=asofs, dtype="float64")
    panel.index.name = "observation_date"
    panel.columns.name = "asof_date"

    df_sorted = df.sort_values(["observation_date", "realtime_start"])
    for asof in asofs:
        asof_ts = pd.Timestamp(asof)
        mask = df_sorted["realtime_start"] <= asof_ts
        if not mask.any():
            continue
        pub = df_sorted.loc[mask]
        snap = pub.groupby("observation_date", as_index=True).tail(1)
        snap = snap.set_index("observation_date")
        panel.loc[snap.index, asof] = snap["value"].astype("float64").values
    return panel


def refresh_all(series_ids: list[str], force: bool = False) -> dict:
    """Fetch/refresh a list of series. Returns manifest snapshot."""
    for sid in series_ids:
        stale = not _is_cache_fresh(sid)
        if force or stale:
            print(f"  fetch {sid} (force={force}, stale={stale})")
            fetch_realtime(sid, force=True)
        else:
            print(f"  cache {sid} (fresh)")
    return _load_manifest()
