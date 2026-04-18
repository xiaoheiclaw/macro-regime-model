#!/usr/bin/env python3
"""
Phase 0b: fetch ALFRED real-time vintages for core macro series used in the
state panel. Caches to ~/.cache/alfred_realtime/.

Run once to seed (or to refresh) the cache. `build_state_features.py` then
resolves vintage-aware values at each asof date.

Usage:
  uv run python scripts/fetch_alfred_vintages.py            # fetch missing/stale
  uv run python scripts/fetch_alfred_vintages.py --force    # re-fetch all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.alfred import refresh_all, fetch_realtime  # type: ignore

# Core revision-prone macro series. Kept deliberately small; expand later if
# regime skill demands it.
VINTAGE_SERIES = {
    # Inflation
    "CPIAUCSL":  "Headline CPI (SA, index)",
    "CPILFESL":  "Core CPI (SA, index)",
    # Growth
    "INDPRO":    "Industrial Production (SA, index)",
    "PAYEMS":    "Nonfarm Payrolls (SA, thousands)",
    "UNRATE":    "Unemployment Rate (SA, %)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-fetch even if cache is fresh")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ALFRED vintage fetch ({len(VINTAGE_SERIES)} series)")
    print("=" * 60)

    for sid, desc in VINTAGE_SERIES.items():
        print(f"\n[{sid}] {desc}")
        df = fetch_realtime(sid, force=args.force)
        n_obs = df["observation_date"].nunique()
        n_vintages = df["realtime_start"].nunique()
        latest_obs = df["observation_date"].max().date()
        latest_vintage = df["realtime_start"].max().date()
        print(f"  {len(df):>6} rows | {n_obs} observation dates | "
              f"{n_vintages} vintages | latest obs {latest_obs} | "
              f"latest vintage {latest_vintage}")


if __name__ == "__main__":
    main()
