#!/usr/bin/env python3
"""
Phase 0: build state_features.parquet + availability_mask.parquet + feature_catalog.json

Consumes merged_data.csv (daily, wide format) and produces the monthly state panel
that all v2 layers (mask / regime / forecast) read from.

Contract (frozen at schema_version v2.1):
  state_features.parquet   index=date(monthly M), columns=F features, dtype=float64
  availability_mask.parquet same shape as state_features, dtype=bool (True if observed)
  feature_catalog.json     metadata per column:
     {
       "feature_name": {
         "start_date": "YYYY-MM-DD",
         "source": "yfinance|FRED|derived",
         "fill_policy": "ffill_5d|none|pca",
         "transformation": "level|log_return|diff|zscore|pca_component",
         "proxy": bool,
         "revision_aware": bool,
         "description": str
       }
     }

Output path: data/state_features.parquet + data/availability_mask.parquet + data/feature_catalog.json
"""
import os, sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR
from lib import alfred
from lib.schema import SCHEMA_VERSION, FEATURE_SET_VERSION, base_meta

# Vintage-aware features: (series_id, out_name, transform, release_lag_days, description)
# transform:
#   yoy_pct — (x_t / x_{t-12}) - 1
#   mom_pct — (x_t / x_{t-1}) - 1  (monthly percent change)
#   level   — use value directly
# release_lag_days: typical publication delay after reference month end.
#   CPI: ~15d; Core CPI: ~15d; IP: ~15d; PAYEMS: ~7d (first release); UNRATE: ~7d
VINTAGE_SPEC = [
    ("CPIAUCSL", "cpi_yoy",      "yoy_pct", 15, "Headline CPI YoY (%, ALFRED vintage)"),
    ("CPILFESL", "core_cpi_yoy", "yoy_pct", 15, "Core CPI YoY (%, ALFRED vintage)"),
    ("INDPRO",   "ip_yoy",       "yoy_pct", 15, "Industrial Production YoY (%, ALFRED vintage)"),
    ("PAYEMS",   "payroll_yoy",  "yoy_pct",  7, "Nonfarm Payrolls YoY (%, ALFRED vintage)"),
    ("PAYEMS",   "payroll_mom",  "mom_pct",  7, "Nonfarm Payrolls MoM (%, ALFRED vintage)"),
    ("UNRATE",   "unrate",       "level",    7, "Unemployment Rate (%, ALFRED vintage)"),
]
BUILT_AT = datetime.now().isoformat(timespec="seconds")

INPUT = Path(DATA_DIR) / "merged_data.csv"
OUT_STATE = Path(DATA_DIR) / "state_features.parquet"
OUT_MASK = Path(DATA_DIR) / "availability_mask.parquet"
OUT_CATALOG = Path(DATA_DIR) / "feature_catalog.json"


# ── Feature spec ────────────────────────────────────────
# (source_col, out_name, transform, fill, source_tag, proxy, revision_aware, description)
# transforms:
#   level        — month-end value
#   log_return   — log(p_t / p_{t-1}) monthly
#   diff         — first difference (for yields, spreads)
#   zscore_60m   — rolling 60-month zscore of level
FEATURE_SPEC = [
    # Risk assets — log return (price level not meaningful as regime state)
    ("SPX",         "spx_ret",      "log_return", "ffill_5d", "yfinance", False, False, "S&P 500 monthly log return"),
    ("HSI",         "hsi_ret",      "log_return", "ffill_5d", "yfinance", False, False, "Hang Seng monthly log return"),
    ("BTC",         "btc_ret",      "log_return", "ffill_5d", "yfinance", True,  False, "BTC monthly log return (history from 2014)"),
    # Commodities — log return
    ("WTI_crude",   "oil_ret",      "log_return", "ffill_5d", "yfinance", False, False, "WTI crude monthly log return"),
    ("NatGas",      "natgas_ret",   "log_return", "ffill_5d", "yfinance", True,  False, "NatGas monthly log return (proxy: continuous contract)"),
    ("Gold",        "gold_ret",     "log_return", "ffill_5d", "yfinance", False, False, "Gold monthly log return"),
    ("Silver",      "silver_ret",   "log_return", "ffill_5d", "yfinance", True,  False, "Silver monthly log return (proxy)"),
    ("Copper",      "copper_ret",   "log_return", "ffill_5d", "yfinance", True,  False, "Copper monthly log return (Dr. Copper proxy)"),
    ("BCOM",        "bcom_ret",     "log_return", "ffill_5d", "yfinance", True,  False, "BCOM commodity index monthly log return (proxy)"),
    # FX
    ("DXY",         "dxy_ret",      "log_return", "ffill_5d", "yfinance", False, False, "DXY monthly log return"),
    # Vol
    ("VIX",         "vix_level",    "level",      "ffill_5d", "yfinance", False, False, "VIX month-end level"),
    # Yield curve raw nodes (will feed PCA)
    ("US3M_yield",  "y3m",          "level",      "ffill_5d", "FRED",     False, True,  "3M Treasury yield (level, %)"),
    ("US2Y_yield",  "y2y",          "level",      "ffill_5d", "FRED",     False, True,  "2Y Treasury yield (level, %)"),
    ("US5Y_FRED",   "y5y",          "level",      "ffill_5d", "FRED",     False, True,  "5Y Treasury yield (level, %)"),
    ("US10Y_FRED",  "y10y",         "level",      "ffill_5d", "FRED",     False, True,  "10Y Treasury yield (level, %)"),
    ("US30Y_yield", "y30y",         "level",      "ffill_5d", "FRED",     False, True,  "30Y Treasury yield (level, %)"),
    # Yield diffs (monthly change)
    ("US10Y_FRED",  "y10y_diff",    "diff",       "ffill_5d", "FRED",     False, True,  "10Y yield monthly change (bps)"),
    # Inflation expectations
    ("BEI_5Y",      "bei_5y",       "level",      "ffill_5d", "FRED",     False, True,  "5Y breakeven inflation (%)"),
    ("BEI_10Y",     "bei_10y",      "level",      "ffill_5d", "FRED",     False, True,  "10Y breakeven inflation (%)"),
    # Credit — ICE BofA OAS (high-quality, post-2023 only due to ICE IP truncation)
    ("HY_OAS",      "hy_oas",       "level",      "ffill_5d", "FRED",     False, True,  "ICE BofA HY OAS (%) — history from 2023 only"),
    ("IG_OAS",      "ig_oas",       "level",      "ffill_5d", "FRED",     False, True,  "ICE BofA IG OAS (%) — history from 2023 only"),
    # Credit — Moody's seasoned yields (long history, for full-sample regimes)
    ("Moody_AAA",   "moody_aaa",    "level",      "ffill_5d", "FRED",     False, True,  "Moody's AAA yield (%, from 1919)"),
    ("Moody_BAA",   "moody_baa",    "level",      "ffill_5d", "FRED",     False, True,  "Moody's BAA yield (%, from 1919)"),
    # Funding / policy
    ("SOFR_rate",   "sofr",         "level",      "ffill_5d", "FRED",     False, True,  "SOFR rate (%, post-2018)"),
    ("USD_broad",   "usd_broad_ret","log_return", "ffill_5d", "FRED",     False, True,  "Trade-weighted USD (broad) monthly log return"),
]


def to_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily panel to month-end (last observation within month)."""
    daily.index = pd.to_datetime(daily.index)
    return daily.resample("ME").last()


def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return s
    if transform == "log_return":
        return np.log(s / s.shift(1))
    if transform == "diff":
        return s.diff()
    if transform == "zscore_60m":
        return (s - s.rolling(60, min_periods=24).mean()) / s.rolling(60, min_periods=24).std()
    raise ValueError(f"Unknown transform: {transform}")


def build_yield_curve_pca(monthly: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    """
    PCA on 5 yield nodes → 3 components (level / slope / curvature in economic terms).
    Fit on full history with >=24 observations required; NaN rows excluded from fit,
    then project all rows (including NaN ones produce NaN via mask).
    """
    from sklearn.decomposition import PCA

    nodes = ["y3m", "y2y", "y5y", "y10y", "y30y"]
    missing = [c for c in nodes if c not in monthly.columns]
    if missing:
        print(f"  ⚠ yield curve PCA skipped, missing nodes: {missing}")
        return pd.DataFrame(index=monthly.index)

    X = monthly[nodes].copy()
    fit_mask = X.notna().all(axis=1)
    if fit_mask.sum() < 24:
        print(f"  ⚠ yield curve PCA skipped, only {fit_mask.sum()} complete rows")
        return pd.DataFrame(index=monthly.index)

    pca = PCA(n_components=3)
    pca.fit(X.loc[fit_mask].values)
    # Sign convention: pc1 positively loaded on all levels; pc2 positive on long end; pc3 positive on mid
    signs = np.ones(3)
    if pca.components_[0].sum() < 0: signs[0] = -1
    if pca.components_[1, -1] < 0:   signs[1] = -1
    if pca.components_[2, 2] < 0:    signs[2] = -1

    projected = pca.transform(X.fillna(X.mean()).values) * signs
    projected[~fit_mask.values, :] = np.nan  # don't fabricate values for partial rows

    out = pd.DataFrame(projected, index=monthly.index,
                       columns=["curve_pc1_level", "curve_pc2_slope", "curve_pc3_curv"])

    evr = pca.explained_variance_ratio_
    print(f"  curve PCA explained variance: "
          f"pc1={evr[0]:.3f} pc2={evr[1]:.3f} pc3={evr[2]:.3f}")

    for i, name in enumerate(out.columns):
        catalog[name] = {
            "start_date": str(out[name].first_valid_index().date()) if out[name].notna().any() else None,
            "source": "derived",
            "fill_policy": "pca",
            "transformation": "pca_component",
            "proxy": False,
            "revision_aware": True,
            "description": f"Yield curve PCA component {i+1} (EVR={evr[i]:.3f}); nodes=3M/2Y/5Y/10Y/30Y",
        }
    return out


def _apply_macro_transform(snap: pd.Series, transform: str) -> pd.Series:
    if transform == "yoy_pct":
        return snap / snap.shift(12) - 1
    if transform == "mom_pct":
        return snap / snap.shift(1) - 1
    if transform == "level":
        return snap
    raise ValueError(f"Unknown macro transform: {transform}")


def add_vintage_features(state: pd.DataFrame, catalog: dict) -> None:
    """
    For each asof t in state.index, pull the vintage snapshot and apply transform.
    Use the LATEST observation available at t (typically 1-2 months lagged).
    This is what an analyst actually saw at month-end t.
    """
    print("\n[vintage] resolving ALFRED vintage features")
    asofs = state.index

    # Pre-load all caches once
    for series_id, *_ in VINTAGE_SPEC:
        alfred.fetch_realtime(series_id)  # ensures cache is hot

    for series_id, out_name, transform, release_lag_days, desc in VINTAGE_SPEC:
        values = np.full(len(asofs), np.nan, dtype="float64")
        latest_obs = [pd.NaT] * len(asofs)
        for i, asof in enumerate(asofs):
            snap = alfred.series_as_of(series_id, asof)
            if snap.empty:
                continue
            transformed = _apply_macro_transform(snap, transform).dropna()
            if transformed.empty:
                continue
            values[i] = float(transformed.iloc[-1])
            latest_obs[i] = transformed.index[-1]
        state[out_name] = values
        first_valid = pd.Series(values, index=asofs).first_valid_index()
        catalog[out_name] = {
            "start_date": str(first_valid.date()) if first_valid is not None else None,
            "source": "ALFRED",
            "fred_series": series_id,
            "fill_policy": "vintage_latest",
            "transformation": transform,
            "proxy": False,
            "revision_aware": True,
            "vintage_resolved": True,
            "release_lag_days": release_lag_days,
            "description": desc,
        }
        non_null = int(np.isfinite(values).sum())
        print(f"  {out_name:<14} ← {series_id:<10} {transform:<8} "
              f"first={first_valid.date() if first_valid is not None else 'NA'} "
              f"n_obs={non_null}")


def main() -> None:
    print("=" * 60)
    print(f"Build state_features | schema={SCHEMA_VERSION} | {BUILT_AT}")
    print("=" * 60)

    if not INPUT.exists():
        sys.exit(f"missing input: {INPUT}. Run data_pipeline.py first.")

    daily = pd.read_csv(INPUT, index_col=0, parse_dates=True)
    print(f"Loaded daily merged_data: {len(daily)} rows × {len(daily.columns)} cols")

    monthly_raw = to_month_end(daily)
    print(f"Resampled to month-end: {len(monthly_raw)} rows")

    state = pd.DataFrame(index=monthly_raw.index)
    catalog: dict = {}

    for src_col, out_name, transform, fill, source, proxy, rev_aware, desc in FEATURE_SPEC:
        if src_col not in monthly_raw.columns:
            print(f"  ⚠ {out_name}: source col '{src_col}' not in merged_data, skipped")
            continue
        s = apply_transform(monthly_raw[src_col], transform)
        state[out_name] = s
        start = s.first_valid_index()
        catalog[out_name] = {
            "start_date": str(start.date()) if start is not None else None,
            "source": source,
            "fill_policy": fill,
            "transformation": transform,
            "proxy": proxy,
            "revision_aware": rev_aware,
            "description": desc,
        }

    # Yield curve PCA (uses the y3m/y2y/y5y/y10y/y30y already in state)
    pca_df = build_yield_curve_pca(state, catalog)
    if len(pca_df.columns):
        state = state.join(pca_df)

    # Vintage-aware macro features (growth + inflation, resolved point-in-time)
    add_vintage_features(state, catalog)

    # Derived spreads
    if {"y10y", "y2y"}.issubset(state.columns):
        state["yc_2s10s"] = state["y10y"] - state["y2y"]
        catalog["yc_2s10s"] = {
            "start_date": str(state["yc_2s10s"].first_valid_index().date()) if state["yc_2s10s"].notna().any() else None,
            "source": "derived", "fill_policy": "none",
            "transformation": "level", "proxy": False, "revision_aware": True,
            "description": "2s10s yield curve spread (10Y - 2Y)",
        }
    if {"y10y", "bei_10y"}.issubset(state.columns):
        state["real_y10y"] = state["y10y"] - state["bei_10y"]
        catalog["real_y10y"] = {
            "start_date": str(state["real_y10y"].first_valid_index().date()) if state["real_y10y"].notna().any() else None,
            "source": "derived", "fill_policy": "none",
            "transformation": "level", "proxy": False, "revision_aware": True,
            "description": "10Y real yield (nominal - BEI10)",
        }

    # Commodity YoY from existing *_ret (12-month cumulative log return)
    for base in ("oil", "copper", "bcom", "gold"):
        ret_col = f"{base}_ret"
        if ret_col not in state.columns:
            continue
        yoy_col = f"{base}_yoy"
        state[yoy_col] = state[ret_col].rolling(12).sum()
        first = state[yoy_col].first_valid_index()
        catalog[yoy_col] = {
            "start_date": str(first.date()) if first is not None else None,
            "source": "derived",
            "fill_policy": "none",
            "transformation": "log_yoy_12m",
            "proxy": catalog[ret_col].get("proxy", False),
            "revision_aware": False,
            "description": f"{ret_col} 12-month cumulative log return (YoY)",
        }

    # Shiller CAPE (valuation) — monthly, ffill for post-Shiller-last-obs staleness
    try:
        from lib import shiller
        cape_df = shiller.load()
        last_shiller = cape_df.index.max()
        for shiller_col, out_col, desc in [
            ("cape", "shiller_cape",
             "Shiller Cyclically Adjusted P/E (10-yr real earnings)"),
            ("earnings_yield", "shiller_earnings_yield",
             "Shiller earnings yield = 1/CAPE (fraction)"),
        ]:
            if shiller_col not in cape_df.columns:
                continue
            series = cape_df[shiller_col].reindex(state.index, method="ffill")
            state[out_col] = series.values
            first_valid = pd.Series(state[out_col].values, index=state.index).first_valid_index()
            stale_count = int((state.index > last_shiller).sum())
            catalog[out_col] = {
                "start_date": str(first_valid.date()) if first_valid is not None else None,
                "source": "Shiller_ie_data",
                "fill_policy": "ffill_from_shiller",
                "transformation": shiller_col,
                "proxy": False,
                "revision_aware": False,
                "stale_after": str(last_shiller.date()),
                "description": f"{desc} (last Shiller obs: {last_shiller.date()}; {stale_count} months ffilled)",
            }
        print(f"  [shiller] loaded {len(cape_df)} monthly obs, "
              f"last {last_shiller.date()}, ffilling {stale_count} months into state")
    except Exception as e:
        print(f"  ⚠ Shiller CAPE not loaded: {e}")

    # Synthetic bond return from yield change (duration-based proxy so
    # allocation can treat "bonds" as a return asset alongside equities).
    if "y10y" in state.columns:
        state["bond_ret"] = -8.0 * state["y10y"].diff() / 100.0
        first_valid = state["bond_ret"].first_valid_index()
        catalog["bond_ret"] = {
            "start_date": str(first_valid.date()) if first_valid is not None else None,
            "source": "derived",
            "fill_policy": "none",
            "transformation": "duration_proxy",
            "proxy": True,
            "revision_aware": True,
            "description": "Synthetic 10Y bond log return (−8 × Δy10y / 100, duration proxy)",
        }

    # Moody's credit spreads (long-history credit risk proxies)
    if {"moody_baa", "moody_aaa"}.issubset(state.columns):
        state["moody_baa_aaa"] = state["moody_baa"] - state["moody_aaa"]
        catalog["moody_baa_aaa"] = {
            "start_date": str(state["moody_baa_aaa"].first_valid_index().date()) if state["moody_baa_aaa"].notna().any() else None,
            "source": "derived", "fill_policy": "none",
            "transformation": "level", "proxy": False, "revision_aware": True,
            "description": "Moody's BAA - AAA credit risk premium (%, long history)",
        }
    if {"moody_baa", "y10y"}.issubset(state.columns):
        state["moody_baa_10y"] = state["moody_baa"] - state["y10y"]
        catalog["moody_baa_10y"] = {
            "start_date": str(state["moody_baa_10y"].first_valid_index().date()) if state["moody_baa_10y"].notna().any() else None,
            "source": "derived", "fill_policy": "none",
            "transformation": "level", "proxy": False, "revision_aware": True,
            "description": "Moody's BAA - 10Y Treasury (%, corporate over Treasury spread)",
        }

    # Availability mask = True where observed (before any downstream imputation)
    mask = state.notna()

    # Metadata
    meta = base_meta(
        layer="state_features",
        data_asof=str(state.index.max().date()) if len(state) else None,
    )
    meta.update({
        "index_type": "DatetimeIndex (month-end)",
        "n_rows": len(state),
        "n_features": len(state.columns),
        "feature_order": list(state.columns),
        "features": catalog,
        "notes": [
            "Return features use log returns on month-end prices.",
            "Yield/spread features are in percent, not bps; yX_diff in pct points.",
            "PCA components signed: pc1=level(+), pc2=slope(long+), pc3=curvature(mid+).",
            "'revision_aware' flags FRED series with vintage histories.",
            "'vintage_resolved=True' means the value at each row uses ALFRED point-in-time lookup — the data an analyst saw at that month-end, not today's revised value.",
            "Other FRED series (y*, bei_*, *_oas, sofr) are revision_aware=True but vintage_resolved=False: revisions are small for yields/spreads so as-is is acceptable.",
            "'proxy' flags series using continuous futures or composite proxies vs professional indices.",
        ],
    })

    state.to_parquet(OUT_STATE)
    mask.to_parquet(OUT_MASK)
    with open(OUT_CATALOG, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✓ state_features  → {OUT_STATE}  ({len(state)} × {len(state.columns)})")
    print(f"✓ availability    → {OUT_MASK}")
    print(f"✓ catalog         → {OUT_CATALOG}")

    # Per-feature availability report
    print(f"\n{'Feature':<22} {'Start':<12} {'End':<12} {'N':>5} {'%':>5}")
    print("-" * 60)
    for col in state.columns:
        non_na = mask[col].sum()
        pct = non_na / len(state) * 100
        start = state[col].first_valid_index()
        end = state[col].last_valid_index()
        print(f"{col:<22} "
              f"{str(start.date()) if start is not None else 'NA':<12} "
              f"{str(end.date()) if end is not None else 'NA':<12} "
              f"{non_na:>5} {pct:>4.0f}%")


if __name__ == "__main__":
    main()
