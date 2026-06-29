"""Tests for scripts/leverage_monitor.py — the margin-debt fragility monitor.

Network fetches (FINRA / Shiller / FRED-via-Yahoo) are mocked, so these run
offline and deterministically. We exercise the pure transform path
(build_series → compute) and assert the output contract the Show Page relies on.
"""
from __future__ import annotations

import os
import importlib.util as _ilu

import numpy as np
import pandas as pd
import pytest

# Load the script as a module (it is not an importable package — same pattern
# used by tests/test_gold_trend_timing.py).
_spec = _ilu.spec_from_file_location(
    "leverage_monitor",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "leverage_monitor.py"),
)
lm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(lm)


# ── synthetic, network-free fixtures ───────────────────────────────────────
def _yms(start="2020-01", n=48):
    return [d.strftime("%Y-%m") for d in pd.date_range(start, periods=n, freq="MS")]


def _fake_finra() -> pd.DataFrame:
    yms = _yms()
    n = len(yms)
    debit = 500_000.0 + np.arange(n) * 5_000.0  # $M, monotonically rising
    return pd.DataFrame({
        "ym": yms,
        "debit_M": debit,
        "cash_credit_M": debit * 0.2,   # cash + margin credit < debit →
        "margin_credit_M": debit * 0.1,  # net/gross = 1 - 0.3 = 0.7 ∈ (0,1)
    })


def _fake_shiller() -> pd.Series:
    yms = _yms()
    p = 3_000.0 + np.arange(len(yms)) * 30.0
    return pd.Series(p, index=yms, name="P")


def _fake_fred() -> pd.Series:
    # Overlaps the tail of Shiller; chain_sp keeps a single level (identity here).
    yms = _yms()[-3:]
    return pd.Series(_fake_shiller().loc[yms].values, index=yms, name="close")


@pytest.fixture()
def signals(monkeypatch):
    monkeypatch.setattr(lm, "fetch_finra", _fake_finra)
    monkeypatch.setattr(lm, "fetch_shiller_sp", _fake_shiller)
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_fred)
    df = lm.build_series()
    return lm.compute(df)


# ── contract assertions ────────────────────────────────────────────────────
def test_latest_gross_positive(signals):
    assert signals["latest"]["gross_B"] > 0


def test_three_layers(signals):
    assert len(signals["layers"]) == 3


def test_chart_columns_equal_length(signals):
    chart = signals["chart"]
    assert len(chart["dt"]) == len(chart["gross"]) == len(chart["ratio"])
    assert len(chart["dt"]) > 0


def test_net_gross_in_unit_interval(signals):
    assert 0.0 <= signals["latest"]["net_gross"] <= 1.0


def test_ratio_percentile_is_a_percentile(signals):
    assert 0.0 <= signals["latest"]["ratio_percentile"] <= 100.0
