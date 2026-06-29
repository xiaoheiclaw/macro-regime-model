"""Tests for scripts/leverage_monitor.py — the margin-debt fragility monitor.

Network fetches (FINRA / Shiller / Yahoo ^GSPC) are mocked, so these run
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


def _fake_finra(date_fmt="ym") -> pd.DataFrame:
    """Synthetic FINRA frame. `date_fmt` controls the month column representation
    so tests can exercise the real Excel parsing path, not just clean 'YYYY-MM':
      "ym"        → '2020-01'      (already canonical)
      "timestamp" → pd.Timestamp   (openpyxl returns these for date cells)
      "datestr"   → '2020-01-01'   (full ISO date string, end-of-month style)
    """
    yms = _yms()
    n = len(yms)
    debit = 500_000.0 + np.arange(n) * 5_000.0  # $M, monotonically rising
    dates = pd.to_datetime(yms)
    if date_fmt == "ym":
        ym_col = yms
    elif date_fmt == "timestamp":
        ym_col = list(dates)
    elif date_fmt == "datestr":
        ym_col = [d.strftime("%Y-%m-%d") for d in dates]
    else:
        raise ValueError(date_fmt)
    return pd.DataFrame({
        "ym": ym_col,
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


@pytest.mark.parametrize("date_fmt", ["ym", "timestamp", "datestr"])
def test_build_series_handles_excel_date_formats(monkeypatch, date_fmt):
    """FINRA month column may arrive as Timestamp or a full date string from the .xlsx —
    build_series must normalize and still align with Shiller/Yahoo (non-empty,
    correct latest month)."""
    monkeypatch.setattr(lm, "fetch_finra", lambda: _fake_finra(date_fmt))
    monkeypatch.setattr(lm, "fetch_shiller_sp", _fake_shiller)
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_fred)
    df = lm.build_series()
    assert not df.empty
    assert df["ym"].iloc[-1] == _yms()[-1]
    assert lm.compute(df)["latest"]["ym"] == _yms()[-1]


def test_to_ym_fails_fast_on_garbage():
    with pytest.raises(ValueError):
        lm._to_ym(pd.Series(["2020-01", "not-a-date", "2020-03"]))


def test_pct_reports_100_for_all_time_high():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert lm.pct(df, "x", 3.0) == 100.0
    assert lm.pct(df, "x", 1.0) <= 100.0


def test_mark_stale_degrades_old_snapshot():
    today = pd.Timestamp("2026-06-29")
    fresh = lm._mark_stale({"status": "red", "note": "x", "asof": "2026-06-26"}, today)
    assert fresh["stale"] is False and fresh["status"] == "red"
    old = lm._mark_stale({"status": "red", "note": "x", "asof": "2026-01-01"}, today)
    assert old["stale"] is True and old["status"] == "muted" and "STALE" in old["note"]


def test_chain_sp_requires_overlap():
    shiller = pd.Series([100.0], index=["2020-01"])
    ext = pd.Series([200.0], index=["2021-06"])  # no overlap
    with pytest.raises(ValueError):
        lm.chain_sp(shiller, ext)
