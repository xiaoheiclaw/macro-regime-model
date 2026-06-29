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


def _fake_yahoo_sp500() -> pd.Series:
    # Yahoo ^GSPC stand-in; overlaps the tail of Shiller so chain_sp anchors cleanly.
    yms = _yms()[-3:]
    return pd.Series(_fake_shiller().loc[yms].values, index=yms, name="close")


@pytest.fixture()
def signals(monkeypatch):
    monkeypatch.setattr(lm, "fetch_finra", _fake_finra)
    monkeypatch.setattr(lm, "fetch_shiller_sp", _fake_shiller)
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_yahoo_sp500)
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
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_yahoo_sp500)
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


def test_mark_stale_handles_missing_note():
    old = lm._mark_stale({"status": "red", "asof": "2026-01-01"}, pd.Timestamp("2026-06-29"))
    assert old["stale"] is True and old["status"] == "muted"
    assert old["note"].startswith("⏳STALE")  # no KeyError, no trailing space


def test_compute_rejects_short_frame():
    df = pd.DataFrame({"ym": ["2020-01"], "dt": pd.to_datetime(["2020-01"]),
                       "gross_B": [1.0], "net_gross": [0.5], "ratio": [0.1],
                       "net_B": [0.5], "sp": [10.0]})
    with pytest.raises(ValueError):
        lm.compute(df)


@pytest.mark.parametrize("payload", [
    {"chart": {"error": {"code": "Not Found"}, "result": None}},
    {"chart": {"error": None, "result": None}},
    {"chart": {"error": None, "result": [
        {"timestamp": [1, 2, 3], "indicators": {"quote": [{"close": [10.0, 11.0]}]}}]}},  # len mismatch
])
def test_fetch_sp500_validates_yahoo_response(monkeypatch, payload):
    import json as _json
    monkeypatch.setattr(lm, "_get", lambda *a, **k: _json.dumps(payload).encode())
    with pytest.raises(ValueError):
        lm.fetch_sp500_monthly()


def test_fetch_sp500_parses_valid_yahoo_response(monkeypatch):
    import json as _json
    # 2020-01-15 and 2020-02-15 UTC midday timestamps → ym 2020-01 / 2020-02.
    payload = {"chart": {"error": None, "result": [{
        "timestamp": [1579089600, 1581768000],
        "indicators": {"quote": [{"close": [3225.5, 3225.5 * 1.02]}]},
    }]}}
    monkeypatch.setattr(lm, "_get", lambda *a, **k: _json.dumps(payload).encode())
    s = lm.fetch_sp500_monthly()
    assert list(s.index) == ["2020-01", "2020-02"]
    assert s.loc["2020-01"] == 3225.5


def test_record_high_streak():
    assert lm._record_high_streak(pd.Series([1.0, 2.0, 3.0, 4.0])) == 4
    assert lm._record_high_streak(pd.Series([1.0, 5.0, 2.0, 3.0])) == 0  # latest not a record
    assert lm._record_high_streak(pd.Series([5.0, 1.0, 2.0, 6.0])) == 1


def test_build_series_rejects_missing_current_credit(monkeypatch):
    def _finra_missing_current():
        df = _fake_finra("ym")
        df.loc[df.index[-1], "cash_credit_M"] = np.nan  # latest month incomplete
        return df
    monkeypatch.setattr(lm, "fetch_finra", _finra_missing_current)
    monkeypatch.setattr(lm, "fetch_shiller_sp", _fake_shiller)
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_yahoo_sp500)
    with pytest.raises(ValueError):
        lm.build_series()


def test_mom_decelerating_not_labeled_accelerating(signals):
    # The fixture rises by a constant absolute amount → mom % is decelerating.
    mom_item = [i for i in signals["layers"][0]["items"] if "mom" in i["label"]][0]
    assert "加速" not in mom_item["note"]
    assert mom_item["status"] != "red"


def test_yoy_muted_when_insufficient_history(monkeypatch):
    monkeypatch.setattr(lm, "fetch_finra", lambda: _fake_finra("ym").iloc[:6].copy())
    monkeypatch.setattr(lm, "fetch_shiller_sp", _fake_shiller)
    monkeypatch.setattr(lm, "fetch_sp500_monthly", _fake_yahoo_sp500)
    sig = lm.compute(lm.build_series())
    yoy_item = [i for i in sig["layers"][0]["items"] if "yoy" in i["label"]][0]
    assert yoy_item["status"] == "muted" and sig["latest"]["yoy_pct"] is None


def test_finra_event_is_dynamic_and_past_filtered(signals):
    data_events = [e for e in signals["events"] if e["type"] == "data"]
    assert data_events, "expected a derived FINRA data event"
    pend = pd.Period(signals["latest"]["ym"], freq="M") + 1
    assert f"{pend.year}-{pend.month:02d}" in data_events[0]["label"]


def test_select_finra_columns_strict_vs_fallback():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})  # no recognizable headers
    with pytest.raises(ValueError):
        lm._select_finra_columns(df, strict=True)
    out = lm._select_finra_columns(df, strict=False)
    assert list(out.columns) == ["ym", "debit_M", "cash_credit_M", "margin_credit_M"]


def test_layer_status_aggregation():
    assert lm._layer_status([{"status": "green"}, {"status": "red"}, {"status": "muted"}]) == "red"
    assert lm._layer_status([{"status": "green"}, {"status": "amber"}]) == "amber"
    assert lm._layer_status([{"status": "green"}, {"status": "muted"}]) == "green"
    assert lm._layer_status([{"status": "muted"}, {"status": "muted"}]) == "muted"


def test_layer_status_matches_items(signals):
    for layer in signals["layers"]:
        assert layer["status"] == lm._layer_status(layer["items"])


def test_finra_monitor_text_tracks_latest_month(signals):
    # First monitor name should reference latest_month + 1, not a hardcoded "6月".
    latest = pd.Period(signals["latest"]["ym"], freq="M")
    pend = latest + 1
    assert f"{pend.year}-{pend.month:02d}" in signals["monitors"][0]["name"]


def test_select_finra_columns_by_header_keyword():
    df = pd.DataFrame({
        "Year-Month": ["2020-01"],
        "Note": ["x"],                                   # inserted junk column
        "Debit Balances in Margin Accounts": [500000],
        "Free Credit Balances in Cash Accounts": [100000],
        "Free Credit Balances in Margin Accounts": [50000],
    })
    out = lm._select_finra_columns(df)
    assert list(out.columns) == ["ym", "debit_M", "cash_credit_M", "margin_credit_M"]
    assert out["ym"].iloc[0] == "2020-01"
    assert out["debit_M"].iloc[0] == 500000
    assert out["cash_credit_M"].iloc[0] == 100000


@pytest.mark.filterwarnings("ignore:Could not infer format")
def test_fetch_finra_rejects_layout_drift(monkeypatch):
    import io as _io
    # A sheet whose first 4 non-empty columns are NOT (date, debit, cash, margin):
    # a text-blurb column has shifted the numeric columns out of place.
    bad = pd.DataFrame({
        "blurb": ["see note"] * 5,
        "debit": [1, 2, 3, 4, 5],
        "cash": [0, 0, 0, 0, 0],
        "margin": [0, 0, 0, 0, 0],
    })
    buf = _io.BytesIO()
    bad.to_excel(buf, index=False)
    monkeypatch.setattr(lm, "_get", lambda *a, **k: buf.getvalue())
    with pytest.raises(ValueError):
        lm.fetch_finra()


def test_write_signals_skips_non_directory_show_src(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lm, "DATA_DIR", tmp_path / "data")
    not_a_dir = tmp_path / "afile"
    not_a_dir.write_text("x")
    lm.write_signals({"ok": 1}, show_src=not_a_dir)  # must NOT raise
    assert (tmp_path / "data" / "leverage_signals.json").exists()
