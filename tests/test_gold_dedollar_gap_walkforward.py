"""Tests for the walk-forward (expanding-window) re-calibration — PR #15.

Offline / synthetic by construction (no network). Covers the task spec:
  1. expanding calibration is ex-ante (no future function): truncating the future
     leaves every past value unchanged, for z AND percentile, include & exclude.
  2. warm-up gating: NaN until `min_periods` observations accrue.
  3. comparability with PR #14's full-sample口径: at the FINAL month the expanding
     (include-current) z/percentile equal the full-sample read (the structural fact
     that the headline current number is not where look-ahead hides).
  4. percentile math: monotone series, ties, include vs exclude convention.
  5. extreme reclassification + walk-forward conditional table contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.gold_dedollar_gap import forward_log_return, full_percentile, full_zscore
from lib.gold_dedollar_gap_walkforward import (
    current_walk_forward_reading,
    expanding_percentile,
    expanding_zscore,
    extreme_reclassification,
    full_percentile_series,
    walk_forward_calibration,
    walk_forward_conditional_table,
    warmup_sensitivity,
)


def _me_index(n, start="2010-01-31"):
    return pd.date_range(start, periods=n, freq=pd.offsets.MonthEnd())


def _close_nan(a: pd.Series, b: pd.Series, *, atol=1e-9):
    assert a.shape == b.shape
    av, bv = a.to_numpy(), b.to_numpy()
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    m = ~(np.isnan(av) | np.isnan(bv))
    np.testing.assert_allclose(av[m], bv[m], rtol=0, atol=atol)


# ── 1. ex-ante / no look-ahead ───────────────────────────────────────────
@pytest.mark.parametrize("exclude", [False, True])
def test_expanding_zscore_no_lookahead(exclude):
    n = 120
    rng = np.random.RandomState(1)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    full = expanding_zscore(s, min_periods=12, exclude_current=exclude)
    cut = 70
    trunc = expanding_zscore(s.iloc[:cut], min_periods=12, exclude_current=exclude)
    _close_nan(full.iloc[:cut], trunc)


@pytest.mark.parametrize("exclude", [False, True])
def test_expanding_percentile_no_lookahead(exclude):
    n = 120
    rng = np.random.RandomState(2)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    full = expanding_percentile(s, min_periods=12, exclude_current=exclude)
    cut = 80
    trunc = expanding_percentile(s.iloc[:cut], min_periods=12, exclude_current=exclude)
    _close_nan(full.iloc[:cut], trunc)


def test_expanding_uses_only_data_up_to_t_explicit():
    """A spike in the FUTURE must not change any earlier expanding value."""
    n = 60
    s = pd.Series(np.linspace(0.0, 1.0, n), index=_me_index(n))
    base_z = expanding_zscore(s, min_periods=6)
    base_p = expanding_percentile(s, min_periods=6)
    s2 = s.copy()
    s2.iloc[50] += 1000.0  # huge future shock
    z2 = expanding_zscore(s2, min_periods=6)
    p2 = expanding_percentile(s2, min_periods=6)
    _close_nan(base_z.iloc[:50], z2.iloc[:50])
    _close_nan(base_p.iloc[:50], p2.iloc[:50])


# ── 2. warm-up gating ─────────────────────────────────────────────────────
@pytest.mark.parametrize("warm", [12, 24, 36])
def test_warmup_gate_blanks_until_enough_history(warm):
    n = 80
    rng = np.random.RandomState(3)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    z = expanding_zscore(s, min_periods=warm)
    p = expanding_percentile(s, min_periods=warm)
    # include-current: the k-th observation (index warm-1) is the first defined one
    assert z.iloc[:warm - 1].isna().all()
    assert np.isfinite(z.iloc[warm - 1])
    assert p.iloc[:warm - 1].isna().all()
    assert np.isfinite(p.iloc[warm - 1])


def test_warmup_gate_counts_nonnan_only():
    """Warm-up counts OBSERVATIONS, not calendar rows: NaN gaps don't satisfy it."""
    n = 40
    s = pd.Series(np.arange(n, dtype=float), index=_me_index(n))
    s.iloc[:10] = np.nan  # first 10 months missing
    p = expanding_percentile(s, min_periods=12)
    # need 12 non-NaN obs; data starts at row 10 → first defined at row 10+12-1 = 21
    assert p.iloc[:21].isna().all()
    assert np.isfinite(p.iloc[21])


def test_warmup_rejects_too_small():
    s = pd.Series([1.0, 2.0, 3.0], index=_me_index(3))
    with pytest.raises(ValueError):
        expanding_zscore(s, min_periods=1)
    with pytest.raises(ValueError):
        expanding_percentile(s, min_periods=1)


# ── 3. comparability with PR #14 full-sample at the final month ───────────
def test_expanding_include_equals_full_sample_at_final_point():
    """The structural fact: at the LAST month the expanding (include-current)
    window IS the full sample, so z/percentile coincide with PR #14's read."""
    n = 100
    rng = np.random.RandomState(4)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    z_wf = expanding_zscore(s, min_periods=24, exclude_current=False)
    p_wf = expanding_percentile(s, min_periods=24, exclude_current=False)
    last = s.index[-1]
    np.testing.assert_allclose(z_wf.loc[last],
                               full_zscore(s).loc[last], atol=1e-9)
    np.testing.assert_allclose(p_wf.loc[last],
                               full_percentile(s, float(s.iloc[-1])), atol=1e-9)


def test_current_reading_incl_matches_full_excl_moves_by_1_over_n():
    n = 138
    rng = np.random.RandomState(5)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    rd = current_walk_forward_reading(s, warmup=24)
    # include-current == full-sample at asof
    np.testing.assert_allclose(rd.pct_wf_incl, rd.pct_full, atol=1e-9)
    np.testing.assert_allclose(rd.z_wf_incl, rd.z_full, atol=1e-9)
    # exclude-current drops just the newest point → within ~1/N of the full read
    assert abs(rd.pct_wf_excl - rd.pct_full) <= 1.5 / n
    assert rd.n_resid == n


# ── 4. percentile / z math ────────────────────────────────────────────────
def test_expanding_percentile_monotone_increasing_is_one():
    n = 30
    s = pd.Series(np.arange(n, dtype=float), index=_me_index(n))
    p_incl = expanding_percentile(s, min_periods=5, exclude_current=False)
    # each new point is the max of history-so-far → percentile 1.0 (include)
    np.testing.assert_allclose(p_incl.dropna().to_numpy(),
                               np.ones(p_incl.notna().sum()), atol=1e-12)
    # exclude-current on a strictly increasing series: today > all prior → 0 of them
    # are <= ... wait: <= prior means none are >= today, fraction(prior <= today)=1.0
    p_excl = expanding_percentile(s, min_periods=5, exclude_current=True)
    np.testing.assert_allclose(p_excl.dropna().to_numpy(),
                               np.ones(p_excl.notna().sum()), atol=1e-12)


def test_expanding_percentile_monotone_decreasing():
    n = 30
    s = pd.Series(np.arange(n, 0, -1, dtype=float), index=_me_index(n))
    p_incl = expanding_percentile(s, min_periods=5, exclude_current=False)
    # each new point is the min so far → only itself <= itself → 1/k
    defined = p_incl.dropna()
    k = np.arange(5, n + 1)  # baseline size at each defined row
    np.testing.assert_allclose(defined.to_numpy(), 1.0 / k, atol=1e-12)
    # exclude-current: today < all prior → fraction(prior <= today) = 0
    p_excl = expanding_percentile(s, min_periods=5, exclude_current=True)
    np.testing.assert_allclose(p_excl.dropna().to_numpy(),
                               np.zeros(p_excl.notna().sum()), atol=1e-12)


def test_full_percentile_series_matches_pointwise_full_percentile():
    n = 50
    rng = np.random.RandomState(6)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    fps = full_percentile_series(s)
    # equals PR #14 full_percentile evaluated at each value
    for ts in s.index[::7]:
        np.testing.assert_allclose(
            fps.loc[ts], full_percentile(s, float(s.loc[ts])), atol=1e-12)


def test_full_percentile_series_excludes_nonfinite_like_expanding():
    """full_percentile_series and the expanding calibrator must filter non-finite
    values the same way (both drop ±inf), so the two口径 stay consistent on a
    residual containing an inf (codex PR#15 R3 P3)."""
    n = 30
    rng = np.random.RandomState(77)
    s = pd.Series(rng.randn(n), index=_me_index(n))
    s.iloc[15] = np.inf
    fps = full_percentile_series(s)
    # the inf row itself is excluded (NaN), and the finite rows are ranked over the
    # finite-only baseline — matching expanding_percentile's isfinite filter.
    assert np.isnan(fps.iloc[15])
    finite_vals = s[np.isfinite(s)].to_numpy()
    ts = s.index[10]
    np.testing.assert_allclose(
        fps.loc[ts], float((finite_vals <= s.loc[ts]).mean()), atol=1e-12)


def test_current_reading_treats_inf_like_nan():
    """current_walk_forward_reading must filter ±inf the SAME way as the
    calibrators (codex PR#15 R4 P2): an inf residual is a non-observation, so an
    inf-injected series gives an IDENTICAL reading to a NaN-injected one — the inf
    is never treated as a real (huge) observation that would distort the rank."""
    n = 60
    rng = np.random.RandomState(123)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    with_inf = resid.copy(); with_inf.iloc[20] = np.inf
    with_nan = resid.copy(); with_nan.iloc[20] = np.nan
    a = current_walk_forward_reading(with_inf, warmup=24)
    b = current_walk_forward_reading(with_nan, warmup=24)
    assert a.n_resid == b.n_resid == int(np.isfinite(resid).sum()) - 1
    np.testing.assert_allclose(a.pct_full, b.pct_full, atol=1e-12)
    np.testing.assert_allclose(a.pct_wf_incl, b.pct_wf_incl, atol=1e-12)
    np.testing.assert_allclose(a.z_wf_excl, b.z_wf_excl, atol=1e-12)
    np.testing.assert_allclose(a.z_full, b.z_full, atol=1e-12)  # finite-only z_full
    assert np.isfinite(a.z_full)
    assert a.asof == b.asof


def test_summarize_excludes_inf_forward_returns():
    """_summarize must drop ±inf (e.g. a 0-price forward return) so mean/quantiles
    are not poisoned (codex PR#15 R4 P3)."""
    from lib.gold_dedollar_gap_walkforward import _summarize
    x = pd.Series([0.1, 0.2, np.inf, -np.inf, 0.3, np.nan])
    out = _summarize(x)
    assert out["n"] == 3
    np.testing.assert_allclose(out["mean"], 0.2, atol=1e-12)
    assert np.isfinite(out["p25"]) and np.isfinite(out["p75"])


def test_expanding_zscore_constant_window_is_nan():
    s = pd.Series(np.full(20, 3.0), index=_me_index(20))
    z = expanding_zscore(s, min_periods=5)
    assert z.dropna().empty  # std == 0 → no information


def test_expanding_percentile_handles_ties():
    s = pd.Series([1.0, 1.0, 1.0, 2.0], index=_me_index(4))
    p = expanding_percentile(s, min_periods=2, exclude_current=False)
    # at row1: two 1s, both <= 1 → 1.0; row3: value 2 is max of 4 → 1.0
    np.testing.assert_allclose(p.iloc[1], 1.0)
    np.testing.assert_allclose(p.iloc[3], 1.0)


# ── 5. trajectory frame + reclassification + conditional table ────────────
def test_walk_forward_calibration_frame_columns_and_gap():
    n = 80
    rng = np.random.RandomState(8)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    frame = walk_forward_calibration(resid, warmup=24)
    assert list(frame.columns) == [
        "resid", "z_full", "pct_full", "z_wf", "pct_wf", "pct_gap"]
    # pct_gap = pct_wf - pct_full where both defined
    both = frame.dropna(subset=["pct_wf", "pct_full"])
    np.testing.assert_allclose(
        both["pct_gap"].to_numpy(),
        (both["pct_wf"] - both["pct_full"]).to_numpy(), atol=1e-12)
    # at the final month the gap is ~0 (include-current == full-sample)
    np.testing.assert_allclose(frame["pct_gap"].iloc[-1], 0.0, atol=1e-9)


def test_extreme_reclassification_contract():
    """A residual that ramps then plateaus high: late months are full-sample
    extreme; early-in-history they were already unprecedented (ex-ante extreme too),
    so agreement should be high and warm-up months excluded from the rate."""
    n = 90
    base = np.concatenate([np.linspace(-1, 2.5, 60), np.full(30, 2.6)])
    resid = pd.Series(base, index=_me_index(n))
    rc = extreme_reclassification(resid, top_q=0.9, warmup=24)
    s = rc.summary
    assert s["n_full_extreme"] > 0
    assert 0.0 <= s["agreement_rate"] <= 1.0
    # evaluable + warmup == full_extreme count (no silent drops)
    assert s["n_evaluable"] + s["n_warmup"] == s["n_full_extreme"]
    # episodes frame has the per-month detail
    for col in ("date", "pct_full", "pct_wf", "wf_extreme", "in_warmup"):
        assert col in rc.episodes.columns


def test_extreme_reclassification_rejects_bad_q():
    resid = pd.Series(np.arange(30.0), index=_me_index(30))
    with pytest.raises(ValueError):
        extreme_reclassification(resid, top_q=0.0)


def test_walk_forward_conditional_table_groups_match_independent_calibrator():
    """The extreme/rest split must match what the (independently-tested) expanding
    calibrator + forward-observability rule produce — not just 'a row exists'
    (codex PR#15 P3). This catches an empty, miscounted, or leaking extreme set."""
    from lib.gold_dedollar_gap_walkforward import expanding_percentile
    n, warm, top_q, h = 90, 24, 0.9, 12
    rng = np.random.RandomState(10)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    resid.iloc[30:40] = 4.0  # an observable mid-sample extreme block
    price = pd.Series(np.exp(np.cumsum(rng.randn(n) * 0.02)), index=_me_index(n))

    tbl = walk_forward_conditional_table(
        resid, price, horizons=(h,), top_q=top_q, warmup=warm)
    assert set(tbl["regime"]) == {"extreme_high_wf", "rest"}

    # reconstruct the expected groups from the calibrator + the same observability
    pct = expanding_percentile(resid, min_periods=warm, exclude_current=False)
    fwd = forward_log_return(price, h)
    valid = pct.dropna().index.intersection(fwd.dropna().index)
    fv = pct.reindex(valid)
    exp_ext = fv[fv >= top_q].index
    exp_rest = fv[fv < top_q].index

    ext_n = int(tbl[tbl["regime"] == "extreme_high_wf"]["n"].iloc[0])
    rest_n = int(tbl[tbl["regime"] == "rest"]["n"].iloc[0])
    assert ext_n == len(exp_ext) > 0          # non-empty AND exactly matches
    assert rest_n == len(exp_rest)
    assert ext_n + rest_n == len(valid)       # exhaustive, no double-count/leak
    for col in ("n", "mean", "median", "p25", "p75", "hit"):
        assert col in tbl.columns


def test_walk_forward_conditional_table_rejects_bad_q():
    n = 40
    resid = pd.Series(np.arange(n, dtype=float), index=_me_index(n))
    price = pd.Series(np.exp(np.arange(n) * 0.01), index=_me_index(n))
    with pytest.raises(ValueError):
        walk_forward_conditional_table(resid, price, top_q=1.0)


# ── 6. monotonic / unique index (no-lookahead) guard ─────────────────────
def test_expanding_rejects_non_monotonic_index():
    """An out-of-order index would let a future month sit in an earlier position
    and leak into a past calibration → must raise (codex PR#15 P2)."""
    idx = _me_index(10)
    shuffled = idx[[0, 1, 5, 2, 3, 4, 6, 7, 8, 9]]  # not increasing
    s = pd.Series(np.arange(10.0), index=shuffled)
    with pytest.raises(ValueError):
        expanding_zscore(s, min_periods=3)
    with pytest.raises(ValueError):
        expanding_percentile(s, min_periods=3)


def test_calibrators_reject_duplicate_index():
    """A duplicate timestamp collapses the percentile baseline and breaks the asof
    read → every entry point must raise (codex PR#15 R2 P2)."""
    idx = _me_index(10)
    dup = idx[[0, 1, 2, 2, 3, 4, 5, 6, 7, 8]]  # sorted but index[3] duplicates [2]
    s = pd.Series(np.arange(10.0), index=dup)
    for fn in (expanding_zscore, expanding_percentile):
        with pytest.raises(ValueError):
            fn(s, min_periods=3)
    with pytest.raises(ValueError):
        full_percentile_series(s)
    with pytest.raises(ValueError):
        current_walk_forward_reading(s, warmup=3)


def test_warmup_sensitivity_table():
    n = 100
    rng = np.random.RandomState(12)
    resid = pd.Series(rng.randn(n), index=_me_index(n))
    tbl = warmup_sensitivity(resid, warmups=(12, 24, 36))
    assert list(tbl["warmup"]) == [12, 24, 36]
    # include-current current read is warm-up invariant (window ends at the same
    # final month, uses full history regardless of the gate)
    assert tbl["pct_wf_incl"].nunique() == 1
    # a larger warm-up starts the ex-ante trajectory later
    firsts = tbl.set_index("warmup")["first_wf_date"]
    assert firsts[12] <= firsts[24] <= firsts[36]


# ── 7. verdict guards (script-level adjudication) ─────────────────────────
def _load_script_module():
    """Load scripts/gold_dedollar_gap_walkforward.py by path (it is a script, not a
    package) so its module-level _verdict can be unit-tested."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "gold_dedollar_gap_walkforward.py")
    spec = importlib.util.spec_from_file_location("_wf_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verdict_not_robust_without_evaluable_extreme():
    """No evaluable historical extreme (warm-up swallows them / warmup > history) ⇒
    NO ex-ante evidence ⇒ must NOT return ROBUST (codex PR#15 P1)."""
    script = _load_script_module()
    n = 40
    rng = np.random.RandomState(99)
    # a non-degenerate residual with a defined current full-sample reading…
    resid = pd.Series(np.sort(rng.randn(n)), index=_me_index(n))
    # …but a warm-up larger than the whole history → every expanding rank is NaN,
    # so any full-sample extreme month is in warm-up → n_evaluable == 0.
    rd = current_walk_forward_reading(resid, warmup=n + 50)
    rc = extreme_reclassification(resid, top_q=0.9, warmup=n + 50)
    assert rc.summary["n_evaluable"] == 0
    assert not np.isfinite(rc.summary["agreement_rate"])
    label, _ = script._verdict(rd, rc, None)
    assert label != "ROBUST"
    assert label == "UNKNOWN"


def test_verdict_unknown_when_strict_excl_current_undefined():
    """If the verdict口径 (strict exclude-current) current pct is NaN — e.g.
    warmup == n_resid leaves include-current defined but exclude-current not — we
    must NOT fall through to ROBUST even when agreement is high (codex PR#15 R2 P2)."""
    from lib.gold_dedollar_gap_walkforward import (
        ExtremeReclassification, WalkForwardReading)
    script = _load_script_module()
    rd = WalkForwardReading(
        asof=pd.Timestamp("2026-05-31"), z_full=1.1, pct_full=0.9,
        z_wf_incl=1.1, pct_wf_incl=0.9, z_wf_excl=np.nan, pct_wf_excl=np.nan,
        n_resid=40, n_wf=40, warmup=40)
    rc = ExtremeReclassification(
        summary={"n_full_extreme": 10, "n_wf_extreme": 10, "n_warmup": 0,
                 "n_evaluable": 10, "agreement_rate": 1.0},
        episodes=pd.DataFrame())
    label, _ = script._verdict(rd, rc, None)
    assert label == "UNKNOWN"


def test_verdict_robust_when_current_stable_and_extremes_agree():
    """Sanity-check the positive branch still fires: current pct ≈ full-sample and
    all evaluable historical extremes agree ex-ante ⇒ ROBUST."""
    script = _load_script_module()
    # ramp then high plateau → late months full-sample extreme AND ex-ante extreme.
    base = np.concatenate([np.linspace(-1, 2.5, 70), np.full(40, 2.6)])
    resid = pd.Series(base, index=_me_index(110))
    rd = current_walk_forward_reading(resid, warmup=24)
    rc = extreme_reclassification(resid, top_q=0.9, warmup=24)
    assert rc.summary["n_evaluable"] > 0
    label, _ = script._verdict(rd, rc, None)
    assert label == "ROBUST"
