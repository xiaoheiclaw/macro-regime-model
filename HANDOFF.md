# HANDOFF — lab-macro-regime

## Current state (2026-04-18)

v2 pipeline complete Stage A + Stage B.0. 17 commits since v1 baseline.

### Pipeline layers (all real, no stubs)

```
mask             MI per-asset × feature × horizon
global_template  K-Means K=4 macro regimes
asset_regime     GMM K=3 per-asset states
template_map     joint conditional distribution (asof-aware)
forecast         KAF + joint-Frobenius regime filter (leak-free)
allocation       MV + SP-CVaR on joint scenarios
```

All 4 rounds of codex review passed. Known issues deferred:
- empirical-joint prior → Phase 3b
- α=3.0 CRPS calibration → Stage B
- schema effective_date elevation → post Stage B
- allocation daily rolling rebalance → Stage B setup
- global_template + asset_regime expanding-window refit → Stage B.1

### Stage B findings (10-year backtest, after Phase 3b series)

Latest overall skill: **-1.1%** (v2 CRPS 0.0957 vs Gaussian 0.0947),
reproducible under fixed seed.

**Honest attribution of the -2.0% → -1.1% improvement** (codex round 5):

- **KAF core remains around -2%**. α / γ / kernel tweaks each deliver
  ≤0.3pp — they don't materially change analog ranking because
  d_state already dominates, d_regime is partially redundant with
  d_state (regime layers derived from state features), and d_traj
  adds tiny marginal information at the chosen tether length.
- The ~0.9pp headline win comes from **switching BTC/Bond out of KAF
  entirely** into a Gaussian parametric fallback that matches the
  benchmark family (same 120m rolling window). This is a real CRPS
  win, but it's "asset-level switch to benchmark", not "KAF got
  better". If you read the headline as "v2 baseline improved", you
  are over-attributing.

**Stress-period breakdown** (exploratory only — post-hoc named
windows, not a validation):
| window | dates | n | skill |
|---|---|---|---|
| Calm expansion | 2015-01 → 2019-12 | 2564 | -7.8% |
| COVID | 2020-01 → 2021-06 | 748 | +4.8% |
| Inflation shock | 2022-01 → 2023-06 | 748 | +3.3% |
| Recent | 2023-07 → 2024-12 | 748 | +5.7% |

Hypothesis-generating: v2 appears to beat Gaussian in every non-calm
window. To turn this into evidence, replace post-hoc window naming
with rule-based segmentation (VIX quantile / NBER dates / rolling
24m skill curve).

**Phase 3b trajectory (reproducible, fixed seed)**:
  Stage A baseline:       -2.0%
  + α calibration:        -2.0%  (null)
  + γ=0.5 tethering:      -1.8%  (+0.2pp)
  + kernel variants:      -2.0%  (null; RBF neutral, Mahal worse)
  + BTC parametric:       -1.3%  (+0.7pp, real win; BTC -7% → -3%)
  + Bond parametric:      -1.1%  (+0.2pp, real but modest)

**Downstream allocation evaluation (Phase 3b.8-3b.10, with stat rigor)**:

  Raw outcomes (overlapping monthly obs — Sharpe SE ≈ 0.11):
    Method (6m)          AnnRet   AnnVol   Sharpe
    60/40                 5.8%    8.3%    0.70
    MVN SP-CVaR           5.1%    5.4%    0.95
    v2 SP-CVaR            6.7%    6.1%    **1.10**
    v2 MV BL             34.4%   46.1%    0.75    ⚠ Gold/BTC cap-bound

  **Paired analysis on COMMON asofs + block-bootstrap 95% CI**
  (block length = horizon; this is the honest test codex Round-6
  demanded):

  v2 SP-CVaR vs MVN SP-CVaR (99 common asofs):
    h=1m:  ann +1.63%, CI [−0.04, +0.32]  not sig
    h=3m:  ann +2.37%, CI [+0.24, +0.97]  **SIGNIFICANT**
    h=6m:  ann +2.33%, CI [+0.48, +1.80]  **SIGNIFICANT**
    h=12m: ann +2.60%, CI [+1.04, +3.38]  **SIGNIFICANT**

  On same asofs, same optimizer, same 50% cap — v2's joint-structure
  scenarios deliver +2.3-2.6% annualized realized return over MVN
  scenarios at 3/6/12m, block-bootstrap CI excludes zero. **First
  statistically significant v2 win.**

  v2 SP-CVaR vs 60/40 (119 obs):
    All horizons: paired return diff CI covers zero.
    → Sharpe 1.10 vs 0.70 advantage comes from vol reduction, not
      mean return premium. That's a valid portfolio property but
      doesn't pass paired-mean significance test.

**Joint Energy Score (Phase 3b.7) — NOT statistically significant**:
  Paired v2 vs MVN, block-bootstrap 95% CI:
    h=1m:  +0.0014 [−0.0003, +0.0030]  not sig
    h=3m:  +0.0029 [−0.0007, +0.0062]  marginal
    h=6m:  +0.0024 [−0.0033, +0.0081]  not sig
    h=12m: +0.0039 [−0.0032, +0.0142]  not sig
  All point estimates positive but 0 inside every CI. Earlier
  "+0.6-1.2% skill" claim was noise, not signal. Retracted.

  Interesting: the joint structure appears to manifest at the
  ALLOCATION layer (significant paired return diff) but not at the
  raw multivariate Energy Score layer. Possible explanation: the CVaR
  optimizer's tail focus amplifies asymmetric joint structure that
  symmetric Energy Score averages out.

**AR(1) benchmark (Phase 3b.7, stronger than unconditional Gaussian)**:
  Many Gaussian "wins" are AR(1) losses — AR(1) captures mean-reverting/
  momentum dynamics Gaussian misses. Under both benchmarks:
    Gold @ 12m:   +7.6% vs AR(1), +16.7% vs Gaussian  ← real v2 win
    Silver @ 12m: +4.8% vs AR(1), +7.5% vs Gaussian   ← real v2 win
    Oil short-h:  +0.3 to +3.2%                       ← marginal
    BTC:          -24.7% vs AR(1), -5.3% vs Gaussian  ← AR(1) captures momentum
    Bond:         -13.6% vs AR(1), -1.2% vs Gaussian  ← AR(1) captures mean reversion
  Honest recalibration: Gold and Silver are the only assets that
  genuinely beat both benchmarks.

**Remaining limitations**:
- Joint structure broken at VALUE level for BTC/Bond (parametric);
  joint for other 8 assets is intact and shown positive by ES. Not
  tested downstream in allocation.
- Stress-period breakdown windows are hand-named; do not cite as
  evidence, only as hypothesis. VIX threshold switch (3b.6) didn't
  help — calm vs KAF-win isn't cleanly separable by VIX.

**Path forward**:
- Regime-conditional KAF/Gaussian switching (use Gaussian when
  global_template probs concentrated, KAF when diffuse) — tests the
  stress-vs-calm hypothesis within the model itself
- Energy Score / multivariate CRPS to validate joint scenarios
- VAR(1) + regime-conditional historical benchmarks for absolute
  assessment
- v2 as tail-risk specialist framing (use for CVaR allocation,
  Gaussian for return point forecasts)

PIT: most assets ~0.5 (calibrated). Bonds ~0.40 (left bias persists).

### Artifacts

Code in git: scripts/{build_state_features,feature_mask,global_template,asset_regime,template_map,forecast,v2_allocation,v2_pipeline,backtest}.py, lib/{alfred,shiller,metrics,schema,paths}.py.

Runtime outputs (not git-tracked, regenerable):
- `data/state_features.parquet` + `data/feature_catalog.json`
- `data/v2/*.parquet` + `data/v2/backtest/results_2015-01_2024-12.parquet`
- `analysis/v2/*.md` incl. `backtest_2015-01_2024-12.md`, `allocation_2026-04-30.md`
- ALFRED cache: `~/.cache/alfred_realtime/`
- Shiller cache: `~/.cache/shiller/`

### Stage B.1 + Phase 3b priorities (from backtest findings)

1. **Expanding-window regime refit** per asof (fixes remaining leakage; Stage B.1)
2. **BTC handling**: asset-specific approach (short history demands different
   analog pool policy; consider excluding from KAF and falling back to
   pure volatility-regime model)
3. **Bond regime features**: current STATE_FEATURES heavy on rate-curve
   features that correlate with bonds themselves; consider different
   feature set for bond_ret prediction
4. **α calibration**: current α=3.0 empirically set; CRPS-optimal α likely
   different per asset class (precious metals benefit most; risk assets less)
5. **Stress-period breakdown** as standard backtest output (2008 Q3-Q4,
   2020 Q1-Q2, 2022)
6. **Additional benchmarks**: VAR(1), regime-conditional empirical, naive
   analog (codex's full benchmark list)

### Prior (v1) context

- v1 HANDOFF preserved below for historical reference
- v1 scripts (regime_switching.py, black_litterman.py, stochastic_programming.py,
  kalman_betas.py, wasserstein_regime.py, daily_dashboard.py) continue to run
  separately; v2 does not replace v1 yet (design decision: parallel run until
  Stage B validation complete)

---

## Prior (v1, 2026-04-10)

### What was completed

**Dashboard output compacted (commit `5144d61`)**
- Rewrote markdown generation in `scripts/daily_dashboard.py` from ~520 lines to ~70 lines
- New sections: 水位 table, 原理 (6 lines), 体制信号 (3-row table), 配置建议 (unified table with Market/BL-Sharpe/SP-CVaR), 关注催化, 反馈闭环, 局限
- Removed: per-Phase verbose methodology, historical beta trajectory tables, four-strategy detailed comparisons, "今日解读" long text, "三层裁判交叉验证", hardcoded regime performance table

**Bug fixes in same commit**
- Fixed BL weight key mapping: BL output uses display names ("S&P 500", "10Y Treasury", "Bitcoin") but allocation table used column names ("SPX", "US10Y_yield", "BTC"). Added `_bl_name_map` dict to translate
- Fixed smart quote syntax errors in Chinese strings (curly quotes inside double-quoted Python strings)
- Added BEI 5Y and 2s10s yield curve to water level table
- Now loads SP-CVaR weights directly from `data/stochastic_prog_weights.csv` instead of only extracting `current_regime`

**File naming standardized**
- Changed `today` format from `%Y%m%d` to `%Y-%m-%d` (single line change, all output paths use this variable)
- Renamed existing files: `*_20260407.md` → `*_2026-04-07.md`
- Cleaned up 10 old/inconsistent files

### Known hook issue

- `~/.claude/hooks/dangerous-cmd-guard.sh` line 25: `'curl.*| *sh'` with `grep -E` parses as `(curl.*)` OR `( *sh)`, blocking any command containing "sh" (like "dashboard"). Fix: escape pipe to `\|`.
