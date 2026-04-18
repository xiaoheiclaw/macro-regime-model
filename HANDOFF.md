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

### Stage B.0 findings (10-year backtest 2015-2024)

Ran `scripts/backtest.py --start 2015-01 --end 2024-12`:

**Overall skill: -2.0%** (v2 CRPS 0.0966 vs Gaussian rolling 0.0947)

Per-asset @ 12m horizon:
| Asset | Skill | Notes |
|---|---|---|
| Gold | **+17.4%** | Regime identification excels for precious metals |
| Silver | +8.5% | Same pattern |
| Copper | +4.7% | Industrial activity → regime |
| Gold @ 6m | +7.8% | |
| BCOM | +3.3% | Commodity composite |
| Oil | +1.2% | Marginal |
| SPX | +0.7% | Marginal |
| DXY | +1.3% | Marginal |
| NatGas | +6.1% | Gas-specific |
| HSI | +0.6% | Marginal |
| Bond | **-4.1%** | Rate mean-reversion → Gaussian sufficient |
| BTC | **-6.8%** | Short history + regime-independent tails |

**Cross-regime heterogeneity**:
- COVID window (2020-01→2021-06): +6.5% overall skill
- Full window (2015-2024): -2.0% overall skill
→ v2's edge is regime-specific (shines in turbulent periods, underperforms
  in calm years when regime info is noise)

PIT most assets ~0.5 (calibrated), bonds ~0.39 (left bias: v2 bond
scenarios skewed above realized).

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
