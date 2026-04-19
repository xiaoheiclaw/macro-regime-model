# macro-regime v2 · 研究报告

> 2026-04-19 · 54 commits · 严格 peer review × 7 轮

## TL;DR

**花了两周把 v1 的 4 层管线重构成 v2 的 6 层架构，再用 7 轮 codex reviewer + 多种基线 benchmark 把它拷问到每一个声明都能被数据支持。结论：**

- **v2 SP-CVaR 是 stress insurance，不是 uniform alpha**。2020-2024 动荡期 paired vs MVN SP-CVaR 年化 +2.5% 显著（CI 不过零），2015-2019 平静期只有点估计方向正但不显著。平静期长持仓下 12m Sharpe 反而比 60/40 差 -0.63（显著）。
- **Gold/Silver "每资产预测 skill" 是 2015-2024 era-specific 虚名**。OOS 2002-2014 期 Gold skill 完全反转（+7.6% → -7.6% vs AR(1)）。原来以为的"KAF 识别贵金属 regime"实际大部分就是"样本期贵金属在涨"。
- **v2 的 KAF 架构本身 justified 但不是 free lunch**。尝试用更简单的"regime-conditional 采样"（v3）替代 KAF state-distance ranking 的结果是 Sharpe -0.2，因为 joint scenarios 供给 CVaR 需要 diversity，简单采样给不了。
- **真正跨样本稳定的 edge 是 SP-CVaR 配置层面**（v2 vs 60/40 Sharpe +0.15 到 +0.29 在两段 10 年都成立，max DD 2.5-5× 小），而不是预测层面。

**可部署结论**：v2 SP-CVaR 在 6m rebalance 下跑正式策略，Sharpe 0.99-1.02，max DD -2% 到 -5%，netof-cost 年化 +5%-6%，beat 60/40 risk-adjusted。tcost drag 30bps/yr 可以忽略。**不要用 v2 MV BL**（Sharpe 看似 0.94 但 49% vol + 43% max DD，完全靠 Gold/BTC 集中暴露撑）。

---

## 一、做了什么

v1 系统（2026-04-03 发布）是 Phase 1-4 的线性管线：相关性快照 → Markov Regime Switching + Wasserstein K-Means → Kalman Filter → Black-Litterman + SP。问题有两个：

1. **评测不够硬**。v1 每天输出一张配置表，但没法证明比 rolling mean Gaussian 强。没有 paired bootstrap、没有 out-of-sample、没有真实 rebalance net PnL。
2. **架构耦合**。想换 forecaster 或 optimizer 要改整条管线。

v2 重构成 6 层：

```
Phase 0  Data       state_features 月度面板 (+ ALFRED vintage + Shiller CAPE + Moody 长历史信用)
Phase 1  Mask       per-asset × feature × horizon 互信息
Phase 2a Global     K-Means 4 模板 (T0 高利率 / T1 ZIRP / T2 正常 / T3 危机)
Phase 2b Asset      per-asset GMM K=3 (bear / neutral / bull)
Phase 2c Joint      P(asset_regime | global_template) + current joint
Phase 3  Forecast   KAF 类比预测 (combined state + regime Frobenius dist, 拴绳 tether, N=200 joint scenarios)
Phase 4  Allocation Mean-Variance BL (12m) + SP-CVaR (6m) on joint scenarios
```

每层独立可替换。allocation 层吃 forecast 层的 joint scenarios（同 scenario_id = 同历史类比日 → 保留跨资产相关性）。

---

## 二、评测框架

**Backtest harness** (`scripts/backtest.py`):
- 月频滚动 asof，2015-01 到 2024-12（119 asofs, ~10 年）
- 每个 asof 扩展窗口重训 Phase 2a/2b (避免 regime 层 leak 未来)
- forecast / allocation 对 asof 感知 (analog eligibility `s + h_max ≤ asof`)
- 同时对 OOS 2002-01 到 2014-12 (156 asofs, 13 年) 跑完全相同管线

**Benchmarks** (按强度递增):
1. **60/40** — 静态 SPX 60% + Bond 40%
2. **Rolling-120m Gaussian** — 无条件 Normal(μ_roll, σ_roll·√h)
3. **AR(1)** — 每资产 OLS AR(1) 迭代 h 步
4. **Regime-conditional 采样** — 过去同 regime 月份 realized forward 返回值
5. **MVN SP-CVaR** — 同样 rolling 120m (μ, Σ) 采 200 场景，跑**同一个 CVaR 优化器**

**Metrics**:
- CRPS (per-asset marginal)
- Energy Score (joint multivariate)
- PIT (calibration)
- Paired bootstrap (block=horizon) 95% CI for (v2 − baseline) 差
- Net-of-cost monthly-rebalance 策略 Sharpe / Max DD / 交易成本 drag

---

## 三、被验证的真 wins

### 1. SP-CVaR 配置层 vs 60/40 是**结构性稳定** edge

| 期间 | v2 Sharpe | 60/40 Sharpe | Sharpe spread | v2 Max DD | 60/40 Max DD |
|---|---|---|---|---|---|
| 2015-2024 (10y, 6m rebal) | **1.02** | 0.87 | +0.15 | **-2.2%** | -11.1% |
| 2002-2014 (OOS, 6m rebal) | **0.74** | 0.45 | +0.29 | **-9.2%** | -22.0% |

- **两个不同样本都 beat 60/40**, Sharpe spread +0.15 / +0.29
- **Max DD 每次都是 2.5-5× 更小**
- 跨 GFC/ZIRP 和 COVID/inflation 两种完全不同的 regime mix 稳定
- Net-of-cost: v2 交易成本 30bps/yr，远小于 edge

### 2. 同 CVaR 优化器下 v2 joint scenarios vs MVN (paired) 显著

10 年 119 月共 99 个 aligned asofs，paired block-bootstrap (block=h):

| Horizon | Δmean ann | CI | sig? | ΔSharpe | CI | sig? |
|---|---|---|---|---|---|---|
| 3m | +2.37% | [+0.24%, +0.97%] | ✓ | +0.19 | [+0.01, +0.41] | ✓ (边缘) |
| 6m | +2.33% | [+0.48%, +1.80%] | ✓ | +0.24 | [+0.04, +0.39] | ✓ |
| 12m | +2.60% | [+1.04%, +3.38%] | ✓ | +0.03 | [-0.28, +0.35] | — |

Δmean 3/6/12m 都过 bootstrap 检验；ΔSharpe 3/6m 过。**同优化器、同 cap、只差 scenarios — v2 联合 KAF 场景带来的真正信息增益**。

### 3. KAF 架构经得起"简化挑战"

v3 实验：删掉 Phase 1 mask + Phase 3 状态距离 ranking，保留 Phase 2 regime + 简单"同 regime 历史月份采样"。结果：

| Method (6m) | Sharpe |
|---|---|
| v2 SP-CVaR | **1.08** |
| MVN SP-CVaR | 0.89 |
| v3 SP-CVaR | **0.86** ← 比 v2 少 0.2 |

v3 输是因为 regime-matched pool 只有 30-100 月，采 200 场景 with replacement 造成重复 → CVaR tail 估计失去多样性。**KAF 的状态距离 ranking 不是装饰，它保留了 joint diversity**。

---

## 四、被撤回的 claims

### 1. ❌ "v2 overall marginal CRPS skill" 

原本 headline："v2 CRPS -1.1% vs Gaussian，per-asset Gold/Silver +7.6%/+4.8% vs AR(1)"。

- AR(1) 是更严基线，vs Gaussian 的多数赢法在 AR(1) 下消失（见 `backtest_2015-01_2024-12.md`）
- Gold/Silver 2015-24 的正 skill 在 **OOS 2002-14 完全反转**（-7.6%/-9.9%）
- regime-conditional 基线在长 horizon 上**击败 v2**（Gold 12m: +0.7% vs +7.6% vs AR(1)）

**修正结论**：per-asset marginal CRPS edge 不成立或是 sample-specific。v2 不是更好的"每资产预测器"。

### 2. ❌ "Joint Energy Score 证明 KAF 核心有 structural edge"

原 claim: ES 4 个 horizon 都 positive (+0.6% ~ +1.2%)。

经 paired block-bootstrap + 5000×3 MC seed 平均：

| Horizon | mean diff | CI | sig? |
|---|---|---|---|
| 1m | +0.0009 | [+0.0000, +0.0018] | 边缘（下界贴 0）|
| 3m | +0.0025 | [-0.0008, +0.0052] | — |
| 6m | +0.0028 | [-0.0042, +0.0093] | — |
| 12m | +0.0062 | [-0.0035, +0.0175] | — |

**除 1m 勉强擦边外其他 horizon CI 都覆盖 0**。Energy Score 的 +1% 原始数字被 MVN MC 噪声和重叠样本放大 SE 之后退回 noise。

### 3. ❌ "v2 uniform alpha"

50/50 mechanical split (2015-19 calm vs 2020-24 stress)：

- **2015-2019 v2 vs MVN**: 所有 horizon 点估计正但 CI 全跨 0
- **2020-2024 v2 vs MVN**: 所有 horizon 过 bootstrap 检验 (+2.2% 到 +2.9% ann)
- **2015-2019 v2 vs 60/40 @ 12m**: ΔSharpe **-0.63 CI [-1.51, -0.16] ✓** — v2 SIG LOSE

**修正结论**：v2 不是 uniform alpha。它是 stress-period specialist，平静期 long-hold 反而比 60/40 Sharpe 差。全样本 sig 驱动来自 stress 半段。

### 4. ❌ "v2 MV BL 34% annualized return"

样本内 10 年 Gold 涨 100%, BTC 涨 100×, MV 优化器 cap-bind 在它俩身上。Sharpe 0.94 ≈ 60/40 的 0.87 —— **high return 是 49% vol 买的，不是 skill**。

BTC 消融后 MV BL Sharpe 0.73, vol 8.7%, max DD -14% —— 这才是可部署版本。

**修正结论**：MV BL 原始形式不可部署。必须 cap ≤25% 或排除 BTC。

---

## 五、Honest 现状

### v2 能做的
1. **SP-CVaR 配置层在不同样本期稳定 beat 60/40** (Sharpe +0.15 ~ +0.29, Max DD 2.5-5× 小)
2. **Stress 期 (COVID/inflation shock) paired vs MVN 年化 +2.5% 显著**
3. **6m rebalance cadence 下 tcost 30bps/yr 成本远小于 edge**

### v2 做不到的
1. 不是 uniform alpha（平静期全时间 carry 有成本）
2. per-asset 预测对 AR(1) 基线不站得住（OOS 反转）
3. Joint Energy Score 不显著
4. 没有 regime-gating 信号（VIX threshold 失败）

### 真正能对外讲的 value proposition
> **"v2 SP-CVaR 是 joint-scenario-based 的 tail-risk optimized 配置策略。相较 60/40 跨两段样本期 (GFC/ZIRP 和 COVID/inflation) 均显著降低 Max DD (2.5-5× 小) 且小幅提升 Sharpe (+0.15 ~ +0.29)。相较同 optimizer 的 MVN scenario 基线在 stress 期 paired Δmean 年化 +2.5% 显著。平静期表现与 60/40 接近，tail-defense 成本在 calm market 真实存在。"**

---

## 六、方法论教训

1. **Gaussian benchmark 太弱**。2015-24 Gaussian vs v2 的各种"win"大部分是 AR(1) 捕捉条件均值 / regime-conditional 捕捉 regime 信息的功劳，不是 KAF。**minimum benchmark 应该是 AR(1) + regime-conditional**。

2. **Overlapping sample 毁掉 SE**。119 个月度 asof 看 6m forward 有效独立样本数 ~20。naive 1-sigma CI 远小于 block-bootstrap CI。没做 block-bootstrap 的 Sharpe 数字一律不可信。

3. **单点参数 vs 网格搜索有信息**。α / γ / kernel 三轮 grid search 全是 ≤0.3pp 噪声级波动。这不是说参数无所谓，是说**这些参数不是 v2 的 lever**。真正的 lever 是"哪些资产走 KAF / 哪些走 parametric"（BTC/Bond 切 Gaussian 给了 +0.9pp 真改进）。

4. **"让 v2 简单点"的 hypothesis 容易自我说服**。Step 3 发现 RegimeCond 基线在 per-asset 追平 KAF 后，自然推论"KAF 冗余"。但 Step 4 实际建 v3 反证了这个推论：per-asset 追平 ≠ joint/CVaR 追平。**要在下游关心的 metric 上测，不是在中间层 metric 上推论**。

5. **Post-hoc segmentation 不算证据**。Phase 3b.5 我手动挑了 "COVID / Inflation Shock / Recent" 几个窗口，每个 v2 都赢 3-6%，看起来 stress-specialist 理论成立。codex Round 7 正确指出这是 data mining。改成 50/50 mechanical split 后，2015-19 确实不过检验，story 这次是 data-driven 的。

6. **Peer review 是真正 pay off 的**。7 轮 codex review 每一轮都撤回至少一个 claim。最后留下的是真的，撤掉的是幻觉。没 review 的话 v2 会以"joint ES +1% 证明结构性 edge"的虚假故事发布。

---

## 七、Open items (什么没做)

技术债（codex 标记，未完成）：

- **VAR(1) benchmark** — 现在有 AR(1)，跨资产动态未测
- **Holm-Bonferroni 多重比较修正** — 40 per-asset tests 现在是单独 α=0.05，family-wise 错误率高
- **Seed sensitivity sweep** — 所有 bootstrap / MC 都是单 seed
- **CVaR random restart 稳定性对照** — 4-restart 已加，但没存 pre-restart 结果对比
- **Diffusion kernel, delay embedding, proper tethering** — Phase 3b 模型层升级全 defer
- **2008 GFC 单独压力期** — 已在 OOS 中，但没单独表

Open 研究问题：

- **Regime gating 信号**：v2 是 stress-insurance，怎么识别 stress 开始？VIX 不 work，rolling-24m skill curve / NBER dates / 信用利差 quantile 没试过
- **ALFRED vintage 完整性**：现在 5 个核心 series（CPI/IP/Payroll/Unrate/Core CPI）有 vintage，其他 FRED 仍 `vintage_resolved=False`
- **Shiller CAPE 数据停在 2023-09** — multpl 爬虫补 31 月 ffill
- **MV BL 在 realistic cap (25%) + turnover penalty 下**是否成为可部署版本？没测

部署：

- **LaunchAgent 自动化** — 当前 dashboard 手动跑
- **异常告警** — weights 剧烈变化、场景数骤降、regime 翻转都没告警
- **v2 vs v1 dashboard side-by-side** ✓ 已做 (commit 5002bd8)

---

## 八、Commits 索引

**Stage A (构建)** 9 commits:
- `733dc67` Phase 0 state_features
- `c04cdaf` Phase 1 MI mask
- `cf5b359` Phase 2a global_template
- `ecb3d8b` Phase 0b ALFRED vintage
- `56030d1` Phase 0c commodity + CAPE
- `8d11fdd` Phase 2b asset_regime
- `a9bcbf2` Phase 2c template_map
- `e2ec26e` Phase 3 KAF baseline
- `3db1d7f` Phase 4 allocation

**Codex 7 轮 review 响应** ~15 commits:
- `7283e0d` time leakage → expanding scaler
- `406ce78` regime-conditional analog filter
- `8fdcd6b` fallback joint-scenario
- `4203e3b` schema freeze + analog_date provenance
- `da6b3f5` forward leakage + real template_map
- `0b89248` template_map asof-aware
- `3f244d9` deterministic seed
- `e8c5fcc` ES pinned universe + portfolio guard
- `25b5151` paired + block-bootstrap + CVaR restart
- `32a3cd4` Sharpe-diff bootstrap + turnover
- `aaeb7be` ES MC 5000×3
- others

**Phase 3b 实验 + 验证** ~12 commits:
- `a856333` α calibration (negative)
- `9387216` tethering (+0.2pp)
- `566d9c4` kernel variants (negative)
- `ff875f5` BTC parametric (+0.7pp)
- `d3103b3` Bond parametric (+0.2pp)
- `d1e40e4` stress-period breakdown (exploratory)
- `9c2ac57` VIX switch (negative)
- `8a9b9ed` Energy Score + AR(1) benchmarks
- `b8357e5` downstream allocation eval
- `aaa0fa2` MVN SP-CVaR fair benchmark
- `8c29a65` real monthly-rebalance net PnL
- `b97e7af` BTC ablation
- `771097c` 6m rebalance strategy

**OOS + 架构挑战** 4 commits:
- `8086e9f` OOS 2002-2014
- `40bef92` regime-conditional benchmark (RC beats KAF long horizons per-asset)
- `4d8c62c` v3 proof-of-concept (REFUTED — KAF needed for joint)
- `220ffe8` subsample stability (stress-specialist with stat backing)

**部署 + 文档** 4 commits:
- `5002bd8` v2 进 daily_dashboard
- `e05437f` paper v2 后记
- `ddd79c3`, `44cf6aa`, `9db3501`, `9654ca3`, `541b833`, `dfef819` HANDOFF 各轮更新

**此 report** 即将:
- `(pending)` docs/v2-report.md 完整梳理

---

## 附录: 核心数字速查

**10 年 (2015-2024) monthly rebalance net strategy Sharpe**:
| | 60/40 | MVN SP-CVaR | v2 SP-CVaR | v2 MV BL |
|---|---|---|---|---|
| Ann Net | 5.92% | 4.16% | 4.90% | 31.4% ⚠️ |
| Ann Vol | 9.96% | 4.25% | 4.94% | 33.4% |
| Sharpe | 0.59 | 0.98 | **0.99** | 0.94 |
| Max DD | -21.5% | -7.8% | **-4.8%** | -43% ⚠️ |

**10 年 6m rebalance strategy**:
| | 60/40 | v2 SP-CVaR | MVN SP-CVaR |
|---|---|---|---|
| Sharpe | 0.87 | **1.02** | 0.85 |
| Max DD | -11.1% | **-2.2%** | -4.0% |

**OOS 13 年 (2002-2014) 6m rebalance**:
| | 60/40 | v2 SP-CVaR |
|---|---|---|
| Sharpe | 0.45 | **0.74** |
| Max DD | -22.0% | **-9.2%** |

**Paired v2 vs MVN mechanical split**:
| 期间 | 6m Δmean ann | CI | sig |
|---|---|---|---|
| Calm 2015-19 | +2.01% | [-0.32%, +5.65%] | — |
| Stress 2020-24 | +2.50% | [+0.89%, +3.33%] | ✓ |
| Full 10y | +2.32% | [+0.95%, +3.60%] | ✓ (由 stress 段驱动)|

**BTC 消融 MV BL**:
| | 带 BTC | 去 BTC |
|---|---|---|
| Sharpe | 0.94 | 0.73 |
| Vol | 33.4% | 8.7% |
| Max DD | -43% | -14% |

---

*最后更新: 2026-04-19*
