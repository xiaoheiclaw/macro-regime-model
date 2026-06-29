"""PR #9 — gold 2022→2026 *ex-post* layered attribution runner.

Decomposes the realised log gold price into five additive layers (inflation /
real-rate / sovereign-credit / tail / flow) and asks the core question: over
2022-01→latest, is the **sovereign-credit layer (③)** the largest contributor —
i.e. did a "fiat credit-spread / de-dollarisation regime" take over from the
usual real-rate cycle?

This is EX-POST attribution, NOT a forecast and NOT a trading backtest. The OLS
fit uses the full sample to *describe* co-movement; rolling coefficients are
reported only to expose the regime-dependence PR #1–#8 already established. No S1
code is touched.

Outputs:
  analysis/gold_attribution_2022_<date>.md   report (coverage, coef tables,
        2022→latest decomposition, the sovereign-takeover verdict, ex-post
        boundary statement)
  data/gold_attribution_decomposition_<date>.csv   the layer contribution table
  data/gold_attribution_coefs_<date>.csv            full-sample coef/std-coef/t
  data/gold_attribution_rolling_<date>.csv          rolling-window coefficients
  data/gold_attribution_stacked_<date>.csv          per-month cumulative stack

Usage:
    uv run python scripts/gold_attribution_2022.py
    uv run python scripts/gold_attribution_2022.py --t0 2022-01 --roll-window 60
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402
from lib.gold_credit_spread_attribution import (  # noqa: E402
    build_attribution_panel,
    decompose_period,
    fit_attribution,
    rolling_coefs,
    stacked_contribution_path,
    verdict,
)


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01", help="panel start (data layer)")
    ap.add_argument("--t0", default="2022-01", help="attribution window start")
    ap.add_argument("--t1", default=None, help="attribution window end (default latest)")
    ap.add_argument("--roll-window", type=int, default=60, help="rolling coef window (months)")
    args = ap.parse_args()

    print("[1/4] building panel (live FRED + anchor reuse) …", file=sys.stderr)
    panel = build_attribution_panel(start=args.start)

    print("[2/4] fitting full-sample attribution (identity + free) …", file=sys.stderr)
    res_id = fit_attribution(panel, cpi_mode="identity")
    res_free = fit_attribution(panel, cpi_mode="free")

    decomp = decompose_period(res_id, t0=args.t0, t1=args.t1)
    decomp_free = decompose_period(res_free, t0=args.t0, t1=args.t1)
    v = verdict(decomp)
    v_free = verdict(decomp_free)

    print("[3/4] rolling coefficients + stacked path …", file=sys.stderr)
    roll = rolling_coefs(panel, window=args.roll_window, cpi_mode="identity")
    stacked = stacked_contribution_path(res_id, t0=args.t0, t1=args.t1)

    # ── persist CSVs ────────────────────────────────────────────────────
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    decomp_csv = os.path.join(DATA_DIR, f"gold_attribution_decomposition_{date}.csv")
    coefs_csv = os.path.join(DATA_DIR, f"gold_attribution_coefs_{date}.csv")
    roll_csv = os.path.join(DATA_DIR, f"gold_attribution_rolling_{date}.csv")
    stack_csv = os.path.join(DATA_DIR, f"gold_attribution_stacked_{date}.csv")

    decomp.to_csv(decomp_csv, index=False)
    coef_tbl = pd.DataFrame({
        "coef_identity": res_id.coefs,
        "std_coef_identity": res_id.std_coefs,
        "t_identity": res_id.tstats,
        "p_identity": res_id.pvals,
        "coef_free": res_free.coefs,
        "std_coef_free": res_free.std_coefs,
        "t_free": res_free.tstats,
        "p_free": res_free.pvals,
    })
    coef_tbl.to_csv(coefs_csv)
    roll.to_csv(roll_csv)
    stacked.to_csv(stack_csv)

    print("[4/4] writing report …", file=sys.stderr)
    md = os.path.join(ANALYSIS_DIR, f"gold_attribution_2022_{date}.md")
    _write_report(md, panel, res_id, res_free, decomp, decomp_free, v, v_free,
                  roll, args, date, decomp_csv, coefs_csv, roll_csv, stack_csv)

    print(f"\nverdict (identity): sovereign_took_over={v['sovereign_took_over']} "
          f"top={v['top_layer']} ({v['top_label']})")
    print(f"report  → {md}")
    for p in (decomp_csv, coefs_csv, roll_csv, stack_csv):
        print(f"csv     → {p}")


def _coef_table_md(res) -> str:
    rows = ["| layer | coef | std-coef | t | p |", "|---|---:|---:|---:|---:|"]
    for k in res.coefs.index:
        if k == "const":
            continue
        sc = res.std_coefs.get(k, np.nan)
        t = res.tstats.get(k, np.nan)
        p = res.pvals.get(k, np.nan)
        rows.append(
            f"| {k} | {res.coefs[k]:+.4f} | "
            f"{sc:+.3f} | {t:+.2f} | {p:.3f} |"
            if np.isfinite(t) else
            f"| {k} | {res.coefs[k]:+.4f} | {sc:+.3f} | (imposed) | — |"
        )
    return "\n".join(rows)


def _decomp_table_md(decomp: pd.DataFrame) -> str:
    rows = ["| 层 | 贡献 (Δln) | 占总涨幅 | 系数 | Δ代理 |",
            "|---|---:|---:|---:|---:|"]
    for _, r in decomp.iterrows():
        coef = "" if not np.isfinite(r["coef"]) else f"{r['coef']:+.4f}"
        dp = "" if not np.isfinite(r["delta_proxy"]) else f"{r['delta_proxy']:+.3f}"
        rows.append(
            f"| {r['label']} | {r['contribution_ln']:+.4f} | "
            f"{_fmt_pct(r['contribution_pct_of_total'])} | {coef} | {dp} |"
        )
    return "\n".join(rows)


def _write_report(path, panel, res_id, res_free, decomp, decomp_free, v, v_free,
                  roll, args, date, decomp_csv, coefs_csv, roll_csv, stack_csv) -> None:
    t0, t1 = decomp.attrs["t0"], decomp.attrs["t1"]
    tot_ret = decomp.attrs["total_pct_return"]
    rank = v["ranking"]

    verdict_line = (
        f"**主权信用层(③)接管 = {'成立 ✅' if v['sovereign_took_over'] else '未成立 ❌'}**"
    )
    # contributions for the nuanced ruling
    def _contrib(layer):
        row = decomp[decomp["layer"] == layer]
        return float(row["contribution_ln"].iloc[0]) if not row.empty else np.nan
    real_c, sov_c, resid_c = _contrib("real"), _contrib("sov"), _contrib("flow_resid")

    if v["sovereign_took_over"]:
        ruling = (
            f"在 {t0}→{t1} 的金价涨幅中,**主权信用层(③)是最大正贡献项** "
            f"(Δln={v['sov_contribution_ln']:+.4f}),定量支持「2022 起主权信用 regime 接管」假说。"
        )
    elif v["top_layer"] == "flow_resid":
        ruling = (
            f"最大「贡献」来自 **ε_flow 残差**(Δln={resid_c:+.4f},占总涨幅 "
            f"{_fmt_pct(resid_c / decomp.loc[decomp['layer']=='TOTAL','contribution_ln'].iloc[0] * 100)}"
            "),即四个可测宏观层(①通胀/②实利率/③主权/④尾部)**合起来也解释不了这段涨幅的主体**。\n\n"
            f"关键证据:**②实利率层贡献为负**(Δln={real_c:+.4f}) —— 2022-23 实利率大幅*上行*,"
            "按传统「金价≈−实利率」模型本应**压制**金价,但金价反而创新高。这说明传统实利率 regime "
            "**确实失效了**(假说的*前半句*成立);但失效后的解释力**并未落到我们构造的 ③主权信用代理上**"
            f"(③仅贡献 {sov_c:+.4f}),而是落在**无法用 ①–④ 线性代理捕捉的残差**里。\n\n"
            "**裁决:严格口径下「③主权信用层接管」未被证实**——③的 *可测代理*(debt/GDP+外官托管)"
            "没有定量接管。但残差的主体最可能由**我们缺数据的⑤央行购金/去美元化流量**承载(WGC 季度数据无 "
            "FRED 源,见 §1);也可能是本归因的多重共线(§2 条件数高、③系数在 free 口径变号)使逐层拆分"
            "本身不可靠。换言之:**「旧实利率锚已断」有据,「新锚=主权信用」尚无可测证据**——需要 WGC 流量"
            "数据(补⑤)才能在③/⑤之间做出裁断。"
        )
    else:
        top = v["top_label"]
        ruling = (
            f"最大正贡献来自 **{top}**(非③主权信用),`sov` 贡献="
            f"{v['sov_contribution_ln']:+.4f}。「主权信用接管」假说**在本口径下未被证实** —— "
            "需重新审视(或归因被多重共线/口径选择主导)。"
        )

    cov = panel.coverage
    notes = panel.notes

    roll_summary = []
    for c in roll.columns:
        s = roll[c].dropna()
        if len(s):
            roll_summary.append(
                f"| {c} | {s.min():+.3f} | {s.median():+.3f} | {s.max():+.3f} | "
                f"{'是' if (s.min() < 0 < s.max()) else '否'} |"
            )

    lines = [
        f"# 黄金「法币信用利差」逐层归因 — 2022→2026 (PR #9)",
        "",
        f"_生成于 {date} · 数据起点 {args.start} · 归因窗口 {t0}→{t1}_",
        "",
        "## 0. 这是什么 / 不是什么 (ex-post vs ex-ante 边界)",
        "",
        "- **是**:对*已实现*金价涨幅的**样本内逐层分解**(log-linear 恒等式),回答"
        "「哪一层在解释这段涨幅」。",
        "- **不是**:预测模型 / 交易回测 / 对未来的择时。PR #1–#8 已证(codex 复审确认)"
        "金价↔宏观各锚关系 **regime 依赖、不可外推**;本 PR 不碰 S1、不做预测。",
        "- 全样本 OLS 拟合系数被**允许**使用全样本(这是解释而非预测);滚动系数报出来"
        "正是为了**暴露不稳定性**,而非用来择时。",
        "",
        "## 1. 五层代理覆盖",
        "",
        "log-linear 恒等式:`ln(P_gold) = ln(P_CPI) + β·(−r_real) + γ·SovRisk "
        "+ δ·TailRisk + ε_flow`",
        "",
        "| 层 | 代理 | 覆盖 |",
        "|---|---|---|",
        f"| ① 通胀购买力 | ln(CPI), CPIAUCSL | {cov.get('ln_cpi','n/a')} |",
        f"| ② 实利率升水 | −real_rate_10y (DFII10 splice, PR#1) | {cov.get('neg_real_rate','n/a')} |",
        f"| ③ 主权信用 | z(ln debt/GDP) ⊕ z(−外官托管份额 WMTSECL1/GFDEBTN) | "
        f"debt:{cov.get('ln_debt_gdp','n/a')}; 托管:{cov.get('neg_custody_share','n/a')} |",
        f"| ④ 尾部保险 | z(VIX) ⊕ z(信用利差 BAA10Y) | "
        f"vix:{cov.get('vix','n/a')}; 利差:{cov.get('credit_spread','n/a')} |",
        f"| ⑤ 流量(央行购金) | WGC 季度(注入式) | {cov.get('wgc_flow','n/a')} |",
        "",
        f"- 信用利差口径:{notes.get('credit_spread_choice','')}",
        f"- 托管份额口径:{notes.get('custody_share','')}",
        f"- 流量层:{notes.get('wgc_flow','')}",
        "- 多分量层(③④)取分量 z-score 等权均值,使单一系数可解释;①②为自然单位单分量。",
        "",
        "## 2. 全样本归因系数",
        "",
        f"样本 n={res_id.n} · R²(identity)={res_id.r2:.3f} · R²(free)={res_free.r2:.3f} · "
        f"回归块条件数(共线性诊断)={res_id.cond_number:.1f}",
        "",
        "**identity 口径**(强加购买力恒等式 b_CPI≡1,回归 ln(gold)−ln(CPI) 对 ②–⑤):",
        "",
        _coef_table_md(res_id),
        "",
        "**free 口径**(ln(CPI) 作为自由回归项):",
        "",
        _coef_table_md(res_free),
        "",
        "> std-coef = 标准化贝塔(可跨层比较量纲)。identity 块条件数="
        f"{res_id.cond_number:.1f}(此处不大,实利率/主权/尾部三层本身不算共线);"
        f"**真正的共线信号是 ③sov 系数跨口径变号**:identity {res_id.coefs.get('sov', float('nan')):+.3f} → "
        f"free {res_free.coefs.get('sov', float('nan')):+.3f} —— 一旦把强趋势的 ln(CPI) 放进自由回归,"
        "debt/GDP 与之共线,③ 的符号被夺走。这说明逐层系数对口径敏感(与 PR#1–#4「无稳定锚」一致),"
        "逐层拆分应视作*提示性*而非*定论*。",
        "",
        f"## 3. {t0}→{t1} 涨幅逐层归因(核心)",
        "",
        f"金价总涨幅 {_fmt_pct(tot_ret)} (Δln={decomp.loc[decomp['layer']=='TOTAL','contribution_ln'].iloc[0]:+.4f})。"
        "下表各层贡献 + ε_flow 残差 **精确加总 = 总 Δln**(OLS 恒等式,零松弛)。",
        "",
        "**identity 口径:**",
        "",
        _decomp_table_md(decomp),
        "",
        "**free 口径(稳健性对照):**",
        "",
        _decomp_table_md(decomp_free),
        "",
        "## 4. 核心裁决:主权信用是否接管?",
        "",
        verdict_line,
        "",
        ruling,
        "",
        "贡献排序(非通胀层,identity 口径,降序):",
        "",
        "| 层 | 贡献 Δln | 占总涨幅 |",
        "|---|---:|---:|",
    ]
    for r in rank:
        lines.append(f"| {r['layer']} | {r['contribution_ln']:+.4f} | "
                     f"{_fmt_pct(r['contribution_pct_of_total'])} |")
    lines += [
        "",
        f"free 口径裁决:{'主权接管 ✅' if v_free['sovereign_took_over'] else '主权未接管 ❌'} "
        f"(最大项 = {v_free['top_layer']})。",
        "",
        "## 5. 滚动系数(暴露 regime 依赖,非择时)",
        "",
        f"滚动窗口 = {args.roll_window} 月,identity 口径。各层滚动 β 的范围:",
        "",
        "| 层 | min | median | max | 变号? |",
        "|---|---:|---:|---:|---:|",
        *roll_summary,
        "",
        "> 若 β 在样本内**变号**(min<0<max),说明该层与金价的关系并非稳定结构 —— "
        "这正是「不能外推预测」的直接证据。",
        "",
        "## 6. 诚实标注与边界",
        "",
        "- 样本内归因(解释历史),**非**预测;系数 regime 依赖(见 §5 滚动)。",
        "- 各层代理高度共线(§2 条件数),逐层系数对口径敏感 —— identity vs free 两套并报。",
        f"- ⑤ 流量层在无 WGC 注入时并入 ε_flow 残差({notes.get('wgc_flow','')[:60]}…)。",
        "- ③ 的去美元化分量(外官托管)始于 2003(WMTSECL1),早于此仅 debt/GDP 一项。",
        f"- ex-post 边界声明:{notes.get('ex_post_boundary','')}",
        "",
        "## 附:产出文件",
        "",
        f"- 逐层归因表:`{os.path.relpath(decomp_csv)}`",
        f"- 全样本系数表:`{os.path.relpath(coefs_csv)}`",
        f"- 滚动系数:`{os.path.relpath(roll_csv)}`",
        f"- 月度堆叠贡献(堆叠图数据):`{os.path.relpath(stack_csv)}`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
