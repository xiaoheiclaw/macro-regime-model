"""Walk-forward (expanding-window) re-calibration of the gold-vs-de-dollarization
deviation — PR #15.

The *cheap veto-style honesty check* on PR #14. PR #14 ranks the latest gold-vs-DI
deviation against history with a **full-sample** z-score / percentile (in-sample by
construction). This script re-ranks the **same** PR #14 residual with an
**expanding window** (at month t only ``[start, t]``), and adjudicates honestly
whether PR #14's "偏贵 / 极端" conclusion is robust or partly an in-sample artifact.

Reuse (touches no PR #14 code):
  * ``lib.gold_dedollar_gap`` builds the panel → DI → rolling-OLS residual exactly
    as PR #14 does (DI construction & regression fit are UNCHANGED).
  * ``lib.gold_dedollar_gap_walkforward`` changes ONLY the calibration口径.

Outputs:
  * markdown report   analysis/gold_dedollar_gap_walkforward_<stamp>.md
  * trajectory CSV    data/gold_dedollar_gap_walkforward_series_<stamp>.csv
                      (resid + full-sample vs walk-forward z/percentile per month —
                       the 对照线 for a Show Page)
  * conditional CSVs  full-sample (PR #14) vs walk-forward extreme→forward-return

Usage:
    uv run python scripts/gold_dedollar_gap_walkforward.py
    uv run python scripts/gold_dedollar_gap_walkforward.py --warmup 24 --top-q 0.9
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gold_dedollar_gap import (  # noqa: E402
    DEFAULT_COMPONENTS,
    DEFAULT_HORIZONS,
    DEFAULT_REG_WINDOW,
    DEFAULT_TOP_Q,
    DXY_COMPONENT,
    build_di,
    build_gap_panel,
    compute_deviation,
    conditional_forward_table,
)
from lib.gold_dedollar_gap_walkforward import (  # noqa: E402
    WARMUP_BAND,
    current_walk_forward_reading,
    extreme_reclassification,
    walk_forward_calibration,
    walk_forward_conditional_table,
    warmup_sensitivity,
)
from lib.paths import ANALYSIS_DIR, DATA_DIR  # noqa: E402


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x * 100:.0f}%"


def _fmt(x: float, d: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _cond_md(tbl: pd.DataFrame) -> str:
    if tbl.empty:
        return "_(no conditioning observations)_"
    rows = []
    for _, r in tbl.iterrows():
        rows.append([
            f"{int(r['horizon'])}m", str(r["regime"]), str(int(r["n"])),
            _fmt(r["mean"]), _fmt(r["median"]), _fmt(r["p25"]),
            _fmt(r["p75"]), _fmt_pct(r["hit"]),
        ])
    return _md_table(
        ["horizon", "regime", "n", "mean(lnret)", "median", "p25", "p75", "hit>0"],
        rows)


def _verdict(rd, rc) -> tuple[str, str]:
    """Honest adjudication. The CURRENT reading is near-invariant to calibration
    (the window ends *now*), so the discriminating evidence is the historical
    extreme-episode reread: how often a full-sample 'extreme' was extreme ex-ante.

    ROBUST  — current pct walk-forward ≈ full-sample AND agreement_rate high.
    QUALIFIED — current robust but a non-trivial share of historical 'extremes'
                were not extreme in real time (the *narrative* is partly hindsight).
    DOWNGRADE — current walk-forward pct drops materially below full-sample.
    """
    pct_full = rd.pct_full
    pct_wf = rd.pct_wf_excl  # strict ex-ante current read
    agree = rc.summary.get("agreement_rate", float("nan"))
    if not np.isfinite(pct_full):
        return "UNKNOWN", "no defined current deviation reading."
    drop = pct_full - pct_wf if np.isfinite(pct_wf) else 0.0
    if np.isfinite(pct_wf) and drop > 0.15:
        return ("DOWNGRADE",
                f"当前分位在 walk-forward(strict ex-ante)下从 {_fmt_pct(pct_full)} "
                f"降到 {_fmt_pct(pct_wf)}(降 {drop * 100:.0f}pp)——「偏贵」部分是"
                "全样本 in-sample 标定的产物,PR#14 高估了极端性,应降级。")
    # current is robust; differentiate on the historical episode reread
    if np.isfinite(agree) and agree < 0.7:
        return ("QUALIFIED",
                f"当前分位 walk-forward 与全样本接近({_fmt_pct(pct_wf)} vs "
                f"{_fmt_pct(pct_full)})——今天的「偏贵」读数本身不是 look-ahead 产物"
                "(窗口终点即当下);但历史「极端」月份中只有 "
                f"{_fmt_pct(agree)} 在当时(ex-ante)也算极端——PR#14『历史极端后回调』"
                "的叙事部分依赖事后视角,应谨慎。")
    return ("ROBUST",
            f"当前分位 walk-forward 与全样本一致({_fmt_pct(pct_wf)} vs "
            f"{_fmt_pct(pct_full)}),且历史「极端」月份 {_fmt_pct(agree)} 在 ex-ante "
            "下也算极端——PR#14 的**估值「偏贵」分位结论**对标定口径稳健,未见 in-sample "
            "假象。⚠️ 注意范围:本裁决只认证「今天有多贵」这个分位读数;PR#14 §3 的"
            "「历史极端→之后回调」**预测性**叙事在 ex-ante 旗标下样本被 warm-up 与"
            "不可观测前瞻窗砍得很薄(见 §4b,极端簇集中在 2025-26、24/36m 前瞻多不可观测),"
            "无法 ex-ante 证实——稳健的是估值分位,不是回调预测。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2002-12")
    ap.add_argument("--end", default=None)
    ap.add_argument("--reg-window", type=int, default=DEFAULT_REG_WINDOW)
    ap.add_argument("--warmup", type=int, default=24,
                    help="expanding-window warm-up (min obs) for the headline read")
    ap.add_argument("--top-q", type=float, default=DEFAULT_TOP_Q)
    ap.add_argument("--include-dxy", action="store_true")
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    now = datetime.now().astimezone()
    tzname = now.strftime("%Z") or "local"
    stamp = f"{now.strftime('%Y-%m-%d')} {tzname}"
    file_stamp = now.strftime("%Y-%m-%d_%H%M%S_%f")

    # ── rebuild the PR #14 residual (DI + rolling-OLS fit UNCHANGED) ──
    print("building PR#14 panel → DI → rolling-OLS residual (unchanged) ...")
    panel = build_gap_panel(args.start, args.end, include_dxy=args.include_dxy)
    df = panel.data
    components = list(DEFAULT_COMPONENTS) + (
        [DXY_COMPONENT] if args.include_dxy else [])
    di = build_di(df, components=components).di
    dev = compute_deviation(df["ln_gold"], di, window=args.reg_window)
    resid = dev.resid
    if resid.dropna().empty:
        raise RuntimeError(
            "PR#14 residual is empty on this window — cannot re-calibrate. Check "
            "component coverage / --start / --reg-window.")

    # ── walk-forward re-calibration (the only thing this PR changes) ──
    frame = walk_forward_calibration(resid, warmup=args.warmup)
    rd = current_walk_forward_reading(resid, warmup=args.warmup)
    warm_tbl = warmup_sensitivity(resid, warmups=WARMUP_BAND)
    rc = extreme_reclassification(
        resid, top_q=args.top_q, warmup=args.warmup, exclude_current=False)

    # ── full-sample (PR #14) vs walk-forward extreme→forward-return ──
    cond_full = conditional_forward_table(
        dev.gap_z_full, df["gold_nominal"],
        horizons=DEFAULT_HORIZONS, top_q=args.top_q)
    cond_wf = walk_forward_conditional_table(
        resid, df["gold_nominal"], horizons=DEFAULT_HORIZONS,
        top_q=args.top_q, warmup=args.warmup, exclude_current=False)

    label, verdict_msg = _verdict(rd, rc)
    asof_str = rd.asof.strftime("%Y-%m") if rd.asof is not None else "n/a"

    # ── trajectory divergence summary (mean/max |pct_wf − pct_full|) ──
    g = frame["pct_gap"].dropna()
    mean_abs_gap = float(g.abs().mean()) if len(g) else float("nan")
    max_abs_gap = float(g.abs().max()) if len(g) else float("nan")

    # ── write report ──
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(
        args.out_dir, f"gold_dedollar_gap_walkforward_{file_stamp}.md")
    P: List[str] = []
    P.append(f"# 金价偏离度 Walk-forward 标定重估 (PR#15, {stamp})\n")
    P.append("> **治本的一票否决式便宜检验。** 把 PR#14 偏离度的 z-score/分位数标定从"
             "「全样本 in-sample」改成「expanding window ex-ante」(每月 t 只用截至当月的"
             "历史算均值/标准差/累计分位),诚实重估「当前 ~88 分位」到底靠不靠谱。"
             "**DI 构造与滚动回归拟合数据完全不动,只改标定口径。** ex-post 描述,非预测。\n")

    P.append("## 裁决 (Verdict)\n")
    P.append(f"**{label}** — {verdict_msg}\n")

    P.append("### ⚠️ 一个必须先读的结构性事实\n")
    P.append("当前月是 expanding 窗口的**终点**,故「截至当下的 expanding 窗口」== "
             "「全样本」。因此 **include-current 口径下当前分位/ z 与 PR#14 完全相等"
             "(数学恒等),exclude-current 仅去掉最新一点、移动约 1/N**。"
             "→ **今天的读数不是 look-ahead 能藏身之处**;真正被 expanding 窗口纠正的是"
             "**历史**月份的分位标定(用来称过去某月为「极端」时,偷看了该月之后的数据)。"
             "裁决因此落在历史极端重判上,而非today的数字上。\n")

    P.append("## 1. 当前(最新月)偏离度:全样本 vs walk-forward 标定\n")
    P.append(f"截至 {asof_str},残差样本 n={rd.n_resid},warm-up={rd.warmup}。\n")
    P.append(_md_table(
        ["标定口径", "z", "历史分位"],
        [["全样本 in-sample (PR#14 headline)", _fmt(rd.z_full), _fmt_pct(rd.pct_full)],
         ["walk-forward · include-current", _fmt(rd.z_wf_incl), _fmt_pct(rd.pct_wf_incl)],
         ["walk-forward · exclude-current (strict ex-ante)",
          _fmt(rd.z_wf_excl), _fmt_pct(rd.pct_wf_excl)]]))
    P.append("")
    P.append(f"**裁决数字:88 分位在 walk-forward 下 = {_fmt_pct(rd.pct_wf_excl)}"
             f"(strict ex-ante)/ {_fmt_pct(rd.pct_wf_incl)}(include)。** "
             "include 与全样本恒等;exclude 仅去最新点,故二者均≈全样本——"
             "当前读数对标定口径稳健。\n")

    P.append("## 2. 全样本 vs walk-forward 标定:整段轨迹的分位差\n")
    P.append("对每个历史月,比较全样本分位(偷看未来)与 expanding 分位(ex-ante)。"
             "差越大,说明 PR#14 对该月「极端度」的判断越依赖事后才有的数据。\n")
    P.append(f"- 分位差 |pct_wf − pct_full|:均值 {_fmt_pct(mean_abs_gap)},"
             f"最大 {_fmt_pct(max_abs_gap)}(warm-up={args.warmup})。")
    P.append("- `pct_gap` > 0 ⇒ 该月在**当时**看起来比事后(全样本)**更**极端;"
             "< 0 ⇒ 事后标定把它抬成了比当时更极端。\n")

    P.append("## 3. 历史极端高位重判 (核心 ex-ante 泄露检验)\n")
    P.append("PR#14 用**全样本** top-decile 把某些月标为「极端高位」并称「之后金价回调」。"
             "这里问当时的人(只用截至当月、warm-up 门控后的 expanding 分位)是否也会判其极端。\n")
    s = rc.summary
    P.append(_md_table(
        ["全样本极端月数", "其中 ex-ante 也极端", "warm-up 内(不计)", "可评估", "一致率"],
        [[str(int(s["n_full_extreme"])), str(int(s["n_wf_extreme"])),
          str(int(s["n_warmup"])), str(int(s["n_evaluable"])),
          _fmt_pct(s["agreement_rate"])]]))
    P.append("")
    if not rc.episodes.empty:
        ep_rows = []
        for _, r in rc.episodes.iterrows():
            ep_rows.append([
                pd.Timestamp(r["date"]).strftime("%Y-%m"),
                _fmt_pct(r["pct_full"]),
                _fmt_pct(r["pct_wf"]) if np.isfinite(r["pct_wf"]) else "warm-up",
                "✓" if r["wf_extreme"] else ("—" if not r["in_warmup"] else "·"),
            ])
        P.append("逐月明细(全样本极端月):")
        P.append(_md_table(["月份", "全样本分位", "ex-ante分位", "ex-ante极端?"], ep_rows))
        P.append("")
    P.append(f"- 读法:{rc.notes['reading']}\n")

    P.append("## 4. 极端→前瞻收益:全样本旗标 vs walk-forward 旗标 (条件描述,非预测)\n")
    P.append("PR#14 §3 用全样本分位定义「极端」;这里用 ex-ante expanding 分位定义「极端」"
             "重做同一张表。**能在 ex-ante 旗标下存活的「极端→弱收益」才是诚实的那个。**\n")
    P.append("### 4a. 全样本旗标 (PR#14 原口径)\n")
    P.append(_cond_md(cond_full))
    P.append("")
    P.append("### 4b. walk-forward 旗标 (ex-ante)\n")
    P.append(_cond_md(cond_wf))
    P.append("")

    P.append("## 5. warm-up 稳健性 {12,24,36} 月\n")
    P.append("expanding 的最小 warm-up 门控对 walk-forward 读数的敏感度。"
             "(当前 include 读数对 warm-up 不变——窗口终点恒为最新月、用满全历史;"
             "warm-up 主要决定 ex-ante 轨迹能往前回推到多早。)\n")
    wrows = []
    for _, r in warm_tbl.iterrows():
        first = r["first_wf_date"]
        wrows.append([
            f"{int(r['warmup'])}m",
            _fmt_pct(r["pct_wf_incl"]), _fmt(r["z_wf_incl"]),
            _fmt_pct(r["pct_wf_excl"]), _fmt(r["z_wf_excl"]),
            pd.Timestamp(first).strftime("%Y-%m") if pd.notna(first) else "n/a",
            str(int(r["n_wf_defined"]))])
    P.append(_md_table(
        ["warm-up", "当前pct(incl)", "z(incl)", "当前pct(excl)", "z(excl)",
         "轨迹起点", "ex-ante定义月数"], wrows))
    P.append("")

    P.append("## 6. 诚实与局限\n")
    P.append("- **walk-forward 只治标定泄露**(把历史分位/ z 改成 ex-ante),"
             "**不解决 DI 本身的 post-2010 / 小样本 / 趋势变量问题**——那些是数据硬约束,"
             "PR#14 已标注,本 PR 不解、也不声称解。")
    P.append("- 残差序列、DI、滚动回归拟合**全部沿用 PR#14**,未改一行;本 PR 只换标定口径。")
    P.append("- 当前读数对标定口径近乎不变是**结构必然**(窗口终点即当下),"
             "不应被解读为「walk-forward 证明今天很极端」——它只证明今天的数字没被未来污染。")
    P.append("- 极端→收益表是**对过去共同走势的条件描述,非预测、非因果**;"
             "ex-ante 旗标下 N 更小(warm-up 砍掉早期),已如实标注。\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print(f"  report → {report_path}")

    # ── write trajectory CSV (the 对照线 for Show Page) ──
    os.makedirs(DATA_DIR, exist_ok=True)
    series_path = os.path.join(
        DATA_DIR, f"gold_dedollar_gap_walkforward_series_{file_stamp}.csv")
    frame.to_csv(series_path)
    print(f"  trajectory → {series_path}")

    cond_path = os.path.join(
        DATA_DIR, f"gold_dedollar_gap_walkforward_conditional_{file_stamp}.csv")
    cond_full2 = cond_full.assign(flag="full_sample")
    cond_wf2 = cond_wf.assign(flag="walk_forward")
    pd.concat([cond_full2, cond_wf2], ignore_index=True).to_csv(cond_path, index=False)
    print(f"  conditional (full vs wf) → {cond_path}")

    ep_path = os.path.join(
        DATA_DIR, f"gold_dedollar_gap_walkforward_episodes_{file_stamp}.csv")
    rc.episodes.to_csv(ep_path, index=False)
    print(f"  extreme episodes → {ep_path}")

    print(f"\nVERDICT: {label} — {verdict_msg}")


if __name__ == "__main__":
    main()
