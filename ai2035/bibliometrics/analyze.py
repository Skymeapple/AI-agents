"""Turn arxiv_counts.json into trend statistics, a figure, and per-report sections.

Statistic: share_t = topic_count_t / baseline_count_t (baseline = cs.LG|cs.AI|stat.ML).
Momentum  = OLS slope of log(share) on year over 2019-2025, reported as an annual
            % change in share with a 95% CI. Positive => the topic is growing
            faster than AI/ML as a whole. Grade: observed. Lexical-matching
            caveats are in arxiv_trends.py and repeated in every injected section.

Usage: python ai2035/bibliometrics/analyze.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.util import plt as get_plt  # noqa: E402

import yaml  # noqa: E402

DATA = json.loads((ROOT / "bibliometrics" / "arxiv_counts.json").read_text())
QS = yaml.safe_load((ROOT / "questions.yaml").read_text())["questions"]
WIN = list(range(2019, 2026))
START, END = "<!-- BIBLIO:start -->", "<!-- BIBLIO:end -->"


def stats_for(qid: str) -> dict:
    base = DATA["baseline"]
    c = DATA[qid]["counts"]
    years = sorted(int(y) for y in c)
    share = {y: c[str(y)] / base[str(y)] for y in years}
    x = np.array(WIN, float)
    ylog = np.log(np.array([max(share[y], 1e-9) for y in WIN]))
    lr = stats.linregress(x, ylog)
    tcrit = stats.t.ppf(0.975, len(WIN) - 2)
    to_pct = lambda b: 100 * (np.exp(b) - 1)
    return {
        "counts": {y: c[str(y)] for y in years},
        "share": share,
        "n_2025": c["2025"],
        "share_2025": share[2025],
        "share_2019": share[2019],
        "momentum_pct": to_pct(lr.slope),
        "momentum_ci": (to_pct(lr.slope - tcrit * lr.stderr), to_pct(lr.slope + tcrit * lr.stderr)),
        "r2": lr.rvalue ** 2,
    }


def verdict(s: dict) -> str:
    lo, hi = s["momentum_ci"]
    if lo > 0:
        return "相对升温（增速显著快于 AI/ML 总体）"
    if hi < 0:
        return "相对降温（增速显著慢于 AI/ML 总体，绝对量仍可能增长）"
    return "与 AI/ML 总体同步（相对份额无显著趋势）"


def section(q: dict, s: dict, rank: int) -> str:
    lo, hi = s["momentum_ci"]
    rows = "\n".join(
        f"| {y} | {s['counts'][y]:,} | {100 * s['share'][y]:.2f}% |" for y in range(2015, 2026)
    )
    phrases = "、".join(f"“{p}”" for p in q["arxiv"])
    return f"""{START}
## 文献计量信号（arXiv，[实测]）

检索式：摘要含 {phrases} 之一；基线为同年 cs.LG / cs.AI / stat.ML 论文总数。数据：`ai2035/bibliometrics/arxiv_counts.json`（检索日 {DATA['_meta']['retrieved']}）。

| 年份 | 命中论文数 | 占 AI/ML 基线比例 |
|---|---|---|
{rows}

- 2019–2025 年份额的对数线性趋势：**每年 {s['momentum_pct']:+.1f}%**（95% CI {lo:+.1f}% ~ {hi:+.1f}%，R²={s['r2']:.2f}），判定：**{verdict(s)}**；在 20 个问题中按相对增速排第 {rank} 位。
- 解读边界：分子覆盖 arXiv 全部类别、分母仅为 AI/ML 三个类别，所以“比例”是相对关注度指数，并非严格的子集占比；关键词匹配既会漏掉使用其他术语的工作，也会混入无关用法；arXiv 偏重机器学习社区，认知科学、神经科学与期刊论文覆盖不足。该信号衡量的是**研究关注度**，不是**问题被解决的程度**。
{END}"""


def inject(slug_dir: Path, text: str) -> bool:
    readme = slug_dir / "README.md"
    if not readme.exists():
        return False
    body = readme.read_text()
    if START in body:
        body = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: text, body, flags=re.S)
    elif "<!-- BIBLIO -->" in body:
        body = body.replace("<!-- BIBLIO -->", text)
    else:
        body = body.replace("## 6. 结论", text + "\n\n## 6. 结论", 1)
    readme.write_text(body)
    return True


def main() -> None:
    allstats = {q["id"]: stats_for(q["id"]) for q in QS}
    order = sorted(allstats, key=lambda k: -allstats[k]["momentum_pct"])
    rank = {k: i + 1 for i, k in enumerate(order)}

    # summary table
    lines = ["| 排名 | 问题 | 2025 论文数 | 2025 份额 | 份额年变化 (95% CI) | 判定 |", "|---|---|---|---|---|---|"]
    for k in order:
        q = next(x for x in QS if x["id"] == k)
        s = allstats[k]
        lo, hi = s["momentum_ci"]
        lines.append(f"| {rank[k]} | {k} {q['title']} | {s['n_2025']:,} | {100*s['share_2025']:.2f}% | "
                     f"{s['momentum_pct']:+.1f}% ({lo:+.1f}~{hi:+.1f}) | {verdict(s)} |")
    b = DATA["baseline"]
    lines.append("")
    lines.append(f"基线（cs.LG|cs.AI|stat.ML）论文数：2015 年 {b['2015']:,} → 2025 年 {b['2025']:,}。")
    (ROOT / "bibliometrics" / "summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "bibliometrics" / "summary.json").write_text(json.dumps(
        {k: {**v, "counts": {str(a): c for a, c in v["counts"].items()},
             "share": {str(a): c for a, c in v["share"].items()}, "rank": rank[k]} for k, v in allstats.items()},
        ensure_ascii=False, indent=1))

    # figure: small multiples of share
    plt = get_plt()
    fig, axes = plt.subplots(4, 5, figsize=(17, 12), sharex=True)
    for ax, q in zip(axes.flat, QS):
        s = allstats[q["id"]]
        ys = list(range(2015, 2026))
        ax.plot(ys, [100 * s["share"][y] for y in ys], marker="o", ms=3)
        ax.set_title(f"{q['id']} {q['title'][:12]}", fontsize=9)
        ax.text(0.03, 0.88, f"{s['momentum_pct']:+.0f}%/年", transform=ax.transAxes, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supylabel("占 cs.LG/cs.AI/stat.ML 论文比例 (%)")
    fig.suptitle("20 个基础问题的 arXiv 相对关注度（2015–2025）")
    fig.tight_layout()
    fig.savefig(ROOT / "bibliometrics" / "trends.png")

    for q in QS:
        d = ROOT / "projects" / f"{q['id']}-{q['slug']}"
        ok = inject(d, section(q, allstats[q["id"]], rank[q["id"]]))
        print(q["id"], "injected" if ok else "README missing")


if __name__ == "__main__":
    main()
