"""Check that decimals quoted in reports/papers appear in the project's results JSON.

For each Markdown file, extract decimals with >=2 decimal places (e.g. 0.585, 469.0 excluded),
skipping the injected bibliometrics block and reference lists. A number is "traced" if some
numeric leaf in the project's results/*.json rounds to it at the quoted precision (also
accepting x100 for percentages and sign flips). Untraced numbers are listed for review —
they include literature figures, theory constants and hypothesis thresholds, which are
legitimately not in the JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-−]?\d+\.(\d{2,})(?![\w.])")


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)


def main() -> None:
    rows, tot_n, tot_ok = [], 0, 0
    for proj in sorted(ROOT.glob("projects/Q*")):
        vals = [v for f in proj.glob("results/*.json") for v in leaves(json.loads(f.read_text()))]
        keys = {}
        for v in vals:
            for d in range(2, 7):
                for x in (v, -v, 100 * v, -100 * v):
                    keys.setdefault(d, set()).add(round(x, d))
        for md in ("README.md", "paper.md", "CONCLUSIONS.md"):
            p = proj / md
            if not p.exists():
                continue
            text = re.sub(r"<!-- BIBLIO:start -->.*?<!-- BIBLIO:end -->", "", p.read_text(), flags=re.S)
            text = text.split("## 参考文献")[0]
            n = ok = 0
            miss = []
            for m in NUM.finditer(text):
                s = m.group(0).replace("−", "-")
                d = len(m.group(1))
                if d > 6 or re.fullmatch(r"\d{4}\.\d{4,5}", s):
                    continue
                n += 1
                if round(float(s), d) in keys.get(d, ()) or round(-float(s), d) in keys.get(d, ()):
                    ok += 1
                else:
                    miss.append(s)
            tot_n += n; tot_ok += ok
            rows.append((f"{proj.name}/{md}", n, ok, miss[:8]))
    out = ["# 数字可追溯审计\n",
           f"报告/论文/结论卡中共抽取 {tot_n} 个 ≥2 位小数（不含文献计量节与参考文献），"
           f"其中 {tot_ok} 个（{100*tot_ok/max(tot_n,1):.1f}%）可在本项目 results/*.json 中按引用精度匹配。",
           "未匹配者多为文献引用值、理论常数、假设阈值或由多个结果推导出的比值，需结合上下文复核。\n",
           "| 文件 | 小数个数 | 可追溯 | 比例 | 未匹配示例 |", "|---|---|---|---|---|"]
    for f, n, ok, miss in rows:
        out.append(f"| `{f}` | {n} | {ok} | {100*ok/max(n,1):.0f}% | {', '.join(miss)} |")
    (ROOT / "audit" / "traceability.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:3]))
    for f, n, ok, miss in sorted(rows, key=lambda r: r[2] / max(r[1], 1))[:8]:
        print(f, n, ok, miss)


if __name__ == "__main__":
    main()
