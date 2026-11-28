"""Compare committed results/*.json with an independent rerun of every experiment.

Usage: python ai2035/audit/compare_repro.py <rerun_ai2035_dir>
For each results JSON present in both trees, walk all numeric leaves (ignoring _meta
and timing fields) and report: number of leaves, share bit-identical, share within
1e-6 relative, and the max relative deviation. Writes audit/reproducibility.md.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = re.compile(r"(_meta|time|runtime|seconds|elapsed|wall|duration)", re.I)


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if not SKIP.search(str(k)):
                yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def main(rerun: Path) -> None:
    rows, total_bad = [], 0
    for f in sorted(ROOT.glob("projects/Q*/results/*.json")):
        g = rerun / f.relative_to(ROOT)
        if not g.exists():
            rows.append((str(f.relative_to(ROOT)), 0, None, None, None, "未重跑"))
            continue
        a = dict(leaves(json.loads(f.read_text())))
        b = dict(leaves(json.loads(g.read_text())))
        keys = a.keys() & b.keys()
        same = close = 0
        maxrel = 0.0
        for k in keys:
            x, y = a[k], b[k]
            if (x == y) or (math.isnan(x) and math.isnan(y)):
                same += 1; close += 1; continue
            rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
            maxrel = max(maxrel, rel)
            close += rel < 1e-6
        n = len(keys)
        status = "逐位一致" if same == n and n else ("数值一致(<1e-6)" if close == n else "存在差异")
        total_bad += status == "存在差异"
        rows.append((str(f.relative_to(ROOT)), n, same / max(n, 1), close / max(n, 1), maxrel, status))
    md = ["# 复现审计\n",
          "在隔离副本中从零重跑全部 `experiment.py` 与 `frontier_experiment.py`（单线程 BLAS），逐个比较 JSON 中的全部数值叶子（忽略运行时间与元数据）。\n",
          "| 结果文件 | 数值数 | 逐位一致比例 | 相对差<1e-6 比例 | 最大相对偏差 | 判定 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r[2] is None:
            md.append(f"| `{r[0]}` | – | – | – | – | {r[5]} |")
        else:
            md.append(f"| `{r[0]}` | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.2e} | {r[5]} |")
    md.append(f"\n存在差异的文件数：{total_bad} / {len(rows)}")
    (ROOT / "audit" / "reproducibility.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
