"""Independent citation audit for every report and paper.

1. Extract every new-style arXiv ID (YYMM.NNNNN) cited in README.md / paper.md / CONCLUSIONS.md.
2. Resolve each ID against the arXiv API (batched id_list) — an ID that resolves to nothing fails.
3. For each citation, compare the title written next to it (same line) with the API title
   using token overlap; low overlap is flagged for manual review (not auto-failed, since
   reference lines are often abbreviated or translated).
Output: audit/citations.json and audit/citations.md
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
A = "{http://www.w3.org/2005/Atom}"
ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
CACHE = ROOT / "audit" / "_arxiv_cache.json"


def norm(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def resolve(ids: list[str]) -> dict[str, str]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [i for i in ids if i not in cache]
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        for attempt in range(6):
            time.sleep(3.2)
            try:
                r = requests.get("https://export.arxiv.org/api/query",
                                 params={"id_list": ",".join(chunk), "max_results": 60}, timeout=90)
                if r.ok:
                    root = ET.fromstring(r.text)
                    got = {}
                    for e in root.findall(f"{A}entry"):
                        eid = re.sub(r"v\d+$", "", e.findtext(f"{A}id", "").rsplit("/abs/", 1)[-1])
                        t = " ".join((e.findtext(f"{A}title") or "").split())
                        if eid and t and t != "Error":
                            got[eid] = t
                    for i in chunk:
                        cache[i] = got.get(i, "")
                    break
            except (requests.RequestException, ET.ParseError):
                pass
            time.sleep(5 * (attempt + 1))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    return cache


def main() -> None:
    cites = []
    for f in sorted(ROOT.glob("projects/Q*/*.md")):
        in_refs = False
        for ln in f.read_text().splitlines():
            if ln.startswith("#"):
                in_refs = "参考文献" in ln or "References" in ln
            for m in ID.finditer(ln):
                yymm = int(m.group(1)[:4])
                if not (701 <= yymm <= 2612) or int(m.group(1)[2:4]) > 12:
                    continue
                cites.append({"file": str(f.relative_to(ROOT)), "id": m.group(1), "line": ln.strip()[:300], "in_refs": in_refs})
    ids = sorted({c["id"] for c in cites})
    titles = resolve(ids)
    unresolved, low = [], []
    for c in cites:
        t = titles.get(c["id"], "")
        c["api_title"] = t
        if not t:
            unresolved.append(c)
            continue
        tt = norm(t)
        c["overlap"] = len(tt & norm(c["line"])) / max(1, len(tt))
        # only judge lines that look like reference entries (contain latin title words)
        if c["in_refs"] and c["overlap"] < 0.3:
            low.append(c)
    (ROOT / "audit" / "citations.json").write_text(json.dumps(
        {"n_citations": len(cites), "n_unique_ids": len(ids), "unresolved": unresolved,
         "low_title_overlap": low}, ensure_ascii=False, indent=1))
    md = [f"# 引文审计\n", f"- 引用次数：{len(cites)}；唯一 arXiv 编号：{len(ids)}",
          f"- 无法在 arXiv 解析的编号：{len(unresolved)}",
          f"- 标题与 API 记录重合度 <30% 的参考文献条目（需人工复核）：{len(low)}\n"]
    for c in unresolved:
        md.append(f"- [未解析] {c['file']} `{c['id']}` — {c['line'][:120]}")
    for c in low:
        md.append(f"- [低重合 {c['overlap']:.2f}] {c['file']} `{c['id']}` API: *{c['api_title']}* — 文中: {c['line'][:120]}")
    (ROOT / "audit" / "citations.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:4]))


if __name__ == "__main__":
    main()
