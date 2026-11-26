"""Harvest recent arXiv papers (2025-01-01 .. today) for each question's frontier queries.

Output: ai2035/frontier/<Qid>.json  — list of {id, title, published, categories, abstract, query}
Grade: observed (records as returned by the arXiv API). Ordering is arXiv relevance,
which is lexical, not a quality judgement; reviewers must read abstracts.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import yaml

API = "https://export.arxiv.org/api/query"
A = "{http://www.w3.org/2005/Atom}"
HERE = Path(__file__).resolve().parent
PER_QUERY = 60
WINDOW = "submittedDate:[202501010000 TO 202612312359]"


def fetch(q: str) -> list[dict]:
    phrase = " AND ".join(f'abs:"{w}"' if " " not in w else f'abs:"{w}"' for w in [q])
    params = {"search_query": f"{phrase} AND {WINDOW}", "max_results": PER_QUERY,
              "sortBy": "relevance", "sortOrder": "descending"}
    for attempt in range(6):
        time.sleep(3.2)
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.ok:
                root = ET.fromstring(r.text)
                out = []
                for e in root.findall(f"{A}entry"):
                    aid = re.sub(r"v\d+$", "", e.findtext(f"{A}id").rsplit("/abs/", 1)[-1])
                    out.append({
                        "id": aid,
                        "title": " ".join(e.findtext(f"{A}title").split()),
                        "published": e.findtext(f"{A}published")[:10],
                        "categories": [c.get("term") for c in e.findall(f"{A}category")],
                        "abstract": " ".join(e.findtext(f"{A}summary").split()),
                        "query": q,
                    })
                return out
        except (requests.RequestException, ET.ParseError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(q)


def main() -> None:
    qs = yaml.safe_load((HERE / "queries.yaml").read_text())
    for qid, queries in qs.items():
        out = HERE / f"{qid}.json"
        if out.exists():
            continue
        seen, recs = set(), []
        for q in queries:
            for rec in fetch(q):
                if rec["id"] not in seen:
                    seen.add(rec["id"]); recs.append(rec)
        out.write_text(json.dumps({"retrieved": time.strftime("%Y-%m-%d"), "queries": queries,
                                   "papers": recs}, ensure_ascii=False, indent=1))
        print(qid, len(recs), flush=True)


if __name__ == "__main__":
    main()
