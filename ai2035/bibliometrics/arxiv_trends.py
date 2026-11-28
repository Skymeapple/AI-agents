"""arXiv publication trends for each of the 20 questions (2015-2025).

For each question and year, counts arXiv records whose abstract contains any of
the question's key phrases, and divides by the count for a broad AI/ML baseline
(cs.LG OR cs.AI OR stat.ML) in the same year. The share controls for the overall
growth of the field, so a rising share means a topic is gaining *relative*
attention, not merely riding the boom.

Grade of evidence: observed (counts from the arXiv API on the run date).
Limitations: lexical matching misses work using other vocabulary and catches
some off-topic uses; arXiv over-represents ML and under-represents journals in
cognitive science and neuroscience. Counts are signals of attention, not of
progress.

Usage:  python ai2035/bibliometrics/arxiv_trends.py   (~12 minutes, rate-limited)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
import yaml

API = "https://export.arxiv.org/api/query"
YEARS = list(range(2015, 2026))
BASELINE = "(cat:cs.LG OR cat:cs.AI OR cat:stat.ML)"
HERE = Path(__file__).resolve().parent
OUT = HERE / "arxiv_counts.json"
_last = [0.0]


def count(query: str) -> int:
    for attempt in range(6):
        wait = 3.1 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(API, params={"search_query": query, "max_results": 1}, timeout=60)
            m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", r.text)
            if r.ok and m:
                return int(m.group(1))
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"arXiv query failed: {query}")


def window(y: int) -> str:
    return f"submittedDate:[{y}01010000 TO {y}12312359]"


def main() -> None:
    qs = yaml.safe_load((HERE.parent / "questions.yaml").read_text())["questions"]
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "baseline" not in data:
        data["baseline"] = {str(y): count(f"{BASELINE} AND {window(y)}") for y in YEARS}
        OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for q in qs:
        if q["id"] in data:
            continue
        phrases = " OR ".join(f'abs:"{p}"' for p in q["arxiv"])
        data[q["id"]] = {
            "title": q["title"],
            "query": phrases,
            "counts": {str(y): count(f"({phrases}) AND {window(y)}") for y in YEARS},
        }
        OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(q["id"], data[q["id"]]["counts"], flush=True)
    data["_meta"] = {"source": API, "retrieved": time.strftime("%Y-%m-%d"), "years": YEARS,
                     "baseline_query": BASELINE, "grade": "observed"}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
