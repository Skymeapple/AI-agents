#!/usr/bin/env python3
"""Market check: who already solves this, and how many people use them?

Usage:
  python3 scripts/npm_market.py "agents.md lint" "cursorrules" [--size 15] [--min 0]
  python3 scripts/npm_market.py --name ctxlint          # exact + scoped name collisions

Uses the npm registry search API (registry.npmjs.org, reachable from the lab sandbox),
which reports weekly/monthly downloads per package. Standard library only.

Gap gate (LAB_PROTOCOL.md, Phase 1): do NOT build if any package with >= 1,000 weekly
downloads already covers >= 70% of the planned features.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "ai-research-lab-market"}


def search(text: str, size: int = 15) -> list[dict]:
    url = f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(text)}&size={size}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        data = json.load(r)
    out = []
    for o in data.get("objects", []):
        p = o["package"]
        dl = o.get("downloads") or {}
        out.append({
            "name": p["name"],
            "weekly": dl.get("weekly", 0) or 0,
            "monthly": dl.get("monthly", 0) or 0,
            "version": p.get("version", ""),
            "date": (p.get("date") or "")[:10],
            "description": (p.get("description") or "").strip(),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--min", type=int, default=0, help="hide packages below this weekly download count")
    ap.add_argument("--name", help="check a planned package name for exact and scoped collisions")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.name:
        hits = [p for p in search(a.name, 25) if p["name"] == a.name or p["name"].split("/")[-1] == a.name]
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://registry.npmjs.org/{a.name}", headers=UA), timeout=30)
            exact = True
        except Exception:
            exact = False
        print(f"exact '{a.name}' taken: {exact}")
        for p in hits:
            print(f"  collision: {p['name']:<36} {p['weekly']:>9}/wk  {p['description'][:70]}")
        if exact or any(p["weekly"] >= 100 for p in hits):
            print("=> pick another name")
            return 1
        print("=> name looks free")
        return 0

    seen: dict[str, dict] = {}
    for q in a.queries:
        for p in search(q, a.size):
            seen.setdefault(p["name"], {**p, "queries": []})["queries"].append(q)
    rows = sorted((p for p in seen.values() if p["weekly"] >= a.min), key=lambda p: -p["weekly"])
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'weekly':>10} {'monthly':>10}  {'package':<38} {'updated':<10}  description")
    for p in rows:
        print(f"{p['weekly']:>10} {p['monthly']:>10}  {p['name']:<38} {p['date']:<10}  {p['description'][:80]}")
    big = [p for p in rows if p["weekly"] >= 1000]
    print(f"\n{len(rows)} packages; {len(big)} with >= 1,000 weekly downloads. "
          "Read the READMEs of the top ones before scoring Gap.")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet when piped into head
    sys.exit(main())
