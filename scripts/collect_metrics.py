#!/usr/bin/env python3
"""Collect performance metrics for every lab project.

Runs daily in GitHub Actions (open internet + GITHUB_TOKEN). Standard library only.

Sources
  * npm:      api.npmjs.org downloads (last-day / last-week / last-month / total)
  * PyPI:     pypistats.org recent downloads
  * GitHub:   repo stars/forks/watchers/issues; traffic views & clones and per-path
              views (needs a token with push access: LAB_STATS_TOKEN secret)

Output
  lab/metrics/latest.json          full snapshot
  lab/metrics/history.jsonl        one line per (date, project) for trend analysis
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "lab" / "registry.json"
METRICS = ROOT / "lab" / "metrics"
REPO = os.environ.get("GITHUB_REPOSITORY", "Skymeapple/ai-research-lab")
TOKEN = os.environ.get("LAB_STATS_TOKEN") or os.environ.get("GITHUB_TOKEN")
UA = {"User-Agent": "ai-research-lab-metrics"}


def get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  ! {url}: {e}", file=sys.stderr)
        return None


def gh(path: str):
    h = {"Accept": "application/vnd.github+json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return get(f"https://api.github.com/repos/{REPO}{path}", h)


def npm_stats(pkg: str, created: str) -> dict:
    out = {}
    for period in ("last-day", "last-week", "last-month"):
        d = get(f"https://api.npmjs.org/downloads/point/{period}/{pkg}")
        out[period.replace("-", "_")] = (d or {}).get("downloads", 0)
    start = created or "2026-01-01"
    today = dt.date.today().isoformat()
    d = get(f"https://api.npmjs.org/downloads/range/{start}:{today}/{pkg}")
    out["total"] = sum(x["downloads"] for x in (d or {}).get("downloads", []))
    return out


def pypi_stats(pkg: str) -> dict:
    d = get(f"https://pypistats.org/api/packages/{pkg.lower()}/recent")
    data = (d or {}).get("data", {})
    out = {"last_day": data.get("last_day", 0), "last_week": data.get("last_week", 0), "last_month": data.get("last_month", 0)}
    d = get(f"https://pypistats.org/api/packages/{pkg.lower()}/overall?mirrors=false")
    out["total"] = sum(x["downloads"] for x in (d or {}).get("data", []))
    return out


def main() -> int:
    registry = json.loads(REGISTRY.read_text())
    today = dt.date.today().isoformat()

    repo = gh("") or {}
    views = gh("/traffic/views") or {}
    clones = gh("/traffic/clones") or {}
    paths = gh("/traffic/popular/paths") or []
    referrers = gh("/traffic/popular/referrers") or []

    snapshot = {
        "date": today,
        "repo": {
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "watchers": repo.get("subscribers_count"),
            "open_issues": repo.get("open_issues_count"),
            "views_14d": views.get("count"), "unique_visitors_14d": views.get("uniques"),
            "clones_14d": clones.get("count"), "unique_cloners_14d": clones.get("uniques"),
            "referrers": [{"source": r.get("referrer"), "count": r.get("count")} for r in referrers if isinstance(r, dict)],
        },
        "projects": [],
    }

    with_history = []
    for p in registry["projects"]:
        m = {"id": p["id"], "downloads": {}}
        for pkg in p.get("packages", []):
            eco, name = pkg["ecosystem"], pkg["name"]
            if eco == "npm":
                m["downloads"][f"npm:{name}"] = npm_stats(name, p.get("date"))
            elif eco == "pypi":
                m["downloads"][f"pypi:{name}"] = pypi_stats(name)
        prefix = f"/{REPO}/tree/main/{p['path']}".lower()
        prefix_blob = f"/{REPO}/blob/main/{p['path']}".lower()
        pv = [x for x in paths if isinstance(x, dict) and str(x.get("path", "")).lower().startswith((prefix, prefix_blob))]
        m["page_views_14d"] = sum(x.get("count", 0) for x in pv)
        m["page_uniques_14d"] = sum(x.get("uniques", 0) for x in pv)
        tot = lambda k: sum(d.get(k, 0) or 0 for d in m["downloads"].values())  # noqa: E731
        m["downloads_last_week"] = tot("last_week")
        m["downloads_last_month"] = tot("last_month")
        m["downloads_total"] = tot("total")
        snapshot["projects"].append(m)
        with_history.append({"date": today, "id": p["id"], "dl_week": m["downloads_last_week"],
                             "dl_month": m["downloads_last_month"], "dl_total": m["downloads_total"],
                             "page_views_14d": m["page_views_14d"], "repo_stars": snapshot["repo"]["stars"]})

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "latest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    hist = METRICS / "history.jsonl"
    lines = [l for l in (hist.read_text().splitlines() if hist.exists() else []) if not l.startswith(f'{{"date": "{today}"')]
    lines += [json.dumps(h) for h in with_history]
    hist.write_text("\n".join(lines) + "\n")
    print(f"metrics: {len(with_history)} project(s), stars={snapshot['repo']['stars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
