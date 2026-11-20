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
  lab/metrics/benchmarks.jsonl     one line per (date, competitor package): weekly downloads
  lab/metrics/launch_curves.json   downloads in the first 7 / 30 days after first publish,
                                   for competitors AND lab projects (same-age comparison)
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
BENCHMARKS = ROOT / "lab" / "benchmarks.json"
METRICS = ROOT / "lab" / "metrics"
REPO = os.environ.get("GITHUB_REPOSITORY", "Skymeapple/AI-agents")
BRANCH = os.environ.get("LAB_BRANCH", "research-lab")
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


def npm_created(pkg: str) -> str | None:
    d = get(f"https://registry.npmjs.org/{pkg.replace('/', '%2F')}")
    created = ((d or {}).get("time") or {}).get("created")
    return created[:10] if created else None


def launch_curve(pkg: str, created: str) -> dict:
    """Downloads in the first 7 / 30 days after first publish, and the weekly rate around day 30."""
    start = dt.date.fromisoformat(created)
    end = start + dt.timedelta(days=29)
    d = get(f"https://api.npmjs.org/downloads/range/{start.isoformat()}:{end.isoformat()}/{pkg}")
    days = [x["downloads"] for x in (d or {}).get("downloads", [])]
    age = (dt.date.today() - start).days
    return {
        "created": created,
        "complete": age >= 30,
        "d7": sum(days[:7]),
        "d30": sum(days[:30]),
        "week_at_day30": sum(days[23:30]),
        "daily": days[:30],
    }


def collect_launch_curves(names: list[str]) -> dict:
    path = METRICS / "launch_curves.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        if cache.get(name, {}).get("complete"):
            continue  # first 30 days are history; never changes
        created = npm_created(name)
        if created:
            cache[name] = launch_curve(name, created)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return cache


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
        prefix = f"/{REPO}/tree/{BRANCH}/{p['path']}".lower()
        prefix_blob = f"/{REPO}/blob/{BRANCH}/{p['path']}".lower()
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

    # Competitor benchmarks: current weekly downloads + launch curves.
    bench = json.loads(BENCHMARKS.read_text()) if BENCHMARKS.exists() else {"categories": {}}
    snapshot["benchmarks"] = {}
    bench_lines = []
    all_names = []
    for cat, spec in bench.get("categories", {}).items():
        rows = []
        for name in spec.get("packages", []):
            d = get(f"https://api.npmjs.org/downloads/point/last-week/{name}")
            wk = (d or {}).get("downloads", 0)
            rows.append({"name": name, "dl_week": wk})
            bench_lines.append({"date": today, "category": cat, "package": name, "dl_week": wk})
            all_names.append(name)
        snapshot["benchmarks"][cat] = rows
    own = [pkg["name"] for p in registry["projects"] for pkg in p.get("packages", [])
           if pkg["ecosystem"] == "npm" and npm_created(pkg["name"])]
    collect_launch_curves(all_names + own)
    bh = METRICS / "benchmarks.jsonl"
    old = [l for l in (bh.read_text().splitlines() if bh.exists() else []) if l.strip() and not l.startswith(f'{{"date": "{today}"')]
    bh.write_text("\n".join(old + [json.dumps(b) for b in bench_lines]) + "\n")

    (METRICS / "latest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    hist = METRICS / "history.jsonl"
    lines = [l for l in (hist.read_text().splitlines() if hist.exists() else []) if not l.startswith(f'{{"date": "{today}"')]
    lines += [json.dumps(h) for h in with_history]
    hist.write_text("\n".join(lines) + "\n")
    print(f"metrics: {len(with_history)} project(s), stars={snapshot['repo']['stars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
