#!/usr/bin/env python3
"""Render DASHBOARD.md and the project table in README.md from registry + metrics.

Value score (per project) = downloads in the last 7 days + 3 x unique page visitors (14d).
Downloads are the north-star metric; page visitors catch projects that are seen but not
yet installed (a README / positioning problem, not a demand problem).
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = json.loads((ROOT / "lab" / "registry.json").read_text())
LATEST_P = ROOT / "lab" / "metrics" / "latest.json"
HIST_P = ROOT / "lab" / "metrics" / "history.jsonl"
LATEST = json.loads(LATEST_P.read_text()) if LATEST_P.exists() else {"date": None, "repo": {}, "projects": []}
HIST = [json.loads(l) for l in HIST_P.read_text().splitlines() if l.strip()] if HIST_P.exists() else []

by_id = {m["id"]: m for m in LATEST["projects"]}
hist_by_id = defaultdict(list)
for h in HIST:
    hist_by_id[h["id"]].append(h)


def score(m: dict) -> int:
    return int((m.get("downloads_last_week") or 0) + 3 * (m.get("page_uniques_14d") or 0))


def trend(pid: str) -> str:
    h = sorted(hist_by_id[pid], key=lambda x: x["date"])
    if len(h) < 2:
        return "-"
    week_ago = [x for x in h if x["date"] <= (dt.date.fromisoformat(h[-1]["date"]) - dt.timedelta(days=7)).isoformat()]
    base = week_ago[-1] if week_ago else h[0]
    d = h[-1]["dl_week"] - base["dl_week"]
    return f"+{d}" if d > 0 else str(d)


def spark(pid: str) -> str:
    bars = "▁▂▃▄▅▆▇█"
    vals = [x["dl_week"] for x in sorted(hist_by_id[pid], key=lambda x: x["date"])][-14:]
    if not vals or max(vals) == 0:
        return ""
    return "".join(bars[min(7, int(v / max(vals) * 7))] for v in vals)


rows = []
for p in REG["projects"]:
    m = by_id.get(p["id"], {})
    rows.append((score(m), p, m))
rows.sort(key=lambda r: (-r[0], r[1]["date"]))

repo = {k: ("-" if v is None else v) for k, v in LATEST.get("repo", {}).items()}
total_dl = sum((m.get("downloads_total") or 0) for _, _, m in rows)
week_dl = sum((m.get("downloads_last_week") or 0) for _, _, m in rows)

out = [
    "# Lab Dashboard",
    "",
    f"_Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · metrics as of {LATEST.get('date') or 'n/a (first run pending)'}_",
    "",
    "| Projects shipped | Downloads (7d) | Downloads (all-time) | ⭐ Stars | Forks | Visitors (14d) | Clones (14d) |",
    "|---:|---:|---:|---:|---:|---:|---:|",
    f"| {len(rows)} | {week_dl} | {total_dl} | {repo.get('stars', '-')} | {repo.get('forks', '-')} | {repo.get('unique_visitors_14d', '-')} | {repo.get('clones_14d', '-')} |",
    "",
    "## Leaderboard",
    "",
    "Value score = 7-day downloads + 3 × unique page visitors (14d).",
    "",
    "| # | Project | Shipped | Score | DL 7d | Δ vs last wk | DL 30d | DL total | Page uniques 14d | Trend | Status |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
]
for i, (s, p, m) in enumerate(rows, 1):
    out.append(
        f"| {i} | [{p['name']}]({p['path']}) | {p['date']} | {s} | {m.get('downloads_last_week', 0)} | {trend(p['id'])} | "
        f"{m.get('downloads_last_month', 0)} | {m.get('downloads_total', 0)} | {m.get('page_uniques_14d', 0)} | {spark(p['id'])} | {p.get('status', 'active')} |"
    )

refs = repo.get("referrers") or []
if refs:
    out += ["", "## Where visitors come from (14d)", "", "| Referrer | Views |", "|---|---:|"]
    out += [f"| {r['source']} | {r['count']} |" for r in refs[:10]]

out += [
    "",
    "## Pain points addressed",
    "",
    "| Project | Pain point | Evidence |",
    "|---|---|---|",
]
for p in sorted(REG["projects"], key=lambda x: x["date"], reverse=True):
    ev = " ".join(f"[{i + 1}]({u})" for i, u in enumerate(p.get("evidence", [])[:4]))
    out.append(f"| {p['name']} | {p['pain_point']} | {ev} |")

(ROOT / "DASHBOARD.md").write_text("\n".join(out) + "\n")

# README project table between markers
readme = ROOT / "README.md"
txt = readme.read_text()
start, end = "<!-- projects:start -->", "<!-- projects:end -->"
if start in txt and end in txt:
    table = ["| Project | What it does | Install |", "|---|---|---|"]
    for p in sorted(REG["projects"], key=lambda x: x["date"], reverse=True):
        table.append(f"| [**{p['name']}**]({p['path']}) | {p['tagline']} | `{p.get('install', '-')}` |")
    pre, rest = txt.split(start, 1)
    post = rest.split(end, 1)[1]
    readme.write_text(f"{pre}{start}\n{chr(10).join(table)}\n{end}{post}")
print(f"dashboard: {len(rows)} project(s)")
