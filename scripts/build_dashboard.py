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

# ---------------------------------------------------------------------------
# Market benchmarks, channels, experiments
# ---------------------------------------------------------------------------
from statistics import median  # noqa: E402

CURVES_P = ROOT / "lab" / "metrics" / "launch_curves.json"
CURVES = json.loads(CURVES_P.read_text()) if CURVES_P.exists() else {}
EXP_P = ROOT / "lab" / "experiments.json"
EXPS = json.loads(EXP_P.read_text()).get("experiments", []) if EXP_P.exists() else []
AUTO = {"hook", "mcp", "action", "pre-commit"}
SMALL = 5000  # "small entrant" = under 5,000 weekly downloads
today = dt.date.today().isoformat()


def own_pkg(p: dict) -> str | None:
    return next((x["name"] for x in p.get("packages", []) if x["ecosystem"] == "npm"), None)


def day30_rate(p: dict) -> int | None:
    """Weekly downloads around day 30 (or the current weekly rate if younger and published)."""
    c = CURVES.get(own_pkg(p) or "")
    if c and c.get("complete"):
        return c["week_at_day30"]
    if c:
        return by_id.get(p["id"], {}).get("downloads_last_week", 0)
    return None  # never published


def med(xs):
    xs = [x for x in xs if x is not None]
    return int(median(xs)) if xs else None


def fmt(x):
    return "-" if x is None else f"{x:,}"


bench_now = LATEST.get("benchmarks", {})
if bench_now:
    out += ["", "## Market benchmarks", "",
            "These are competitors in each project's category. *Day-30 rate* is a package's downloads in days 24–30 after its first publish, which lets the lab compare a new project against established ones at the same age.", ""]
    for cat, rows in bench_now.items():
        ours = [p for p in REG["projects"] if p.get("category") == cat]
        small = [CURVES.get(r["name"], {}).get("week_at_day30") for r in rows if r["dl_week"] < SMALL and CURVES.get(r["name"], {}).get("complete")]
        out += [f"### {cat}", "", f"Median day-30 rate of small entrants (< {SMALL:,}/wk now): **{fmt(med(small))}**", "",
                "| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |", "|---|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: -r["dl_week"]):
            c = CURVES.get(r["name"], {})
            out.append(f"| {r['name']} | {fmt(r['dl_week'])} | {fmt(c.get('d7'))} | {fmt(c.get('d30'))} | {fmt(c.get('week_at_day30'))} |")
        for p in ours:
            c = CURVES.get(own_pkg(p) or "", {})
            m = by_id.get(p["id"], {})
            label = f"**{p['name']} (lab)**" + ("" if c else f" _{p.get('status', 'unpublished')}_")
            out.append(f"| {label} | {fmt(m.get('downloads_last_week'))} | {fmt(c.get('d7'))} | {fmt(c.get('d30'))} | {fmt(c.get('week_at_day30'))} |")
        out.append("")

published = [p for p in REG["projects"] if CURVES.get(own_pkg(p) or "")]
out += ["## Downloads by distribution channel", "",
        "This tests experiment E2: tools that run automatically (agent hooks, MCP servers, Actions, pre-commit) should earn recurring downloads; manual CLIs shouldn't.", "",
        "| Channel group | Published projects | Median weekly downloads |", "|---|---:|---:|"]
groups = {"auto-invoked (hook/mcp/action/pre-commit)": [p for p in published if AUTO & set(p.get("distribution", []))],
          "cli-only": [p for p in published if not AUTO & set(p.get("distribution", []))]}
for g, ps in groups.items():
    out.append(f"| {g} | {len(ps)} | {fmt(med([by_id.get(p['id'], {}).get('downloads_last_week', 0) for p in ps]))} |")


def reading(e: dict) -> str:
    t = e.get("metric", {}).get("type")
    if t == "gated_vs_ungated_median":
        g = med([day30_rate(p) for p in published if p.get("gated")])
        u = med([day30_rate(p) for p in published if not p.get("gated")])
        return f"gated median {fmt(g)} vs ungated {fmt(u)} ({sum(1 for p in published if p.get('gated'))} gated published)"
    if t == "channel_median_dl_week":
        return "; ".join(f"{g.split(' ')[0]}: {fmt(med([by_id.get(p['id'], {}).get('downloads_last_week', 0) for p in ps]))} (n={len(ps)})" for g, ps in groups.items())
    if t == "releases_vs_downloads":
        many = [day30_rate(p) for p in published if len(p.get("releases", [])) >= 3]
        one = [day30_rate(p) for p in published if len(p.get("releases", [])) < 3]
        return f">=3 releases: {fmt(med(many))} (n={len(many)}); fewer: {fmt(med(one))} (n={len(one)})"
    if t == "project_dl_week":
        return fmt(by_id.get(e["metric"].get("project"), {}).get("downloads_last_week"))
    return "-"


if EXPS:
    out += ["", "## Experiments", "", "These were written down before the data arrived. A daily mission evaluates each one once its date passes (see `lab/experiments.json`).", "",
            "| ID | Change | Evaluate on | Status | Current reading |", "|---|---|---|---|---|"]
    for e in EXPS:
        status = f"✅ {e['verdict']}" if e.get("verdict") else ("⏰ **DUE**" if e["evaluate_on"] <= today else "running")
        out.append(f"| {e['id']} | {e['change'][:90]}{'…' if len(e['change']) > 90 else ''} | {e['evaluate_on']} | {status} | {reading(e)} |")

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
