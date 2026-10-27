"""Export a day's radar output as the data files behind the shared web page.

The page (a claude.ai Dashboard artifact) holds no numbers of its own: every
figure on it is read from one of these files, so refreshing the page is
re-uploading the files and pointing the page's datasets at them. Most files are
computed from the store. Two things are written by the daily agent, because they
come from its research rather than from the radar: the headline takeaway with
the key findings, and the caveats for the day. Those arrive in
``reports/DATE-dashboard.json`` and are held to the inbox's standard -- a
finding without a source URL is refused, not published.

```json
{
  "takeaway": "One or two sentences: what is growing fastest and what changed.",
  "findings": [{"id": "video", "heading": "...", "text": "...", "source": "https://..."}],
  "caveats": ["..."]
}
```
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from pathlib import Path

from sciscout.provenance import Grade

from .params import FIXED, KNOBS, Params
from .pipeline import Analysis, analyse
from .report import TOP_N, _evidence_ledger
from .store import Store

ID_RE = re.compile(r"^[a-z0-9-]{1,40}$")

FILES = {
    "summary": ["report_date", "takeaway", "apps_tracked", "with_growth_evidence", "baseline_only",
                "measured_share", "observations", "directions", "hypotheses", "matured_forecasts",
                "latest_spearman", "latest_precision", "first_scorecard_expected", "tuning_status",
                "tuning_reason"],
    "apps": ["rank", "app", "url", "kind", "tags", "score", "evidence", "signal", "evidence_type",
             "weekly_growth_pct", "flag"],
    "directions": ["direction", "status", "change", "top_quartile_share", "mean_score", "apps",
                   "open_source", "commercial", "leaders"],
    "hypotheses": ["pattern", "direction", "rationale", "leaders", "researched"],
    "findings": ["id", "heading", "text", "source"],
    "caveats": ["id", "text"],
    "scorecard": ["made_on", "apps_compared", "spearman", "precision_at_k", "k"],
    "movers": ["app", "url", "previous_date", "previous", "now", "change", "per_day", "rounded"],
    "params": ["parameter", "value", "basis"],
    "sources": ["collector", "last_ok", "consecutive_failures", "last_count", "last_error"],
}


class DashboardError(ValueError):
    pass


def load_editorial(path: Path) -> dict:
    """Read and validate the agent-written takeaway, findings and caveats."""
    if not path.exists():
        raise DashboardError(
            f"{path} is missing: the daily agent writes it with the analysis (DAILY_AGENT.md step 5)"
        )
    raw = json.loads(path.read_text())
    takeaway = " ".join(str(raw.get("takeaway", "")).split())
    if not takeaway:
        raise DashboardError(f"{path}: 'takeaway' is required")
    findings = raw.get("findings") or []
    if not findings:
        raise DashboardError(f"{path}: at least one finding is required")
    seen = set()
    for i, f in enumerate(findings):
        where = f"{path} finding #{i}"
        for key in ("id", "heading", "text", "source"):
            if not str(f.get(key, "")).strip():
                raise DashboardError(f"{where}: '{key}' is required")
        if not ID_RE.match(f["id"]) or f["id"] in seen:
            raise DashboardError(f"{where}: id must be unique, lowercase letters, digits and hyphens")
        if not str(f["source"]).startswith(("https://", "http://")):
            raise DashboardError(f"{where}: source must be a URL")
        seen.add(f["id"])
    caveats = [" ".join(str(c).split()) for c in raw.get("caveats") or [] if str(c).strip()]
    return {"takeaway": takeaway, "findings": findings, "caveats": caveats}


def _rows(analysis: Analysis, store: Store, editorial: dict) -> dict[str, list[dict]]:
    a = analysis
    apps = []
    for rank, s in enumerate(a.scores, 1):
        app = a.apps[s.app_id]
        best = s.signals[max(range(len(s.signals)), key=lambda k: s.signal_scores[k])]
        evidence_type = {
            "api": "Measured (API)", "page_extract": "Page read", "reported": "Company-reported",
        }[best.method.value]
        weekly = round(math.expm1(best.rel_per_day * 7) * 100, 1) if best.rel_per_day is not None else ""
        apps.append({
            "rank": rank, "app": app.name, "url": app.url, "kind": app.kind.value.replace("_", " "),
            "tags": ", ".join(app.tags[:3]), "score": round(s.score, 4), "evidence": round(s.evidence, 3),
            "signal": s.headline(), "evidence_type": evidence_type, "weekly_growth_pct": weekly,
            "flag": "; ".join(s.anomalies),
        })

    directions = [{
        "direction": d.key, "status": d.status,
        "change": round(d.change, 4) if d.change is not None else "",
        "top_quartile_share": round(d.top_share, 4), "mean_score": round(d.mean_score, 4),
        "apps": len(d.members), "open_source": d.open_count, "commercial": d.commercial_count,
        "leaders": ", ".join(a.apps[i].name for i, _ in d.scored[:3]),
    } for d in a.directions]

    hypotheses = [{
        "pattern": o.kind, "direction": o.direction, "rationale": o.rationale,
        "leaders": ", ".join(o.evidence),
        "researched": (f"{o.verdict['verdict']} ({o.verdict['observed_on']}): {o.verdict['note']}".rstrip(": ")
                       if o.verdict else ""),
    } for o in a.opportunities]

    scorecard = [{
        "made_on": e.made_on.isoformat(), "apps_compared": e.apps,
        "spearman": round(e.spearman, 4) if e.spearman is not None else "",
        "precision_at_k": round(e.precision_at_k, 4) if e.precision_at_k is not None else "",
        "k": e.k,
    } for e in a.evaluations]

    movers = [{
        "app": a.apps[m.app_id].name, "url": a.apps[m.app_id].url,
        "previous_date": m.start.isoformat(), "previous": round(m.before), "now": round(m.after),
        "change": round(m.change), "per_day": round(m.per_day), "rounded": "yes" if m.rounded else "",
    } for m in a.movers[:TOP_N]]

    predictions = store.load_predictions()
    first_expected = ""
    if not a.evaluations and predictions:
        first = min(dt.date.fromisoformat(p["made_on"]) for p in predictions)
        first_expected = (first + dt.timedelta(days=FIXED["horizon_days"])).isoformat()
    latest = a.evaluations[-1] if a.evaluations else None

    iterations = [i for i in store.load_iterations() if i["date"] == a.date.isoformat()]
    tuning = iterations[-1] if iterations else None

    ledger = _evidence_ledger(a)
    counts = ledger.counts()
    total = sum(counts.values())
    summary = [{
        "report_date": a.date.isoformat(),
        "takeaway": editorial["takeaway"],
        "apps_tracked": len(a.apps),
        "with_growth_evidence": len(a.scores),
        "baseline_only": len(a.baseline_only),
        "measured_share": round(counts[Grade.OBSERVED] / total, 4) if total else 0,
        "observations": sum(1 for o in store.load_observations() if o.known_on <= a.date),
        "directions": len(a.directions),
        "hypotheses": len(a.opportunities),
        "matured_forecasts": len(a.evaluations),
        "latest_spearman": round(latest.spearman, 4) if latest and latest.spearman is not None else "",
        "latest_precision": round(latest.precision_at_k, 4) if latest and latest.precision_at_k is not None else "",
        "first_scorecard_expected": first_expected,
        "tuning_status": ("Adopted" if tuning["adopted"] else "No change") if tuning else "Not run",
        "tuning_reason": tuning["reason"] if tuning else "tuning did not run today",
    }]

    params = [{"parameter": k, "value": a.params[k], "basis": a.params.provenance[k]} for k in KNOBS]
    sources = [{
        "collector": h.collector, "last_ok": h.last_ok.isoformat() if h.last_ok else "",
        "consecutive_failures": h.consecutive_failures, "last_count": h.last_count,
        "last_error": (h.last_errors[0][:200] if h.last_errors else ""),
    } for h in a.health]

    findings = [{k: " ".join(str(f[k]).split()) for k in FILES["findings"]} for f in editorial["findings"]]
    caveats = [{"id": f"c{i}", "text": text} for i, text in enumerate(editorial["caveats"], 1)]
    return {
        "summary": summary, "apps": apps, "directions": directions, "hypotheses": hypotheses,
        "findings": findings, "caveats": caveats, "scorecard": scorecard, "params": params,
        "sources": sources, "movers": movers,
    }


def export(store: Store, date: dt.date, out: Path) -> dict[str, Path]:
    """Write one CSV per dataset into ``out``; returns name -> path.

    Read-only on the store: it re-runs the analysis for ``date`` without
    recording a forecast or tuning, so it can be run as often as needed.
    """
    editorial = load_editorial(store.reports / f"{date.isoformat()}-dashboard.json")
    # Rank with the parameters the day's report was made with. If the tuner
    # moved a knob after that, the page must still show the published ranking.
    params = Params.load(store.params_path)
    snapshot = store.snapshots / f"{date.isoformat()}.json"
    if snapshot.exists():
        params.values.update(json.loads(snapshot.read_text())["params"])
    rows = _rows(analyse(store, date, params), store, editorial)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, columns in FILES.items():
        path = out / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows[name]:
                writer.writerow({c: row.get(c, "") for c in columns})
        written[name] = path
    return written
