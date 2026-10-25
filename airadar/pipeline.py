"""One day of the radar: ingest, score, find directions, check itself, tune.

The order matters. The day's prediction is recorded with the parameters that
produced the day's ranking *before* tuning runs, so tomorrow's scorecard grades
what was actually published, not what the tuner wished it had published.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .collectors.base import load_watch, save_watch
from .directions import (
    Direction, EmergentGroup, Opportunity, emergent_groups, opportunities, rising_pairs,
    tag_directions,
)
from .growth import AppScore, extract_signals, measured_movers, score_apps
from .inbox import InboxError, load_inbox_file
from .iterate import (
    Evaluation, SourceHealth, TuningOutcome, apply_tuning, record_prediction, scorecard,
    source_health, tune,
)
from .models import App
from .params import Params
from .store import RunRecord, Store

MAX_AUTO_TERMS = 40
NEW_TERMS_PER_DAY = 3


@dataclass
class Analysis:
    date: dt.date
    apps: dict[str, App]
    scores: list[AppScore]
    baseline_only: list[str]
    directions: list[Direction]
    emergent: list[EmergentGroup]
    opportunities: list[Opportunity]
    pairs: list[tuple[str, str, int]]
    params: Params
    evaluations: list[Evaluation] = field(default_factory=list)
    tuning: TuningOutcome | None = None
    health: list[SourceHealth] = field(default_factory=list)
    ingest_log: list[str] = field(default_factory=list)
    watch_added: list[str] = field(default_factory=list)
    # Apps whose page was found gone (deleted, made private), with the record
    # that says so. They leave the ranking: growth claimed for a page that no
    # longer exists cannot be checked, and no one can build on or back it.
    gone: dict[str, dict] = field(default_factory=dict)
    # Growth the radar measured between its own last two readings.
    movers: list = field(default_factory=list)


def ingest_inbox(store: Store, date: dt.date) -> list[str]:
    """Load every pending inbox file. A file that fails validation is left in
    place and logged as a failed run for its collector, so the report shows it."""
    log = []
    if not store.inbox.exists():
        return log
    for path in sorted(store.inbox.glob("*.json")):
        try:
            batch = load_inbox_file(path)
        except InboxError as exc:
            store.record_run(RunRecord(date, f"inbox:{path.stem}", 0, [str(exc)]))
            log.append(f"REJECTED {path.name}: {exc}")
            continue
        added = store.ingest_batch(batch)
        log.append(f"{path.name}: {len(batch.observations)} observations ({added} new), "
                   f"{len(batch.apps)} apps, collector {batch.collector}")
    return log


def expand_watch(store: Store, groups: list[EmergentGroup], scores: list[AppScore]) -> list[str]:
    """Add the distinctive terms of fast-growing emergent groupings to the live
    collectors' watch list, so tomorrow's search looks where today's growth was.

    Only groups whose members average in the top half are mined, at most a few
    terms a day, and the list is capped -- the oldest auto terms drop first.
    """
    path = store.root / "watch.json"
    watch = load_watch(path)
    score_of = {s.app_id: s.score for s in scores}
    existing = set(watch["auto_terms"]) | set(watch["github_topics"]) | set(watch["hn_queries"])
    added = []
    for group in sorted(groups, key=lambda g: -sum(score_of.get(m, 0) for m in g.members) / len(g.members)):
        mean = sum(score_of.get(m, 0) for m in group.members) / len(group.members)
        if mean < 0.55 or group.cohesion < 0.3:
            continue
        for term in group.terms[:2]:
            if term not in existing and len(added) < NEW_TERMS_PER_DAY:
                added.append(term)
                existing.add(term)
    if added:
        watch["auto_terms"] = (watch["auto_terms"] + added)[-MAX_AUTO_TERMS:]
        save_watch(path, watch)
    return added


def analyse(store: Store, date: dt.date, params: Params | None = None) -> Analysis:
    params = params or Params.load(store.params_path)
    apps = store.load_apps()
    observations = store.load_observations()
    gone = store.gone_as_of(date)
    signals = {
        app_id: rows for app_id, rows in extract_signals(observations, asof=date).items()
        if app_id not in gone
    }
    scores = [s for s in score_apps(signals, params, asof=date) if s.app_id in apps]
    seen = {o.app_id for o in observations if o.known_on <= date} - set(gone)
    baseline_only = sorted(seen - set(signals))
    previous = store.previous_snapshot(date)
    directions = tag_directions(apps, scores, previous)
    groups = emergent_groups(apps, scores)
    hypotheses = opportunities(directions, apps)
    for o in hypotheses:
        o.verdict = store.verdict_for(o.direction, o.kind, date)
    return Analysis(
        date=date,
        apps=apps,
        scores=scores,
        baseline_only=baseline_only,
        directions=directions,
        emergent=groups,
        opportunities=hypotheses,
        pairs=rising_pairs(apps, scores),
        params=params,
        evaluations=scorecard(store, date, observations),
        health=source_health(store.load_runs()),
        gone=gone,
        movers=[m for m in measured_movers(observations, date) if m.app_id in apps and m.app_id not in gone],
    )


def snapshot_payload(analysis: Analysis) -> dict:
    return {
        "date": analysis.date.isoformat(),
        "params": analysis.params.values,
        "scores": [
            {"app_id": s.app_id, "score": round(s.score, 6), "evidence": round(s.evidence, 4)}
            for s in analysis.scores
        ],
        "directions": [d.to_dict() for d in analysis.directions],
        "emergent": [{"label": g.label, "members": g.members} for g in analysis.emergent],
    }


def run_day(store: Store, date: dt.date, tune_params: bool = True) -> Analysis:
    ingest_log = ingest_inbox(store, date)
    analysis = analyse(store, date)
    analysis.ingest_log = ingest_log
    record_prediction(store, date, analysis.scores, analysis.params)
    store.write_snapshot(date, snapshot_payload(analysis))
    if tune_params:
        outcome = tune(analysis.params, store.load_observations(), date)
        analysis.params = apply_tuning(store, analysis.params, outcome, date)
        analysis.tuning = outcome
    analysis.watch_added = expand_watch(store, analysis.emergent, analysis.scores)
    return analysis
