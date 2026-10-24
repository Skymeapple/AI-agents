"""Command line interface.

``collect``
    Run the live API collectors (needs outbound network) and store what comes
    back. Failures are logged as source health, not raised.
``ingest``
    Validate and load pending inbox files without running analysis.
``validate``
    Check inbox files against the format without loading them.
``daily``
    The whole day: ingest the inbox, score, find directions, record the
    forecast, score matured forecasts, tune, write the snapshot and report.
``report``
    Re-render a day's report without tuning or recording a forecast.
``dashboard``
    Export a day's output as the CSV files behind the shared web page.
``log-failure``
    Record that a source could not be collected today, so source health and
    the page show it.
``mark-gone``
    Record that an app's page is gone (404, deleted, made private), or back
    with ``--back``. A gone app leaves the ranking until it is seen again.
``mark-moved``
    Record that an app moved to a new url (a repo transferred to another
    owner). Its history under the old url and the new one become one app.
``verdict``
    Record what research found when it tested a hypothesis, so the report
    shows it beside the rule each day the rule fires again.
``recheck``
    List the apps worth re-observing today -- the current leaders and every app
    seen only once -- with the metric to record. Re-observation is what turns a
    stated claim into growth the radar measured itself.

Run ``python -m airadar.cli <verb> --help`` for arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from .store import DEFAULT_ROOT, RunRecord, Store


def _date(text: str | None) -> dt.date:
    return dt.date.fromisoformat(text) if text else dt.datetime.now(dt.timezone.utc).date()


def cmd_collect(args) -> int:
    import requests

    from .collectors.base import load_watch
    from .collectors.live import (
        collect_github, collect_hackernews, collect_huggingface, collect_packages,
    )

    store = Store(args.root)
    watch = load_watch(store.root / "watch.json")
    session = requests.Session()
    runners = {
        "github": lambda: collect_github(watch, session, os.environ.get("GITHUB_TOKEN")),
        "huggingface": lambda: collect_huggingface(session),
        "hackernews": lambda: collect_hackernews(watch, session),
        "packages": lambda: collect_packages(watch, session),
    }
    chosen = args.only or list(runners)
    for name in chosen:
        collected = runners[name]()
        added = store.add(collected.apps, collected.observations)
        store.record_run(RunRecord(_date(None), collected.collector, len(collected.observations),
                                   collected.errors[:5]))
        status = f"{len(collected.errors)} error(s)" if collected.errors else "ok"
        print(f"{collected.collector}: {len(collected.observations)} observations ({added} new), {status}")
        for error in collected.errors[:3]:
            print(f"  warning: {error}", file=sys.stderr)
    return 0


def cmd_validate(args) -> int:
    from .inbox import InboxError, load_inbox_file

    failed = 0
    for path in args.files:
        try:
            batch = load_inbox_file(Path(path))
            print(f"ok   {path}: {len(batch.observations)} observations, {len(batch.apps)} apps")
        except InboxError as exc:
            failed += 1
            print(f"FAIL {exc}")
    return 1 if failed else 0


def cmd_ingest(args) -> int:
    from .pipeline import ingest_inbox

    for line in ingest_inbox(Store(args.root), _date(args.date)):
        print(line)
    return 0


def cmd_daily(args) -> int:
    from .pipeline import run_day
    from .report import render

    store = Store(args.root)
    date = _date(args.date)
    analysis = run_day(store, date, tune_params=not args.no_tune)
    path = store.write_report(date, render(analysis))
    for line in analysis.ingest_log:
        print(line)
    print(f"{len(analysis.scores)} apps scored, {len(analysis.directions)} directions, "
          f"{len(analysis.opportunities)} hypotheses")
    if analysis.tuning:
        print(f"tuning: {analysis.tuning.reason}")
    print(f"report: {path}")
    return 0


def cmd_report(args) -> int:
    from .pipeline import analyse
    from .report import render

    store = Store(args.root)
    text = render(analyse(store, _date(args.date)))
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import DashboardError, export

    try:
        written = export(Store(args.root), _date(args.date), Path(args.out))
    except DashboardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, path in written.items():
        print(f"{name}\t{path}")
    return 0


def cmd_log_failure(args) -> int:
    Store(args.root).record_run(RunRecord(_date(args.date), args.collector, 0, [args.error]))
    print(f"recorded failure for {args.collector}")
    return 0


def cmd_mark_gone(args) -> int:
    from .models import canonical_id
    app_id = canonical_id(args.url)
    status = "back" if args.back else "gone"
    Store(args.root).record_availability(app_id, status, _date(args.date), args.source_url, args.note)
    print(f"recorded {app_id} as {status}")
    return 0


def cmd_mark_moved(args) -> int:
    Store(args.root).record_move(args.old_url, args.new_url, _date(args.date), args.source_url, args.note)
    print(f"recorded {args.old_url} -> {args.new_url}")
    return 0


def cmd_verdict(args) -> int:
    Store(args.root).record_verdict(args.direction, args.pattern, args.verdict, _date(args.date),
                                    args.source_url, args.note)
    print(f"recorded {args.verdict} for {args.pattern} / {args.direction}")
    return 0


RECHECK_METRIC = {"repo": "stars", "space": "likes", "model": "likes"}


def cmd_recheck(args) -> int:
    import json

    from .pipeline import analyse

    store = Store(args.root)
    analysis = analyse(store, _date(args.date))
    chosen = [s.app_id for s in analysis.scores[: args.top]] + analysis.baseline_only
    seen = set()
    for app_id in chosen:
        app = analysis.apps.get(app_id)
        if app is None or app_id in seen or app.kind.value not in RECHECK_METRIC:
            continue
        seen.add(app_id)
        print(json.dumps({"name": app.name, "url": app.url, "kind": app.kind.value,
                          "metric": RECHECK_METRIC[app.kind.value]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airadar", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="radar data directory")
    sub = parser.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("collect", help="run live API collectors")
    p.add_argument("--only", nargs="+", choices=["github", "huggingface", "hackernews", "packages"])
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("validate", help="check inbox files")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ingest", help="load pending inbox files")
    p.add_argument("--date")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("daily", help="run the full daily cycle")
    p.add_argument("--date", help="ISO date to run as (default: today, UTC)")
    p.add_argument("--no-tune", action="store_true")
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("report", help="render a report without side effects")
    p.add_argument("--date")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="export the shared page's data files")
    p.add_argument("--date")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("log-failure", help="record a source that could not be collected")
    p.add_argument("collector", help="e.g. agent:huggingface")
    p.add_argument("error", help="what went wrong, e.g. DNS failure through WebFetch")
    p.add_argument("--date")
    p.set_defaults(func=cmd_log_failure)

    p = sub.add_parser("mark-gone", help="record that an app's page is gone, or back")
    p.add_argument("url", help="the app's url, as in apps.json")
    p.add_argument("source_url", help="the page that was checked")
    p.add_argument("note", nargs="?", default="", help="what was seen, e.g. HTTP 404")
    p.add_argument("--back", action="store_true", help="the page is reachable again")
    p.add_argument("--date")
    p.set_defaults(func=cmd_mark_gone)

    p = sub.add_parser("mark-moved", help="record that an app moved to a new url")
    p.add_argument("old_url")
    p.add_argument("new_url")
    p.add_argument("source_url", help="the page that shows the move")
    p.add_argument("note", nargs="?", default="")
    p.add_argument("--date")
    p.set_defaults(func=cmd_mark_moved)

    p = sub.add_parser("verdict", help="record what research found about a hypothesis")
    p.add_argument("direction", help="e.g. agent-memory")
    p.add_argument("pattern", help='e.g. "fast and uncrowded", or "*" for every pattern')
    p.add_argument("verdict", choices=Store.VERDICTS)
    p.add_argument("source_url", help="the analysis or source the verdict rests on")
    p.add_argument("note", nargs="?", default="")
    p.add_argument("--date")
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("recheck", help="list apps to re-observe today")
    p.add_argument("--date")
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(func=cmd_recheck)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
