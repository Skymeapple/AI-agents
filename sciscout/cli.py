"""Command line interface.

Four verbs, matching the four things the platform does:

``harvest``
    Pull works from a source into a local corpus file. Separated from analysis so
    that scoring is reproducible: the corpus is the input you can diff.
``rank``
    Score research directions by importance, urgency and priority.
``model``
    Run the investment simulation for one direction.
``report``
    Do all of it and write a markdown report.

Run ``python -m sciscout.cli <verb> --help`` for arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .commercial.profile import PriorLibrary
from .invest.model import DealTerms, MarketModel
from .pipeline import Pipeline
from .report import render_portfolio, render_track
from .sources.base import dump_works
from .sources.local import LocalCorpus


def _load_tracks(pipeline: Pipeline, corpus: Path, label_key: str):
    result = LocalCorpus(corpus).harvest()
    for error in result.errors:
        print(f"warning: {error}", file=sys.stderr)
    if not result.works:
        print(f"error: no works loaded from {corpus}", file=sys.stderr)
        raise SystemExit(2)
    return result.works, pipeline.tracks_from_works(result.works, label_key)


def _market_from_args(args) -> MarketModel:
    if args.market_json:
        spec = json.loads(Path(args.market_json).read_text())
        model = MarketModel.placeholder(tam_usd=float(spec.get("tam_usd", 2.0e9)))
        # Only TAM is overridable from the simple JSON form. Anything richer
        # should be built in Python, where the provenance of each input can be
        # stated properly rather than inferred from a bare number in a file.
        return model
    return MarketModel.placeholder(tam_usd=args.tam)


def cmd_harvest(args) -> int:
    """Fetch works from a live source into a corpus file."""
    if args.source == "arxiv":
        from .sources.arxiv import ArxivSource

        source = ArxivSource()
    elif args.source == "openalex":
        from .sources.openalex import OpenAlexSource

        source = OpenAlexSource(mailto=args.mailto)
    else:
        print(f"error: unknown source {args.source}", file=sys.stderr)
        return 2

    result = source.harvest(args.query, limit=args.limit)
    for error in result.errors:
        print(f"warning: {error}", file=sys.stderr)
    if not result.works:
        print(
            "error: no works retrieved. If this environment blocks outbound "
            "network access, generate a local corpus instead:\n"
            "    python scripts/make_demo_corpus.py",
            file=sys.stderr,
        )
        return 1

    # Stamp the track label onto every work so the pipeline can group them.
    for work in result.works:
        work.extra.setdefault("track", args.track or "unclassified")
        work.extra.setdefault("track_name", args.track_name or args.query)
        work.extra.setdefault("sector", args.sector)

    dump_works(result.works, Path(args.out))
    print(f"{result.summary()} -> {args.out}")
    return 0


def cmd_rank(args) -> int:
    """Score and rank research directions."""
    pipeline = Pipeline(asof=args.asof, draws=args.draws, seed=args.seed)
    _, tracks = _load_tracks(pipeline, Path(args.corpus), args.label_key)
    reports = pipeline.analyse(tracks, simulate=not args.no_simulate)

    width = max(len(r.name) for r in reports)
    print(f"{'direction'.ljust(width)}  prio   imp   urg   conf  TRL  works")
    for report in reports:
        a = report.assessment
        print(
            f"{report.name.ljust(width)}  {a.priority:.2f}  {a.importance:.2f}  "
            f"{a.urgency:.2f}  {a.confidence:.0%}   {report.profile.trl.mode}   "
            f"{len(report.track.works)}"
        )
    return 0


def cmd_model(args) -> int:
    """Run the investment simulation for one direction."""
    pipeline = Pipeline(asof=args.asof, draws=args.draws, seed=args.seed)
    _, tracks = _load_tracks(pipeline, Path(args.corpus), args.label_key)
    selected = [t for t in tracks if t.id == args.track]
    if not selected:
        print(
            f"error: no track {args.track!r}; available: "
            f"{', '.join(t.id for t in tracks)}",
            file=sys.stderr,
        )
        return 2

    reports = pipeline.analyse(
        selected,
        markets={args.track: _market_from_args(args)},
        terms={
            args.track: DealTerms.placeholder(
                check_usd=args.check, entry_ownership=args.ownership
            )
        },
    )
    print(render_track(reports[0]))
    return 0


def cmd_report(args) -> int:
    """Run everything and write a markdown report."""
    pipeline = Pipeline(asof=args.asof, draws=args.draws, seed=args.seed)
    _, tracks = _load_tracks(pipeline, Path(args.corpus), args.label_key)
    reports = pipeline.analyse(tracks)
    markdown = render_portfolio(reports, asof=args.asof)
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"wrote report for {len(reports)} directions to {args.out}")
    else:
        print(markdown)
    return 0


def cmd_sectors(args) -> int:
    """List the sectors that commercialisation priors exist for."""
    library = PriorLibrary()
    for sector in library.sectors():
        stages = library.stages(sector)
        print(f"{sector}: {len(stages)} stages")
        for stage in stages:
            print(
                f"    TRL {stage.from_trl}-{stage.to_trl}  {stage.name}  "
                f"[{stage.provenance.grade.value}]"
            )
    return 0


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sciscout",
        description=(
            "Track research directions, score their importance and urgency, and "
            "model the investment case for pursuing them."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", default="data/demo_corpus.json", help="corpus JSON file")
        p.add_argument(
            "--label-key",
            default="track",
            help="key in Work.extra used to group works into directions",
        )
        p.add_argument("--asof", type=_date, default=dt.date.today(), help="analysis date")
        p.add_argument("--draws", type=int, default=20_000, help="Monte Carlo draws")
        p.add_argument("--seed", type=int, default=20260918, help="random seed")

    harvest = sub.add_parser("harvest", help="fetch works from a live source")
    harvest.add_argument("--source", choices=["arxiv", "openalex"], required=True)
    harvest.add_argument("--query", required=True)
    harvest.add_argument("--limit", type=int, default=200)
    harvest.add_argument("--out", required=True)
    harvest.add_argument("--track", help="track id to stamp onto harvested works")
    harvest.add_argument("--track-name", help="human-readable track name")
    harvest.add_argument("--sector", default="generic")
    harvest.add_argument("--mailto", help="contact email for the OpenAlex polite pool")
    harvest.set_defaults(func=cmd_harvest)

    rank = sub.add_parser("rank", help="score and rank research directions")
    add_common(rank)
    rank.add_argument("--no-simulate", action="store_true", help="skip Monte Carlo")
    rank.set_defaults(func=cmd_rank)

    model = sub.add_parser("model", help="run the investment simulation for one direction")
    add_common(model)
    model.add_argument("--track", required=True)
    model.add_argument("--tam", type=float, default=2.0e9, help="median addressable market (USD)")
    model.add_argument("--market-json", help="JSON file of market assumptions")
    model.add_argument("--check", type=float, default=5.0e6, help="entry cheque (USD)")
    model.add_argument("--ownership", type=float, default=0.15, help="entry ownership fraction")
    model.set_defaults(func=cmd_model)

    report = sub.add_parser("report", help="full markdown report")
    add_common(report)
    report.add_argument("--out", help="output file; prints to stdout if omitted")
    report.set_defaults(func=cmd_report)

    sectors = sub.add_parser("sectors", help="list commercialisation priors by sector")
    sectors.set_defaults(func=cmd_sectors)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # Piping into `head` closes stdout early. Exit quietly rather than
        # tracebacking, and redirect the fd so the interpreter's own flush at
        # shutdown does not raise a second time.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
