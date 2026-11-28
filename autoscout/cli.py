"""Command line interface for autoscout.

``init``     create a state directory from the seed agenda
``cycle``    run one gather -> learn -> propose -> gate cycle
``run``      run several cycles (offline: until the stream is exhausted)
``status``   print the latest cycle and the current champion's scores
``queue``    write the papers the agent is least sure about, for a human to label
``label``    import labels, which become the real-paper benchmark

Run ``python -m autoscout.cli <verb> --help`` for arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import DEFAULT_BENCH, MIN_HUMAN_LABELS, load_human_labels
from .config import AGENDA_PATH
from .gate import GatePolicy
from .loop import CycleOptions, init_state, run_cycle
from .scout import DEFAULT_STREAM
from .state import StateDir

# Local runs default to an ignored directory. The tracked state that the
# scheduled workflow maintains lives in autoscout_state/ and is always named
# explicitly, so an offline synthetic run can't be committed over it by accident.
DEFAULT_STATE = Path("autoscout_runs/local")


def _options(args) -> CycleOptions:
    return CycleOptions(
        live=args.live,
        candidates=args.candidates,
        stream_path=Path(args.stream),
        bench_path=Path(args.bench),
        agenda_path=Path(args.agenda),
        seed=args.seed,
        policy=GatePolicy(min_effect=args.min_effect),
        allow_synthetic_gate=args.allow_synthetic_gate,
    )


def _print_summary(summary: dict) -> None:
    triage = summary["scores"].get("triage", {})
    line = (f"cycle {summary['cycle']}: {summary['new_papers']} new, {summary['flagged']} flagged, "
            f"{summary['mined_terms']} mined terms, {summary['evaluated']} evaluated, "
            f"{summary['passed_tune']} passed tune -> {summary['decision']}")
    print(line)
    if triage:
        print(f"  triage utility tune {triage['tune']['mean']:.4f} holdout "
              f"{triage['holdout']['mean']:.4f} sealed {triage['sealed']['mean']:.4f} "
              f"({summary['seconds']}s)")


def cmd_init(args) -> int:
    state = StateDir(args.state)
    try:
        init_state(state, Path(args.agenda), force=args.force, bench_path=Path(args.bench))
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"initialised {state.path} from {args.agenda}")
    return 0


def cmd_cycle(args) -> int:
    summary = run_cycle(StateDir(args.state), _options(args))
    _print_summary(summary)
    return 0


def cmd_run(args) -> int:
    state = StateDir(args.state)
    for _ in range(args.cycles):
        summary = run_cycle(state, _options(args))
        _print_summary(summary)
        if summary["exhausted"] and not args.live:
            print("offline stream exhausted")
            break
    print(f"report: {state.path / 'REPORT.md'}")
    return 0


def cmd_status(args) -> int:
    state = StateDir(args.state)
    if not state.exists():
        print(f"no agent in {state.path}; run init or cycle first", file=sys.stderr)
        return 2
    cycles = state.read_jsonl("cycles.jsonl")
    if cycles:
        _print_summary(cycles[-1])
    champion, scores = state.load_champion()
    print(f"champion {champion.fingerprint()}")
    print(json.dumps(scores, indent=2))
    labels = load_human_labels(state.labels_path)
    print(f"human labels: {len(labels)} (real-paper benchmark activates at {MIN_HUMAN_LABELS})")
    return 0


def cmd_queue(args) -> int:
    """Active learning: the papers nearest the flagging threshold are worth most."""
    state = StateDir(args.state)
    champion, _ = state.load_champion()
    knowledge = state.load_knowledge()
    labelled = {w.id for w in load_human_labels(state.labels_path)}
    threshold = champion.get("triage.threshold")
    pool = [r for r in knowledge.recent_works
            if r["work"]["id"] not in labelled and not r["work"]["id"].startswith("synthetic:")]
    pool.sort(key=lambda r: abs(r["score"] - threshold))
    out = Path(args.out)
    with out.open("w") as fh:
        for r in pool[: args.n]:
            fh.write(json.dumps({"work": r["work"], "predicted_topic": r["topic"],
                                 "score": r["score"], "label": None}) + "\n")
    print(f"wrote {min(args.n, len(pool))} papers to {out}. Set each \"label\" to a topic "
          f"name ({', '.join(champion.topics())}) or \"irrelevant\", then run `label --import {out}`.")
    return 0


def cmd_label(args) -> int:
    state = StateDir(args.state)
    champion, _ = state.load_champion()
    allowed = set(champion.topics()) | {"irrelevant"}
    added = skipped = 0
    lines = []
    for line in Path(args.import_path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("label") is None:
            skipped += 1
            continue
        if raw["label"] not in allowed:
            print(f"error: {raw['work']['id']}: label {raw['label']!r} not in {sorted(allowed)}",
                  file=sys.stderr)
            return 2
        lines.append(json.dumps({"work": raw["work"], "label": raw["label"]}))
        added += 1
    state.path.mkdir(parents=True, exist_ok=True)
    with state.labels_path.open("a") as fh:
        fh.writelines(line + "\n" for line in lines)
    print(f"added {added} labels ({skipped} unlabelled rows skipped)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoscout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="state directory")
    sub = parser.add_subparsers(dest="command", required=True)

    def cycle_args(p):
        p.add_argument("--live", action="store_true", help="harvest from arXiv instead of the stream")
        p.add_argument("--candidates", type=int, default=12, help="candidate changes per cycle")
        p.add_argument("--stream", default=str(DEFAULT_STREAM))
        p.add_argument("--bench", default=str(DEFAULT_BENCH))
        p.add_argument("--agenda", default=str(AGENDA_PATH))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--min-effect", type=float, default=GatePolicy.min_effect)
        p.add_argument("--allow-synthetic-gate", action="store_true",
                       help="in live mode, let the synthetic benchmark gate changes "
                            "before real labels exist (not recommended)")

    p = sub.add_parser("init", help="create a state directory")
    p.add_argument("--agenda", default=str(AGENDA_PATH))
    p.add_argument("--bench", default=str(DEFAULT_BENCH))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("cycle", help="run one cycle")
    cycle_args(p)
    p.set_defaults(func=cmd_cycle)

    p = sub.add_parser("run", help="run several cycles")
    cycle_args(p)
    p.add_argument("--cycles", type=int, default=12)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show the latest cycle and champion")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("queue", help="export uncertain papers for labelling")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--out", default="label_queue.jsonl")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("label", help="import labelled papers")
    p.add_argument("--import", dest="import_path", required=True)
    p.set_defaults(func=cmd_label)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
