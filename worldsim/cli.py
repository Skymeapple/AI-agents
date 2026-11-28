"""Command line for the world model.

    python -m worldsim.cli fetch          # refresh data/world from the public sources
    python -m worldsim.cli run            # simulate and write the report and JSON
    python -m worldsim.cli run --lever peacebuilding=1 --lever development=0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import LEVERS, OUTCOMES, load_calibration, run_scenarios, simulate
from .report import write

ROOT = Path(__file__).resolve().parent.parent


def _lever(text: str) -> tuple[str, float]:
    name, _, value = text.partition("=")
    if name not in LEVERS or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE with NAME one of {', '.join(LEVERS)}")
    return name, float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worldsim", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the observed series")
    run = sub.add_parser("run", help="simulate the scenarios")
    run.add_argument("--draws", type=int, default=4000)
    run.add_argument("--seed", type=int, default=20261128)
    run.add_argument("--lever", type=_lever, action="append", default=[], help="NAME=VALUE; prints a custom scenario instead")
    run.add_argument("--report", type=Path, default=ROOT / "examples" / "world-problems-simulation.md")
    run.add_argument("--json", type=Path, default=ROOT / "data" / "world" / "simulation.json")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        from .fetch import DATA_DIR, fetch_all

        print(f"fetched {len(fetch_all())} series into {DATA_DIR}")
        return 0

    calibration = load_calibration()
    if args.lever:
        custom = simulate(calibration, dict(args.lever), args.draws, args.seed, name="custom")
        base = simulate(calibration, {}, args.draws, args.seed, name="current")
        print(f"{'indicator':42s} {'current':>12s} {'custom':>12s}   in {calibration.horizon}, median")
        for o in OUTCOMES:
            a, b = np.median(base.final(o.key)), np.median(custom.final(o.key))
            print(f"{o.label[:42]:42s} {a:12,.{o.digits}f} {b:12,.{o.digits}f}   {o.unit}")
        return 0

    runs = run_scenarios(calibration, args.draws, args.seed)
    write(calibration, runs, args.draws, args.seed, args.report, args.json)
    print(f"wrote {args.report} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
