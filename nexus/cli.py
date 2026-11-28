"""Command line for nexus: ``python -m nexus.cli <command>``."""

from __future__ import annotations

import argparse
import importlib

COMMANDS = ("loop", "matching", "impact", "ledger", "bottleneck")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus", description="Connections between the projects in this repository."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        try:
            module = importlib.import_module(f"nexus.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"nexus.{name}":
                raise
            continue
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
