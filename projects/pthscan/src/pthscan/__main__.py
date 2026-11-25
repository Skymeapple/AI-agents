"""Command-line interface for pthscan."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__
from .scanner import Finding, default_customize_dirs, default_targets, find_venvs, interpreter_site_dirs, interpreter_stdlib_dirs, scan, venv_site_dirs

INSTALL_CMD = re.compile(r"\b(?:pip3?|uv|poetry|pdm|conda|mamba|pipx|rye|hatch)\b[^;&|\n]*\b(?:install|add|sync|update|upgrade)\b|python3?\s+-m\s+pip\s+install")

COLORS = {"critical": "31;1", "warning": "33", "ok": "32"}


def color(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else s


def targets_from(args) -> list[Path]:
    dirs: list[Path] = []
    for p in args.paths:
        path = Path(p).expanduser()
        if path.name in ("site-packages", "dist-packages") and path.is_dir():
            dirs.append(path)
        elif (path / "pyvenv.cfg").is_file():
            dirs += venv_site_dirs(path)
        elif path.is_dir():
            for v in find_venvs(path, depth=args.depth):
                dirs += venv_site_dirs(v)
        else:
            sys.exit(f"pthscan: not a directory: {p}")
    for py in args.python or []:
        dirs += interpreter_site_dirs(py)
    if not args.paths and not args.python:
        dirs = default_targets(Path.cwd())
    return dirs


def report(findings: list[Finding], sites: int, args) -> None:
    shown = [f for f in findings if args.all or f.severity != "ok"]
    if args.format == "json":
        print(json.dumps({"version": __version__, "site_dirs": sites, "findings": [f.to_dict() for f in shown]}, indent=2))
        return
    if args.format == "github":
        for f in shown:
            lvl = {"critical": "error", "warning": "warning", "ok": "notice"}[f.severity]
            print(f"::{lvl} file={f.file},title=pthscan {f.title}::{'; '.join(f.reasons) or f.title}")
    for f in shown:
        if args.format == "github":
            break
        owner = f" ({f.owner})" if f.owner else ""
        print(f"{color(COLORS[f.severity], f.severity.upper()):<10} {f.file}{owner}")
        print(f"           {f.title}")
        for r in f.reasons:
            print(f"           - {r}")
        for ln in f.lines[:3]:
            print(f"           > {ln}")
    hooks = sum(1 for f in findings if f.file.endswith((".pth", ".start", "customize.py")))
    crit = sum(1 for f in findings if f.severity == "critical")
    warn = sum(1 for f in findings if f.severity == "warning")
    summary = f"{sites} site-packages dir(s), {hooks} startup file(s): {crit} critical, {warn} warning"
    if crit:
        print("\n" + color(COLORS["critical"], summary))
        print("Do not start Python in an affected environment. From a clean machine: rotate every credential the host or runner\n"
              "could reach (cloud, CI, git, package registries, AI keys), then delete and rebuild the environment.")
    elif warn:
        print("\n" + color(COLORS["warning"], summary) + "\nReview each unrecognised hook: it runs on every interpreter start.")
    else:
        print(color(COLORS["ok"], f"✓ {summary}"))


def hook() -> int:
    """Claude Code PostToolUse hook: after a package install, block the agent if a hostile hook appeared."""
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    cmd = str((data.get("tool_input") or {}).get("command", ""))
    if data.get("tool_name") != "Bash" or not INSTALL_CMD.search(cmd):
        return 0
    try:
        cwd = Path(data.get("cwd") or os.getcwd())
        findings = [f for f in scan(default_targets(cwd), default_customize_dirs()) if f.severity == "critical"]
    except Exception as e:  # never break the agent loop on our own bug
        print(f"pthscan: hook error ({e}); not blocking", file=sys.stderr)
        return 0
    if not findings:
        return 0
    lines = "\n".join(f"  - {f.file}: {f.title}; {'; '.join(f.reasons)}" for f in findings[:6])
    print("pthscan: STOP. The install just put code in place that runs on every Python start and looks like a credential stealer:\n"
          f"{lines}\nDo not run python, pip or tests in this environment. Tell the user immediately: they must rotate credentials "
          "this machine can reach and rebuild the environment from a clean lockfile.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["hook"]:
        return hook()
    ap = argparse.ArgumentParser(prog="pthscan", description="Audit Python startup hooks (.pth / .start) for malicious code, without running them.")
    ap.add_argument("paths", nargs="*", help="site-packages dirs, virtualenvs, or folders to search for virtualenvs (default: python on PATH + venvs under .)")
    ap.add_argument("--python", action="append", help="also scan this interpreter's site dirs (queried with -I -S, so no hooks run)")
    ap.add_argument("--format", choices=["text", "json", "github"], default="text")
    ap.add_argument("--all", action="store_true", help="also list recognised, harmless hooks")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--depth", type=int, default=3, help="how deep to look for virtualenvs (default 3)")
    ap.add_argument("-V", "--version", action="version", version=__version__)
    args = ap.parse_args(argv)
    dirs = targets_from(args)
    extra = [] if args.paths else (default_customize_dirs() if not args.python else [d for py in args.python for d in interpreter_stdlib_dirs(py)])
    findings = scan(dirs, extra)
    report(findings, len({d.resolve() for d in dirs}), args)
    if any(f.severity == "critical" for f in findings):
        return 1
    if args.strict and any(f.severity == "warning" for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
