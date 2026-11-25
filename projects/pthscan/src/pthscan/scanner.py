"""Find and judge Python startup hooks (.pth / .start files) without running them."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .known import COMPROMISED, INDICATORS, KNOWN_GOOD_CUSTOMIZE, KNOWN_GOOD_RE, MALICIOUS_FILE_PATTERNS, MALICIOUS_FILES

MAX_MODULE_BYTES = 2 * 1024 * 1024
SEVERITY_ORDER = {"critical": 0, "warning": 1, "ok": 2}


@dataclass
class Finding:
    severity: str            # critical | warning | ok
    file: str                # absolute path of the .pth/.start file (or dist-info)
    site_dir: str
    owner: str | None        # "name==version" of the distribution that installed it
    title: str
    reasons: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)   # executable lines, truncated

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Locating site-packages without executing anything hostile
# ---------------------------------------------------------------------------

def venv_site_dirs(venv: Path) -> list[Path]:
    """site-packages of a virtualenv, found by path only (never runs its interpreter)."""
    out = sorted(venv.glob("lib/python*/site-packages")) + sorted(venv.glob("lib64/python*/site-packages"))
    out += [p for p in [venv / "Lib" / "site-packages"] if p.is_dir()]
    return [p for p in out if p.is_dir()]


def find_venvs(root: Path, depth: int = 3) -> list[Path]:
    found: list[Path] = []

    def walk(d: Path, level: int) -> None:
        if (d / "pyvenv.cfg").is_file():
            found.append(d)
            return
        if level >= depth:
            return
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        for e in entries:
            if e.is_dir() and not e.is_symlink() and e.name not in {"node_modules", ".git", "__pycache__", "site-packages"}:
                walk(e, level + 1)

    walk(root, 0)
    return found


def interpreter_stdlib_dirs(python: str) -> list[Path]:
    """The stdlib dir, where distros put sitecustomize.py (queried with -I -S: no hooks run)."""
    code = "import sysconfig,json;print(json.dumps([sysconfig.get_paths()['stdlib']]))"
    try:
        out = subprocess.run([python, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=20, check=True).stdout
        return [Path(p) for p in json.loads(out) if Path(p).is_dir()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def interpreter_site_dirs(python: str) -> list[Path]:
    """Ask an interpreter for its site dirs with -I -S, which skips .pth processing."""
    code = "import site,json;print(json.dumps(site.getsitepackages()+[site.getusersitepackages()]))"
    try:
        out = subprocess.run([python, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=20, check=True).stdout
        return [Path(p) for p in json.loads(out) if Path(p).is_dir()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Ownership: which installed distribution put a file there
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_index(site: Path) -> tuple[dict[str, str], list[tuple[str, str, Path]]]:
    """Map top-level file name -> 'name==version', and list (name, version, dist-info path)."""
    owners: dict[str, str] = {}
    dists: list[tuple[str, str, Path]] = []
    for info in site.glob("*.dist-info"):
        name, version = None, None
        meta = info / "METADATA"
        try:
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                if name and version:
                    break
        except OSError:
            m = re.match(r"(.+?)-([^-]+)\.dist-info$", info.name)
            if m:
                name, version = m.group(1), m.group(2)
        if not name:
            continue
        dists.append((name, version or "?", info))
        try:
            rows = csv.reader(io.StringIO((info / "RECORD").read_text(encoding="utf-8", errors="replace")))
            for row in rows:
                if row and "/" not in row[0]:
                    owners.setdefault(row[0], f"{name}=={version}")
                elif row:
                    owners.setdefault(row[0].split("/")[0], f"{name}=={version}")
        except OSError:
            pass
    return owners, dists


# ---------------------------------------------------------------------------
# Judging one hook file
# ---------------------------------------------------------------------------

def executable_lines(text: str) -> list[str]:
    # site.py executes lines that start with "import " or "import\t"
    return [ln.rstrip("\n") for ln in text.splitlines() if ln.startswith(("import ", "import\t"))]


def imported_modules(line: str) -> list[str]:
    mods = []
    for stmt in line.split(";"):
        stmt = stmt.strip()
        m = re.match(r"import\s+([\w.]+(?:\s*,\s*[\w.]+)*)", stmt)
        if m:
            mods += [x.strip().split(".")[0] for x in m.group(1).split(",")]
        mods += re.findall(r"__import__\(\s*['\"]([\w]+)", stmt)
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"os", "sys", "types", "site"}
    return [m for m in dict.fromkeys(mods) if m not in stdlib]


def indicator_hits(text: str) -> list[str]:
    return [label for label, rx in INDICATORS if rx.search(text)]


def module_source(site: Path, mod: str) -> tuple[Path | None, bool]:
    """Return (path, is_single_file_module)."""
    f = site / f"{mod}.py"
    if f.is_file():
        return f, True
    pkg = site / mod / "__init__.py"
    if pkg.is_file():
        return pkg, False
    return None, False


def judge_hook(path: Path, site: Path, owners: dict[str, str]) -> Finding:
    owner = owners.get(path.name)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return Finding("warning", str(path), str(site), owner, "unreadable startup file", [str(e)])

    if path.suffix == ".start":
        # PEP 829 entry points: "package.module:function"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        lines = executable_lines(text)

    reasons: list[str] = []
    sev = "ok"
    if path.name in MALICIOUS_FILES:
        return Finding("critical", str(path), str(site), owner, f"known malicious file: {MALICIOUS_FILES[path.name]}",
                       ["file name matches a published incident"], [ln[:160] for ln in lines])
    for rx, why in MALICIOUS_FILE_PATTERNS:
        if rx.match(path.name) and lines:
            reasons.append(why)
            sev = "warning"

    if not lines:
        return Finding("ok", str(path), str(site), owner, "path entries only (no code runs)")

    unknown = []
    labels = []
    for ln in lines:
        good = next((label for label, rx in KNOWN_GOOD_RE if rx.fullmatch(ln)), None)
        if good:
            labels.append(good)
        else:
            unknown.append(ln)
    if not unknown and sev == "ok":
        # A recognised line can still import a planted module (e.g. a malicious _virtualenv.py).
        planted = _planted_module_reasons(lines, path, site, owners)
        if planted:
            return Finding("critical", str(path), str(site), owner, "known hook imports a suspicious module", planted, [ln[:160] for ln in lines])
        return Finding("ok", str(path), str(site), owner, f"known hook: {', '.join(dict.fromkeys(labels))}", [], [ln[:160] for ln in lines])

    # Unknown executable code at interpreter startup.
    line_hits = sorted({h for ln in unknown for h in indicator_hits(ln)})
    if line_hits:
        sev = "critical"
        reasons.append("startup line " + ", ".join(line_hits))
    for ln in unknown:
        for mod in imported_modules(ln if path.suffix != ".start" else "import " + ln.split(":")[0]):
            src, single = module_source(site, mod)
            if not src:
                continue
            try:
                if src.stat().st_size > MAX_MODULE_BYTES:
                    continue
                body = src.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hits = indicator_hits(body)
            if not hits:
                continue
            mod_owner = owners.get(src.name if single else mod)
            if single and (not mod_owner or len(hits) >= 3 or STRONG & set(hits)):
                sev = "critical"
            elif sev != "critical":
                sev = "warning"
            reasons.append(f"imported module {src.name} ({'unowned' if not mod_owner else mod_owner}) {', '.join(hits)}")
    if not owner:
        reasons.append("no installed package claims this file (left behind after uninstall, or dropped by something else)")
        if sev == "ok":
            sev = "warning"
    if sev == "ok":
        sev = "warning"
        reasons.append("runs unrecognised code at every interpreter start")
    title = "malicious-looking startup hook" if sev == "critical" else "unrecognised startup hook"
    return Finding(sev, str(path), str(site), owner, title, reasons, [ln[:160] + ("…" if len(ln) > 160 else "") for ln in unknown])


STRONG = {"decodes base64", "opens network connections", "reads other processes or credential stores", "contains an encoded blob"}


def _planted_module_reasons(lines: list[str], path: Path, site: Path, owners: dict[str, str]) -> list[str]:
    out = []
    for ln in lines:
        for mod in imported_modules(ln if path.suffix != ".start" else "import " + ln.split(":")[0]):
            src, single = module_source(site, mod)
            if not src or not single:
                continue
            try:
                body = src.read_text(encoding="utf-8", errors="replace") if src.stat().st_size <= MAX_MODULE_BYTES else ""
            except OSError:
                continue
            hits = [h for h in indicator_hits(body) if h in STRONG]
            if hits:
                out.append(f"imported module {src.name} ({owners.get(src.name, 'unowned')}) {', '.join(hits)}")
    return out


CUSTOMIZE_NAMES = ("sitecustomize.py", "usercustomize.py")


def _normalize(text: str) -> str:
    lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
    return " ".join(" ".join(lines).split())


def judge_customize(path: Path, site: Path, owners: dict[str, str]) -> Finding:
    """sitecustomize.py / usercustomize.py: imported by the site module at every start."""
    owner = owners.get(path.name)
    try:
        text = path.read_text(encoding="utf-8", errors="replace") if path.stat().st_size <= MAX_MODULE_BYTES else ""
    except OSError as e:
        return Finding("warning", str(path), str(site), owner, f"unreadable {path.name}", [str(e)])
    norm = _normalize(text)
    if not norm:
        return Finding("ok", str(path), str(site), owner, f"empty {path.name}")
    known = next((label for label, sig in KNOWN_GOOD_CUSTOMIZE if norm == sig), None)
    if known:
        return Finding("ok", str(path), str(site), owner, f"known {path.name}: {known}")
    hits = indicator_hits(text)
    reasons = [f"runs at every interpreter start ({len(text.splitlines())} lines)"]
    if not owner:
        reasons.append("no installed package claims this file")
    if hits:
        reasons.append(", ".join(hits))
    critical = bool(STRONG & set(hits)) or len(hits) >= 3
    sev = "critical" if critical else "warning"
    title = f"malicious-looking {path.name}" if critical else f"unrecognised {path.name}"
    return Finding(sev, str(path), str(site), owner, title, reasons)


def scan_customize_dir(d: Path) -> list[Finding]:
    owners, _ = dist_index(d)
    return [judge_customize(d / n, d, owners) for n in CUSTOMIZE_NAMES if (d / n).is_file()]


def scan_site(site: Path) -> list[Finding]:
    owners, dists = dist_index(site)
    findings = []
    for name, version, info in dists:
        bad = COMPROMISED.get(_norm(name), {}).get(version)
        if bad:
            findings.append(Finding("critical", str(info), str(site), f"{name}=={version}",
                                    "known-compromised release installed",
                                    [f"{name} {version} shipped a credential stealer ({bad}). Treat every secret on this host or runner as stolen."]))
    for f in sorted(list(site.glob("*.pth")) + list(site.glob("*.start"))):
        findings.append(judge_hook(f, site, owners))
    for n in CUSTOMIZE_NAMES:
        if (site / n).is_file():
            findings.append(judge_customize(site / n, site, owners))
    for name, why in MALICIOUS_FILES.items():
        p = site / name
        if p.suffix == ".py" and p.is_file():
            findings.append(Finding("critical", str(p), str(site), owners.get(name), f"known malicious file: {why}", ["file name matches a published incident"]))
    return findings


def scan(site_dirs: list[Path], customize_dirs: list[Path] | None = None) -> list[Finding]:
    seen: set[Path] = set()
    out: list[Finding] = []
    for s in site_dirs:
        r = s.resolve()
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out += scan_site(r)
    seen_files = {Path(f.file).resolve() for f in out}
    for d in customize_dirs or []:
        r = d.resolve()
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out += [f for f in scan_customize_dir(r) if Path(f.file).resolve() not in seen_files]
    out.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.file))
    return out


def default_customize_dirs() -> list[Path]:
    import shutil
    dirs: list[Path] = []
    for exe in dict.fromkeys(filter(None, [shutil.which("python3"), shutil.which("python"), sys.executable])):
        dirs += interpreter_stdlib_dirs(exe)
    return dirs


def default_targets(root: Path, include_current: bool = True) -> list[Path]:
    import shutil
    dirs: list[Path] = []
    if include_current:
        for exe in dict.fromkeys(filter(None, [shutil.which("python3"), shutil.which("python"), sys.executable])):
            dirs += interpreter_site_dirs(exe)
    for v in find_venvs(root):
        dirs += venv_site_dirs(v)
    if os.environ.get("VIRTUAL_ENV"):
        dirs += venv_site_dirs(Path(os.environ["VIRTUAL_ENV"]))
    return dirs
