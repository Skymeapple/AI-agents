"""Known-good startup hooks and known-compromised releases.

Known-good hooks are matched on CONTENT, never on file name alone: an attacker
can name a file distutils-precedence.pth. Each entry is a regex that the whole
executable line must match.
"""
import re

# (label, regex for one executable line)
KNOWN_GOOD = [
    ("setuptools distutils shim (distutils-precedence.pth)",
     r"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os\.environ\.get\(var, '(?:local|stdlib)'\) == 'local'; enabled and __import__\('_distutils_hack'\)\.add_shim\(\);?\s*"),
    ("virtualenv bootstrap (_virtualenv.pth)", r"import _virtualenv\s*"),
    ("setuptools editable install finder",
     r"import __editable___[\w.-]+_finder; __editable___[\w.-]+_finder\.install\([^)]*\)\s*"),
    ("coverage.py subprocess measurement", r"import coverage; coverage\.process_startup\(\)\s*"),
    ("coverage.py subprocess measurement (a1_coverage.pth)",
     r"import sys; exec\('import os\\n\\nif os\.getenv\(\"COVERAGE_PROCESS_START\"\)(?: or os\.getenv\(\"COVERAGE_PROCESS_CONFIG\"\))?:\\n try:\\n  import coverage\\n except:\\n  pass\\n else:\\n  coverage\.process_startup\((?:slug=\"pth\")?\)'\)\s*"),
    ("wrapt post-import hooks (autowrapt)", r"import autowrapt; autowrapt\.init\(\)\s*"),
    ("wrapt post-import hooks (autowrapt, PEP 829 .start)", r"autowrapt:init\s*"),
    ("pytest-cov subprocess support",
     r"import os, sys;exec\('if \\'COV_CORE_SOURCE\\' in os\.environ:.*'\)\s*"),
    ("pytest-cov subprocess support (new)", r"import pytest_cov\.embed.*"),
    ("setuptools namespace package (-nspkg.pth)",
     r"import sys, types, os;(?:has_mfs = sys\.version_info > \(3, 5\);)?p = os\.path\.join\(sys\._getframe\(1\)\.f_locals\['sitedir'\], \*\(.*"),
    ("legacy easy-install.pth", r"import sys; sys\.__plen = len\(sys\.path\)\s*"),
    ("legacy easy-install.pth", r"import sys; new = sys\.path\[sys\.__plen:\]; del sys\.path\[sys\.__plen:\]; p = getattr\(sys, '__egginsert', 0\); sys\.path\[p:p\] = new; sys\.__egginsert = p \+ len\(new\)\s*"),
    ("pywin32 bootstrap", r"import pywin32_bootstrap\s*"),
    ("matplotlib / mpl_toolkits namespace", r"import sys, types, os;.*mpl_toolkits.*"),
    ("Datadog ddtrace auto-instrumentation", r"import ddtrace\.bootstrap\.sitecustomize.*"),
    ("hatch/uv editable finder", r"import _[\w]+_editable(?:_finder)?; _[\w]+_editable(?:_finder)?\.install\(\)\s*"),
    ("pdm editable finder", r"import _pdm_editable_\w+.*"),
]
KNOWN_GOOD_RE = [(label, re.compile(rx)) for label, rx in KNOWN_GOOD]

# Releases known to ship malicious startup hooks. Source URLs are kept so every
# entry can be checked. (normalized name -> {version: source})
COMPROMISED = {
    "agent-framework": {
        "1.26.1": "https://www.stepsecurity.io/blog/microsoft-agent-framework-pypi-package-compromised",
        "1.26.2": "https://www.stepsecurity.io/blog/microsoft-agent-framework-pypi-package-compromised",
    },
    "litellm": {
        "1.82.7": "https://www.stepsecurity.io/blog/litellm-credential-stealer-hidden-in-pypi-wheel",
        "1.82.8": "https://www.stepsecurity.io/blog/litellm-credential-stealer-hidden-in-pypi-wheel",
    },
}

# File names seen in real incidents.
MALICIOUS_FILES = {
    "litellm_init.pth": "LiteLLM 1.82.8 credential stealer",
    "af_runtime_diag.pth": "agent-framework 1.26.2 credential stealer",
    "af_runtime_diag.py": "agent-framework 1.26.2 credential stealer",
}
MALICIOUS_FILE_PATTERNS = [
    (re.compile(r".+-setup\.pth$"), "matches the 'Hades' campaign naming (<name>-setup.pth)"),
]

# Indicators of hostile startup code: (label, regex). Any hit in a .pth line is critical.
INDICATORS = [
    ("decodes base64", re.compile(r"b64decode|base64\.b(?:32|16|85)?decode|decodebytes|codecs\.decode\([^)]*base64")),
    ("runs dynamic code (exec/eval/compile)", re.compile(r"\b(?:exec|eval|compile)\s*\(")),
    ("spawns processes", re.compile(r"\bsubprocess\b|os\.system\s*\(|os\.popen\s*\(|os\.exec[lv]p?e?\s*\(|\bpty\.spawn|os\.fork\s*\(|setsid")),
    ("opens network connections", re.compile(r"\bsocket\b|urllib\.request|urlopen\s*\(|http\.client|\brequests\.(?:get|post|put)\b|httpx\.|webhook|discord(?:app)?\.com/api")),
    ("decompresses or unmarshals code", re.compile(r"zlib\.decompress|marshal\.loads|lzma\.decompress|bz2\.decompress|pickle\.loads")),
    ("reads other processes or credential stores", re.compile(r"/proc/\S*environ|/proc/\[0-9\]|\.aws/credentials|\.azure/|\.docker/config|\.git-credentials|\.npmrc|\.pypirc|ACTIONS_ID_TOKEN_REQUEST|IDENTITY_ENDPOINT|169\.254\.169\.254")),
    ("loads native code", re.compile(r"\bctypes\b|cffi")),
    ("hides imports", re.compile(r"__import__\(\s*['\"](?:os|subprocess|socket|base64|urllib|zlib|marshal)['\"]|getattr\(__import__")),
    ("contains an encoded blob", re.compile(r"[A-Za-z0-9+/]{240,}={0,2}")),
]
