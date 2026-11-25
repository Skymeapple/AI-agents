# pthscan

**Check whether a Python package planted code that runs every time Python starts (`.pth`, `.start`, `sitecustomize.py`), without running it.**

```bash
pipx run pthscan            # python on PATH + every virtualenv under the current folder
pthscan .venv/ ~/projects/  # specific virtualenvs, or folders to search for them
```

Pure standard library, Python ≥ 3.9, no network calls.

## Why

Python runs every line that starts with `import` in any `.pth` file in `site-packages`, **on every interpreter start**. You don't have to import the package; `pip`, `pytest` or your IDE starting Python is enough. Attackers keep using this:

* **Microsoft `agent-framework` 1.26.1 / 1.26.2 (17 Nov 2026).** A `.pth` hook stole Azure, Azure DevOps and GitHub Actions OIDC tokens and read the environments of up to 300 processes. The package has about 200k downloads a month ([StepSecurity](https://www.stepsecurity.io/blog/microsoft-agent-framework-pypi-package-compromised), [Elastic](https://www.elastic.co/security-labs/threat-command/we-caught-the-agent-framework-pypi-compromise-six-minutes-after-it-published)).
* **LiteLLM 1.82.8.** `litellm_init.pth` ran a base64-hidden credential stealer ([StepSecurity](https://www.stepsecurity.io/blog/litellm-credential-stealer-hidden-in-pypi-wheel), [Snyk](https://snyk.io/blog/poisoned-security-scanner-backdooring-litellm)).
* **The June 2026 "Hades" campaign** shipped `<name>-setup.pth` hooks.

The usual advice is to run `find … -name "*.pth" | xargs grep import` by hand. That drowns you in legitimate hooks (setuptools, coverage, editable installs, virtualenv) and misses malware that hides in the *module* the hook imports. As Snyk put it, *"No widely deployed pip plugin currently does this automatically."*

## What it does

* **Never triggers what it scans.** Virtualenvs are found by path; interpreters are queried with `python -I -S`, which skips `.pth` processing. Running `python -S -m pthscan` inside a suspect environment is safe as well.
* **Recognises legitimate hooks by content, not by name.** An attacker can call a file `distutils-precedence.pth`. Recognised: setuptools, virtualenv, setuptools / hatch / uv / pdm editable installs, coverage (`a1_coverage.pth`), pytest-cov, namespace packages, autowrapt, ddtrace and pywin32.
* **Follows the import.** `import af_runtime_diag` looks harmless, but the stealer was in `af_runtime_diag.py`. pthscan reads the imported module and checks it for base64 decoding, `exec`, subprocesses, network calls, credential-store access (`/proc/*/environ`, `ACTIONS_ID_TOKEN_REQUEST`, cloud metadata) and encoded blobs.
* **Knows who installed what.** Each hook is mapped to its distribution through `RECORD`. A hook that no installed package claims is flagged: it was left behind after an uninstall, or something else dropped it.
* **Knows the incidents.** Flags installed known-compromised releases (`agent-framework` 1.26.1/1.26.2, `litellm` 1.82.7/1.82.8) and the incidents' file names.
* **Checks `sitecustomize.py` and `usercustomize.py`**, the other modules Python imports at every start. Empty files and Debian/Ubuntu's apport hook are recognised; package-internal copies (ddtrace, OpenTelemetry) that only run when you opt in are ignored.
* **Handles PEP 829 `.start` files**, the newer startup entry points, which already appear in the wild (`autowrapt-init.start`).

```
$ pthscan
CRITICAL   .venv/lib/python3.13/site-packages/zz_runtime_telemetry.pth (fancy-agents==2.0.0)
           malicious-looking startup hook
           - imported module zz_runtime_telemetry.py (fancy-agents==2.0.0) opens network connections, reads other processes or credential stores
           > import zz_runtime_telemetry

1 site-packages dir(s), 1 startup file(s): 1 critical, 0 warning
Do not start Python in an affected environment. From a clean machine: rotate every credential the host or runner
could reach (cloud, CI, git, package registries, AI keys), then delete and rebuild the environment.
```

*That example is from the test suite, which rebuilds the real attack shapes with inert payloads.*

**Results on real environments:** across 3 virtualenvs (800 MB+ of popular packages: numpy, pandas, scikit-learn, Django, FastAPI, Jupyter, boto3, OpenTelemetry, coverage, autowrapt, setuptools / hatch / uv editable installs) and a system Python, it judged all 10 real startup files correctly in under 0.2 s, with no false alarms.

## Severity

| Level | Meaning | Exit code |
|---|---|---|
| `CRITICAL` | A known-compromised release or incident file, or startup code / an imported module showing stealer behaviour | 1 |
| `WARNING` | Unrecognised code that runs at every start, or a hook no package claims | 0 (1 with `--strict`) |
| `OK` | Path entries only, or a recognised legitimate hook (shown with `--all`) | 0 |

## Run it automatically

**In CI, after installing dependencies.** This catches a poisoned release before your tests run and before the runner's OIDC tokens get stolen:

```yaml
- run: pip install -r requirements.txt
- uses: Skymeapple/AI-agents/projects/pthscan@research-lab
  with:
    args: --strict
```

Or with no Action: `pipx run pthscan --format github`.

**After every package install your coding agent runs** (Claude Code `PostToolUse` hook). If an install drops a hostile hook, the agent is told to stop and alert you:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "pipx run pthscan hook" }] }
    ]
  }
}
```

It reacts to `pip`, `uv`, `poetry`, `pdm`, `conda`, `pipx`, `rye` and `hatch` install, add and sync commands, and ignores everything else.

## Options

```
pthscan [paths ...] [--python PATH] [--format text|json|github] [--all] [--strict] [--depth 3]
pthscan hook
```

* `paths` can be `site-packages` dirs, virtualenvs, or folders to search for virtualenvs.
* `--python` also scans that interpreter's site dirs (queried with `-I -S`).

## If it finds something critical

1. **Don't start Python in that environment again.** Every start re-runs the hook.
2. From a clean machine, **rotate every credential the host or CI runner could reach**: cloud, CI and OIDC trust, git, PyPI / npm tokens, AI API keys.
3. Delete the environment and rebuild it from a lockfile pinned to a known-good version.
4. Check for persistence beyond Python. LiteLLM's payload, for example, added a systemd user service.

## Limits

This is static analysis of startup hooks. It isn't a full malware scanner. Obfuscation that avoids every indicator, or a hook that imports a large, legitimately owned package and hides its payload deep inside, can slip past. Treat `WARNING`s as "read this file".

---

Built by the [AI Research Lab](https://github.com/Skymeapple/AI-agents/tree/research-lab). Seen a hook it gets wrong? Open an issue with the file's contents (minus any secrets).
