# Changelog

## 0.1.0 (2026-11-21)

First release.

* Finds `.pth` and PEP 829 `.start` startup files in virtualenvs (located by path, never executed) and in interpreters (queried with `python -I -S`).
* Maps each file to the distribution that installed it (`RECORD`), and flags orphans.
* Recognises legitimate hooks by content (setuptools, virtualenv, editable installs, coverage, pytest-cov, namespace packages, autowrapt and others).
* Flags unknown startup code, inline indicators (base64, exec, subprocess, network, credential stores, encoded blobs), and suspicious modules that hooks import.
* Known-compromised releases: agent-framework 1.26.1/1.26.2 and litellm 1.82.7/1.82.8. Known incident file names.
* Output formats: text, json and github. `--strict`. Claude Code PostToolUse hook mode. GitHub Action.
