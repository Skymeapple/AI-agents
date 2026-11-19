# AI Research Lab: agent instructions

This repo is an autonomous research lab that ships one developer tool a day.

* Daily mission: follow `LAB_PROTOCOL.md` exactly (sync → review metrics → research → build → test → ship → report).
* Each project lives in `projects/<name>/` and is self-contained, with its own manifest, tests, README and LICENSE.
* Register every project in `lab/registry.json`, then run `python3 scripts/build_dashboard.py`.
* Test a project: `cd projects/<name> && npm test` (or `pytest`).
* Lint these instruction files: `node projects/ctxlint/bin/ctxlint.js .`
* Never commit secrets. Publishing uses the `NPM_TOKEN` / `PYPI_TOKEN` Actions secrets via `.github/workflows/publish.yml`.
* Metrics (`lab/metrics/`, `DASHBOARD.md`) are written by the `metrics` workflow. Never edit the numbers by hand.
