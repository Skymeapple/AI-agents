"""On-disk state of the radar.

Everything lives in plain files under one directory so the whole history can be
committed to git and diffed. A day's run is then a reviewable commit, and the
self-tuner's decisions sit in the same history as the data that justified them.

```
data/radar/
  apps.json             app registry, keyed by canonical id
  observations.jsonl    append-only measurements; growth is derived, never stored
  runs.jsonl            one line per collector per run: what came back, what failed
  params.json           current scoring parameters, each with its provenance
  predictions.jsonl     what each day's ranking predicted, for scoring later
  iterations.jsonl      every parameter change the tuner made, with its evidence
  availability.jsonl    append-only: an app's page found gone (404) or back, with evidence
  snapshots/DATE.json   that day's scores and directions, for day-on-day comparison
  reports/DATE.md       the human-readable daily report
  inbox/                pending inbox files; moved to inbox/archive once ingested
```
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .inbox import InboxBatch
from .models import App, Observation

DEFAULT_ROOT = Path("data/radar")


@dataclass
class RunRecord:
    """Outcome of one collector on one day -- the raw material of source health."""

    date: dt.date
    collector: str
    observations: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.observations > 0 and not self.errors

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "collector": self.collector,
            "observations": self.observations,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RunRecord":
        return cls(
            date=dt.date.fromisoformat(raw["date"]),
            collector=raw["collector"],
            observations=int(raw["observations"]),
            errors=list(raw.get("errors", [])),
        )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class Store:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.apps_path = self.root / "apps.json"
        self.observations_path = self.root / "observations.jsonl"
        self.runs_path = self.root / "runs.jsonl"
        self.params_path = self.root / "params.json"
        self.predictions_path = self.root / "predictions.jsonl"
        self.iterations_path = self.root / "iterations.jsonl"
        self.availability_path = self.root / "availability.jsonl"
        self.snapshots = self.root / "snapshots"
        self.reports = self.root / "reports"

    # -- apps and observations ------------------------------------------------

    def load_apps(self) -> dict[str, App]:
        if not self.apps_path.exists():
            return {}
        raw = json.loads(self.apps_path.read_text())
        return {key: App.from_dict(value) for key, value in raw.items()}

    def save_apps(self, apps: dict[str, App]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {key: apps[key].to_dict() for key in sorted(apps)}
        self.apps_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    def load_observations(self) -> list[Observation]:
        return [Observation.from_dict(r) for r in _read_jsonl(self.observations_path)]

    def add(self, apps: list[App], observations: list[Observation]) -> int:
        """Merge apps into the registry and append new observations.

        Returns the number of observations actually added; re-ingesting the same
        file adds nothing.
        """
        registry = self.load_apps()
        for app in apps:
            if app.id in registry:
                registry[app.id].merge(app)
            else:
                registry[app.id] = app
        self.save_apps(registry)

        seen = {o.key() for o in self.load_observations()}
        fresh = []
        for observation in observations:
            if observation.key() not in seen:
                seen.add(observation.key())
                fresh.append(observation)
        _append_jsonl(self.observations_path, [o.to_dict() for o in fresh])
        return len(fresh)

    def ingest_batch(self, batch: InboxBatch, archive: bool = True) -> int:
        added = self.add(batch.apps, batch.observations)
        self.record_run(RunRecord(batch.collected_at, batch.collector, len(batch.observations)))
        if archive and batch.path and batch.path.parent == self.inbox:
            target = self.inbox / "archive" / batch.path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(batch.path), target)
            notes = batch.path.with_suffix(".notes.md")
            if notes.exists():
                shutil.move(str(notes), target.with_suffix(".notes.md"))
        return added

    # -- run log --------------------------------------------------------------

    def record_run(self, run: RunRecord) -> None:
        _append_jsonl(self.runs_path, [run.to_dict()])

    def load_runs(self) -> list[RunRecord]:
        return [RunRecord.from_dict(r) for r in _read_jsonl(self.runs_path)]

    # -- predictions and iterations --------------------------------------------

    def append_predictions(self, rows: list[dict]) -> None:
        _append_jsonl(self.predictions_path, rows)

    def load_predictions(self) -> list[dict]:
        return _read_jsonl(self.predictions_path)

    def append_iteration(self, row: dict) -> None:
        _append_jsonl(self.iterations_path, [row])

    def load_iterations(self) -> list[dict]:
        return _read_jsonl(self.iterations_path)

    # -- availability -----------------------------------------------------------

    def record_availability(self, app_id: str, status: str, observed_on: dt.date,
                            source_url: str, note: str = "") -> None:
        """Record that an app's page was found gone, or back. Append-only, like
        observations: the record of what the radar saw stays intact."""
        if status not in ("gone", "back"):
            raise ValueError(f"status must be 'gone' or 'back', got {status!r}")
        if not source_url.strip():
            raise ValueError("an availability record needs the source_url that was checked")
        _append_jsonl(self.availability_path, [{
            "app_id": app_id, "status": status, "observed_on": observed_on.isoformat(),
            "source_url": source_url, "note": note,
        }])

    def gone_as_of(self, asof: dt.date) -> dict[str, dict]:
        """Apps whose latest availability record on or before ``asof`` says gone."""
        latest: dict[str, dict] = {}
        for row in _read_jsonl(self.availability_path):
            if dt.date.fromisoformat(row["observed_on"]) > asof:
                continue
            current = latest.get(row["app_id"])
            if current is None or row["observed_on"] >= current["observed_on"]:
                latest[row["app_id"]] = row
        return {app_id: row for app_id, row in latest.items() if row["status"] == "gone"}

    # -- snapshots and reports --------------------------------------------------

    def write_snapshot(self, date: dt.date, payload: dict) -> Path:
        path = self.snapshots / f"{date.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        return path

    def previous_snapshot(self, before: dt.date) -> dict | None:
        """The most recent snapshot strictly before ``before``."""
        if not self.snapshots.exists():
            return None
        candidates = sorted(
            p for p in self.snapshots.glob("*.json") if p.stem < before.isoformat()
        )
        return json.loads(candidates[-1].read_text()) if candidates else None

    def write_report(self, date: dt.date, markdown: str) -> Path:
        path = self.reports / f"{date.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown)
        return path
