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
  aliases.jsonl         append-only: an app that moved (repo transfer, rename) to a new url
  verdicts.jsonl        append-only: what research found when it tested a hypothesis
  snapshots/DATE.json   that day's scores and directions, for day-on-day comparison
  reports/DATE.md       the human-readable daily report
  inbox/                pending inbox files; moved to inbox/archive once ingested
```
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from .inbox import InboxBatch
from .models import App, Observation, canonical_id

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
        self.aliases_path = self.root / "aliases.jsonl"
        self.verdicts_path = self.root / "verdicts.jsonl"
        self.snapshots = self.root / "snapshots"
        self.reports = self.root / "reports"

    # -- apps and observations ------------------------------------------------

    def load_apps(self) -> dict[str, App]:
        if not self.apps_path.exists():
            return {}
        raw = json.loads(self.apps_path.read_text())
        registry = {key: App.from_dict(value) for key, value in raw.items()}
        for old_id, (new_id, new_url) in self.aliases().items():
            if old_id not in registry:
                continue
            app = registry.pop(old_id)
            if new_id in registry:
                registry[new_id].merge(app)
            else:
                app.id, app.url = new_id, new_url
                app.name = urlparse(new_url if "//" in new_url else "//" + new_url).path.strip("/") or app.name
                registry[new_id] = app
        return registry

    def save_apps(self, apps: dict[str, App]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {key: apps[key].to_dict() for key in sorted(apps)}
        self.apps_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    def load_observations(self) -> list[Observation]:
        """Every observation, with the app ids of moved apps resolved to where
        they live now. The file itself is never rewritten."""
        rows = [Observation.from_dict(r) for r in _read_jsonl(self.observations_path)]
        return self._resolve(rows)

    def _resolve(self, rows: list[Observation]) -> list[Observation]:
        aliases = self.aliases()
        if not aliases:
            return rows
        return [replace(o, app_id=aliases[o.app_id][0]) if o.app_id in aliases else o for o in rows]

    def add(self, apps: list[App], observations: list[Observation]) -> int:
        """Merge apps into the registry and append new observations.

        Returns the number of observations actually added; re-ingesting the same
        file adds nothing.
        """
        registry = self.load_apps()
        aliases = self.aliases()
        for app in apps:
            if app.id in aliases:
                app.id, app.url = aliases[app.id]
            if app.id in registry:
                registry[app.id].merge(app)
            else:
                registry[app.id] = app
        self.save_apps(registry)

        seen = {o.key() for o in self.load_observations()}
        fresh = []
        for observation in self._resolve(observations):
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

    # -- aliases ---------------------------------------------------------------

    def record_move(self, old_url: str, new_url: str, observed_on: dt.date,
                    source_url: str, note: str = "") -> None:
        """Record that an app moved (a repo transferred to another owner, a
        product renamed its domain). Its old and new history become one app."""
        old_id, new_id = canonical_id(old_url), canonical_id(new_url)
        if old_id == new_id:
            raise ValueError(f"{old_url} and {new_url} are the same app ({old_id})")
        if not source_url.strip():
            raise ValueError("a move record needs the source_url that shows it")
        if self.aliases().get(new_id, (None,))[0] == old_id:
            raise ValueError(f"{new_id} already moved to {old_id}; a move back would loop")
        _append_jsonl(self.aliases_path, [{
            "from_id": old_id, "to_id": new_id, "to_url": new_url,
            "observed_on": observed_on.isoformat(), "source_url": source_url, "note": note,
        }])

    def aliases(self) -> dict[str, tuple[str, str]]:
        """old app id -> (current app id, current url), following chains of moves."""
        direct: dict[str, tuple[str, str]] = {}
        for row in _read_jsonl(self.aliases_path):
            direct[row["from_id"]] = (row["to_id"], row["to_url"])
        resolved = {}
        for old_id, target in direct.items():
            seen = {old_id}
            while target[0] in direct and target[0] not in seen:
                seen.add(target[0])
                target = direct[target[0]]
            if target[0] != old_id:
                resolved[old_id] = target
        return resolved

    # -- research verdicts --------------------------------------------------------

    VERDICTS = ("disproved", "partly disproved", "holds")

    def record_verdict(self, direction: str, pattern: str, verdict: str, observed_on: dt.date,
                       source_url: str, note: str = "") -> None:
        """Record what research found when it tested a hypothesis. ``pattern``
        is the rule's name ("fast and uncrowded") or "*" for every pattern on
        that direction."""
        if verdict not in self.VERDICTS:
            raise ValueError(f"verdict must be one of {self.VERDICTS}, got {verdict!r}")
        if not source_url.strip():
            raise ValueError("a verdict needs the source_url of the research behind it")
        _append_jsonl(self.verdicts_path, [{
            "direction": direction, "pattern": pattern, "verdict": verdict,
            "observed_on": observed_on.isoformat(), "source_url": source_url, "note": note,
        }])

    def verdict_for(self, direction: str, pattern: str, asof: dt.date) -> dict | None:
        """The latest verdict on or before ``asof`` for this direction and
        pattern, a pattern-specific one winning a tie with a "*" one."""
        best = None
        for row in _read_jsonl(self.verdicts_path):
            if row["direction"] != direction or row["pattern"] not in (pattern, "*"):
                continue
            if dt.date.fromisoformat(row["observed_on"]) > asof:
                continue
            rank = (row["observed_on"], row["pattern"] == pattern)
            if best is None or rank >= best[0]:
                best = (rank, row)
        return best[1] if best else None

    def gone_as_of(self, asof: dt.date) -> dict[str, dict]:
        """Apps whose latest availability record on or before ``asof`` says gone."""
        latest: dict[str, dict] = {}
        aliases = self.aliases()
        for row in _read_jsonl(self.availability_path):
            if dt.date.fromisoformat(row["observed_on"]) > asof:
                continue
            app_id = aliases.get(row["app_id"], (row["app_id"],))[0]
            current = latest.get(app_id)
            if current is None or row["observed_on"] >= current["observed_on"]:
                latest[app_id] = row
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
