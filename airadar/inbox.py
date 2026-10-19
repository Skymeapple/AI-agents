"""Loading inbox files: observations collected outside the Python collectors.

The daily agent session reads the web with tools the Python process cannot use,
and writes what it finds here. That makes the inbox the widest door into the
radar for an unfounded number, so the loader is strict in the same way
sciscout's scenario loader is: a field it does not understand, a metric outside
the vocabulary or an observation without a source URL fails the whole file with
the offending entry named. A loader that skipped bad rows quietly would let a
half-broken collector look like a quiet day.

The format is documented in ``INBOX.md`` next to this module.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import App, AppKind, Method, Observation, canonical_id

APP_FIELDS = {"name", "url", "kind", "description", "tags"}
OBS_FIELDS = {
    "app", "metric", "value", "period_days", "observed_on", "source_url",
    "method", "note", "via",
}


@dataclass
class InboxBatch:
    collector: str
    collected_at: dt.date
    apps: list[App] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    path: Path | None = None


class InboxError(ValueError):
    pass


def _app(raw: dict, where: str, collected_at: dt.date) -> App:
    unknown = set(raw) - APP_FIELDS
    if unknown:
        raise InboxError(f"{where}: unknown app field(s) {sorted(unknown)}")
    for required in ("name", "url", "kind"):
        if not str(raw.get(required, "")).strip():
            raise InboxError(f"{where}: app.{required} is required")
    try:
        kind = AppKind(raw["kind"])
    except ValueError:
        raise InboxError(
            f"{where}: unknown app.kind {raw['kind']!r}; expected one of "
            f"{[k.value for k in AppKind]}"
        ) from None
    tags = raw.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise InboxError(f"{where}: app.tags must be a list of strings")
    try:
        app_id = canonical_id(raw["url"])
    except ValueError as exc:
        raise InboxError(f"{where}: {exc}") from None
    return App(
        id=app_id,
        name=str(raw["name"]).strip(),
        url=str(raw["url"]).strip(),
        kind=kind,
        description=" ".join(str(raw.get("description", "")).split()),
        tags=[t.strip().lower() for t in tags if t.strip()],
        first_seen=collected_at,
    )


def parse_batch(payload: dict, origin: str = "<inbox>") -> InboxBatch:
    """Validate an inbox payload and convert it to apps and observations."""
    if not isinstance(payload, dict):
        raise InboxError(f"{origin}: top level must be an object")
    collector = str(payload.get("collector", "")).strip()
    if not collector:
        raise InboxError(f"{origin}: 'collector' is required")
    try:
        collected_at = dt.date.fromisoformat(str(payload.get("collected_at")))
    except ValueError:
        raise InboxError(f"{origin}: 'collected_at' must be an ISO date") from None
    raw_obs = payload.get("observations")
    if not isinstance(raw_obs, list):
        raise InboxError(f"{origin}: 'observations' must be a list")

    batch = InboxBatch(collector=collector, collected_at=collected_at)
    apps: dict[str, App] = {}
    for i, raw in enumerate(raw_obs):
        where = f"{origin} observation #{i}"
        if not isinstance(raw, dict):
            raise InboxError(f"{where}: must be an object")
        unknown = set(raw) - OBS_FIELDS
        if unknown:
            raise InboxError(f"{where}: unknown field(s) {sorted(unknown)}")
        app = _app(raw.get("app") or {}, where, collected_at)
        where = f"{where} ({app.name})"
        try:
            method = Method(raw.get("method"))
        except ValueError:
            raise InboxError(
                f"{where}: method must be one of {[m.value for m in Method]}"
            ) from None
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InboxError(f"{where}: value must be a number, got {value!r}")
        try:
            observed_on = dt.date.fromisoformat(str(raw.get("observed_on")))
        except ValueError:
            raise InboxError(f"{where}: observed_on must be an ISO date") from None
        if observed_on > collected_at:
            raise InboxError(f"{where}: observed_on {observed_on} is after collected_at")
        period = raw.get("period_days")
        try:
            observation = Observation(
                app_id=app.id,
                metric=str(raw.get("metric")),
                value=float(value),
                observed_on=observed_on,
                source_url=str(raw.get("source_url") or ""),
                method=method,
                collector=collector,
                period_days=float(period) if period is not None else None,
                via=raw.get("via"),
                note=str(raw.get("note") or ""),
            )
        except ValueError as exc:
            raise InboxError(f"{where}: {exc}") from None
        if app.id in apps:
            apps[app.id].merge(app)
        else:
            apps[app.id] = app
        batch.observations.append(observation)
    batch.apps = list(apps.values())
    return batch


def load_inbox_file(path: Path) -> InboxBatch:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InboxError(f"{path}: not valid JSON ({exc})") from None
    batch = parse_batch(payload, origin=path.name)
    batch.path = path
    return batch
