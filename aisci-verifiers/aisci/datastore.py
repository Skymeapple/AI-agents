"""I4: a metadata standard and an append-only experiment store (tasks 4.2, 4.3).

Every experiment is written, including failures, negative results and runs the
safety layer blocked. Records are hash-chained so a missing or edited record
is detectable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import jsonschema

RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ExperimentRecord v1",
    "type": "object",
    "required": ["record_id", "campaign_id", "protocol", "outcome", "result", "agent",
                 "hypothesis_id", "timestamp_cycle", "instrument", "prev_hash"],
    "properties": {
        "record_id": {"type": "string"},
        "campaign_id": {"type": "string"},
        "hypothesis_id": {"type": ["string", "null"]},
        "protocol": {"type": "object"},
        "outcome": {"enum": ["success", "negative", "failed_run", "blocked_by_safety"]},
        "result": {"type": ["object", "null"]},
        "units": {"type": "object", "additionalProperties": {"type": "string"}},
        "agent": {"type": "string"},
        "instrument": {"type": "string"},
        "timestamp_cycle": {"type": "integer"},
        "prev_hash": {"type": "string"},
        "license": {"type": "string", "default": "CC-BY-4.0"},
    },
}


@dataclass
class DataStore:
    records: list[dict] = field(default_factory=list)
    issued: set[str] = field(default_factory=set)

    def issue_id(self, campaign: str) -> str:
        rid = f"{campaign}-{len(self.issued):05d}"
        self.issued.add(rid)
        return rid

    def _hash(self, rec: dict) -> str:
        return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()

    def append(self, rec: dict):
        rec = dict(rec)
        rec["prev_hash"] = self._hash(self.records[-1]) if self.records else "genesis"
        jsonschema.validate(rec, RECORD_SCHEMA)
        self.records.append(rec)

    def audit(self) -> dict:
        written = {r["record_id"] for r in self.records}
        chain_ok = all(self.records[i]["prev_hash"] == self._hash(self.records[i - 1])
                       for i in range(1, len(self.records)))
        by_outcome = {}
        for r in self.records:
            by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
        return {"issued": len(self.issued), "written": len(written),
                "missing": sorted(self.issued - written), "chain_intact": chain_ok,
                "by_outcome": by_outcome}
