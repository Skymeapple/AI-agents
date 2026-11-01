"""I2: a structured world model with enforced provenance (tasks 2.1 and 2.3).

Every claim must carry evidence (a code run, a dataset, or a citation). A
synthesis claim must also name at least two supporting claims (RP3's
chain-of-evidence rule). Claims are retracted, never deleted, so the history
stays auditable, and retracting a claim flags everything derived from it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Literal

EvidenceKind = Literal["code", "data", "citation", "experiment"]
ClaimType = Literal["data", "literature", "synthesis"]
Status = Literal["active", "retracted", "suspect"]

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorldModelClaim",
    "type": "object",
    "required": ["id", "text", "type", "confidence", "evidence", "depends_on", "cycle", "status"],
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string", "minLength": 1},
        "type": {"enum": ["data", "literature", "synthesis"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["kind", "ref"],
            "properties": {"kind": {"enum": ["code", "data", "citation", "experiment"]},
                           "ref": {"type": "string", "minLength": 1},
                           "digest": {"type": "string"}}}},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "cycle": {"type": "integer", "minimum": 0},
        "status": {"enum": ["active", "retracted", "suspect"]},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "allOf": [{"if": {"properties": {"type": {"const": "synthesis"}}},
               "then": {"properties": {"depends_on": {"minItems": 2}}}}],
}


class ProvenanceError(ValueError):
    pass


@dataclass
class Evidence:
    kind: EvidenceKind
    ref: str
    digest: str = ""

    def __post_init__(self):
        if not self.digest:
            self.digest = hashlib.sha256(f"{self.kind}:{self.ref}".encode()).hexdigest()[:16]


@dataclass
class Claim:
    id: str
    text: str
    type: ClaimType
    confidence: float
    evidence: list[Evidence]
    depends_on: list[str] = field(default_factory=list)
    cycle: int = 0
    status: Status = "active"
    open_questions: list[str] = field(default_factory=list)


@dataclass
class WorldModel:
    claims: dict[str, Claim] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def add(self, c: Claim) -> Claim:
        if not c.evidence:
            raise ProvenanceError(f"{c.id}: claim has no evidence")
        if c.type == "synthesis" and len(c.depends_on) < 2:
            raise ProvenanceError(f"{c.id}: synthesis claim needs >= 2 supporting claims")
        missing = [d for d in c.depends_on if d not in self.claims]
        if missing:
            raise ProvenanceError(f"{c.id}: depends on unknown claims {missing}")
        if c.id in self.claims:
            raise ProvenanceError(f"{c.id}: duplicate id")
        self.claims[c.id] = c
        self.events.append({"event": "add", "id": c.id, "cycle": c.cycle})
        return c

    def descendants(self, cid: str) -> set[str]:
        out, frontier = set(), {cid}
        while frontier:
            nxt = {k for k, c in self.claims.items() if frontier & set(c.depends_on)} - out
            out |= nxt
            frontier = nxt
        return out

    def retract(self, cid: str, reason: str, cycle: int = 0):
        self.claims[cid].status = "retracted"
        self.events.append({"event": "retract", "id": cid, "reason": reason, "cycle": cycle})
        for d in self.descendants(cid):
            if self.claims[d].status == "active":
                self.claims[d].status = "suspect"
                self.events.append({"event": "suspect", "id": d, "because": cid, "cycle": cycle})

    def traceability(self) -> float:
        act = [c for c in self.claims.values() if c.status != "retracted"]
        ok = [c for c in act if c.evidence and all(e.ref for e in c.evidence)]
        return len(ok) / len(act) if act else 1.0

    def report(self) -> str:
        lines = []
        for c in self.claims.values():
            if c.status == "retracted":
                continue
            refs = "; ".join(f"{e.kind}:{e.ref}" for e in c.evidence)
            dep = f" (from {', '.join(c.depends_on)})" if c.depends_on else ""
            flag = " [SUSPECT]" if c.status == "suspect" else ""
            lines.append(f"- {c.text}{flag} [{refs}]{dep}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({k: asdict(v) for k, v in self.claims.items()}, indent=2)
