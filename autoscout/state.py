"""Persistent state: what the agent knows and what it has tried.

Everything lives as plain JSON and Markdown in one directory, so that the
agent's entire history is diffable and reviewable in a pull request. That
matters more than it might seem. The scheduled workflow proposes each cycle's
state as a PR; a human merging it is the checkpoint that stops an automated
loop from compounding a mistake indefinitely.

Layout::

    champion.json     the current best configuration and its scores
    knowledge.json    seen papers, term statistics, proposer track record
    history.jsonl     one line per candidate ever evaluated
    cycles.jsonl      one line per cycle
    labels.jsonl      human relevance labels on real papers (optional)
    ideas.md          changes a proposer wanted but config cannot express
    digests/          per-cycle digest of the latest advancements
    REPORT.md         the trajectory, regenerated each cycle
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import AgentConfig

MAX_RECENT_WORKS = 1500
MAX_EVIDENCE = 5


@dataclass
class Knowledge:
    cycle: int = 0
    stream_cursor: int = 0
    seen: set[str] = field(default_factory=set)
    n_docs: int = 0
    background_df: dict[str, int] = field(default_factory=dict)
    topic_df: dict[str, dict[str, int]] = field(default_factory=dict)
    topic_docs: dict[str, int] = field(default_factory=dict)
    term_evidence: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Terms whose addition lost at the gate, with the cycle it happened, so the
    # literature proposer does not re-propose them every cycle.
    rejected_terms: dict[str, dict[str, int]] = field(default_factory=dict)
    proposer_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    mutation_step: float = 0.25
    recent_works: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "stream_cursor": self.stream_cursor,
            "seen": sorted(self.seen),
            "n_docs": self.n_docs,
            "background_df": dict(sorted(self.background_df.items())),
            "topic_df": {t: dict(sorted(d.items())) for t, d in sorted(self.topic_df.items())},
            "topic_docs": dict(sorted(self.topic_docs.items())),
            "term_evidence": self.term_evidence,
            "rejected_terms": self.rejected_terms,
            "proposer_stats": self.proposer_stats,
            "mutation_step": self.mutation_step,
            "recent_works": self.recent_works[-MAX_RECENT_WORKS:],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Knowledge":
        k = cls(**{key: raw[key] for key in raw if key != "seen"})
        k.seen = set(raw.get("seen", []))
        return k


class StateDir:
    def __init__(self, path: Path):
        self.path = Path(path)

    # -- paths ---------------------------------------------------------------
    @property
    def champion_path(self) -> Path:
        return self.path / "champion.json"

    @property
    def knowledge_path(self) -> Path:
        return self.path / "knowledge.json"

    @property
    def labels_path(self) -> Path:
        return self.path / "labels.jsonl"

    def exists(self) -> bool:
        return self.champion_path.exists()

    # -- champion ------------------------------------------------------------
    def load_champion(self) -> tuple[AgentConfig, dict]:
        raw = json.loads(self.champion_path.read_text())
        return AgentConfig.from_dict(raw["config"]), raw.get("scores", {})

    def save_champion(self, config: AgentConfig, scores: dict, cycle: int, reason: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": config.fingerprint(),
            "promoted_cycle": cycle,
            "reason": reason,
            "scores": scores,
            "config": config.to_dict(),
        }
        self.champion_path.write_text(json.dumps(payload, indent=2) + "\n")

    # -- knowledge -----------------------------------------------------------
    def load_knowledge(self) -> Knowledge:
        if not self.knowledge_path.exists():
            return Knowledge()
        return Knowledge.from_dict(json.loads(self.knowledge_path.read_text()))

    def save_knowledge(self, knowledge: Knowledge) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.write_text(json.dumps(knowledge.to_dict(), indent=1) + "\n")

    # -- logs ----------------------------------------------------------------
    def append(self, name: str, record: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / name).open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def read_jsonl(self, name: str) -> list[dict]:
        path = self.path / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def write_text(self, relative: str, text: str) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def append_idea(self, cycle: int, source: str, idea: str) -> None:
        path = self.path / "ideas.md"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Ideas backlog\n\n"
                "Changes a proposer suggested that the agent cannot make to itself, "
                "because they need code rather than configuration. For human review.\n\n"
            )
        with path.open("a") as fh:
            fh.write(f"- cycle {cycle} ({source}, {dt.date.today().isoformat()}): {idea}\n")
