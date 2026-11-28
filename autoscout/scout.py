"""Gathering: fetch what is new since the last cycle.

Two modes, one interface:

* **live** -- query arXiv, newest first, with queries built from the agent's
  *current* topic profiles. As profiles learn new vocabulary, the harvest
  follows it, so what the agent looks for and what it finds co-evolve.
* **offline** -- replay a local stream file one batch per cycle. This is how the
  loop runs where network egress is blocked (common in managed environments),
  and how it is tested: a replayed stream is deterministic.

Either way, papers already seen are dropped, so each cycle works only on what
is genuinely new to the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sciscout.models import Work
from sciscout.sources.local import _parse_work

from .config import AgentConfig
from .state import Knowledge

DEFAULT_STREAM = Path(__file__).resolve().parent.parent / "data" / "agent_stream.json"


@dataclass
class Gathered:
    works: list[Work]
    source: str
    queries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exhausted: bool = False  # offline stream has no more batches


def build_queries(config: AgentConfig, categories: list[str]) -> list[str]:
    """One arXiv query per topic from its highest-weighted terms."""
    per_query = int(config.get("scout.terms_per_query"))
    cats = " OR ".join(f"cat:{c}" for c in categories)
    queries = []
    for topic in config.topics():
        profile = config.profiles[topic]
        terms = sorted(profile, key=lambda t: (-profile[t], t))[:per_query]
        phrase = " OR ".join(f'all:"{t.replace("-", " ")}"' for t in terms)
        queries.append(f"({phrase}) AND ({cats})")
    return queries


def gather_live(config: AgentConfig, knowledge: Knowledge, categories: list[str]) -> Gathered:
    from sciscout.sources.arxiv import ArxivSource

    source = ArxivSource()
    limit = int(config.get("scout.limit_per_query"))
    works: dict[str, Work] = {}
    errors: list[str] = []
    queries = build_queries(config, categories)
    for query in queries:
        result = source.harvest(query, limit=limit)
        errors.extend(result.errors)
        for work in result.works:
            if work.id not in knowledge.seen:
                works.setdefault(work.id, work)
    return Gathered(list(works.values()), "arxiv", queries, errors)


def gather_offline(knowledge: Knowledge, stream_path: Path = DEFAULT_STREAM) -> Gathered:
    """Release the next weekly batch of the stream file."""
    raw = json.loads(Path(stream_path).read_text())
    weeks = sorted({r["extra"].get("synthetic_week", 0) for r in raw})
    if knowledge.stream_cursor >= len(weeks):
        return Gathered([], f"stream:{Path(stream_path).name}", exhausted=True)
    week = weeks[knowledge.stream_cursor]
    works = [_parse_work(r, "stream") for r in raw if r["extra"].get("synthetic_week", 0) == week]
    works = [w for w in works if w.id not in knowledge.seen]
    knowledge.stream_cursor += 1
    return Gathered(works, f"stream:{Path(stream_path).name}#week{int(week)}")
