"""Source adapter protocol and shared plumbing.

A source turns some external corpus into :class:`~sciscout.models.Work` records.
Adapters are deliberately thin: they fetch, parse and normalise, and do nothing
else. Scoring never talks to the network, which keeps every score reproducible
from a cached corpus.

Network failures are expected rather than exceptional -- rate limits, outages,
egress policy -- so :class:`HarvestResult` carries partial results alongside
errors instead of raising. A harvest that got 80 works from one source and
failed on another should still produce a report, with the gap stated.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Work


@dataclass
class HarvestResult:
    """Outcome of a harvest: what came back, and what went wrong."""

    works: list[Work] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source: str = "unknown"
    query: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "HarvestResult") -> "HarvestResult":
        return HarvestResult(
            works=self.works + other.works,
            errors=self.errors + other.errors,
            source=f"{self.source}+{other.source}",
            query=self.query or other.query,
        )

    def summary(self) -> str:
        status = "ok" if self.ok else f"{len(self.errors)} error(s)"
        return f"{self.source}: {len(self.works)} works, {status}"


class Source(abc.ABC):
    """Fetches works matching a query."""

    name: str = "source"

    @abc.abstractmethod
    def harvest(self, query: str, limit: int = 100) -> HarvestResult:
        """Fetch up to ``limit`` works matching ``query``."""


def dump_works(works: list[Work], path: Path) -> None:
    """Write works to a JSON corpus file.

    Caching harvests to disk is what makes a run reproducible: the same corpus
    file gives the same scores, which is not true of a live API whose citation
    counts move daily.
    """
    payload = [
        {
            "id": w.id,
            "title": w.title,
            "abstract": w.abstract,
            "authors": [
                {"name": a.name, "affiliation": a.affiliation, "is_industry": a.is_industry}
                for a in w.authors
            ],
            "published": w.published.isoformat() if w.published else None,
            "venue": w.venue,
            "maturity": w.maturity.value,
            "disciplines": w.disciplines,
            "citations": w.citations,
            "url": w.url,
            "source": w.source,
            "extra": w.extra,
        }
        for w in works
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
