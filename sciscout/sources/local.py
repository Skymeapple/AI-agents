"""Local JSON corpus source.

Reads works from a file on disk. This is the source that makes the platform
testable and reproducible: harvests are cached here, and every downstream score
is then a pure function of a file you can diff.

It is also the fallback when network egress is unavailable, which in many managed
environments is the normal case rather than the exception.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..models import Author, Maturity, Work
from .base import HarvestResult, Source


def _parse_work(raw: dict, source_label: str) -> Work:
    published = None
    if raw.get("published"):
        published = dt.date.fromisoformat(raw["published"])
    return Work(
        id=raw["id"],
        title=raw["title"],
        abstract=raw.get("abstract", ""),
        authors=[
            Author(
                name=a["name"],
                affiliation=a.get("affiliation"),
                is_industry=bool(a.get("is_industry", False)),
            )
            for a in raw.get("authors", [])
        ],
        published=published,
        venue=raw.get("venue"),
        maturity=Maturity(raw.get("maturity", "preprint")),
        disciplines=raw.get("disciplines", []),
        citations=raw.get("citations"),
        url=raw.get("url"),
        source=raw.get("source", source_label),
        extra=raw.get("extra", {}),
    )


class LocalCorpus(Source):
    """Serves works from a JSON file."""

    name = "local"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def harvest(self, query: str = "", limit: int = 10_000) -> HarvestResult:
        """Return works from the corpus, optionally filtered by substring.

        ``query`` matches against title and abstract, case-insensitively. An
        empty query returns everything, which is the usual case -- the corpus
        file is already the result of a query.
        """
        if not self.path.exists():
            return HarvestResult(
                errors=[f"corpus file not found: {self.path}"], source=self.name
            )
        try:
            raw_works = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            return HarvestResult(
                errors=[f"corpus file {self.path} is not valid JSON: {exc}"],
                source=self.name,
            )

        works: list[Work] = []
        errors: list[str] = []
        needle = query.lower().strip()
        for raw in raw_works:
            try:
                work = _parse_work(raw, self.name)
            except (KeyError, ValueError) as exc:
                errors.append(f"skipped malformed record {raw.get('id', '?')}: {exc}")
                continue
            if needle and needle not in work.text().lower():
                continue
            works.append(work)
            if len(works) >= limit:
                break

        return HarvestResult(works=works, errors=errors, source=self.name, query=query)
