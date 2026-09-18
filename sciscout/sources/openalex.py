"""OpenAlex source adapter.

OpenAlex is the citation and metadata backbone here: it covers roughly every
indexed work across all disciplines, carries citation counts, institutional
affiliations and concept labels, and unlike the commercial indices it is openly
licensed and needs no key.

It supplies the three things arXiv cannot: citation counts (so impact can be
normalised), institution records (so industry participation and independent-group
counts are real rather than guessed), and concept labels (so cross-disciplinary
breadth is measurable).

OpenAlex asks for a contact email in requests, which buys access to a faster
pool. Set ``SCISCOUT_CONTACT_EMAIL`` or pass ``mailto``.
"""

from __future__ import annotations

import datetime as dt
import os

import requests

from ..models import Author, Maturity, Work
from .base import HarvestResult, Source

API = "https://api.openalex.org/works"
PAGE_SIZE = 100

# OpenAlex institution types that indicate a commercial rather than academic
# actor. Industry participation is one of the stronger readiness signals.
INDUSTRY_TYPES = {"company", "facility"}


class OpenAlexSource(Source):
    """Harvests works from the OpenAlex REST API."""

    name = "openalex"

    def __init__(
        self,
        mailto: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.mailto = mailto or os.environ.get("SCISCOUT_CONTACT_EMAIL")
        self.timeout = timeout
        self.session = session or requests.Session()

    def harvest(
        self,
        query: str,
        limit: int = 100,
        from_date: dt.date | None = None,
        extra_filters: str | None = None,
    ) -> HarvestResult:
        """Fetch works matching a full-text ``query``.

        Uses cursor pagination, which is the only method OpenAlex supports past
        10,000 results and is well-behaved below that too.
        """
        works: list[Work] = []
        errors: list[str] = []
        cursor = "*"

        filters = []
        if from_date:
            filters.append(f"from_publication_date:{from_date.isoformat()}")
        if extra_filters:
            filters.append(extra_filters)

        while len(works) < limit:
            params = {
                "search": query,
                "per-page": min(PAGE_SIZE, limit - len(works)),
                "cursor": cursor,
            }
            if filters:
                params["filter"] = ",".join(filters)
            if self.mailto:
                params["mailto"] = self.mailto

            try:
                response = self.session.get(API, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                errors.append(f"OpenAlex request failed: {exc}")
                break
            except ValueError as exc:
                errors.append(f"OpenAlex returned invalid JSON: {exc}")
                break

            results = payload.get("results", [])
            if not results:
                break
            for raw in results:
                try:
                    works.append(self._parse(raw))
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"skipped an OpenAlex record: {exc}")

            cursor = payload.get("meta", {}).get("next_cursor")
            if not cursor:
                break

        return HarvestResult(
            works=works[:limit], errors=errors, source=self.name, query=query
        )

    def _parse(self, raw: dict) -> Work:
        published = None
        if raw.get("publication_date"):
            published = dt.date.fromisoformat(raw["publication_date"])

        authors: list[Author] = []
        for authorship in raw.get("authorships", []) or []:
            person = authorship.get("author") or {}
            institutions = authorship.get("institutions") or []
            primary = institutions[0] if institutions else {}
            authors.append(
                Author(
                    name=person.get("display_name", "unknown"),
                    affiliation=primary.get("display_name"),
                    is_industry=any(
                        (inst.get("type") or "").lower() in INDUSTRY_TYPES
                        for inst in institutions
                    ),
                )
            )

        # OpenAlex scores concepts by confidence; below ~0.3 they are noise and
        # would inflate the cross-disciplinary breadth score.
        disciplines = sorted(
            {
                concept["display_name"].lower()
                for concept in (raw.get("concepts") or [])
                if concept.get("score", 0) >= 0.3 and concept.get("level", 9) <= 1
            }
        )

        primary_location = raw.get("primary_location") or {}
        venue_source = primary_location.get("source") or {}
        work_type = (raw.get("type") or "").lower()
        if work_type == "patent":
            maturity = Maturity.PATENT
        elif raw.get("type_crossref") == "posted-content" or venue_source.get(
            "display_name", ""
        ).lower() in {"arxiv", "biorxiv", "medrxiv", "ssrn"}:
            maturity = Maturity.PREPRINT
        elif work_type in {"report", "dataset"}:
            maturity = Maturity.REPORT
        else:
            maturity = Maturity.PEER_REVIEWED

        return Work(
            id=(raw.get("id") or "").rsplit("/", 1)[-1],
            title=raw.get("display_name") or raw.get("title") or "(untitled)",
            abstract=_reconstruct_abstract(raw.get("abstract_inverted_index")),
            authors=authors,
            published=published,
            venue=venue_source.get("display_name"),
            maturity=maturity,
            disciplines=disciplines,
            citations=raw.get("cited_by_count"),
            url=raw.get("doi") or raw.get("id"),
            source=self.name,
            extra={"type": work_type, "open_access": (raw.get("open_access") or {}).get("is_oa")},
        )


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild abstract text from OpenAlex's inverted index.

    OpenAlex stores abstracts as ``{word: [positions]}`` for licensing reasons.
    The TRL signal extractor needs running text, so we invert it back.
    """
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = [
        (position, word)
        for word, occurrences in inverted_index.items()
        for position in occurrences
    ]
    positions.sort()
    return " ".join(word for _, word in positions)
