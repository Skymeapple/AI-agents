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

# Venues whose works are preprints whatever OpenAlex calls the record type.
PREPRINT_VENUES = {"arxiv", "biorxiv", "medrxiv", "ssrn"}


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
        filters = _build_filters(from_date, extra_filters)

        while cursor and len(works) < limit:
            params = {
                "search": query,
                "per-page": min(PAGE_SIZE, limit - len(works)),
                "cursor": cursor,
            }
            if filters:
                params["filter"] = filters
            if self.mailto:
                params["mailto"] = self.mailto

            payload = self._fetch_page(params, errors)
            results = payload.get("results", []) if payload is not None else []
            if not results:
                break
            self._parse_page(results, works, errors)
            cursor = payload.get("meta", {}).get("next_cursor")

        return HarvestResult(
            works=works[:limit], errors=errors, source=self.name, query=query
        )

    def _fetch_page(self, params: dict, errors: list[str]) -> dict | None:
        """One page of results, or ``None`` with the failure recorded in ``errors``."""
        try:
            response = self.session.get(API, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            errors.append(f"OpenAlex request failed: {exc}")
        except ValueError as exc:
            errors.append(f"OpenAlex returned invalid JSON: {exc}")
        return None

    def _parse_page(
        self, results: list[dict], works: list[Work], errors: list[str]
    ) -> None:
        """Parse a page into ``works``; a malformed record is skipped, not fatal."""
        for raw in results:
            try:
                works.append(self._parse(raw))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"skipped an OpenAlex record: {exc}")

    def _parse(self, raw: dict) -> Work:
        published = None
        if raw.get("publication_date"):
            published = dt.date.fromisoformat(raw["publication_date"])
        authors = _parse_authors(raw)
        disciplines = _parse_disciplines(raw)

        venue_source = (raw.get("primary_location") or {}).get("source") or {}
        work_type = (raw.get("type") or "").lower()

        return Work(
            id=(raw.get("id") or "").rsplit("/", 1)[-1],
            title=raw.get("display_name") or raw.get("title") or "(untitled)",
            abstract=_reconstruct_abstract(raw.get("abstract_inverted_index")),
            authors=authors,
            published=published,
            venue=venue_source.get("display_name"),
            maturity=_infer_maturity(raw, work_type, venue_source),
            disciplines=disciplines,
            citations=raw.get("cited_by_count"),
            url=raw.get("doi") or raw.get("id"),
            source=self.name,
            extra={"type": work_type, "open_access": (raw.get("open_access") or {}).get("is_oa")},
        )


def _build_filters(from_date: dt.date | None, extra_filters: str | None) -> str:
    """Comma-joined OpenAlex filter expression; empty when nothing is filtered."""
    filters = []
    if from_date:
        filters.append(f"from_publication_date:{from_date.isoformat()}")
    if extra_filters:
        filters.append(extra_filters)
    return ",".join(filters)


def _parse_authors(raw: dict) -> list[Author]:
    authors: list[Author] = []
    for authorship in raw.get("authorships") or []:
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
    return authors


def _parse_disciplines(raw: dict) -> list[str]:
    # OpenAlex scores concepts by confidence; below ~0.3 they are noise and
    # would inflate the cross-disciplinary breadth score.
    return sorted(
        {
            concept["display_name"].lower()
            for concept in (raw.get("concepts") or [])
            if concept.get("score", 0) >= 0.3 and concept.get("level", 9) <= 1
        }
    )


def _infer_maturity(raw: dict, work_type: str, venue_source: dict) -> Maturity:
    if work_type == "patent":
        return Maturity.PATENT
    is_preprint = (
        raw.get("type_crossref") == "posted-content"
        or venue_source.get("display_name", "").lower() in PREPRINT_VENUES
    )
    if is_preprint:
        return Maturity.PREPRINT
    if work_type in {"report", "dataset"}:
        return Maturity.REPORT
    return Maturity.PEER_REVIEWED


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
