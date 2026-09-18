"""arXiv source adapter.

arXiv is the fastest signal in physics, mathematics, computer science and
quantitative biology -- results appear here months to years before the journal
version, which is precisely what a platform watching for cutting-edge
developments wants. The trade-off is that nothing on arXiv is refereed, which is
why :class:`~sciscout.models.Maturity` marks these works as preprints and the
evidence sub-score discounts them accordingly.

arXiv provides no citation counts. Works harvested here carry ``citations=None``
rather than 0 -- an absent measurement, not a measurement of zero -- and the
scoring code treats the two differently. Pair with OpenAlex when impact matters.

The API asks for no more than one request every three seconds; this client
honours that. It is deliberately the only rate limit hard-coded anywhere.
"""

from __future__ import annotations

import datetime as dt
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import requests

from ..models import Author, Maturity, Work
from .base import HarvestResult, Source

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
MIN_INTERVAL_SECONDS = 3.0

# arXiv primary categories mapped to coarse disciplines, for breadth scoring.
CATEGORY_DISCIPLINES: dict[str, str] = {
    "cs": "computer science",
    "math": "mathematics",
    "physics": "physics",
    "cond-mat": "condensed matter physics",
    "astro-ph": "astrophysics",
    "quant-ph": "quantum physics",
    "hep": "high energy physics",
    "nucl": "nuclear physics",
    "q-bio": "quantitative biology",
    "q-fin": "quantitative finance",
    "stat": "statistics",
    "eess": "electrical engineering",
    "econ": "economics",
    "nlin": "nonlinear sciences",
}


def _discipline(category: str) -> str:
    prefix = category.split(".")[0]
    return CATEGORY_DISCIPLINES.get(prefix, prefix)


class ArxivSource(Source):
    """Harvests preprints from the arXiv Atom API."""

    name = "arxiv"

    def __init__(self, timeout: float = 30.0, session: requests.Session | None = None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        self._last_request = time.monotonic()

    def harvest(self, query: str, limit: int = 100) -> HarvestResult:
        """Fetch preprints matching ``query`` (arXiv search syntax).

        Paginates in blocks of 100, the largest arXiv reliably serves.
        """
        works: list[Work] = []
        errors: list[str] = []
        start = 0
        page_size = min(100, limit)

        while len(works) < limit:
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, limit - len(works)),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            self._throttle()
            try:
                response = self.session.get(
                    f"{API}?{urlencode(params)}", timeout=self.timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                errors.append(f"arXiv request failed at offset {start}: {exc}")
                break

            try:
                root = ET.fromstring(response.text)
            except ET.ParseError as exc:
                errors.append(f"arXiv returned unparseable XML at offset {start}: {exc}")
                break

            entries = root.findall(f"{ATOM}entry")
            if not entries:
                break
            for entry in entries:
                try:
                    works.append(self._parse_entry(entry))
                except (AttributeError, ValueError) as exc:
                    errors.append(f"skipped an arXiv entry: {exc}")
            start += len(entries)

        return HarvestResult(
            works=works[:limit], errors=errors, source=self.name, query=query
        )

    def _parse_entry(self, entry: ET.Element) -> Work:
        def text(tag: str) -> str:
            node = entry.find(f"{ATOM}{tag}")
            return (node.text or "").strip() if node is not None else ""

        raw_id = text("id")
        published_text = text("published")
        published = None
        if published_text:
            published = dt.datetime.fromisoformat(
                published_text.replace("Z", "+00:00")
            ).date()

        authors = [
            Author(
                name=(node.find(f"{ATOM}name").text or "").strip(),
                affiliation=(
                    node.find(f"{ARXIV_NS}affiliation").text
                    if node.find(f"{ARXIV_NS}affiliation") is not None
                    else None
                ),
            )
            for node in entry.findall(f"{ATOM}author")
            if node.find(f"{ATOM}name") is not None
        ]

        categories = [
            node.attrib["term"]
            for node in entry.findall(f"{ATOM}category")
            if "term" in node.attrib
        ]
        disciplines = sorted({_discipline(c) for c in categories})

        # A journal reference means the preprint has since been published, which
        # is a genuine step up in evidence weight.
        journal_ref = entry.find(f"{ARXIV_NS}journal_ref")
        has_journal = journal_ref is not None and (journal_ref.text or "").strip()

        return Work(
            id=raw_id.rsplit("/", 1)[-1],
            title=" ".join(text("title").split()),
            abstract=" ".join(text("summary").split()),
            authors=authors,
            published=published,
            venue=(journal_ref.text.strip() if has_journal else "arXiv"),
            maturity=Maturity.PEER_REVIEWED if has_journal else Maturity.PREPRINT,
            disciplines=disciplines,
            # arXiv exposes no citation data. None, not 0: we did not measure it.
            citations=None,
            url=raw_id,
            source=self.name,
            extra={"categories": categories},
        )
