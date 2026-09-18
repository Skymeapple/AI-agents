"""Domain model: what the platform tracks.

Two levels matter, and conflating them is the usual modelling mistake:

*A* :class:`Work` *is a single artefact* -- one paper, one preprint, one patent.
Works are what we ingest. They are facts with dates and citation counts.

*A* :class:`Track` *is a research direction* -- "room-temperature ambient-pressure
superconductivity", "solid-state sodium batteries". A track is a cluster of works
plus the groups pursuing them. Tracks are what we score for importance, what we
estimate commercial potential for, and what we build investment cases around.
You cannot sensibly ask for the ROI of a single paper; you can ask it of a
direction.
"""

from __future__ import annotations

import datetime as dt
import enum
import math
import re
from dataclasses import dataclass, field


class Maturity(enum.Enum):
    """Publication status, a coarse proxy for how vetted a result is."""

    PREPRINT = "preprint"
    PEER_REVIEWED = "peer_reviewed"
    PATENT = "patent"
    CLINICAL_REGISTRY = "clinical_registry"
    REPORT = "report"

    @property
    def evidence_weight(self) -> float:
        """Relative credence, in ``[0, 1]``, for evidence-strength scoring.

        A preprint is not worthless -- in fast-moving fields it is where the
        result appears first -- but it has not been refereed. A registered
        clinical trial ranks highest because it carries a pre-committed protocol.
        """
        return {
            "preprint": 0.45,
            "report": 0.5,
            "patent": 0.6,
            "peer_reviewed": 0.85,
            "clinical_registry": 0.95,
        }[self.value]


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str | None = None
    # Set when the affiliation resolves to a company rather than a university or
    # national lab. Industry participation is one of the stronger signals that a
    # research direction has left the purely academic phase.
    is_industry: bool = False

    def group_key(self) -> str:
        """Key identifying the research *group*, for counting independent teams.

        Affiliation where we have it, falling back to surname. Imperfect -- two
        unrelated labs at one university collapse together -- but it is the
        distinction that matters for competitive density, where the question is
        how many separate organisations are pursuing a direction.
        """
        if self.affiliation:
            return self.affiliation.strip().lower()
        return self.name.strip().lower().split()[-1] if self.name.strip() else "unknown"


@dataclass
class Work:
    """A single research artefact."""

    id: str
    title: str
    abstract: str = ""
    authors: list[Author] = field(default_factory=list)
    published: dt.date | None = None
    venue: str | None = None
    maturity: Maturity = Maturity.PREPRINT
    disciplines: list[str] = field(default_factory=list)
    citations: int | None = None
    url: str | None = None
    source: str = "unknown"
    # Free-form extras an adapter wants to preserve (arXiv categories, OpenAlex
    # concept scores) without forcing every source into one schema.
    extra: dict = field(default_factory=dict)

    def age_years(self, asof: dt.date | None = None) -> float | None:
        """Age in years, or None when the publication date is unknown.

        Returns 0.0 rather than a negative age for works dated in the future,
        which occurs routinely: journals stamp issues ahead of print.
        """
        if self.published is None:
            return None
        asof = asof or dt.date.today()
        return max(0.0, (asof - self.published).days / 365.25)

    def text(self) -> str:
        """Title and abstract together, for keyword signal extraction."""
        return f"{self.title}\n{self.abstract}"

    def industry_fraction(self) -> float:
        """Share of authors with a company affiliation, in ``[0, 1]``."""
        if not self.authors:
            return 0.0
        return sum(1 for a in self.authors if a.is_industry) / len(self.authors)


@dataclass
class Track:
    """A research direction: a cluster of works plus the groups pursuing it.

    This is the unit of analysis. Importance, urgency, technology readiness,
    commercial potential and investment cases all attach here.
    """

    id: str
    name: str
    description: str = ""
    works: list[Work] = field(default_factory=list)
    # Sector key used to select commercialisation priors (see priors/sectors.yaml).
    sector: str = "generic"
    # Optional analyst-supplied TRL, which overrides textual inference when set.
    # Inference from abstracts is a weak signal; a human who knows the field is
    # a much better one, so we let them say so.
    trl_override: int | None = None

    def add(self, work: Work) -> None:
        self.works.append(work)

    def dated_works(self) -> list[Work]:
        return [w for w in self.works if w.published is not None]

    def works_per_year(self) -> dict[int, int]:
        """Publication counts by calendar year."""
        counts: dict[int, int] = {}
        for work in self.dated_works():
            year = work.published.year  # type: ignore[union-attr]
            counts[year] = counts.get(year, 0) + 1
        return counts

    def recent_works(self, months: int = 12, asof: dt.date | None = None) -> list[Work]:
        asof = asof or dt.date.today()
        cutoff = asof - dt.timedelta(days=int(months * 30.44))
        return [w for w in self.dated_works() if w.published >= cutoff]  # type: ignore[operator]

    def independent_groups(self, months: int | None = 12, asof: dt.date | None = None) -> int:
        """Count of distinct research groups active in the window.

        ``months=None`` counts over the whole track history.
        """
        works = self.works if months is None else self.recent_works(months, asof)
        return len({a.group_key() for w in works for a in w.authors})

    def disciplines(self) -> set[str]:
        return {d for w in self.works for d in w.disciplines}

    def total_citations(self) -> int:
        return sum(w.citations or 0 for w in self.works)

    def growth_rate(self, min_years: int = 3) -> float | None:
        """Compound annual growth rate of publication volume.

        Fits ``log(count) = a + b*year`` by least squares and returns
        ``exp(b) - 1``. Returns None when there are too few years to fit, which
        is the honest answer for a track with two data points -- a slope through
        two years is not a trend.
        """
        counts = self.works_per_year()
        if len(counts) < min_years:
            return None
        years = sorted(counts)
        # +1 keeps a zero-count year finite; it biases the fit slightly downward
        # for small counts, which is the conservative direction for a growth claim.
        xs = [float(y) for y in years]
        ys = [math.log(counts[y] + 1) for y in years]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        # Clamp before exp() so a pathological fit cannot overflow.
        return math.exp(max(-5.0, min(5.0, slope))) - 1.0

    def industry_fraction(self) -> float:
        """Share of works with at least one industry-affiliated author."""
        if not self.works:
            return 0.0
        return sum(1 for w in self.works if w.industry_fraction() > 0) / len(self.works)

    def matches(self, patterns: list[str]) -> int:
        """Count works whose text matches any of ``patterns`` (regex, case-insensitive)."""
        if not patterns:
            return 0
        combined = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
        return sum(1 for w in self.works if combined.search(w.text()))
