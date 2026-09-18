"""Importance and urgency scoring for research tracks.

Importance and urgency are kept as separate axes because they answer different
questions and routinely disagree. A foundational result in a slow-moving field
is important and not urgent. A crowded race toward a modest prize is urgent and
not important. Collapsing them into one number destroys exactly the distinction
a prioritisation decision turns on.

Each axis is a weighted mean of sub-scores in ``[0, 1]``. Every sub-score reports
its own provenance, so a high importance score built entirely from assumptions is
visibly different from one built from citation data.

Priority combines the two geometrically::

    priority = importance ** w_i * urgency ** w_u

Geometric rather than arithmetic so that a zero on either axis cannot be
compensated by the other: something with no importance is not worth doing
quickly, and a weighted sum would let a high urgency hide that.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from ..models import Track
from ..provenance import Estimate, Grade, ProvenanceLedger, assumed, observed
from .baseline import CitationBaseline


def _logistic(x: float, midpoint: float, steepness: float) -> float:
    """Map an unbounded quantity to ``[0, 1]``, crossing 0.5 at ``midpoint``."""
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


def _saturating(x: float, scale: float) -> float:
    """Map a non-negative count to ``[0, 1]`` with diminishing returns."""
    return 1.0 - math.exp(-max(0.0, x) / scale)


@dataclass
class Weights:
    """Tunable weights. Exposed so a user can argue with the model, not just read it."""

    # Importance axis.
    impact: float = 0.30
    momentum: float = 0.25
    breadth: float = 0.15
    evidence: float = 0.20
    foundationality: float = 0.10
    # Urgency axis.
    competition: float = 0.45
    acceleration: float = 0.35
    industry_entry: float = 0.20
    # Priority combination exponents.
    importance_exponent: float = 0.6
    urgency_exponent: float = 0.4

    def importance_weights(self) -> dict[str, float]:
        return {
            "impact": self.impact,
            "momentum": self.momentum,
            "breadth": self.breadth,
            "evidence": self.evidence,
            "foundationality": self.foundationality,
        }

    def urgency_weights(self) -> dict[str, float]:
        return {
            "competition": self.competition,
            "acceleration": self.acceleration,
            "industry_entry": self.industry_entry,
        }


# Works describing a method, platform or tool tend to enable downstream work
# rather than being an endpoint, which is what "foundational" means for our
# purposes. This is a crude lexical proxy for a property that really needs
# citation-graph analysis; it is marked as such in the provenance.
_FOUNDATIONAL_PATTERNS = [
    r"\bframework\b", r"\bplatform\b", r"\bgeneral(?:is|iz)ed?\b", r"\bmethod\b",
    r"\btoolkit\b", r"\barchitecture\b", r"\bprotocol\b", r"\bscalable\b",
    r"\bfoundation model\b", r"\benabl(?:es|ing)\b", r"\bfirst demonstration\b",
]


@dataclass
class Assessment:
    """Scored result for one track."""

    track_id: str
    track_name: str
    importance: float
    urgency: float
    priority: float
    importance_parts: dict[str, Estimate[float]] = field(default_factory=dict)
    urgency_parts: dict[str, Estimate[float]] = field(default_factory=dict)
    ledger: ProvenanceLedger = field(default_factory=ProvenanceLedger)

    @property
    def confidence(self) -> float:
        """How much of this score rests on evidence rather than assumption.

        ``1 - assumption_share``. A priority of 0.8 at confidence 0.3 and one at
        confidence 0.9 are not the same claim, and ranking on priority alone
        hides that.
        """
        return 1.0 - self.ledger.assumption_share()

    def summary(self) -> str:
        return (
            f"{self.track_name}: priority {self.priority:.2f} "
            f"(importance {self.importance:.2f}, urgency {self.urgency:.2f}, "
            f"confidence {self.confidence:.2f})"
        )


class Scorer:
    """Scores tracks for importance, urgency and priority."""

    def __init__(
        self,
        baseline: CitationBaseline,
        weights: Weights | None = None,
        asof: dt.date | None = None,
    ) -> None:
        self.baseline = baseline
        self.weights = weights or Weights()
        self.asof = asof or dt.date.today()

    # -- importance sub-scores ------------------------------------------------

    def _impact(self, track: Track) -> Estimate[float]:
        """Field-normalised citation impact, aggregated over the track's works.

        Uses the 75th percentile of per-work normalised impact rather than the
        mean. Citation distributions are heavy-tailed, and what makes a direction
        important is usually that its *best* results landed, not that its median
        one did -- but a raw max would let one fluke paper carry a whole track.
        """
        ratios = [
            e.value for w in track.works if (e := self.baseline.normalised_impact(w))
        ]
        if not ratios:
            return Estimate(
                0.0,
                assumed(
                    "no work in this track carries both a publication date and a "
                    "citation count, so impact could not be measured; scored 0 "
                    "rather than imputed, which will understate a genuinely "
                    "high-impact track with missing metadata"
                ),
                label="impact",
            )
        ratios.sort()
        idx = min(len(ratios) - 1, int(0.75 * len(ratios)))
        p75 = ratios[idx]
        # log1p compresses the heavy tail; /log1p(10) puts "10x expected" near 1.0.
        score = min(1.0, math.log1p(max(0.0, p75)) / math.log1p(10.0))
        return Estimate(
            score,
            observed(
                f"75th-percentile normalised citation impact {p75:.2f}x expected "
                f"across {len(ratios)} works with citation data"
            ),
            label="impact",
        )

    def _momentum(self, track: Track) -> Estimate[float]:
        """Growth in publication volume. A direction heating up scores higher."""
        cagr = track.growth_rate()
        if cagr is None:
            return Estimate(
                0.5,
                assumed(
                    f"only {len(track.works_per_year())} distinct publication years "
                    "in this track, too few to fit a trend; scored neutral"
                ),
                label="momentum",
            )
        # Crosses 0.5 at 25% annual growth: brisk but not remarkable for an
        # active field. Steepness 3 keeps 0% growth near 0.3 and 100% near 0.9.
        score = _logistic(cagr, midpoint=0.25, steepness=3.0)
        return Estimate(
            score,
            observed(
                f"publication volume CAGR {cagr:+.0%} fitted over "
                f"{len(track.works_per_year())} years"
            ),
            label="momentum",
        )

    def _breadth(self, track: Track) -> Estimate[float]:
        """Cross-disciplinary reach. Directions that spread tend to matter more."""
        disciplines = track.disciplines()
        if not disciplines:
            return Estimate(
                0.3,
                assumed(
                    "no discipline labels on any work in this track; scored below "
                    "neutral since breadth is unevidenced rather than absent"
                ),
                label="breadth",
            )
        score = _saturating(len(disciplines), scale=3.0)
        return Estimate(
            score,
            observed(
                f"{len(disciplines)} distinct disciplines represented: "
                + ", ".join(sorted(disciplines)[:6])
            ),
            label="breadth",
        )

    def _evidence(self, track: Track) -> Estimate[float]:
        """How vetted the underlying results are, and by how many separate groups.

        Two components multiplied: the mean publication-status weight, and
        whether more than one group has produced results. A striking claim from
        a single group is weaker evidence than a modest one replicated widely,
        and this is the sub-score that expresses that.
        """
        if not track.works:
            return Estimate(
                0.0, assumed("track has no works"), label="evidence"
            )
        mean_weight = sum(w.maturity.evidence_weight for w in track.works) / len(
            track.works
        )
        groups = track.independent_groups(months=None)
        # Independent replication: saturates around 4 groups.
        replication = _saturating(max(0, groups - 1), scale=2.0)
        # Floor of 0.4 on the replication factor so a genuinely new result from
        # one strong group is not scored as near-worthless.
        score = mean_weight * (0.4 + 0.6 * replication)
        return Estimate(
            score,
            observed(
                f"mean publication-status weight {mean_weight:.2f} across "
                f"{len(track.works)} works, from {groups} independent groups"
            ),
            label="evidence",
        )

    def _foundationality(self, track: Track) -> Estimate[float]:
        """Whether the direction enables downstream work or is an endpoint."""
        if not track.works:
            return Estimate(0.0, assumed("track has no works"), label="foundationality")
        hits = track.matches(_FOUNDATIONAL_PATTERNS)
        fraction = hits / len(track.works)
        return Estimate(
            min(1.0, fraction * 1.5),
            assumed(
                f"{hits} of {len(track.works)} works use platform/method/enabling "
                "language. This is a lexical proxy: properly, foundationality is "
                "how much downstream work builds on a result, which needs "
                "citation-graph data this corpus does not carry",
                needs_review=True,
            ),
            label="foundationality",
        )

    # -- urgency sub-scores ---------------------------------------------------

    def _competition(self, track: Track) -> Estimate[float]:
        """How many separate groups are actively pursuing this right now.

        Crowding cuts both ways and the model takes a position: high competitive
        density raises urgency (the window is closing) without raising
        importance. An investor arriving late to a crowded race faces worse
        entry terms for the same science.
        """
        groups = track.independent_groups(months=12, asof=self.asof)
        if groups == 0:
            return Estimate(
                0.1,
                observed(
                    "no groups published in this track in the last 12 months, "
                    "which is either a dormant direction or a stale corpus"
                ),
                label="competition",
            )
        score = _saturating(groups, scale=4.0)
        return Estimate(
            score,
            observed(f"{groups} independent groups published in the last 12 months"),
            label="competition",
        )

    def _acceleration(self, track: Track) -> Estimate[float]:
        """Whether recent activity outpaces the track's own history.

        Compares the last 12 months against the track's historical annual mean.
        A ratio above 1 means the direction is accelerating relative to itself,
        which is a sharper urgency signal than raw volume.
        """
        recent = len(track.recent_works(12, self.asof))
        per_year = track.works_per_year()
        if len(per_year) < 2:
            return Estimate(
                0.5,
                assumed("too little history to judge acceleration; scored neutral"),
                label="acceleration",
            )
        historical_mean = sum(per_year.values()) / len(per_year)
        if historical_mean == 0:
            return Estimate(
                0.5, assumed("no historical baseline"), label="acceleration"
            )
        ratio = recent / historical_mean
        score = _logistic(ratio, midpoint=1.2, steepness=2.0)
        return Estimate(
            score,
            observed(
                f"{recent} works in the last 12 months vs historical mean "
                f"{historical_mean:.1f}/year (ratio {ratio:.2f})"
            ),
            label="acceleration",
        )

    def _industry_entry(self, track: Track) -> Estimate[float]:
        """Industry participation, a strong tell that a direction is being claimed."""
        fraction = track.industry_fraction()
        if not track.works:
            return Estimate(0.0, assumed("track has no works"), label="industry_entry")
        any_affiliation = any(a.affiliation for w in track.works for a in w.authors)
        if not any_affiliation:
            return Estimate(
                0.3,
                assumed(
                    "no author affiliations in this corpus, so industry entry "
                    "could not be measured; scored below neutral",
                ),
                label="industry_entry",
            )
        return Estimate(
            min(1.0, fraction * 2.0),
            observed(
                f"{fraction:.0%} of works have at least one industry-affiliated author"
            ),
            label="industry_entry",
        )

    # -- combination ----------------------------------------------------------

    def score(self, track: Track) -> Assessment:
        """Score one track, returning sub-scores and a provenance ledger."""
        importance_parts = {
            "impact": self._impact(track),
            "momentum": self._momentum(track),
            "breadth": self._breadth(track),
            "evidence": self._evidence(track),
            "foundationality": self._foundationality(track),
        }
        urgency_parts = {
            "competition": self._competition(track),
            "acceleration": self._acceleration(track),
            "industry_entry": self._industry_entry(track),
        }

        ledger = ProvenanceLedger()
        ledger.extend(importance_parts.values())
        ledger.extend(urgency_parts.values())

        importance = self._weighted(importance_parts, self.weights.importance_weights())
        urgency = self._weighted(urgency_parts, self.weights.urgency_weights())
        priority = (
            importance ** self.weights.importance_exponent
            * urgency ** self.weights.urgency_exponent
        )

        return Assessment(
            track_id=track.id,
            track_name=track.name,
            importance=importance,
            urgency=urgency,
            priority=priority,
            importance_parts=importance_parts,
            urgency_parts=urgency_parts,
            ledger=ledger,
        )

    @staticmethod
    def _weighted(parts: dict[str, Estimate[float]], weights: dict[str, float]) -> float:
        total_weight = sum(weights.get(k, 0.0) for k in parts)
        if total_weight == 0:
            return 0.0
        return sum(parts[k].value * weights.get(k, 0.0) for k in parts) / total_weight

    def rank(self, tracks: list[Track]) -> list[Assessment]:
        """Score every track, highest priority first."""
        return sorted(
            (self.score(t) for t in tracks), key=lambda a: a.priority, reverse=True
        )
