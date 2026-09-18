"""End-to-end pipeline: corpus in, ranked investment cases out.

Ties the stages together: harvest works, cluster them into tracks, score
importance and urgency, infer readiness, build a commercialisation profile, and
run the investment simulation.

Clustering is the weakest link and is treated as such. Grouping works into
research directions properly needs embeddings or citation-graph community
detection; what ships here is grouping by an explicit label, which means a human
decides what counts as a direction. That is a limitation, and also a defensible
default: the boundary of a "research direction" is a judgement call that a
clustering algorithm makes silently and usually badly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .commercial.profile import CommercialProfile, PriorLibrary
from .invest.engine import MonteCarlo, MonteCarloResult
from .invest.model import DealTerms, MarketModel
from .models import Track, Work
from .scoring.baseline import CitationBaseline
from .scoring.importance import Assessment, Scorer, Weights


@dataclass
class TrackReport:
    """Everything the platform knows about one research direction."""

    track: Track
    assessment: Assessment
    profile: CommercialProfile
    simulation: MonteCarloResult | None = None

    @property
    def name(self) -> str:
        return self.track.name


@dataclass
class Pipeline:
    """Runs the full analysis over a corpus."""

    priors: PriorLibrary = field(default_factory=PriorLibrary)
    weights: Weights = field(default_factory=Weights)
    asof: dt.date = field(default_factory=dt.date.today)
    draws: int = 20_000
    seed: int = 20260918

    def tracks_from_works(
        self, works: list[Work], label_key: str = "track"
    ) -> list[Track]:
        """Group works into tracks by a label carried in ``Work.extra``.

        Works without the label are collected into a single ``unclassified``
        track rather than dropped, so nothing disappears silently -- an
        unclassified pile that keeps growing is itself a signal that the label
        scheme needs work.
        """
        grouped: dict[str, list[Work]] = {}
        for work in works:
            key = str(work.extra.get(label_key, "unclassified"))
            grouped.setdefault(key, []).append(work)

        tracks: list[Track] = []
        for key, members in sorted(grouped.items()):
            first = members[0]
            tracks.append(
                Track(
                    id=key,
                    name=str(first.extra.get("track_name", key.replace("-", " ").title())),
                    sector=str(first.extra.get("sector", "generic")),
                    works=members,
                )
            )
        return tracks

    def analyse(
        self,
        tracks: list[Track],
        markets: dict[str, MarketModel] | None = None,
        terms: dict[str, DealTerms] | None = None,
        simulate: bool = True,
    ) -> list[TrackReport]:
        """Score, profile and simulate every track, ranked by priority.

        ``markets`` and ``terms`` are keyed by track id. Anything missing falls
        back to the clearly-labelled placeholders, whose outputs are marked as
        resting entirely on assumption.
        """
        markets = markets or {}
        terms = terms or {}

        # The citation baseline is learned across the whole corpus, so tracks are
        # normalised against each other rather than against an external table.
        all_works = [w for t in tracks for w in t.works]
        baseline = CitationBaseline.from_works(all_works, asof=self.asof)
        scorer = Scorer(baseline, self.weights, asof=self.asof)
        engine = MonteCarlo(draws=self.draws, seed=self.seed)

        reports: list[TrackReport] = []
        for track in tracks:
            assessment = scorer.score(track)
            profile = self.priors.profile(track)
            simulation = None
            if simulate and profile.remaining_stages:
                simulation = engine.run(
                    profile,
                    markets.get(track.id, MarketModel.placeholder()),
                    terms.get(track.id, DealTerms.placeholder()),
                )
            reports.append(TrackReport(track, assessment, profile, simulation))

        reports.sort(key=lambda r: r.assessment.priority, reverse=True)
        return reports
