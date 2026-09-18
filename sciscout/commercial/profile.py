"""From technology readiness to a commercialisation profile.

Given where a research direction currently sits (a TRL distribution) and what
kind of technology it is (a sector), this module assembles the remaining path to
market: which development stages are still ahead, how long each takes, what each
costs, how likely each is to be survived, and which structural constraints apply.

The output is not a forecast. It is a *structured set of distributions* that the
Monte Carlo engine turns into a forecast. Keeping the two separate matters: the
profile is auditable on its own, and two analysts who disagree can point at the
specific stage assumption they disagree about rather than arguing about a
headline ROI number.

A note on stage overlap: stages are modelled as sequential and non-overlapping.
Real programmes overlap them (long-lead manufacturing work starts during
trials), which makes this a conservative bias on timelines. It is a simplifying
assumption, recorded here rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..invest.dist import Distribution, from_three_point
from ..models import Track
from ..provenance import Grade, Provenance, ProvenanceLedger
from ..scoring.trl import TRLBelief, infer_trl

PRIORS_PATH = Path(__file__).resolve().parent.parent / "priors" / "sectors.yaml"


@dataclass(frozen=True)
class Constraint:
    """A structural obstacle between the science and the revenue."""

    kind: str
    severity: str
    note: str

    @property
    def severity_rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(self.severity, 2)


@dataclass
class Stage:
    """One development stage, spanning a TRL band."""

    name: str
    from_trl: int
    to_trl: int
    duration_years: Distribution
    cost_usd: Distribution
    p_success: Distribution
    provenance: Provenance

    def applies_from(self, trl: int) -> bool:
        """Whether a programme at ``trl`` still has this stage ahead of it.

        A programme already at or past ``to_trl`` has cleared the stage. One
        partway through the band still faces it -- we do not pro-rate, which
        slightly overstates remaining time and cost. Conservative, and stated.
        """
        return trl < self.to_trl


@dataclass
class CommercialProfile:
    """Everything needed to build an investment case for one track."""

    track_id: str
    track_name: str
    sector: str
    sector_label: str
    trl: TRLBelief
    remaining_stages: list[Stage]
    constraints: list[Constraint]
    ledger: ProvenanceLedger = field(default_factory=ProvenanceLedger)

    def nominal_years_to_market(self) -> float:
        """Sum of stage duration means. Ignores attrition -- the success case only."""
        return sum(s.duration_years.mean() for s in self.remaining_stages)

    def nominal_cost_usd(self) -> float:
        """Sum of stage cost means, again assuming every stage is survived."""
        return sum(s.cost_usd.mean() for s in self.remaining_stages)

    def cumulative_probability(self) -> float:
        """Product of stage success means: the chance of reaching market at all."""
        result = 1.0
        for stage in self.remaining_stages:
            result *= stage.p_success.mean()
        return result

    def binding_constraints(self) -> list[Constraint]:
        """Constraints rated high severity, worst first."""
        return sorted(
            (c for c in self.constraints if c.severity == "high"),
            key=lambda c: c.kind,
        )

    def summary(self) -> str:
        return (
            f"{self.track_name} [{self.sector_label}]: {self.trl.describe()}; "
            f"{len(self.remaining_stages)} stages remaining, nominally "
            f"{self.nominal_years_to_market():.1f} years and "
            f"${self.nominal_cost_usd() / 1e6:,.0f}M to market, "
            f"cumulative technical success {self.cumulative_probability():.1%}"
        )


class PriorLibrary:
    """Loads and serves sector commercialisation priors."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PRIORS_PATH
        with open(self.path) as handle:
            self._data = yaml.safe_load(handle)
        if not self._data.get("sectors"):
            raise ValueError(f"{self.path} defines no sectors")

    def sectors(self) -> list[str]:
        return sorted(self._data["sectors"])

    def _provenance(self, raw: dict, context: str) -> Provenance:
        grade = Grade(raw.get("grade", "assumed"))
        detail = " ".join(raw.get("detail", "").split()) or f"prior for {context}"
        return Provenance(
            grade=grade,
            detail=f"{context}: {detail}",
            reference=raw.get("reference"),
            needs_review=bool(raw.get("needs_review", grade is Grade.ASSUMED)),
        )

    def stages(self, sector: str) -> list[Stage]:
        entry = self._data["sectors"].get(sector)
        if entry is None:
            raise KeyError(
                f"unknown sector {sector!r}; known sectors: {', '.join(self.sectors())}"
            )
        stages: list[Stage] = []
        for raw in entry["stages"]:
            provenance = self._provenance(raw.get("provenance", {}), raw["name"])
            stages.append(
                Stage(
                    name=raw["name"],
                    from_trl=int(raw["from_trl"]),
                    to_trl=int(raw["to_trl"]),
                    duration_years=from_three_point(
                        raw["duration_years"], provenance, f"{raw['name']} duration (y)"
                    ),
                    cost_usd=from_three_point(
                        raw["cost_usd"], provenance, f"{raw['name']} cost (USD)"
                    ),
                    p_success=from_three_point(
                        raw["p_success"],
                        provenance,
                        f"{raw['name']} P(technical success)",
                        kind="probability",
                    ),
                    provenance=provenance,
                )
            )
        return sorted(stages, key=lambda s: s.from_trl)

    def constraints(self, sector: str) -> list[Constraint]:
        entry = self._data["sectors"].get(sector, {})
        return [
            Constraint(kind=c["kind"], severity=c["severity"], note=c["note"])
            for c in entry.get("constraints", [])
        ]

    def profile(self, track: Track, trl: TRLBelief | None = None) -> CommercialProfile:
        """Build the commercialisation profile for ``track``.

        Remaining stages are selected against the TRL *mode*. The full TRL
        distribution still reaches the Monte Carlo engine, which re-selects
        stages per draw -- so a track that might be at TRL 4 or might be at TRL 7
        produces a genuinely bimodal cost distribution rather than an averaged
        one that describes no possible world.
        """
        sector = track.sector if track.sector in self._data["sectors"] else "generic"
        belief = trl or infer_trl(track)
        stages = [s for s in self.stages(sector) if s.applies_from(belief.mode)]

        ledger = ProvenanceLedger()
        ledger.record(belief.as_estimate())
        for stage in stages:
            ledger.record(stage.duration_years.as_estimate())
            ledger.record(stage.cost_usd.as_estimate())
            ledger.record(stage.p_success.as_estimate())

        return CommercialProfile(
            track_id=track.id,
            track_name=track.name,
            sector=sector,
            sector_label=self._data["sectors"][sector].get("label", sector),
            trl=belief,
            remaining_stages=stages,
            constraints=self.constraints(sector),
            ledger=ledger,
        )
