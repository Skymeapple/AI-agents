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

A note on stage overlap: stages are sequential unless a stage declares an
``overlap`` -- the fraction of its own duration that runs before the previous
stage's gate. Real programmes do this (long-lead manufacturing work starts
during trials), and it is a trade, not a free lunch: it shortens the path and
puts the early part of the next stage's budget at risk if the gate fails. The
shipped priors declare no overlap, because an invented overlap figure would be
one more unfounded number; supply one in a scenario where you know it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..invest.dist import Distribution, from_three_point
from ..models import Track
from ..provenance import Grade, Provenance, ProvenanceLedger, parse_provenance
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
    # Fraction of this stage's duration that runs concurrently with the previous
    # stage, i.e. before that stage's gate is known. None means strictly
    # sequential. See the engine for how the at-risk spend is charged.
    overlap: Distribution | None = None

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
    # The sector's whole development path, including stages the modal TRL has
    # already cleared. The engine needs it to propagate readiness uncertainty:
    # a draw that starts at a lower TRL than the mode faces stages that
    # ``remaining_stages`` leaves out. Empty means "no path supplied", and the
    # engine then treats ``remaining_stages`` as certain.
    stage_path: list[Stage] = field(default_factory=list)

    def possible_stages(self) -> list[Stage]:
        """Stages some TRL the belief gives weight to would still have ahead."""
        if not self.stage_path:
            return list(self.remaining_stages)
        levels = [t for t, p in self.trl.probabilities.items() if p > 0]
        return [s for s in self.stage_path if any(s.applies_from(t) for t in levels)]

    def stage_reach_probability(self, stage: Stage) -> float:
        """Probability, under the TRL belief, that ``stage`` is still ahead."""
        if not self.stage_path:
            return 1.0 if stage in self.remaining_stages else 0.0
        return sum(
            p for t, p in self.trl.probabilities.items() if stage.applies_from(t)
        )

    def expected_success_probability(self) -> float:
        """Chance of reaching market, averaged over the TRL belief.

        :meth:`cumulative_probability` answers for the modal TRL only. This is
        the figure the simulation's success rate converges to.
        """
        if not self.stage_path:
            return self.cumulative_probability()
        total = 0.0
        for level, weight in self.trl.probabilities.items():
            chance = 1.0
            for stage in self.stage_path:
                if stage.applies_from(level):
                    chance *= stage.p_success.mean()
            total += weight * chance
        return total

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
        return parse_provenance(raw, context)

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
                    overlap=(
                        from_three_point(
                            raw["overlap"],
                            provenance,
                            f"{raw['name']} overlap with previous stage",
                            kind="probability",
                        )
                        if "overlap" in raw
                        else None
                    ),
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

        ``remaining_stages`` is selected against the TRL *mode* and is what the
        nominal figures and the report's stage table describe. ``stage_path``
        carries the sector's whole path alongside the full TRL distribution, and
        the Monte Carlo engine draws a starting TRL per draw and selects stages
        from the path -- so a track that might be at TRL 4 or might be at TRL 7
        produces a genuinely bimodal cost distribution rather than an averaged
        one that describes no possible world.
        """
        sector = track.sector if track.sector in self._data["sectors"] else "generic"
        belief = trl or infer_trl(track)
        path = self.stages(sector)
        stages = [s for s in path if s.applies_from(belief.mode)]

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
            stage_path=path,
        )
