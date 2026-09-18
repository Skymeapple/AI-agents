"""Named scenarios: comparing investment cases under different assumptions.

A single simulation answers "what does this look like?". The question that
actually drives a decision is "what would have to be true for this to work, and
how far off are we?" -- and that needs several cases run side by side against
the same science.

A scenario file supplies the assumptions the platform cannot derive from a
research corpus: market size, deal terms, and any stage priors you know better
than the shipped defaults. Every value must carry a ``source`` block. That is
enforced rather than encouraged: a config loader that quietly accepted bare
numbers would undo the provenance discipline the rest of the system is built on,
because it is exactly where an unfounded figure would enter.

Example::

    track: solid-state-sodium
    scenarios:
      - name: Base case
        market:
          tam_usd:
            median: 12.0e9
            spread: 3.0
            source:
              grade: sourced
              detail: global grid storage market, mid estimate
              reference: <your market study>
        deal:
          check_usd: 5.0e6
          entry_ownership: 0.15
        stages:
          "Pilot and scale-up":
            cost_usd:
              low: 30.0e6
              mode: 60.0e6
              high: 150.0e6
              source:
                grade: observed
                detail: from three comparable pilot lines in our portfolio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .commercial.profile import CommercialProfile, Stage
from .invest.dist import Distribution, LogNormal, from_three_point
from .invest.engine import MonteCarlo, MonteCarloResult
from .invest.model import DealTerms, MarketModel
from .provenance import Grade, parse_provenance

# Which distribution kind each market field expects, so a probability is not
# silently built as an unbounded PERT.
_MARKET_FIELDS: dict[str, str] = {
    "tam_usd": "lognormal",
    "peak_share": "probability",
    "gross_margin": "probability",
    "ramp_years": "pert",
    "exit_revenue_multiple": "pert",
    "years_to_exit": "pert",
}

_STAGE_FIELDS: dict[str, str] = {
    "duration_years": "pert",
    "cost_usd": "pert",
    "p_success": "probability",
}


def _build_distribution(spec: dict, context: str, kind: str) -> Distribution:
    """Build a distribution from a config mapping, provenance required."""
    if not isinstance(spec, dict):
        raise ValueError(
            f"{context}: expected a mapping with a 'source' block, got {type(spec).__name__}"
        )
    provenance = parse_provenance(spec.get("source"), context)
    if kind == "lognormal":
        missing = {"median", "spread"} - set(spec)
        if missing:
            raise ValueError(
                f"{context}: a lognormal needs 'median' and 'spread' "
                f"(p95/median ratio); missing {sorted(missing)}"
            )
        return LogNormal(
            float(spec["median"]), float(spec["spread"]), provenance, context
        )
    return from_three_point(
        {k: spec[k] for k in ("low", "mode", "high") if k in spec},
        provenance,
        context,
        kind=kind,
    )


@dataclass
class Scenario:
    """One named set of assumptions."""

    name: str
    description: str = ""
    market: MarketModel | None = None
    deal: DealTerms | None = None
    # Stage prior overrides, keyed by stage name then field.
    stage_overrides: dict[str, dict[str, Distribution]] = field(default_factory=dict)

    def market_model(self) -> MarketModel:
        return self.market or MarketModel.placeholder()

    def deal_terms(self) -> DealTerms:
        return self.deal or DealTerms.placeholder()

    def apply_to(self, profile: CommercialProfile) -> CommercialProfile:
        """Return ``profile`` with this scenario's stage overrides applied.

        Overrides name a stage that must exist and still be ahead of the
        programme. A name that matches nothing is an error rather than a
        silent no-op: a typo in a stage name would otherwise leave you
        believing you had replaced a prior when you had not.
        """
        if not self.stage_overrides:
            return profile

        available = {stage.name for stage in profile.remaining_stages}
        unknown = set(self.stage_overrides) - available
        if unknown:
            raise ValueError(
                f"scenario {self.name!r} overrides stage(s) "
                f"{sorted(unknown)} which are not in this track's remaining "
                f"path; remaining stages are {sorted(available)}"
            )

        patched: list[Stage] = []
        for stage in profile.remaining_stages:
            overrides = self.stage_overrides.get(stage.name)
            if not overrides:
                patched.append(stage)
                continue
            patched.append(
                Stage(
                    name=stage.name,
                    from_trl=stage.from_trl,
                    to_trl=stage.to_trl,
                    duration_years=overrides.get("duration_years", stage.duration_years),
                    cost_usd=overrides.get("cost_usd", stage.cost_usd),
                    p_success=overrides.get("p_success", stage.p_success),
                    provenance=stage.provenance,
                )
            )

        return CommercialProfile(
            track_id=profile.track_id,
            track_name=profile.track_name,
            sector=profile.sector,
            sector_label=profile.sector_label,
            trl=profile.trl,
            remaining_stages=patched,
            constraints=profile.constraints,
            ledger=profile.ledger,
        )


@dataclass
class ScenarioOutcome:
    """A scenario paired with what it produced."""

    scenario: Scenario
    result: MonteCarloResult

    @property
    def assumption_share(self) -> float:
        return self.result.variance_by_grade()[Grade.ASSUMED]


@dataclass
class ScenarioSet:
    """A track's scenarios, loaded from a file."""

    track_id: str
    scenarios: list[Scenario]
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ScenarioSet":
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or "scenarios" not in data:
            raise ValueError(
                f"{path}: expected a mapping with a 'scenarios' list at the top level"
            )
        track_id = str(data.get("track", "")).strip()
        if not track_id:
            raise ValueError(f"{path}: needs a 'track' naming the direction to model")

        scenarios: list[Scenario] = []
        seen: set[str] = set()
        for index, raw in enumerate(data["scenarios"], start=1):
            name = str(raw.get("name", f"scenario {index}")).strip()
            if name in seen:
                raise ValueError(f"{path}: duplicate scenario name {name!r}")
            seen.add(name)
            scenarios.append(cls._parse_scenario(raw, name, path))
        if not scenarios:
            raise ValueError(f"{path}: defines no scenarios")
        return cls(track_id=track_id, scenarios=scenarios, source_path=path)

    @staticmethod
    def _parse_scenario(raw: dict, name: str, path: Path) -> Scenario:
        market = None
        if "market" in raw:
            base = MarketModel.placeholder()
            for field_name, spec in raw["market"].items():
                if field_name not in _MARKET_FIELDS:
                    raise ValueError(
                        f"{path}: scenario {name!r} sets unknown market field "
                        f"{field_name!r}; known fields are "
                        f"{', '.join(sorted(_MARKET_FIELDS))}"
                    )
                setattr(
                    base,
                    field_name,
                    _build_distribution(
                        spec, f"{name}/{field_name}", _MARKET_FIELDS[field_name]
                    ),
                )
            market = base

        deal = None
        if "deal" in raw:
            spec = raw["deal"]
            base = DealTerms.placeholder(
                check_usd=float(spec.get("check_usd", 5.0e6)),
                entry_ownership=float(spec.get("entry_ownership", 0.15)),
            )
            for field_name in ("valuation_step_up", "discount_rate"):
                if field_name in spec:
                    setattr(
                        base,
                        field_name,
                        _build_distribution(
                            spec[field_name], f"{name}/{field_name}", "pert"
                        ),
                    )
            deal = base

        stage_overrides: dict[str, dict[str, Distribution]] = {}
        for stage_name, fields in (raw.get("stages") or {}).items():
            built: dict[str, Distribution] = {}
            for field_name, spec in fields.items():
                if field_name not in _STAGE_FIELDS:
                    raise ValueError(
                        f"{path}: scenario {name!r} sets unknown stage field "
                        f"{field_name!r}; known fields are "
                        f"{', '.join(sorted(_STAGE_FIELDS))}"
                    )
                built[field_name] = _build_distribution(
                    spec,
                    f"{name}/{stage_name}/{field_name}",
                    _STAGE_FIELDS[field_name],
                )
            stage_overrides[str(stage_name)] = built

        return Scenario(
            name=name,
            description=" ".join(str(raw.get("description", "")).split()),
            market=market,
            deal=deal,
            stage_overrides=stage_overrides,
        )

    def run(
        self, profile: CommercialProfile, draws: int = 20_000, seed: int = 20260918
    ) -> list[ScenarioOutcome]:
        """Run every scenario against the same science.

        All scenarios share one seed so that differences between them come from
        the assumptions rather than from sampling noise. Comparing scenarios run
        on different seeds would mix the two, and with heavy-tailed market
        distributions the noise can be large enough to reverse a ranking.
        """
        engine = MonteCarlo(draws=draws, seed=seed)
        return [
            ScenarioOutcome(
                scenario,
                engine.run(
                    scenario.apply_to(profile),
                    scenario.market_model(),
                    scenario.deal_terms(),
                ),
            )
            for scenario in self.scenarios
        ]


def render_comparison(outcomes: list[ScenarioOutcome], track_name: str) -> str:
    """Markdown table comparing scenarios side by side."""
    if not outcomes:
        return "_No scenarios to compare._"

    lines = [
        f"# Scenario comparison: {track_name}",
        "",
        f"_{outcomes[0].result.draws:,} draws per scenario, shared seed "
        f"{outcomes[0].result.seed} so differences reflect assumptions rather "
        "than sampling noise._",
        "",
        "| Scenario | P(market) | NPV P10 | NPV P50 | NPV P90 | E[MOIC] | MOIC if it works | P(loss) | Assumption-driven |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def money(value: float) -> str:
        magnitude = abs(value)
        sign = "-" if value < 0 else ""
        if magnitude >= 1e9:
            return f"{sign}${magnitude / 1e9:,.2f}B"
        return f"{sign}${magnitude / 1e6:,.0f}M"

    for outcome in outcomes:
        result = outcome.result
        npv = result.percentiles(result.project_npv)
        lines.append(
            f"| {outcome.scenario.name} | {result.success_rate:.0%} "
            f"| {money(npv[10])} | {money(npv[50])} | {money(npv[90])} "
            f"| {result.expected_moic():.2f}x | {result.conditional_moic():.2f}x "
            f"| {result.probability_of_loss():.0%} "
            f"| {outcome.assumption_share:.0%} |"
        )
    lines.append("")

    described = [o for o in outcomes if o.scenario.description]
    if described:
        for outcome in described:
            lines.append(f"- **{outcome.scenario.name}** — {outcome.scenario.description}")
        lines.append("")

    # The spread across scenarios is usually the most decision-relevant number
    # on the page: it says how much the answer depends on what you assumed.
    moics = [o.result.expected_moic() for o in outcomes]
    best = max(outcomes, key=lambda o: o.result.expected_moic())
    worst = min(outcomes, key=lambda o: o.result.expected_moic())
    lines.append(
        f"Expected multiple ranges from **{min(moics):.2f}x** ({worst.scenario.name}) "
        f"to **{max(moics):.2f}x** ({best.scenario.name}). "
        "The gap between scenarios is the part of the answer that comes from your "
        "assumptions rather than from the science, so it is the first thing to "
        "narrow if the decision is close."
    )
    lines.append("")
    return "\n".join(lines)
