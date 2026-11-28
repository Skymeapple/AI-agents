"""Pricing a research direction by what it would do to the world.

sciscout values a direction as an investment: expected multiple, expected NPV.
That says who gets paid, not what changes. worldsim models thirteen 2050
outcomes -- warming, poverty, resistance deaths -- and what each policy lever
does to them. This module joins the two: a direction that reaches market is
treated as a partial move of a worldsim lever, starting when sciscout says it
arrives, and the change in each 2050 outcome is reported beside the financial
return.

Three steps, and where each number comes from:

1. *Whether and when.* ``sciscout.Pipeline`` simulates the direction's staged
   development. Each draw either reaches market in some year or fails.
2. *Which lever, how far.* ``impact_map.yaml`` says which worldsim lever a
   sector's success moves and how large a share of a full push one technology
   is. This is the weak link and every value in it is graded ``assumed``.
3. *What it changes.* worldsim is run with and without that lever move on the
   same random numbers, so the per-draw difference is the direction's effect
   and nothing else.

Timing is an approximation. worldsim holds each lever constant from its base
year to its horizon and has no way to switch one on in 2038. A direction that
arrives late is therefore given the *constant lever with the same lever-years*:
full size times the share of the horizon left after arrival and half the ramp.
For an outcome whose lever enters as a plain yearly rate (emissions, the AI
gap, resistance deaths) the 2050 value depends only on total lever-years, so
this is exact. For a cumulative outcome it is not: warming depends on *when*
the emissions were cut, and spreading a late cut back to 2025 overstates the
benefit. The report marks those outcomes.

worldsim also takes one lever value for all draws, while here each draw has
its own. Rather than change worldsim, it is run on a grid of lever values with
the same seed; because every draw's inputs are then identical across the grid,
each draw's outcome is read off by interpolating along its own row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sciscout.invest.dist import Distribution, Point, from_three_point
from sciscout.invest.engine import MonteCarloResult, _rank, _stream
from sciscout.models import Track
from sciscout.pipeline import Pipeline, TrackReport
from sciscout.provenance import Grade, ProvenanceLedger, parse_provenance
from sciscout.sources.local import LocalCorpus
from worldsim.model import LEVERS, OUTCOME_BY_KEY, Calibration, Run, load_calibration, simulate

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DEFAULT_MAP = PACKAGE_DIR / "impact_map.yaml"
DEFAULT_CORPUS = ROOT / "data" / "demo_corpus.json"
DEFAULT_REPORT = ROOT / "examples" / "nexus-impact.md"

# Outcomes each lever acts on directly, primary first, and whether the
# lever-years approximation is exact for the lever's own term (see the module
# docstring). Read off the equations in worldsim/model.py; outcomes reached only
# through couplings are picked up by the "moved at all" test instead.
LEVER_OUTCOMES: dict[str, dict[str, bool]] = {
    "mitigation": {"emissions": True, "temperature": False},
    "peacebuilding": {"conflict_deaths": False, "displaced": False},
    "development": {"poverty_people": True, "hunger_people": True},
    "health": {"amr_deaths": True, "pandemic_deaths": False},
    "conservation": {"forest_share": True},
    "wash": {"water_people": True, "sanitation_people": True},
    "ai_governance": {"ai_gap": True},
    "redistribution": {"wealth_share": True, "poverty_people": True},
}
GRID_POINTS = 9
_VALUE_FIELDS = {"low", "mode", "high", "value", "kind", "unit", "source"}


# ----------------------------------------------------------------------- mapping


def _value(context: str, spec) -> Distribution:
    """A sampled or fixed quantity from the map, refused without a source."""
    if not isinstance(spec, dict):
        raise ValueError(f"{context}: expected a mapping with a 'source' block, got {type(spec).__name__}")
    source = spec.get("source")
    if isinstance(source, dict):
        # An unquoted comma inside a flow-style detail silently becomes extra
        # keys and truncates the rationale; refuse it instead.
        stray = set(source) - {"grade", "detail", "reference", "retrieved", "needs_review"}
        if stray:
            raise ValueError(f"{context}: unknown field(s) {sorted(stray)} in 'source' (quote a detail containing commas)")
    provenance = parse_provenance(source, context)
    unknown = set(spec) - _VALUE_FIELDS
    if unknown:
        raise ValueError(f"{context}: unknown field(s) {sorted(unknown)}")
    if "value" in spec:
        return Point(float(spec["value"]), provenance, context)
    return from_three_point(spec, provenance, context, kind=spec.get("kind", "pert"))


@dataclass
class Rule:
    """One line of the map: which directions it covers and what they move."""

    name: str
    sector: str
    levers: dict[str, Distribution]
    size: Distribution
    ramp_years: Distribution
    terms: list[str] = field(default_factory=list)
    min_share: Distribution | None = None

    def covers(self, track: Track) -> bool:
        if track.sector != self.sector:
            return False
        if not self.terms:
            return True
        share = track.matches(self.terms) / len(track.works) if track.works else 0.0
        return share >= self.min_share.mean()

    def distributions(self) -> list[Distribution]:
        extra = [self.min_share] if self.min_share is not None else []
        return [self.size, self.ramp_years, *self.levers.values(), *extra]


@dataclass
class ImpactMap:
    rules: list[Rule]
    unpriced: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_MAP) -> "ImpactMap":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {}, Path(path))

    @classmethod
    def from_dict(cls, raw: dict, path: Path | None = None) -> "ImpactMap":
        unknown = set(raw) - {"rules", "unpriced"}
        if unknown:
            raise ValueError(f"impact map: unknown section(s) {sorted(unknown)}")
        rules = []
        for index, spec in enumerate(raw.get("rules") or []):
            name = str(spec.get("name") or f"rule {index + 1}")
            context = f"impact map, {name}"
            extra = set(spec) - {"name", "sector", "terms", "min_share", "levers", "size", "ramp_years"}
            if extra:
                raise ValueError(f"{context}: unknown field(s) {sorted(extra)}")
            for required in ("sector", "levers", "size", "ramp_years"):
                if required not in spec:
                    raise ValueError(f"{context}: missing '{required}'")
            bad = set(spec["levers"]) - set(LEVERS)
            if bad:
                raise ValueError(f"{context}: unknown lever(s) {sorted(bad)}; expected {sorted(LEVERS)}")
            terms = [str(t) for t in spec.get("terms") or []]
            if terms and "min_share" not in spec:
                raise ValueError(f"{context}: a rule with 'terms' needs a 'min_share'")
            rules.append(
                Rule(
                    name=name,
                    sector=str(spec["sector"]),
                    levers={k: _value(f"{context}, weight on {k}", v) for k, v in spec["levers"].items()},
                    size=_value(f"{context}, lever size", spec["size"]),
                    ramp_years=_value(f"{context}, ramp years", spec["ramp_years"]),
                    terms=terms,
                    min_share=_value(f"{context}, minimum term share", spec["min_share"]) if terms else None,
                )
            )
        return cls(rules, {str(k): str(v) for k, v in (raw.get("unpriced") or {}).items()}, path)

    def rule_for(self, track: Track) -> Rule | None:
        return next((rule for rule in self.rules if rule.covers(track)), None)

    def reason_unpriced(self, track: Track) -> str:
        return self.unpriced.get(track.sector, f"no rule in the impact map covers sector '{track.sector}'")

    def ledger(self) -> ProvenanceLedger:
        ledger = ProvenanceLedger()
        for rule in self.rules:
            ledger.extend(d.as_estimate() for d in rule.distributions())
        return ledger


# ------------------------------------------------------------------------ pricing


@dataclass
class OutcomeImpact:
    """Change in one 2050 outcome, per draw, against the shared-draw baseline."""

    key: str
    delta: np.ndarray
    baseline_median: float
    exact_timing: bool
    direct: bool

    @property
    def expected(self) -> float:
        """Mean change over every draw, failures included as zero."""
        return float(self.delta.mean())

    def if_success(self, reached: np.ndarray, points=(10, 50, 90)) -> tuple[float, ...] | None:
        hit = self.delta[reached]
        return tuple(float(v) for v in np.percentile(hit, points)) if len(hit) else None

    @property
    def expected_percent(self) -> float:
        """Expected change as % of the baseline median; NaN where that median is zero."""
        return 100.0 * self.expected / self.baseline_median if self.baseline_median else float("nan")


@dataclass
class DirectionImpact:
    """A direction priced both ways, or the reason it could not be."""

    report: TrackReport
    rule: Rule | None = None
    reason: str = ""
    effective_lever: np.ndarray | None = None
    outcomes: dict[str, OutcomeImpact] = field(default_factory=dict)
    spread: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def priced(self) -> bool:
        return self.rule is not None and not self.reason

    @property
    def simulation(self) -> MonteCarloResult | None:
        return self.report.simulation

    @property
    def primary(self) -> OutcomeImpact | None:
        """The outcome the rule's first lever acts on most directly."""
        for lever in self.rule.levers if self.rule else ():
            for key in LEVER_OUTCOMES[lever]:
                if key in self.outcomes:
                    return self.outcomes[key]
        return None

    def goodness(self) -> float | None:
        """Expected change in the primary outcome as % of baseline, signed so up is better."""
        primary = self.primary
        if primary is None:
            return None
        sign = -1.0 if OUTCOME_BY_KEY[primary.key].lower_is_better else 1.0
        return sign * primary.expected_percent


def effective_lever(
    reached: np.ndarray,
    years_to_market: np.ndarray,
    size: np.ndarray,
    ramp_years: np.ndarray,
    start_year: float,
    base_year: int,
    horizon: int,
) -> np.ndarray:
    """Constant lever with the same lever-years as a late, ramped arrival.

    A draw that never reaches market gets exactly zero. One that does arrives
    at ``start_year + years_to_market`` and ramps linearly to ``size`` over
    ``ramp_years``, which costs half the ramp in lever-years.
    """
    full_from = start_year + np.where(reached, years_to_market, 0.0) + ramp_years / 2.0
    share = np.clip((horizon - full_from) / (horizon - base_year), 0.0, 1.0)
    return np.where(reached, size * share, 0.0)


def _interpolate(grid: np.ndarray, stack: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Each draw's value at its own lever level. ``stack`` is ``(grid, draws)``."""
    upper = np.clip(np.searchsorted(grid, at, side="left"), 1, len(grid) - 1)
    lower = upper - 1
    rows = np.arange(stack.shape[1])
    weight = (at - grid[lower]) / (grid[upper] - grid[lower])
    return stack[lower, rows] + weight * (stack[upper, rows] - stack[lower, rows])


def _spread_by_grade(target: np.ndarray, columns: dict[str, tuple[np.ndarray, Grade]]) -> dict[str, float]:
    """Share of the spread in ``target`` carried by each grade of input.

    Same method as ``worldsim.Run.spread_by_grade``: squared rank correlations
    summed by grade, the remainder being chance and interaction.
    """
    shares = {grade.value: 0.0 for grade in Grade}
    if len(target) >= 30 and np.ptp(target) > 0:
        ranks = _rank(target)
        for values, grade in columns.values():
            if np.ptp(values) == 0:
                continue
            shares[grade.value] += float(np.corrcoef(_rank(values), ranks)[0, 1]) ** 2
    total = sum(shares.values())
    if total > 1.0:
        shares = {k: v / total for k, v in shares.items()}
        total = 1.0
    shares["chance and interaction"] = 1.0 - total
    return shares


@dataclass
class ImpactPricer:
    """Prices sciscout track reports against worldsim outcomes."""

    calibration: Calibration
    mapping: ImpactMap
    draws: int = 4000
    seed: int = 20261128
    asof: dt.date = field(default_factory=dt.date.today)
    _baseline: Run | None = field(default=None, repr=False)

    @property
    def baseline(self) -> Run:
        if self._baseline is None:
            self._baseline = simulate(self.calibration, {}, self.draws, self.seed, name="baseline")
        return self._baseline

    def price(self, report: TrackReport) -> DirectionImpact:
        track, sim = report.track, report.simulation
        rule = self.mapping.rule_for(track)
        if rule is None:
            return DirectionImpact(report, reason=self.mapping.reason_unpriced(track))
        if sim is None:
            return DirectionImpact(report, rule, reason="sciscout produced no simulation for this direction")
        if sim.draws != self.draws:
            raise ValueError(
                f"{track.id}: sciscout ran {sim.draws} draws and worldsim {self.draws}; "
                "they are paired draw for draw and must match"
            )

        n = self.draws
        size = rule.size.sample(n, _stream(self.seed, f"{track.id}: lever size"))
        ramp = rule.ramp_years.sample(n, _stream(self.seed, f"{track.id}: ramp years"))
        start = self.asof.year + (self.asof.timetuple().tm_yday - 1) / 365.25
        level = effective_lever(
            sim.reached_market, sim.years_to_market, size, ramp,
            start, self.calibration.base_year, self.calibration.horizon,
        )
        result = DirectionImpact(report, rule, effective_lever=level)
        weights = {lever: dist.mean() for lever, dist in rule.levers.items()}
        direct = {key: exact for lever in weights for key, exact in LEVER_OUTCOMES[lever].items()}

        base = self.baseline
        top = float(level.max())
        if top > 0:
            grid = np.linspace(0.0, top, GRID_POINTS)
            runs = [base] + [
                simulate(self.calibration, {k: w * g for k, w in weights.items()}, n, self.seed, name=track.id)
                for g in grid[1:]
            ]
        columns: dict[str, tuple[np.ndarray, Grade]] = {
            f"sciscout: {label}": (values, sim.input_grades[label]) for label, values in sim.inputs.items()
        }
        columns["impact map: lever size"] = (size, rule.size.grade)
        columns["impact map: ramp years"] = (ramp, rule.ramp_years.grade)
        columns.update({f"worldsim: {label}": (values, base.grades[label]) for label, values in base.inputs.items()})

        for key in OUTCOME_BY_KEY:
            base_final = base.final(key)
            if top > 0:
                stack = np.stack([run.final(key) for run in runs])
                delta = _interpolate(grid, stack, level) - base_final
                delta[level == 0] = 0.0
            else:
                delta = np.zeros(n)
            if key not in direct and not np.any(np.abs(delta) > 1e-9 * max(1.0, float(np.median(np.abs(base_final))))):
                continue
            result.outcomes[key] = OutcomeImpact(
                key, delta, float(np.median(base_final)), direct.get(key, False), key in direct
            )
            reached = sim.reached_market
            result.spread[key] = {
                "all": _spread_by_grade(delta, columns),
                "if_success": _spread_by_grade(
                    delta[reached], {k: (v[reached], g) for k, (v, g) in columns.items()}
                ),
            }
        return result

    def price_all(self, reports: list[TrackReport]) -> list[DirectionImpact]:
        return [self.price(report) for report in reports]


def analyse(
    corpus: Path = DEFAULT_CORPUS,
    map_path: Path = DEFAULT_MAP,
    draws: int = 4000,
    seed: int = 20261128,
    asof: dt.date | None = None,
) -> tuple[ImpactPricer, list[DirectionImpact]]:
    """Corpus in, every direction priced both ways out."""
    asof = asof or dt.date.today()
    harvest = LocalCorpus(Path(corpus)).harvest()
    if not harvest.works:
        raise ValueError(f"no works loaded from {corpus}: {'; '.join(harvest.errors) or 'empty corpus'}")
    pipeline = Pipeline(asof=asof, draws=draws, seed=seed)
    reports = pipeline.analyse(pipeline.tracks_from_works(harvest.works))
    pricer = ImpactPricer(load_calibration(), ImpactMap.load(map_path), draws, seed, asof)
    return pricer, pricer.price_all(reports)


# ---------------------------------------------------------------------- rendering


def _fmt(key: str, value: float, signed: bool = False) -> str:
    """Three significant figures: a share of a lever moves outcomes by small amounts."""
    if value == 0:
        return "0"
    digits = max(0, 2 - int(np.floor(np.log10(abs(value)))))
    return f"{value:{'+' if signed else ''},.{digits}f}"


def _pct(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:+.3f}%"


def _ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda k: values[k], reverse=True)
    return {key: index + 1 for index, key in enumerate(ordered)}


def render(pricer: ImpactPricer, results: list[DirectionImpact], corpus: Path) -> str:
    cal = pricer.calibration
    priced = [r for r in results if r.priced]
    unpriced = [r for r in results if not r.priced]
    synthetic = any(str(w.id).startswith("synthetic:") for r in results for w in r.report.track.works[:1])
    map_ledger = pricer.mapping.ledger()
    map_assumed = map_ledger.counts().get(Grade.ASSUMED, 0)

    lines = [
        "# Research directions priced by world impact",
        "",
        f"_Generated by `python -m nexus.cli impact` from `{Path(corpus).name}`; "
        f"{pricer.draws:,} paired draws, seed {pricer.seed}, as of {pricer.asof.isoformat()}._",
        "",
        "## Read this first",
        "",
    ]
    if synthetic:
        lines.append(
            "- **The corpus is synthetic.** Every work in it is machine-generated, so the directions, "
            "their success rates and their arrival dates describe no real field. This page shows the "
            "method working end to end; it is not a finding about any technology."
        )
    lines += [
        f"- **The lever mapping is assumed.** {map_assumed} of the {len(map_ledger.entries)} values in "
        f"`nexus/{pricer.mapping.path.name if pricer.mapping.path else 'impact_map.yaml'}` are judgement "
        "calls with no data behind them, including how much of a lever one technology is worth. "
        "Every energy direction is given the same size, so differences between them below come only "
        "from sciscout's success rate and timing.",
        "- **worldsim is not a forecast.** Most of its links and all of its lever effects are assumed; "
        "its own report says so.",
        f"- **Late arrival is approximated.** worldsim holds a lever constant from {cal.base_year} to "
        f"{cal.horizon}, so a direction arriving later is given the constant lever with the same "
        "lever-years. For outcomes marked `exact` the lever's own term gives the same 2050 value "
        "either way; for those marked `approx` it does not, and for warming, where an early cut "
        "counts for more than a late one, it overstates the benefit.",
        "",
        "## Two valuations side by side",
        "",
        "Financial return is sciscout's, on its placeholder market and deal terms. World impact is the "
        f"expected change in the direction's primary {cal.horizon} outcome as a share of the baseline "
        "median, failures counted as no change. Ranks are among priced directions only, best first.",
        "",
        "| Direction | Sector | P(market) | Expected MOIC | Expected NPV | World impact | Rank by MOIC | Rank by impact |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    moic_rank = _ranks({r.report.track.id: r.simulation.expected_moic() for r in priced})
    impact_rank = _ranks({r.report.track.id: r.goodness() or 0.0 for r in priced})
    for r in sorted(priced, key=lambda r: impact_rank[r.report.track.id]):
        sim, tid = r.simulation, r.report.track.id
        lines.append(
            f"| {r.report.name} | {r.report.track.sector} | {sim.expected_success_rate():.1%} "
            f"| {sim.expected_moic():.2f}x | ${sim.expected_npv() / 1e6:,.0f}M "
            f"| {OUTCOME_BY_KEY[r.primary.key].label} {r.primary.expected_percent:+.3f}% "
            f"| {moic_rank[tid]} | {impact_rank[tid]} |"
        )
    for r in unpriced:
        sim = r.simulation
        moic = f"{sim.expected_moic():.2f}x" if sim else "-"
        npv = f"${sim.expected_npv() / 1e6:,.0f}M" if sim else "-"
        reach = f"{sim.expected_success_rate():.1%}" if sim else "-"
        lines.append(f"| {r.report.name} | {r.report.track.sector} | {reach} | {moic} | {npv} | unpriced | - | - |")
    lines.append("")

    moved = [r for r in priced if moic_rank[r.report.track.id] != impact_rank[r.report.track.id]]
    if len(priced) < 2:
        lines.append("Fewer than two directions were priced, so the two rankings cannot be compared.")
    elif moved:
        worst = max(moved, key=lambda r: abs(moic_rank[r.report.track.id] - impact_rank[r.report.track.id]))
        tid = worst.report.track.id
        lines.append(
            f"The two rankings disagree for {len(moved)} of {len(priced)} priced directions. The largest "
            f"gap is {worst.report.name}: rank {moic_rank[tid]} by expected MOIC, rank {impact_rank[tid]} "
            "by world impact. With one lever size shared by every energy direction, impact here follows "
            "the chance of arriving and how early; MOIC also depends on capital needs and dilution."
        )
    else:
        lines.append("The two rankings agree for every priced direction in this run.")
    lines.append("")

    lines += ["## What each priced direction changes", ""]
    for r in sorted(priced, key=lambda r: impact_rank[r.report.track.id]):
        sim = r.simulation
        reached = sim.reached_market
        arrival = pricer.asof.year + float(np.nanmedian(sim.years_to_market)) if reached.any() else float("nan")
        lines += [
            f"### {r.report.name}",
            "",
            f"Rule: {r.rule.name} ({', '.join(f'{LEVERS[k].lower()} x{d.mean():g}' for k, d in r.rule.levers.items())}). "
            f"Reaches market in {reached.mean():.1%} of draws, median arrival {arrival:.0f}. "
            f"Lever size {r.rule.size} and ramp {r.rule.ramp_years} years, both assumed. "
            f"Mean effective lever when it succeeds: {float(r.effective_lever[reached].mean()) if reached.any() else 0.0:.4f} "
            "of a full push.",
            "",
            f"| {cal.horizon} outcome | Baseline median | Expected change | If it succeeds: P10 / P50 / P90 | Timing | Spread from assumed inputs | from chance |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key, impact in r.outcomes.items():
            o = OUTCOME_BY_KEY[key]
            band = impact.if_success(reached)
            band_text = " / ".join(_fmt(key, v, True) for v in band) if band else "never succeeded"
            spread = r.spread[key]["if_success"]
            timing = ("exact" if impact.exact_timing else "approx") if impact.direct else "indirect"
            lines.append(
                f"| {o.label} ({o.unit}) | {_fmt(key, impact.baseline_median)} "
                f"| {_fmt(key, impact.expected, True)} ({_pct(impact.expected_percent)}) | {band_text} "
                f"| {timing} | {spread[Grade.ASSUMED.value]:.0%} | {spread['chance and interaction']:.0%} |"
            )
        by_grade = sim.variance_by_grade()
        lines += [
            "",
            f"Financial side: expected MOIC {sim.expected_moic():.2f}x, expected NPV "
            f"${sim.expected_npv() / 1e6:,.0f}M; {by_grade.get(Grade.ASSUMED, 0.0):.0%} of the "
            "input-attributable NPV spread comes from assumed inputs.",
            "",
        ]
    lines += [
        "Expected change averages over every draw, with failures counted as no change. The spread "
        "columns are for the draws that reach market: squared rank correlations between the change "
        "and every sampled input from sciscout, the impact map and worldsim, summed by provenance "
        "grade. `indirect` outcomes are reached only through worldsim's couplings between problems.",
        "",
        "## Unpriced directions",
        "",
        "Unpriced does not mean zero impact. It means no worldsim lever represents what the direction "
        "would change, so this method has nothing to say about it.",
        "",
        "| Direction | Sector | Why it is unpriced |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {r.report.name} | {r.report.track.sector} | {r.reason} |" for r in unpriced] or ["| (none) | | |"]
    lines += ["", "## Where the mapping's numbers come from", "", map_ledger.audit_table(), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- cli


def cmd_impact(args) -> int:
    try:
        pricer, results = analyse(args.corpus, args.map, args.draws, args.seed, args.asof)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render(pricer, results, args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    priced = [r for r in results if r.priced]
    print(f"priced {len(priced)} of {len(results)} directions against worldsim; wrote {args.out}")
    for r in sorted(priced, key=lambda r: r.goodness() or 0.0, reverse=True):
        p = r.primary
        print(
            f"  {r.report.name[:38]:38s} P(market) {r.simulation.expected_success_rate():5.1%}  "
            f"MOIC {r.simulation.expected_moic():5.2f}x  "
            f"{OUTCOME_BY_KEY[p.key].label}: {_fmt(p.key, p.expected, True)} {OUTCOME_BY_KEY[p.key].unit}"
        )
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "impact", help="price research directions by their effect on worldsim's 2050 outcomes"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="lever mapping; every value needs a source")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--asof", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.set_defaults(func=cmd_impact)
