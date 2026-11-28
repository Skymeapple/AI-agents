"""Physical bottlenecks: how fast deployed hardware actually follows an idea.

sciscout's commercialisation priors and market placeholders are mostly graded
``assumed``. This module replaces the ones that data in this repository can
speak to, and says plainly which ones it cannot.

Two mechanisms:

*Deployment ramps.* From country-level electricity data, how many years a
technology took to grow its share of a national grid tenfold (by default from
1% to 10%). Countries still on the way are right-censored, so the distribution
is a Kaplan-Meier estimate -- dropping them would keep only the fast ramps and
bias the answer short. The result replaces ``MarketModel.ramp_years`` for the
``energy_hardware`` sector, graded ``observed``.

*Experience curves.* Wright's law: unit cost falls by a fixed fraction with
every doubling of cumulative production. The fit and its back-test are here and
tested, but **no cost series is shipped** -- the host that publishes them was
refused by this session's network policy -- so no stage cost prior is touched.
Add a series to ``data/nexus/bottleneck/manifest.json`` with ``kind: cost`` and
the report picks it up.

What this does not do: it changes no stage duration, cost or success
probability. Nothing in the repository measures those. The manufacturing,
precision, chip and technology-transfer models that would have informed them
were built elsewhere and are not in this tree.

Both mechanisms are back-tested by refitting on data up to a cutoff year and
scoring against the years held out.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sciscout.commercial.profile import CommercialProfile, PriorLibrary, Stage
from sciscout.invest.dist import Distribution, LogNormal
from sciscout.invest.engine import MonteCarlo
from sciscout.invest.model import DealTerms, MarketModel
from sciscout.pipeline import Pipeline
from sciscout.provenance import Grade, Provenance, ProvenanceLedger, assumed
from sciscout.sources.local import LocalCorpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nexus" / "bottleneck"
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.json"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "examples" / "nexus-bottleneck.md"

# Judgement calls this module makes. Each is graded ``assumed`` and shown in the
# report; none is hidden in a formula.
BAND = (1.0, 10.0)
BAND_NOTE = assumed(
    "a tenfold rise in share of national electricity, from 1% to 10%, stands for "
    "the ramp from first commercial sales to peak share. The band is a choice; "
    "the report shows two others beside it.",
)
MIN_GRID_TWH = 10.0
MIN_GRID_NOTE = assumed(
    "grids generating under 10 TWh a year are left out: one plant can move a "
    "small grid's share by several points in a year, which measures the size of "
    "the grid rather than the pace of deployment.",
)
# Which measured technologies stand in for which sciscout sector.
SECTOR_ANALOGUES: dict[str, tuple[str, ...]] = {"energy_hardware": ("solar", "wind")}
ANALOGUE_NOTE = assumed(
    "solar and wind deployment stand in for energy hardware generally. A new "
    "chemistry or reactor may ramp slower (no supply chain yet) or faster "
    "(drop-in to existing lines); nothing here measures which.",
)
Z80 = 1.2815515655446004  # standard normal quantile for a central 80% interval
Z95 = 1.6448536269514722
Z75 = 0.6744897501960817

Series = dict[tuple[str, str], list[tuple[int, float, float]]]


# -- data ---------------------------------------------------------------------


def load_manifest(data_dir: Path = DATA_DIR) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def load_deployment(data_dir: Path = DATA_DIR) -> Series:
    """``(technology, iso_code) -> [(year, share %, grid TWh), ...]`` by year."""
    entry = load_manifest(data_dir)["deployment"]
    series: Series = {}
    with open(data_dir / entry["file"], newline="") as handle:
        for row in csv.DictReader(handle):
            grid = float(row["electricity_generation_twh"] or "nan")
            series.setdefault((row["technology"], row["iso_code"]), []).append(
                (int(row["year"]), float(row["share_elec_pct"]), grid)
            )
    for points in series.values():
        points.sort()
    return series


def truncate(series: Series, cutoff: int) -> Series:
    """The series as it stood at the end of ``cutoff``. Later years are gone."""
    cut = {key: [p for p in points if p[0] <= cutoff] for key, points in series.items()}
    return {key: points for key, points in cut.items() if points}


def last_year(series: Series) -> int:
    return max(points[-1][0] for points in series.values())


# -- deployment ramps ---------------------------------------------------------


@dataclass(frozen=True)
class Ramp:
    """One technology in one country crossing the band, or still trying to."""

    technology: str
    country: str
    start: float  # year the share first reached the lower bound
    duration: float  # years to the upper bound, or to the last observation
    completed: bool


def _crossing(points: list[tuple[int, float, float]], level: float, after: float = -math.inf):
    """First upward passage through ``level``, linearly interpolated.

    Returns ``None`` if the level is never reached, or if the series is already
    above it at its first point -- then the crossing happened before the data
    starts and its date is unknown.
    """
    for (y0, s0, _), (y1, s1, _) in zip(points, points[1:]):
        if s0 < level <= s1:
            when = y0 + (level - s0) / (s1 - s0) * (y1 - y0)
            if when >= after:
                return when
    return None


def ramps(
    series: Series,
    band: tuple[float, float] = BAND,
    min_grid_twh: float = MIN_GRID_TWH,
    technologies: tuple[str, ...] | None = None,
) -> list[Ramp]:
    low, high = band
    found = []
    for (technology, country), points in sorted(series.items()):
        if technologies and technology not in technologies:
            continue
        start = _crossing(points, low)
        if start is None:
            continue
        grid = next((g for y, _, g in points if y >= start), math.nan)
        if not grid >= min_grid_twh:  # also drops a missing grid size
            continue
        end = _crossing(points, high, after=start)
        if end is None:
            found.append(Ramp(technology, country, start, points[-1][0] - start, False))
        else:
            found.append(Ramp(technology, country, start, end - start, True))
    return found


@dataclass
class KaplanMeier:
    """Survival curve for time-to-completion with right-censoring."""

    times: np.ndarray  # distinct completion times, ascending
    survival: np.ndarray  # S(t) just after each of ``times``
    n: int
    completed: int

    @classmethod
    def fit(cls, observations: list[Ramp]) -> "KaplanMeier":
        events = sorted({r.duration for r in observations if r.completed})
        s, curve = 1.0, []
        for t in events:
            at_risk = sum(r.duration >= t for r in observations)
            died = sum(r.completed and r.duration == t for r in observations)
            s *= 1.0 - died / at_risk
            curve.append(s)
        return cls(np.array(events), np.array(curve), len(observations),
                   sum(r.completed for r in observations))

    def s(self, t: float) -> float:
        index = int(np.searchsorted(self.times, t, side="right")) - 1
        return 1.0 if index < 0 else float(self.survival[index])

    def quantile(self, q: float, given: float = 0.0) -> float | None:
        """Time by which a share ``q`` have completed, among those not done by ``given``.

        ``None`` when the curve never gets that far -- too many are still
        censored to say.
        """
        target = self.s(given) * (1.0 - q)
        reached = np.nonzero((self.survival <= target) & (self.times > given))[0]
        return float(self.times[reached[0]]) if len(reached) else None

    def completes_within(self, elapsed: float, horizon: float) -> float:
        """P(done by ``elapsed + horizon`` | not done by ``elapsed``)."""
        base = self.s(elapsed)
        return 0.0 if base <= 0 else 1.0 - self.s(elapsed + horizon) / base


def ramp_distribution(
    observations: list[Ramp], evidence: str, band: tuple[float, float] = BAND
) -> LogNormal:
    """Years-to-ramp as a distribution sciscout can sample, graded ``observed``.

    A log-normal with the Kaplan-Meier median and interquartile range. The
    quartiles are used, not the tails, because the slowest ramps are still
    running and the curve does not reach its 90th percentile; the upper tail
    of the result is therefore the log-normal's shape, not a measurement.
    Raises if even the quartiles are not reached.
    """
    km = KaplanMeier.fit(observations)
    q1, median, q3 = km.quantile(0.25), km.quantile(0.5), km.quantile(0.75)
    if q1 is None or median is None or q3 is None or q3 <= q1:
        raise ValueError(
            f"{km.completed} completed ramps of {km.n} is too few to place the "
            "quartiles"
        )
    sigma = math.log(q3 / q1) / (2 * Z75)
    technologies = ", ".join(sorted({r.technology for r in observations}))
    provenance = Provenance(
        Grade.OBSERVED,
        f"Kaplan-Meier over {km.n} country ramps of {technologies} from "
        f"{band[0]:g}% to {band[1]:g}% of national electricity ({km.completed} "
        f"completed, {km.n - km.completed} censored); median {median:.1f} y, quartiles "
        f"{q1:.1f} and {q3:.1f} y; series '{evidence}' in data/nexus/bottleneck. "
        "Measured: how fast these technologies were deployed. Not measured: that "
        "a different technology, or one firm's sales, ramps the same way.",
        needs_review=True,
    )
    return LogNormal(median, math.exp(Z95 * sigma), provenance, "ramp (y)")


def backtest_ramps(
    series: Series,
    cutoff: int,
    band: tuple[float, float] = BAND,
    min_grid_twh: float = MIN_GRID_TWH,
    technologies: tuple[str, ...] | None = None,
) -> dict:
    """Fit on data to ``cutoff``; score against what happened afterwards.

    The units scored are ramps that had started but not finished at the cutoff.
    For each, the curve fitted on the truncated data gives a probability of
    finishing by the last year on record and a 50% interval for the total
    duration. Only :func:`truncate`'d data reaches the fit.
    """
    end = last_year(series)
    known = truncate(series, cutoff)
    train = ramps(known, band, min_grid_twh, technologies)
    km = KaplanMeier.fit(train)
    truth = {(r.technology, r.country): r for r in ramps(series, band, min_grid_twh, technologies)}

    predicted, actual, inside, judged, early = [], [], 0, 0, 0
    for unit in train:
        if unit.completed:
            continue
        outcome = truth[(unit.technology, unit.country)]
        predicted.append(km.completes_within(unit.duration, end - cutoff))
        actual.append(float(outcome.completed))
        low = km.quantile(0.25, given=unit.duration)
        high = km.quantile(0.75, given=unit.duration)
        if low is None or high is None:
            continue
        # A ramp still running past the upper bound is a definite miss; one
        # still running inside the interval cannot be judged yet.
        if outcome.completed:
            judged += 1
            inside += low <= outcome.duration <= high
            early += outcome.duration < low
        elif outcome.duration > high:
            judged += 1
    p, a = np.array(predicted), np.array(actual)
    return {
        "cutoff": cutoff,
        "scored_to": end,
        "train_ramps": len(train),
        "train_completed": km.completed,
        "at_risk": len(p),
        "predicted_completions": float(p.sum()) if len(p) else 0.0,
        "actual_completions": int(a.sum()) if len(a) else 0,
        "brier": float(np.mean((p - a) ** 2)) if len(p) else math.nan,
        "brier_base_rate": float(np.mean((a.mean() - a) ** 2)) if len(a) else math.nan,
        "interval_judged": judged,
        "interval_covered": int(inside),
        "finished_early": int(early),
    }


# -- experience curves --------------------------------------------------------


@dataclass(frozen=True)
class WrightFit:
    """``log cost = intercept + slope * log cumulative``, by least squares."""

    intercept: float
    slope: float
    residual_sd: float
    n: int
    mean_log_cumulative: float
    sxx: float

    @property
    def learning_rate(self) -> float:
        """Fractional cost fall per doubling of cumulative production."""
        return 1.0 - 2.0 ** self.slope

    def predict(self, cumulative: float, z: float = Z80) -> tuple[float, float, float]:
        """Median cost and a prediction interval at ``cumulative``."""
        x = math.log(cumulative)
        centre = self.intercept + self.slope * x
        half = z * self.residual_sd * math.sqrt(
            1.0 + 1.0 / self.n + (x - self.mean_log_cumulative) ** 2 / self.sxx
        )
        return math.exp(centre), math.exp(centre - half), math.exp(centre + half)

    def decline_after(self, doublings: float) -> float:
        """Cost as a multiple of today's after ``doublings`` of cumulative output."""
        return 2.0 ** (self.slope * doublings)


def fit_wright(cumulative, cost) -> WrightFit:
    x, y = np.log(np.asarray(cumulative, float)), np.log(np.asarray(cost, float))
    if len(x) < 4:
        raise ValueError("an experience curve needs at least four points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("cumulative production must be strictly increasing")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    sd = float(np.sqrt(residual @ residual / (len(x) - 2)))
    return WrightFit(float(intercept), float(slope), sd, len(x), float(x.mean()),
                     float(((x - x.mean()) ** 2).sum()))


def backtest_wright(years, cumulative, cost, cutoff: int) -> dict:
    """Fit to ``cutoff``; predict later costs from the cumulative output reached.

    This tests the cost-given-production relation only. Forecasting a cost at a
    date also needs a forecast of production, which this does not make.
    """
    years = np.asarray(years)
    cumulative, cost = np.asarray(cumulative, float), np.asarray(cost, float)
    seen = years <= cutoff
    if (~seen).sum() == 0:
        raise ValueError(f"no years after {cutoff} to score against")
    fit = fit_wright(cumulative[seen], cost[seen])
    errors, covered = [], 0
    for c, actual in zip(cumulative[~seen], cost[~seen]):
        median, low, high = fit.predict(c)
        errors.append(abs(math.log(median / actual)))
        covered += low <= actual <= high
    return {
        "cutoff": cutoff,
        "train_points": int(seen.sum()),
        "held_out": int((~seen).sum()),
        "learning_rate": fit.learning_rate,
        "mean_abs_log_error": float(np.mean(errors)),
        "interval_covered": int(covered),
    }


def load_cost_series(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Cost series listed in the manifest (``kind: cost``). None ship today."""
    found = {}
    for name, entry in load_manifest(data_dir).items():
        if not isinstance(entry, dict) or entry.get("kind") != "cost":
            continue
        with open(data_dir / entry["file"], newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: int(r["year"]))
        found[name] = {
            "years": [int(r["year"]) for r in rows],
            "cumulative": [float(r["cumulative"]) for r in rows],
            "cost": [float(r["cost"]) for r in rows],
        }
    return found


# -- applying evidence to a sciscout case -------------------------------------

STAGE_FIELDS = ("duration_years", "cost_usd", "p_success")


@dataclass(frozen=True)
class Override:
    """One prior replaced by one distribution.

    ``target`` is ``market.<field>`` or ``stage.<stage name>.<field>``.
    ``evidence`` names the manifest series a non-assumed value was computed
    from; it is what stops a guess from being relabelled as a measurement.
    """

    target: str
    distribution: Distribution
    evidence: str | None = None


def _check(override: Override, manifest: dict) -> None:
    provenance = getattr(override.distribution, "provenance", None)
    if not isinstance(provenance, Provenance):
        raise ValueError(f"override of {override.target} carries no provenance")
    if provenance.grade is Grade.OBSERVED:
        entry = manifest.get(override.evidence or "")
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(
                f"override of {override.target} is graded observed but names no "
                "series in the bottleneck manifest; grade it assumed, or add the data"
            )


@dataclass
class Adjusted:
    profile: CommercialProfile
    market: MarketModel
    replaced: list[str]
    still_assumed: list[str]


def apply_overrides(
    profile: CommercialProfile,
    market: MarketModel,
    overrides: list[Override],
    manifest: dict | None = None,
) -> Adjusted:
    """Return the case with ``overrides`` applied and everything else untouched.

    An unknown target is an error, not a silent no-op.
    """
    manifest = load_manifest() if manifest is None else manifest
    market_changes: dict[str, Distribution] = {}
    stage_changes: dict[str, dict[str, Distribution]] = {}
    stage_names = {s.name for s in profile.stage_path or profile.remaining_stages}
    for override in overrides:
        _check(override, manifest)
        kind, _, rest = override.target.partition(".")
        if kind == "market" and rest in market.distributions():
            market_changes[rest] = override.distribution
            continue
        name, _, fieldname = rest.rpartition(".")
        if kind == "stage" and name in stage_names and fieldname in STAGE_FIELDS:
            stage_changes.setdefault(name, {})[fieldname] = override.distribution
            continue
        raise ValueError(f"override target {override.target!r} matches nothing in this case")

    def patch(stage: Stage) -> Stage:
        return replace(stage, **stage_changes.get(stage.name, {}))

    remaining = [patch(s) for s in profile.remaining_stages]
    ledger = ProvenanceLedger()
    ledger.record(profile.trl.as_estimate())
    for stage in remaining:
        for fieldname in STAGE_FIELDS:
            ledger.record(getattr(stage, fieldname).as_estimate())
    new_profile = replace(
        profile, remaining_stages=remaining, ledger=ledger,
        stage_path=[patch(s) for s in profile.stage_path],
    )
    new_market = replace(market, **market_changes)

    still = [f"market.{name}" for name, d in new_market.distributions().items()
             if d.grade is Grade.ASSUMED]
    still += [f"stage.{s.name}.{f}" for s in new_profile.possible_stages()
              for f in STAGE_FIELDS if getattr(s, f).grade is Grade.ASSUMED]
    return Adjusted(new_profile, new_market, [o.target for o in overrides], still)


def evidence_for(sector: str, series: Series | None = None) -> list[Override]:
    """Overrides the shipped data supports for ``sector``. Empty for most."""
    analogues = SECTOR_ANALOGUES.get(sector)
    if not analogues:
        return []
    series = load_deployment() if series is None else series
    observations = ramps(series, technologies=analogues)
    return [Override("market.ramp_years", ramp_distribution(observations, "deployment"),
                     "deployment")]


# -- report -------------------------------------------------------------------


def _ramp_table(series: Series) -> list[str]:
    lines = ["| Technologies | Band | Ramps | Completed | Median (y) | Quartiles (y) |",
             "| --- | --- | --- | --- | --- | --- |"]

    def fmt(value):
        return "not reached" if value is None else f"{value:.1f}"

    for technologies in (("solar", "wind"), ("solar",), ("wind",)):
        for band in (BAND, (0.5, 5.0), (2.0, 20.0)):
            if technologies != ("solar", "wind") and band != BAND:
                continue
            km = KaplanMeier.fit(ramps(series, band, technologies=technologies))
            lines.append(
                f"| {' + '.join(technologies)} | {band[0]:g}% to {band[1]:g}% | {km.n} "
                f"| {km.completed} | {fmt(km.quantile(0.5))} "
                f"| {fmt(km.quantile(0.25))} to {fmt(km.quantile(0.75))} |"
            )
    return lines


def _cohort_table(series: Series) -> list[str]:
    observations = ramps(series, technologies=SECTOR_ANALOGUES["energy_hardware"])
    lines = ["| Ramp started | Ramps | Completed | First quartile (y) | Median (y) |",
             "| --- | --- | --- | --- | --- |"]

    def fmt(value):
        return "not reached" if value is None else f"{value:.1f}"

    for first, last in ((1985, 2009), (2010, 2014), (2015, 2019)):
        cohort = [r for r in observations if first <= r.start < last + 1]
        km = KaplanMeier.fit(cohort)
        lines.append(f"| {first}-{last} | {km.n} | {km.completed} "
                     f"| {fmt(km.quantile(0.25))} | {fmt(km.quantile(0.5))} |")
    return lines


def _backtest_table(series: Series, cutoffs: tuple[int, ...]) -> list[str]:
    lines = ["| Fitted to | Ramps (completed) | Unfinished at cutoff | Finished since: "
             "predicted | actual | Brier | Brier, base rate | 50% interval covered | Finished before interval |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cutoff in cutoffs:
        b = backtest_ramps(series, cutoff, technologies=SECTOR_ANALOGUES["energy_hardware"])
        lines.append(
            f"| {cutoff} | {b['train_ramps']} ({b['train_completed']}) | {b['at_risk']} "
            f"| {b['predicted_completions']:.1f} | {b['actual_completions']} "
            f"| {b['brier']:.3f} | {b['brier_base_rate']:.3f} "
            f"| {b['interval_covered']} of {b['interval_judged']} "
            f"| {b['finished_early']} |"
        )
    return lines


def _comparison(corpus: Path, series: Series, draws: int, seed: int) -> list[str]:
    works = LocalCorpus(corpus).harvest().works
    pipeline = Pipeline()
    tracks = pipeline.tracks_from_works(works)
    engine = MonteCarlo(draws=draws, seed=seed)
    terms = DealTerms.placeholder()
    lines = ["| Direction | Ramp to peak, prior | Ramp, evidence | Expected MOIC | "
             "Expected NPV ($M) | Assumed share of input spread |",
             "| --- | --- | --- | --- | --- | --- |"]
    untouched = []
    for track in tracks:
        overrides = evidence_for(track.sector, series)
        if not overrides:
            untouched.append(track.sector)
            continue
        profile = pipeline.priors.profile(track)
        if not profile.remaining_stages:
            continue
        market = MarketModel.placeholder()
        adjusted = apply_overrides(profile, market, overrides)
        before = engine.run(profile, market, terms)
        after = engine.run(adjusted.profile, adjusted.market, terms)
        share = lambda r: r.variance_by_grade()[Grade.ASSUMED]  # noqa: E731
        lines.append(
            f"| {track.name} | {market.ramp_years} | {adjusted.market.ramp_years} "
            f"| {before.expected_moic():.2f}x -> {after.expected_moic():.2f}x "
            f"| {before.expected_npv() / 1e6:,.0f} -> {after.expected_npv() / 1e6:,.0f} "
            f"| {share(before):.1%} -> {share(after):.1%} |"
        )
    sectors = sorted(set(untouched))
    lines += ["", "Sectors with no evidence here, left exactly as shipped: "
              + (", ".join(sectors) if sectors else "none") + "."]
    return lines


def render(corpus: Path = DEFAULT_CORPUS, data_dir: Path = DATA_DIR,
           draws: int = 20_000, seed: int = 20260918) -> str:
    manifest = load_manifest(data_dir)
    series = load_deployment(data_dir)
    entry = manifest["deployment"]
    priors = PriorLibrary()
    lines = [
        "# Physical bottlenecks", "",
        "_Generated by `python -m nexus.cli bottleneck`._", "",
        "How much of sciscout's commercialisation case can be moved from assumption "
        "to measurement with the data in this repository. The answer is: one input, "
        "for one sector.", "",
        "## What was measured", "",
        f"- **Data:** {entry['title']}. {entry['publisher']}; retrieved "
        f"{entry['retrieved']} from <{entry['url']}>. {len(series)} country series, "
        f"to {last_year(series)}.",
        f"- **Assumed:** {BAND_NOTE.detail}",
        f"- **Assumed:** {MIN_GRID_NOTE.detail}",
        f"- **Assumed:** {ANALOGUE_NOTE.detail}", "",
        "Years for a technology's share of a national grid to rise across the band. "
        "Countries still inside the band are censored, not dropped.", "",
        *_ramp_table(series), "",
        "## Back-test", "",
        "The curve is refitted on data up to a cutoff year. Each ramp that had "
        "started but not finished by then gets a probability of finishing by "
        f"{last_year(series)} and a 50% interval (conditional quartiles) for its total length; both are "
        "scored against what happened. A Brier score below the base-rate column "
        "means the curve tells unfinished ramps apart better than one shared guess "
        "(the base rate is computed from the outcomes, so it is not a forecast "
        "anyone could have made).", "",
        *_backtest_table(series, (2012, 2015, 2018)), "",
        "The interval column counts only ramps whose outcome is settled: finished, "
        "or still running past the interval's upper end.", "",
        "**Read this as a weak result.** The curve does no better than the base "
        "rate at saying which unfinished ramps will finish, and at the later "
        "cutoffs many ramps finished before its 50% interval even began. Ramps are "
        "not drawn from one fixed distribution. Split by the year a ramp started, "
        "later starters have moved faster, and a pooled curve fitted on the early "
        "ones cannot see that:", "",
        *_cohort_table(series), "",
        "So treat the pooled median as an order of magnitude -- around a decade, "
        "not the four years of the placeholder -- and as biased long for a "
        "technology starting now. It is not a calibrated forecast.", "",
        "## Effect on the investment case", "",
        "`MarketModel.ramp_years` for energy hardware, placeholder against "
        "evidence, same random draws for every other input.", "",
        *_comparison(corpus, series, draws, seed), "",
        "The last column is the share of input-attributable NPV spread that comes "
        "from assumed inputs. It falls because ramp time now counts as observed, "
        "and stays well above zero because every other market and stage input is "
        "still a judgement call. The fall in expected MOIC is the substantive "
        "result: a ramp of about twelve years instead of four pushes revenue past "
        "the exit date in most draws. It inherits the assumption that a new "
        "energy technology ramps the way solar and wind did.", "",
        "## Experience curves", "",
    ]
    costs = load_cost_series(data_dir)
    if costs:
        lines += ["| Series | Points | Learning rate | Back-test: mean abs log error | "
                  "80% interval covered |", "| --- | --- | --- | --- | --- |"]
        for name, s in costs.items():
            fit = fit_wright(s["cumulative"], s["cost"])
            cutoff = s["years"][int(len(s["years"]) * 0.7)]
            b = backtest_wright(s["years"], s["cumulative"], s["cost"], cutoff)
            lines.append(f"| {name} | {fit.n} | {fit.learning_rate:.1%} "
                         f"| {b['mean_abs_log_error']:.2f} (fitted to {cutoff}) "
                         f"| {b['interval_covered']} of {b['held_out']} |")
    else:
        lines.append("No cost series is in the repository, so no learning rate is "
                     "reported and no stage cost is changed. The fit and its "
                     "back-test are implemented and tested on planted data only.")
        for gap in manifest.get("_not_obtained", []):
            lines.append(f"- Wanted: {gap['wanted']}. Tried: {gap['tried']}; "
                         f"{gap['result']}.")
    lines += ["", "## Still assumed", ""]
    for sector in priors.sectors():
        stages = priors.stages(sector)
        assumed_n = sum(s.provenance.grade is Grade.ASSUMED for s in stages)
        covered = "ramp time replaced" if sector in SECTOR_ANALOGUES else "nothing replaced"
        lines.append(f"- `{sector}`: {covered}; {assumed_n} of {len(stages)} stage "
                     "priors assumed, all unchanged.")
    lines += ["", "Stage durations, costs and success rates are untouched in every "
              "sector. Deployment data says how fast a working product spreads, not "
              "how long a pilot line takes or what it costs.", ""]
    return "\n".join(lines)


def run(args) -> int:
    text = render(Path(args.corpus), Path(args.data), args.draws, args.seed)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "bottleneck",
        help="replace assumed scale-up priors with measured deployment ramps",
    )
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="markdown file to write; empty string prints instead")
    parser.set_defaults(func=run)
