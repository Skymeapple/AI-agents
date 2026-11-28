"""A coupled, stochastic model of ten global problems, 2025-2050.

Each problem is one or two state variables stepped a year at a time:

* where a global series exists, its baseline rate of change is *fitted* from the
  data in ``data/world`` (graded ``observed``);
* where only report figures exist, they anchor the level (``sourced``);
* the links between problems and the effect of policy levers come from
  ``params.yaml``, and are almost all ``assumed``.

That last point decides how the output should be read. The direction of each
link is well supported; its size is not. So the model is useful for two things
-- seeing how far the world is from its goals if measured trends persist, and
seeing which results would survive a change of assumptions -- and it is not a
forecast. Every run reports what share of its spread traces to assumption.

Couplings act on *changes* in a driver relative to the conditions the trends
were measured under, because the fitted trends already contain whatever effect
conflict, warming and inequality had over the fitting window. Feeding the level
back in as well would count it twice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sciscout.invest.dist import PERT, Distribution, LogNormal, Point, from_three_point
from sciscout.invest.engine import _rank, _stream  # shared so draws follow the same common-random-number scheme
from sciscout.provenance import Grade, Provenance, ProvenanceLedger, observed, parse_provenance

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR.parent / "data" / "world"

LEVERS: dict[str, str] = {
    "mitigation": "Cut emissions",
    "peacebuilding": "Reduce conflict",
    "development": "Finance development",
    "health": "Invest in health security",
    "conservation": "Protect ecosystems",
    "wash": "Build water and sanitation",
    "ai_governance": "Govern advanced AI",
    "redistribution": "Tax and redistribute",
}


@dataclass(frozen=True)
class Outcome:
    key: str
    problem: str
    label: str
    unit: str
    lower_is_better: bool = True
    digits: int = 1


OUTCOMES: tuple[Outcome, ...] = (
    Outcome("temperature", "Climate change", "Global warming", "degC above pre-industrial", digits=2),
    Outcome("emissions", "Climate change", "CO2 emissions", "GtCO2 per year"),
    Outcome("conflict_deaths", "War", "Deaths from organised violence", "thousand per year", digits=0),
    Outcome("poverty_people", "Extreme poverty", "People under $3 a day", "million", digits=0),
    Outcome("hunger_people", "Hunger", "Undernourished people", "million", digits=0),
    Outcome("amr_deaths", "Pandemics and resistance", "Deaths from drug-resistant infections", "million per year", digits=2),
    Outcome("pandemic_deaths", "Pandemics and resistance", "Cumulative pandemic deaths since 2025", "million"),
    Outcome("forest_share", "Biodiversity loss", "Forest cover (habitat proxy)", "% of land", lower_is_better=False, digits=2),
    Outcome("water_people", "Water and sanitation", "People without safe drinking water", "million", digits=0),
    Outcome("sanitation_people", "Water and sanitation", "People without safe sanitation", "million", digits=0),
    Outcome("displaced", "Forced displacement", "Forcibly displaced people", "million"),
    Outcome("ai_gap", "Governing AI", "AI oversight gap index (2025 = 1)", "index", digits=2),
    Outcome("wealth_share", "Inequality", "Wealth share of the richest 10%", "%"),
)
OUTCOME_BY_KEY = {o.key: o for o in OUTCOMES}


@dataclass
class Normal(Distribution):
    """Gaussian, for a fitted coefficient with a standard error."""

    mu: float
    sd: float
    provenance: Provenance
    label: str = "normal"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sd, size=n)

    def mean(self) -> float:
        return float(self.mu)

    def __str__(self) -> str:
        return f"Normal({self.mu:.4g}, sd {self.sd:.2g})"


# --------------------------------------------------------------------------- data


def load_observed(path: Path | None = None) -> dict[str, dict[int, float]]:
    """Long-form ``observed.csv`` -> ``{series: {year: value}}``."""
    path = path or DATA_DIR / "observed.csv"
    series: dict[str, dict[int, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row["series"], {})[int(row["year"])] = float(row["value"])
    return series


def _slope(years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """OLS slope of ``values`` on ``years`` and its standard error."""
    x = years - years.mean()
    slope = float((x * (values - values.mean())).sum() / (x**2).sum())
    resid = values - values.mean() - slope * x
    se = float(np.sqrt((resid**2).sum() / max(len(x) - 2, 1) / (x**2).sum()))
    return slope, se


def _typical_change(years: np.ndarray, logs: np.ndarray) -> tuple[float, float]:
    """Median annual log change and a robust standard error for it.

    The median rather than a fitted slope, because these series contain one-off
    steps -- the pandemic lifted undernourishment by a tenth in a single year --
    and a least-squares line through a step reports a trend that is not there.
    The model draws pandemics separately, so a trend that absorbed the last one
    would also count it twice.
    """
    changes = np.diff(logs) / np.diff(years)
    centre = float(np.median(changes))
    mad = float(np.median(np.abs(changes - centre)))
    return centre, 1.2533 * 1.4826 * mad / np.sqrt(len(changes))


def fit_trend(
    points: dict[int, float], label: str, lack_of: bool = False, extended_with: Provenance | None = None
) -> PERT:
    """Annual log rate of change, with the choice of window as the uncertainty.

    A trend is not one number: the last ten years and the last twenty often
    disagree, and which one persists is exactly what nobody knows. So the mode
    is the typical annual change over the last ten years (the most recent
    regime) and the range spans both windows, widened by a standard error on
    each side. A series shorter than fifteen years uses two standard errors
    around the ten-year figure instead.

    Args:
        points: ``{year: value}``.
        label: Name carried into the audit table.
        lack_of: Fit the trend of ``100 - value``, for coverage percentages
            where the quantity of interest is the share still without.
        extended_with: Provenance of a report figure appended to the ingested
            series. The fit then takes that weaker grade: a trend is only as
            well founded as its least founded point.
    """
    years = np.array(sorted(points), dtype=float)
    values = np.array([points[int(y)] for y in years], dtype=float)
    if lack_of:
        values = 100.0 - values
    logs = np.log(values)
    last = int(years[-1])
    recent = years >= last - 10
    r10, se10 = _typical_change(years[recent], logs[recent])
    long_window = years >= last - 20
    if long_window.sum() >= 16:
        r20, se20 = _typical_change(years[long_window], logs[long_window])
        low, high = min(r10 - se10, r20 - se20), max(r10 + se10, r20 + se20)
        basis = f"10-year {r10:+.4f}, 20-year {r20:+.4f}, to {last}"
    else:
        low, high = r10 - 2 * se10, r10 + 2 * se10
        basis = f"10-year {r10:+.4f} +/- 2 s.e., to {last} (series too short for a 20-year window)"
    detail = f"{label}: median annual log change; {basis}"
    if extended_with is not None:
        provenance = Provenance(
            extended_with.grade,
            f"{detail}; last point from {extended_with.detail}",
            extended_with.reference,
            extended_with.retrieved,
            extended_with.needs_review,
        )
    else:
        provenance = observed(detail)
    return PERT(low, r10, high, provenance, label)


# -------------------------------------------------------------------- calibration


@dataclass
class Calibration:
    """Everything the simulation needs, each quantity with its provenance."""

    base_year: int
    horizon: int
    params: dict[str, Distribution]
    trends: dict[str, PERT]
    start: dict[str, float]
    start_provenance: dict[str, Provenance]
    conflict_log_mean: float
    conflict_log_sd: float
    displaced_history: np.ndarray
    population: np.ndarray
    scenarios: list[dict]
    context: dict[str, dict]
    observed: dict[str, dict[int, float]]
    sourced_series: dict[str, dict[int, float]]

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.base_year, self.horizon + 1)

    def ledger(self) -> ProvenanceLedger:
        """Every sampled input, for the audit table."""
        ledger = ProvenanceLedger()
        for dist in list(self.trends.values()) + list(self.params.values()):
            ledger.record(dist.as_estimate())
        return ledger


def _parameter(name: str, spec: dict) -> Distribution:
    if not isinstance(spec, dict):
        raise ValueError(f"{name}: expected a mapping with a 'source' block")
    provenance = parse_provenance(spec.get("source"), name)
    kind = spec.get("kind", "pert")
    unknown = set(spec) - {"low", "mode", "high", "median", "spread", "value", "kind", "unit", "source"}
    if unknown:
        raise ValueError(f"{name}: unknown field(s) {sorted(unknown)}")
    if kind == "lognormal":
        return LogNormal(float(spec["median"]), float(spec["spread"]), provenance, name)
    if "value" in spec:
        return Point(float(spec["value"]), provenance, name)
    return from_three_point(spec, provenance, name, kind=kind)


def _latest(points: dict[int, float]) -> tuple[int, float]:
    year = max(points)
    return year, points[year]


def load_calibration(
    params_path: Path | None = None, data_dir: Path | None = None
) -> Calibration:
    """Fit trends from the observed series and load everything else from config."""
    data_dir = data_dir or DATA_DIR
    config = yaml.safe_load((params_path or PACKAGE_DIR / "params.yaml").read_text())
    sourced = yaml.safe_load((data_dir / "sourced.yaml").read_text())
    obs = load_observed(data_dir / "observed.csv")

    params = {name: _parameter(name, spec) for name, spec in config["parameters"].items()}
    points = {name: (float(p["value"]), parse_provenance(p.get("source"), name)) for name, p in sourced["points"].items()}
    series_prov = {name: parse_provenance(s.get("source"), name) for name, s in sourced["series"].items()}
    sourced_series = {name: {int(y): float(v) for y, v in s["values"].items()} for name, s in sourced["series"].items()}

    base_year, horizon = int(config["base_year"]), int(config["horizon"])

    # The ingested undernourishment series stops a year before the latest
    # published figure, and that year matters: it is the third straight fall.
    hunger_value, hunger_prov = points["undernourishment_2025"]
    hunger_series = {**obs["undernourishment_rate"], base_year: hunger_value}

    trends = {
        "emissions_trend": fit_trend(obs["co2_emissions"], "emissions_trend"),
        "poverty_trend": fit_trend(obs["poverty_rate"], "poverty_trend"),
        "hunger_trend": fit_trend(hunger_series, "hunger_trend", extended_with=hunger_prov),
        "water_trend": fit_trend(obs["safe_water_rate"], "water_trend", lack_of=True),
        "sanitation_trend": fit_trend(obs["safe_sanitation_rate"], "sanitation_trend", lack_of=True),
        "forest_trend": fit_trend(obs["forest_share"], "forest_trend"),
    }

    start: dict[str, float] = {}
    start_prov: dict[str, Provenance] = {}

    def carry(key: str, series: str, transform=lambda v: v) -> None:
        year, value = _latest(obs[series])
        start[key] = transform(value)
        note = "" if year == base_year else f", carried forward to {base_year}"
        start_prov[key] = observed(f"{series} in {year}{note}")

    carry("emissions", "co2_emissions", lambda v: v / 1000.0)
    carry("poverty_rate", "poverty_rate")
    carry("water_lack", "safe_water_rate", lambda v: 100.0 - v)
    carry("sanitation_lack", "safe_sanitation_rate", lambda v: 100.0 - v)
    carry("forest_share", "forest_share")
    for key, point in (
        ("warming_prior_year", "warming_2024"),
        ("hunger_rate", "undernourishment_2025"),
        ("wealth_share", "top10_wealth_share_2025"),
        ("amr_2021", "amr_deaths_2021"),
    ):
        start[key], start_prov[key] = points[point]
    conflict, displaced = sourced_series["conflict_deaths"], sourced_series["displaced_total"]
    start["conflict_deaths"], start_prov["conflict_deaths"] = conflict[base_year], series_prov["conflict_deaths"]
    start["displaced"], start_prov["displaced"] = displaced[base_year], series_prov["displaced_total"]

    logs = np.log(np.array([conflict[y] for y in sorted(conflict)]))
    pop_now = obs["population"][max(obs["population"])] / 1e9
    pop_end = points["population_2050"][0]
    population = np.linspace(pop_now, pop_end, horizon - base_year + 1)

    return Calibration(
        base_year=base_year,
        horizon=horizon,
        params=params,
        trends=trends,
        start=start,
        start_provenance=start_prov,
        conflict_log_mean=float(logs.mean()),
        conflict_log_sd=float(logs.std(ddof=1)),
        displaced_history=np.array([displaced[y] for y in sorted(displaced)]),
        population=population,
        scenarios=config["scenarios"],
        context=sourced.get("context", {}),
        observed=obs,
        sourced_series=sourced_series,
    )


# --------------------------------------------------------------------- simulation


@dataclass
class Run:
    """One scenario's Monte Carlo paths. Arrays are ``(draws, years)``."""

    name: str
    levers: dict[str, float]
    years: np.ndarray
    paths: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    grades: dict[str, Grade] = field(repr=False, default_factory=dict)
    pandemic_occurred: np.ndarray | None = None

    def final(self, key: str) -> np.ndarray:
        return self.paths[key][:, -1]

    def quantiles(self, key: str, points=(10, 50, 90)) -> np.ndarray:
        """Percentile paths, shape ``(len(points), years)``."""
        return np.percentile(self.paths[key], points, axis=0)

    def sensitivity(self, key: str, top: int = 4) -> list[tuple[str, float, Grade]]:
        """Inputs ranked by rank correlation with the outcome in the final year."""
        target = _rank(self.final(key))
        if np.ptp(target) == 0:
            return []
        scored = []
        for label, column in self.inputs.items():
            if np.ptp(column) == 0:
                continue
            scored.append((label, float(np.corrcoef(_rank(column), target)[0, 1]), self.grades[label]))
        scored.sort(key=lambda item: -abs(item[1]))
        return scored[:top]

    def spread_by_grade(self, key: str) -> dict[str, float]:
        """Approximate share of final-year spread traced to each kind of input.

        Squared rank correlations, summed by grade. What they leave unexplained
        is chance (conflict shocks, whether a pandemic strikes) and interaction
        between inputs. First-order only: read it as a guide to what the result
        rests on, not as a variance decomposition.
        """
        shares = {grade.value: 0.0 for grade in Grade}
        for _, rho, grade in self.sensitivity(key, top=len(self.inputs)):
            shares[grade.value] += rho**2
        total = sum(shares.values())
        if total > 1.0:
            shares = {k: v / total for k, v in shares.items()}
            total = 1.0
        shares["chance and interaction"] = 1.0 - total
        return shares


def simulate(
    calibration: Calibration,
    levers: dict[str, float] | None = None,
    draws: int = 4000,
    seed: int = 20261128,
    name: str = "run",
) -> Run:
    """Step the coupled system from the base year to the horizon.

    Every input draws from its own stream keyed by its name, so two scenarios
    share identical draws for everything they have in common and differ only
    through their levers.
    """
    levers = dict(levers or {})
    unknown = set(levers) - set(LEVERS)
    if unknown:
        raise ValueError(f"unknown lever(s) {sorted(unknown)}; expected {sorted(LEVERS)}")
    L = {key: float(levers.get(key, 0.0)) for key in LEVERS}

    cal = calibration
    n, steps = draws, len(cal.years)
    inputs: dict[str, np.ndarray] = {}
    grades: dict[str, Grade] = {}
    for label, dist in {**cal.trends, **cal.params}.items():
        if label == "pandemic_deaths":
            continue  # drawn per event below
        inputs[label] = dist.sample(n, _stream(seed, label))
        grades[label] = dist.grade
    p = inputs

    conflict_noise = _stream(seed, "conflict_shocks").normal(size=(n, steps))
    pandemic_roll = _stream(seed, "pandemic_roll").random(size=(n, steps))
    pandemic_size = cal.params["pandemic_deaths"].sample(n * steps, _stream(seed, "pandemic_size")).reshape(n, steps)

    def blank() -> np.ndarray:
        return np.empty((n, steps))

    E, T, C, P, H, A = blank(), blank(), blank(), blank(), blank(), blank()
    F, W, S, D, G, I, PD = blank(), blank(), blank(), blank(), blank(), blank(), blank()

    warming_per_gt = p["tcre"] * p["nonco2_multiplier"] / 1000.0
    s = cal.start
    E[:, 0] = s["emissions"]
    T[:, 0] = s["warming_prior_year"] + warming_per_gt * s["emissions"]
    C[:, 0] = s["conflict_deaths"]
    P[:, 0] = s["poverty_rate"]
    H[:, 0] = s["hunger_rate"]
    A[:, 0] = s["amr_2021"] * np.exp(p["amr_growth"] * (cal.base_year - 2021))
    F[:, 0] = s["forest_share"]
    W[:, 0] = s["water_lack"]
    S[:, 0] = s["sanitation_lack"]
    D[:, 0] = s["displaced"]
    G[:, 0] = 1.0
    I[:, 0] = s["wealth_share"]
    PD[:, 0] = 0.0

    # Conflict reverts to its long-run log level. The shock size is set so the
    # stationary spread matches the spread of the historical series, whatever
    # reversion speed was drawn.
    kappa = p["conflict_reversion"]
    shock_sd = cal.conflict_log_sd * np.sqrt(1.0 - (1.0 - kappa) ** 2)
    conflict_ref = np.exp(cal.conflict_log_mean)

    # New displacement per year at historical conflict levels, backed out of
    # the stock series for the return rate drawn: inflow = D[t] - (1 - r) D[t-1].
    history = cal.displaced_history
    return_rate = p["displacement_return_rate"]
    base_inflow = history[1:].mean() - (1.0 - return_rate) * history[:-1].mean()
    base_inflow = np.maximum(base_inflow, 0.0)
    return_eff = np.clip(return_rate + L["peacebuilding"] * p["peacebuilding_return_effect"], 0.0, 0.5)
    pandemic_factor = np.clip(1.0 - L["health"] * p["health_pandemic_effect"], 0.0, None)

    for t in range(1, steps):
        d_temp = T[:, t - 1] - T[:, 0]
        z_conflict = np.log(C[:, t - 1] / conflict_ref)
        z_hunger = np.log(H[:, t - 1] / H[:, 0])
        z_wealth = I[:, t - 1] - I[:, 0]

        E[:, t] = E[:, t - 1] * np.exp(p["emissions_trend"] - L["mitigation"] * p["mitigation_effect"])
        T[:, t] = T[:, t - 1] + warming_per_gt * E[:, t]

        I[:, t] = np.clip(
            I[:, t - 1] + p["inequality_drift"] - L["redistribution"] * p["redistribution_inequality_effect"], 0.0, 100.0
        )

        struck = pandemic_roll[:, t] < p["pandemic_annual_probability"]
        deaths = np.where(struck, pandemic_size[:, t] * pandemic_factor, 0.0)
        PD[:, t] = PD[:, t - 1] + deaths

        dln_poverty = (
            p["poverty_trend"]
            + p["conflict_to_poverty"] * z_conflict
            + p["climate_to_poverty"] * d_temp
            + p["inequality_to_poverty"] * z_wealth
            - L["development"] * p["development_poverty_effect"]
            - L["redistribution"] * p["redistribution_poverty_effect"]
            + p["pandemic_to_poverty"] * deaths / 10.0
        )
        P[:, t] = np.minimum(P[:, t - 1] * np.exp(dln_poverty), 100.0)

        dln_hunger = (
            p["hunger_trend"]
            + p["poverty_to_hunger"] * (dln_poverty - p["poverty_trend"])
            + p["conflict_to_hunger"] * z_conflict
            + p["climate_to_hunger"] * d_temp
            - L["development"] * p["development_hunger_effect"]
        )
        H[:, t] = np.minimum(H[:, t - 1] * np.exp(dln_hunger), 100.0)

        long_run = (
            cal.conflict_log_mean
            + p["hunger_to_conflict"] * z_hunger
            + p["climate_to_conflict"] * d_temp
            + p["inequality_to_conflict"] * z_wealth
            - L["peacebuilding"] * p["peacebuilding_effect"]
        )
        log_c = np.log(C[:, t - 1])
        C[:, t] = np.exp(log_c + kappa * (long_run - log_c) + shock_sd * conflict_noise[:, t])

        stress = p["conflict_to_water"] * z_conflict + p["climate_to_water"] * d_temp - L["wash"] * p["wash_effect"]
        W[:, t] = np.minimum(W[:, t - 1] * np.exp(p["water_trend"] + stress), 100.0)
        dln_sanitation = p["sanitation_trend"] + stress
        S[:, t] = np.minimum(S[:, t - 1] * np.exp(dln_sanitation), 100.0)

        A[:, t] = A[:, t - 1] * np.exp(
            p["amr_growth"]
            + p["sanitation_to_amr"] * (dln_sanitation - p["sanitation_trend"])
            - L["health"] * p["health_amr_effect"]
        )

        F[:, t] = np.minimum(
            F[:, t - 1]
            * np.exp(p["forest_trend"] - p["climate_to_forest"] * d_temp + L["conservation"] * p["conservation_effect"]),
            100.0,
        )

        inflow = (
            base_inflow
            * (C[:, t] / conflict_ref) ** p["conflict_to_displacement"]
            * (1.0 + p["climate_to_displacement"] * d_temp)
        )
        D[:, t] = D[:, t - 1] * (1.0 - return_eff) + inflow

        G[:, t] = G[:, t - 1] * np.exp(p["ai_gap_growth"] - L["ai_governance"] * p["ai_governance_effect"])

    people = cal.population * 10.0  # percent of billions -> millions
    paths = {
        "temperature": T,
        "emissions": E,
        "conflict_deaths": C,
        "poverty_people": P * people,
        "hunger_people": H * people,
        "amr_deaths": A,
        "pandemic_deaths": PD,
        "forest_share": F,
        "water_people": W * people,
        "sanitation_people": S * people,
        "displaced": D,
        "ai_gap": G,
        "wealth_share": I,
    }
    return Run(name, L, cal.years, paths, inputs, grades, pandemic_occurred=PD[:, -1] > 0)


def run_scenarios(calibration: Calibration, draws: int = 4000, seed: int = 20261128) -> list[Run]:
    return [
        simulate(calibration, sc.get("levers") or {}, draws, seed, name=sc["name"])
        for sc in calibration.scenarios
    ]


def lever_effects(
    calibration: Calibration, baseline: Run, draws: int = 4000, seed: int = 20261128
) -> dict[str, dict[str, float]]:
    """Median percentage change in each final-year outcome from one lever alone.

    Each lever is set to +1 with the rest at zero and compared draw by draw
    against the baseline, which shares its random numbers -- so the difference
    is the lever's effect and nothing else.
    """
    effects: dict[str, dict[str, float]] = {}
    for lever in LEVERS:
        run = simulate(calibration, {lever: 1.0}, draws, seed, name=lever)
        row = {}
        for outcome in OUTCOMES:
            base, new = baseline.final(outcome.key), run.final(outcome.key)
            if outcome.key == "pandemic_deaths":
                hit = base > 0  # undefined in worlds where no pandemic strikes
                row[outcome.key] = float(np.median(new[hit] / base[hit] - 1.0) * 100) if hit.any() else 0.0
            else:
                row[outcome.key] = float(np.median(new / base - 1.0) * 100)
        effects[lever] = row
    return effects


def climate_hindcast(calibration: Calibration, start: int = 1995) -> dict[str, float]:
    """Check the warming equation against a period it was not fitted to.

    Warming implied by observed emissions since ``start`` (central TCRE and
    non-CO2 multiplier) against the fitted GISTEMP trend over the same years.
    """
    obs = calibration.observed
    temps = {y: v for y, v in obs["temperature_anomaly"].items() if y >= start}
    end = min(max(temps), max(obs["co2_emissions"]))
    years = np.array([y for y in sorted(temps) if y <= end])
    slope, _ = _slope(years.astype(float), np.array([temps[y] for y in years]))
    emitted = sum(v for y, v in obs["co2_emissions"].items() if start < y <= end) / 1000.0
    central = calibration.params["tcre"].mode * calibration.params["nonco2_multiplier"].mode
    return {
        "start": start,
        "end": int(end),
        "emitted_gt": emitted,
        "modelled_warming": central * emitted / 1000.0,
        "observed_warming": slope * (end - start),
    }
