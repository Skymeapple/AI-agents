"""Render a simulation as markdown, and as JSON for charting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciscout.provenance import Grade

from .model import LEVERS, OUTCOMES, Calibration, Run, climate_hindcast, lever_effects

# Published projections to hold the model against. Each is a figure the model
# was not calibrated to.
FAO_2030_HUNGER = (510.0, 520.0)


def _fmt(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}"


def _band(run: Run, key: str, digits: int, year_index: int = -1) -> str:
    low, mid, high = np.percentile(run.paths[key][:, year_index], [10, 50, 90])
    return f"**{_fmt(mid, digits)}** ({_fmt(low, digits)} to {_fmt(high, digits)})"


def _signed(value: float) -> str:
    return "·" if abs(value) < 0.5 else f"{value:+.0f}%"


def build_results(calibration: Calibration, runs: list[Run], draws: int, seed: int) -> dict:
    """Everything the report and the charts need, as plain data."""
    baseline = runs[0]
    effects = lever_effects(calibration, baseline, draws, seed)
    history_sources = {
        "emissions": ("co2_emissions", 1e-3),
        "conflict_deaths": ("conflict_deaths", 1.0),
        "displaced": ("displaced_total", 1.0),
        "forest_share": ("forest_share", 1.0),
    }
    population = {y: v / 1e9 for y, v in calibration.observed["population"].items()}
    history: dict[str, dict[int, float]] = {}
    for key, (series, scale) in history_sources.items():
        source = calibration.observed.get(series) or calibration.sourced_series[series]
        history[key] = {y: v * scale for y, v in source.items() if y >= 2000}
    for key, series, lack in (
        ("poverty_people", "poverty_rate", False),
        ("hunger_people", "undernourishment_rate", False),
        ("water_people", "safe_water_rate", True),
        ("sanitation_people", "safe_sanitation_rate", True),
    ):
        history[key] = {
            y: ((100.0 - v) if lack else v) * population[y] * 10.0
            for y, v in calibration.observed[series].items()
            if y >= 2000 and y in population
        }

    outcomes = []
    for o in OUTCOMES:
        outcomes.append(
            {
                "key": o.key,
                "problem": o.problem,
                "label": o.label,
                "unit": o.unit,
                "digits": o.digits,
                "lower_is_better": o.lower_is_better,
                "start": float(np.median(baseline.paths[o.key][:, 0])),
                "history": {str(y): round(v, 4) for y, v in sorted(history.get(o.key, {}).items())},
                "scenarios": {
                    run.name: {
                        name: [round(float(v), 4) for v in row]
                        for name, row in zip(("p10", "p50", "p90"), run.quantiles(o.key))
                    }
                    for run in runs
                },
                "drivers": [
                    {"input": label, "correlation": round(rho, 3), "grade": grade.value}
                    for label, rho, grade in baseline.sensitivity(o.key, top=4)
                    if abs(rho) >= 0.1
                ],
                "spread_by_grade": {k: round(v, 3) for k, v in baseline.spread_by_grade(o.key).items()},
            }
        )
    return {
        "years": [int(y) for y in calibration.years],
        "draws": draws,
        "seed": seed,
        "scenarios": [
            {"name": sc["name"], "description": sc["description"], "levers": sc.get("levers") or {}}
            for sc in calibration.scenarios
        ],
        "levers": LEVERS,
        "outcomes": outcomes,
        "lever_effects": {lever: {k: round(v, 2) for k, v in row.items()} for lever, row in effects.items()},
        "pandemic_probability": float(baseline.pandemic_occurred.mean()),
        "hindcast": climate_hindcast(calibration),
    }


def render(calibration: Calibration, runs: list[Run], results: dict) -> str:
    cal, baseline = calibration, runs[0]
    year = cal.horizon
    ledger = cal.ledger()
    counts = ledger.counts()
    lines: list[str] = []
    add = lines.append

    add(f"# Ten world problems, simulated to {year}")
    add("")
    add(
        f"A coupled model of ten global problems, stepped a year at a time from {cal.base_year} to {year} "
        f"across {results['draws']:,} random draws. Baseline rates of change are fitted from global data "
        "(World Bank, Our World in Data, NASA); levels with no open series are anchored on cited agency "
        "reports; the links between problems and the effect of each policy lever are assumptions."
    )
    add("")
    add(
        f"**This is not a forecast.** Of the {len(ledger.entries)} uncertain inputs, {counts[Grade.OBSERVED]} are fitted "
        f"from data, {counts[Grade.SOURCED]} come from a cited source and {counts[Grade.ASSUMED]} are judgement calls. "
        "The current-trajectory numbers say where measured trends lead if they persist. The lever and scenario numbers "
        "say what the assumptions imply, and are only as good as those assumptions."
    )
    add("")

    add("## Where the world stands")
    add("")
    add("| Problem | Indicator | Latest value | Grade | Basis |")
    add("| --- | --- | --- | --- | --- |")
    start_keys = {
        "temperature": "warming_prior_year",
        "emissions": "emissions",
        "conflict_deaths": "conflict_deaths",
        "poverty_people": "poverty_rate",
        "hunger_people": "hunger_rate",
        "amr_deaths": "amr_2021",
        "forest_share": "forest_share",
        "water_people": "water_lack",
        "sanitation_people": "sanitation_lack",
        "displaced": "displaced",
        "wealth_share": "wealth_share",
    }
    for o in OUTCOMES:
        if o.key not in start_keys:
            continue
        prov = cal.start_provenance[start_keys[o.key]]
        value = float(np.median(baseline.paths[o.key][:, 0]))
        add(f"| {o.problem} | {o.label} | {_fmt(value, o.digits)} {o.unit} | {prov.grade.value} | {prov.detail} |")
    for name, item in cal.context.items():
        source = item["source"]
        add(
            f"| Context | {name.replace('_', ' ')} | {item['value']:,} {item['unit']} | {source['grade']} | "
            f"{source['detail']} |"
        )
    add("")
    add(
        "People counts for the base year are the latest observed rate applied to the 2025 population, so they "
        "differ slightly from agency headline counts."
    )
    add("")

    add(f"## Three scenarios in {year}")
    add("")
    for sc in results["scenarios"]:
        add(f"- **{sc['name']}**: {sc['description']}")
    add("")
    add("Median, with the 10th to 90th percentile range in brackets.")
    add("")
    add("| Indicator | " + f"{cal.base_year} | " + " | ".join(run.name for run in runs) + " |")
    add("| --- | --- | " + " | ".join("---" for _ in runs) + " |")
    for o in OUTCOMES:
        if o.key == "pandemic_deaths":
            continue
        start = _fmt(float(np.median(baseline.paths[o.key][:, 0])), o.digits)
        add(f"| {o.label} ({o.unit}) | {start} | " + " | ".join(_band(run, o.key, o.digits) for run in runs) + " |")
    add("")
    hit = baseline.final("pandemic_deaths")
    add(
        f"**Pandemics** are drawn as chance events. A pandemic on the scale of COVID-19 strikes at least once by "
        f"{year} in {results['pandemic_probability']:.0%} of draws. Where one does, median deaths are "
        f"{np.median(hit[hit > 0]):.0f} million on the current trajectory and "
        f"{np.median(runs[1].final('pandemic_deaths')[hit > 0]):.0f} million under {runs[1].name}."
    )
    add("")

    add("## Where the model disagrees with published projections")
    add("")
    i2030 = list(cal.years).index(2030)
    low, mid, high = np.percentile(baseline.paths["hunger_people"][:, i2030], [10, 50, 90])
    add(
        f"- **Hunger in 2030.** The model has {mid:,.0f} million undernourished ({low:,.0f} to {high:,.0f}); FAO projects "
        f"{FAO_2030_HUNGER[0]:,.0f} to {FAO_2030_HUNGER[1]:,.0f} million. The model is more pessimistic because the "
        "typical annual change over the last ten years is close to flat, and it then adds assumed climate and conflict "
        "pressure. Hunger is the result most sensitive to which years the trend is fitted on."
    )
    h = results["hindcast"]
    add(
        f"- **Warming, checked against the past.** Fed observed emissions for {h['start']}-{h['end']} "
        f"({h['emitted_gt']:,.0f} GtCO2), the warming equation gives {h['modelled_warming']:.2f} degC. The fitted NASA "
        f"temperature trend over those years is {h['observed_warming']:.2f} degC. The central setting therefore "
        f"under-predicts by {1 - h['modelled_warming'] / h['observed_warming']:.0%}, inside the sampled range but worth knowing."
    )
    add(
        "- **Emissions.** Extending the recent trend has emissions still rising in 2050. Published current-policy "
        "pathways have them flat or falling, because they count policies not yet visible in the data."
    )
    add("")

    add("## What each lever moves")
    add("")
    add(
        f"Each lever is pushed to full extra effort on its own and compared draw by draw with the current trajectory. "
        f"Cells are the median change in {year}; a dot is under half a percent."
    )
    add("")
    shown = [o for o in OUTCOMES if o.key != "emissions"]
    add("| Lever | " + " | ".join(o.label for o in shown) + " | Indicators moved 5%+ |")
    add("| --- | " + " | ".join("---" for _ in shown) + " | --- |")
    effects = results["lever_effects"]
    breadth = {
        lever: sum(1 for o in shown if abs(effects[lever][o.key]) >= 5.0) for lever in LEVERS
    }
    for lever in sorted(LEVERS, key=lambda k: -breadth[k]):
        add(
            f"| {LEVERS[lever]} | "
            + " | ".join(_signed(effects[lever][o.key]) for o in shown)
            + f" | {breadth[lever]} |"
        )
    add("")
    add(
        "The breadth ranking is a property of the assumed links, not a finding about the world: a lever reaches other "
        "problems only through couplings whose sizes are judgement calls. What it does show is which claims about "
        "cross-cutting leverage the assumptions support and which they do not. Forest cover moves slowly, so the "
        "ecosystem lever falls under the 5% bar even though it reverses twenty-five years of loss."
    )
    add("")

    add("## What each result rests on")
    add("")
    add(
        "Approximate share of the spread in each current-trajectory result traced to inputs of each grade, and the "
        "inputs that matter most. Chance covers conflict shocks and whether a pandemic strikes."
    )
    add("")
    add("| Indicator | Observed | Sourced | Assumed | Chance and interaction | Main drivers |")
    add("| --- | --- | --- | --- | --- | --- |")
    for item in results["outcomes"]:
        share = item["spread_by_grade"]
        drivers = ", ".join(f"{d['input']} ({d['grade']})" for d in item["drivers"][:3]) or "chance"
        add(
            f"| {item['label']} | {share['observed']:.0%} | {share['sourced']:.0%} | {share['assumed']:.0%} | "
            f"{share['chance and interaction']:.0%} | {drivers} |"
        )
    add("")

    add("## Limitations")
    add("")
    for text in (
        f"**The couplings are assumed.** {counts[Grade.ASSUMED]} of the {len(ledger.entries)} inputs are judgement "
        "calls, including every link between problems and every lever effect. Change them in "
        "`worldsim/params.yaml` and rerun.",
        "**Trends are extrapolated.** A rate measured over ten or twenty years is carried for twenty-five. Nothing in "
        "the model knows about policies adopted but not yet visible in the data.",
        "**Two problems have no usable series.** The AI oversight gap and the wealth-share drift are placeholders that "
        "show a direction, and their results are 100% assumption.",
        "**Forest cover stands in for biodiversity.** It is the only global habitat series available here. It says "
        "nothing about species loss inside standing forest, or about oceans.",
        "**Global averages hide the problem.** Poverty, hunger and unsafe water are concentrated in fragile states; a "
        "world model cannot say where.",
        "**Levers have no cost.** The model says what an effort would move, not what it would take to mount.",
        "**Some history is recalled, not ingested.** Conflict and displacement totals before 2024 are approximate "
        "report figures flagged for review in `data/world/sourced.yaml`. They set the spread of the conflict process.",
        "**Nuclear risk is not simulated.** It is a low-probability catastrophe with no base rate to fit.",
    ):
        add(f"- {text}")
    add("")

    add("## Every uncertain input")
    add("")
    add(ledger.audit_table())
    add("")
    return "\n".join(lines)


def write(calibration: Calibration, runs: list[Run], draws: int, seed: int, report_path: Path, json_path: Path) -> dict:
    results = build_results(calibration, runs, draws, seed)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(calibration, runs, results))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=1) + "\n")
    return results
