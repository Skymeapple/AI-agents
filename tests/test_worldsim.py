"""The world model.

These pin the modelling claims the report leans on: that a number cannot enter
without a source, that a one-off shock is not read as a trend, that scenarios
differ only through their levers, and that the one equation we can check against
history is roughly right.
"""

import csv
import json
import textwrap

import numpy as np
import pytest
import yaml

from sciscout.provenance import Grade, sourced
from worldsim.model import (
    DATA_DIR,
    LEVERS,
    OUTCOMES,
    PACKAGE_DIR,
    climate_hindcast,
    fit_trend,
    lever_effects,
    load_calibration,
    run_scenarios,
    simulate,
)

DRAWS = 800


@pytest.fixture(scope="module")
def cal():
    return load_calibration()


@pytest.fixture(scope="module")
def runs(cal):
    return run_scenarios(cal, draws=DRAWS)


def test_every_fetched_series_is_in_the_manifest():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    with (DATA_DIR / "observed.csv").open() as handle:
        names = {row["series"] for row in csv.DictReader(handle)}
    assert names == set(manifest)
    assert all(entry["url"].startswith("https://") and entry["retrieved"] for entry in manifest.values())


def test_every_report_figure_names_its_source():
    raw = yaml.safe_load((DATA_DIR / "sourced.yaml").read_text())
    for section in ("series", "points", "context"):
        for name, item in raw[section].items():
            assert item["source"]["grade"] == "sourced", name
            assert item["source"].get("reference"), name


def test_parameter_without_a_source_is_refused(tmp_path):
    config = yaml.safe_load((PACKAGE_DIR / "params.yaml").read_text())
    del config["parameters"]["tcre"]["source"]
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="source"):
        load_calibration(params_path=path)


def test_misspelled_parameter_field_is_refused(tmp_path):
    config = yaml.safe_load((PACKAGE_DIR / "params.yaml").read_text())
    config["parameters"]["tcre"]["hgih"] = 0.7
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hgih"):
        load_calibration(params_path=path)


def test_unknown_lever_is_refused(cal):
    with pytest.raises(ValueError, match="unknown lever"):
        simulate(cal, {"mitigaton": 1.0}, draws=10)


def test_a_one_off_shock_is_not_read_as_a_trend():
    # A steady 2% a year decline with one 10% jump, the shape the pandemic left
    # in the undernourishment series. A least-squares line through the last ten
    # years of this reports a flat or rising trend.
    points, value = {}, 20.0
    for year in range(2000, 2025):
        value *= np.exp(0.10 if year == 2020 else -0.02)
        points[year] = value
    trend = fit_trend(points, "stepped")
    assert trend.mode == pytest.approx(-0.02, abs=1e-9)


def test_trend_extended_with_a_report_figure_takes_the_weaker_grade():
    points = {year: 10.0 * 0.98 ** (year - 2000) for year in range(2000, 2026)}
    assert fit_trend(points, "t").grade is Grade.OBSERVED
    extended = fit_trend(points, "t", extended_with=sourced("a report", "the report"))
    assert extended.grade is Grade.SOURCED


def test_hunger_trend_is_not_graded_observed(cal):
    # Its last point is a report figure, so the fit is only as good as that.
    assert cal.trends["hunger_trend"].grade is Grade.SOURCED


def test_same_seed_reproduces_exactly(cal):
    a = simulate(cal, {}, draws=200, seed=7)
    b = simulate(cal, {}, draws=200, seed=7)
    for outcome in OUTCOMES:
        assert np.array_equal(a.paths[outcome.key], b.paths[outcome.key])


def test_a_lever_moves_only_what_it_is_linked_to(cal):
    base = simulate(cal, {}, draws=DRAWS)
    cut = simulate(cal, {"mitigation": 1.0}, draws=DRAWS)
    # Unlinked outcomes are bit-identical: the difference between scenarios is
    # the lever, never resampling noise.
    for key in ("wealth_share", "ai_gap", "pandemic_deaths"):
        assert np.array_equal(base.paths[key], cut.paths[key]), key
    assert (cut.final("temperature") < base.final("temperature")).all()
    # Linked, but only through a chain: warming -> sanitation -> infections.
    assert (cut.final("amr_deaths") <= base.final("amr_deaths")).all()
    # Warming feeds poverty in every draw where the coupling is non-zero.
    assert (cut.final("poverty_people") <= base.final("poverty_people")).all()


def test_cooperation_beats_and_fragmentation_trails_the_current_trajectory(runs):
    current, cooperation, fragmentation = runs
    for outcome in OUTCOMES:
        if outcome.key == "pandemic_deaths":
            continue  # median is zero: most draws have no pandemic
        mid = [np.median(run.final(outcome.key)) for run in (cooperation, current, fragmentation)]
        if not outcome.lower_is_better:
            mid.reverse()
        assert mid[0] < mid[1] < mid[2], outcome.key


def test_health_lever_cuts_pandemic_deaths_without_changing_whether_one_strikes(cal):
    base = simulate(cal, {}, draws=DRAWS)
    safer = simulate(cal, {"health": 1.0}, draws=DRAWS)
    assert np.array_equal(base.pandemic_occurred, safer.pandemic_occurred)
    hit = base.pandemic_occurred
    assert hit.any()
    assert (safer.final("pandemic_deaths")[hit] < base.final("pandemic_deaths")[hit]).all()


def test_pandemic_frequency_matches_the_annual_probability(cal):
    run = simulate(cal, {}, draws=4000)
    p = cal.params["pandemic_annual_probability"].mean()
    expected = 1.0 - (1.0 - p) ** (len(cal.years) - 1)
    assert run.pandemic_occurred.mean() == pytest.approx(expected, abs=0.04)


def test_warming_equation_is_within_a_third_of_observed_history(cal):
    check = climate_hindcast(cal)
    ratio = check["modelled_warming"] / check["observed_warming"]
    assert 0.67 < ratio < 1.33


def test_warming_spread_rests_mostly_on_the_sourced_climate_sensitivity(runs):
    shares = runs[0].spread_by_grade("temperature")
    assert shares["sourced"] > 0.5
    assert sum(shares.values()) == pytest.approx(1.0)


def test_placeholder_problems_report_themselves_as_pure_assumption(runs):
    for key in ("ai_gap", "wealth_share"):
        assert runs[0].spread_by_grade(key)["assumed"] > 0.95, key


def test_lever_effects_show_peacebuilding_reaching_displacement(cal, runs):
    effects = lever_effects(cal, runs[0], draws=DRAWS)
    assert set(effects) == set(LEVERS)
    assert effects["peacebuilding"]["displaced"] < -20
    assert effects["ai_governance"]["displaced"] == 0.0
