"""Scenario loading and comparison.

The loader is the boundary where an unfounded number would enter the system, so
most of these tests are about what it refuses.
"""

import textwrap

import numpy as np
import pytest

from sciscout.commercial.profile import PriorLibrary
from sciscout.models import Track
from sciscout.provenance import Grade, parse_provenance
from sciscout.scenarios import ScenarioSet, render_comparison


def write(tmp_path, body: str):
    path = tmp_path / "scenarios.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def profile():
    return PriorLibrary().profile(
        Track(id="solid-state-sodium", name="Sodium", sector="energy_hardware", trl_override=4)
    )


MINIMAL = """
    track: solid-state-sodium
    scenarios:
      - name: Base
        market:
          tam_usd:
            median: 10.0e9
            spread: 3.0
            source: {grade: assumed, detail: illustrative}
    """


# -- provenance enforcement ------------------------------------------------


def test_a_value_without_a_source_block_is_rejected():
    """The whole point: a config file cannot smuggle in an ungrounded number."""
    with pytest.raises(ValueError, match="needs a 'source' block"):
        parse_provenance(None, "tam_usd")


def test_a_sourced_value_without_a_reference_is_rejected(tmp_path):
    path = write(
        tmp_path,
        """
        track: t
        scenarios:
          - name: Base
            market:
              tam_usd:
                median: 1.0e9
                spread: 2.0
                source: {grade: sourced, detail: from a study}
        """,
    )
    with pytest.raises(ValueError, match="must name its reference"):
        ScenarioSet.from_file(path)


def test_an_unknown_provenance_grade_is_rejected():
    with pytest.raises(ValueError, match="unknown provenance grade"):
        parse_provenance({"grade": "probably-fine", "detail": "x"}, "ctx")


def test_provenance_without_detail_is_rejected():
    with pytest.raises(ValueError, match="needs a 'detail'"):
        parse_provenance({"grade": "assumed"}, "ctx")


def test_assumed_values_are_flagged_for_review_by_default():
    assert parse_provenance({"grade": "assumed", "detail": "x"}, "c").needs_review
    assert not parse_provenance({"grade": "observed", "detail": "x"}, "c").needs_review


# -- structural validation -------------------------------------------------


def test_a_misspelled_market_field_is_rejected(tmp_path):
    """Silently ignoring it would leave you believing you had set something."""
    path = write(
        tmp_path,
        """
        track: t
        scenarios:
          - name: Base
            market:
              total_market:
                median: 1.0e9
                spread: 2.0
                source: {grade: assumed, detail: x}
        """,
    )
    with pytest.raises(ValueError, match="unknown market field"):
        ScenarioSet.from_file(path)


def test_an_override_of_a_stage_that_is_not_ahead_is_rejected(tmp_path):
    path = write(
        tmp_path,
        """
        track: solid-state-sodium
        scenarios:
          - name: Base
            stages:
              Nonexistent stage:
                cost_usd:
                  low: 1.0e6
                  mode: 2.0e6
                  high: 3.0e6
                  source: {grade: assumed, detail: x}
        """,
    )
    scenario_set = ScenarioSet.from_file(path)
    with pytest.raises(ValueError, match="not in this track's remaining path"):
        scenario_set.scenarios[0].apply_to(profile())


def test_duplicate_scenario_names_are_rejected(tmp_path):
    path = write(
        tmp_path,
        """
        track: t
        scenarios:
          - name: Base
          - name: Base
        """,
    )
    with pytest.raises(ValueError, match="duplicate scenario name"):
        ScenarioSet.from_file(path)


def test_a_file_without_a_track_is_rejected(tmp_path):
    path = write(tmp_path, "scenarios:\n  - name: Base\n")
    with pytest.raises(ValueError, match="needs a 'track'"):
        ScenarioSet.from_file(path)


def test_a_lognormal_missing_its_parameters_is_rejected(tmp_path):
    path = write(
        tmp_path,
        """
        track: t
        scenarios:
          - name: Base
            market:
              tam_usd:
                median: 1.0e9
                source: {grade: assumed, detail: x}
        """,
    )
    with pytest.raises(ValueError, match="needs 'median' and 'spread'"):
        ScenarioSet.from_file(path)


# -- behaviour -------------------------------------------------------------


def test_scenarios_share_a_seed_so_differences_are_not_noise(tmp_path):
    """Two identical scenarios must produce identical results.

    If they did not, any comparison between different scenarios would be
    confounded by sampling noise, which with heavy-tailed market distributions
    is large enough to reverse a ranking.
    """
    path = write(
        tmp_path,
        """
        track: solid-state-sodium
        scenarios:
          - name: A
            market:
              tam_usd: {median: 8.0e9, spread: 3.0, source: {grade: assumed, detail: x}}
          - name: B
            market:
              tam_usd: {median: 8.0e9, spread: 3.0, source: {grade: assumed, detail: x}}
        """,
    )
    outcomes = ScenarioSet.from_file(path).run(profile(), draws=4_000)
    assert np.array_equal(outcomes[0].result.project_npv, outcomes[1].result.project_npv)


def test_a_larger_market_scenario_produces_a_better_case(tmp_path):
    path = write(
        tmp_path,
        """
        track: solid-state-sodium
        scenarios:
          - name: Small
            market:
              tam_usd: {median: 2.0e9, spread: 2.5, source: {grade: assumed, detail: x}}
          - name: Large
            market:
              tam_usd: {median: 40.0e9, spread: 2.5, source: {grade: assumed, detail: x}}
        """,
    )
    small, large = ScenarioSet.from_file(path).run(profile(), draws=8_000)
    assert large.result.expected_moic() > small.result.expected_moic()


def test_a_stage_cost_override_actually_changes_spend(tmp_path):
    """Guards against an override that parses but never reaches the engine."""
    path = write(
        tmp_path,
        """
        track: solid-state-sodium
        scenarios:
          - name: Cheap
            stages:
              Pilot and scale-up:
                cost_usd:
                  low: 5.0e6
                  mode: 8.0e6
                  high: 12.0e6
                  source: {grade: assumed, detail: x}
          - name: Expensive
            stages:
              Pilot and scale-up:
                cost_usd:
                  low: 400.0e6
                  mode: 600.0e6
                  high: 900.0e6
                  source: {grade: assumed, detail: x}
        """,
    )
    cheap, expensive = ScenarioSet.from_file(path).run(profile(), draws=6_000)

    # Isolate the overridden stage: compare draws that died at that gate, which
    # paid that stage's cost and nothing else. Comparing total means instead
    # would be diluted by the untouched downstream stage.
    def first_stage_spend(outcome):
        died = outcome.result.failure_stage == 0
        return outcome.result.total_spend[died].mean()

    assert first_stage_spend(expensive) > 10 * first_stage_spend(cheap)
    assert expensive.result.total_spend.mean() > cheap.result.total_spend.mean()


def test_overriding_cost_alone_leaves_technical_risk_untouched(tmp_path):
    """Money and physics are separate levers, and must stay separate."""
    path = write(
        tmp_path,
        """
        track: solid-state-sodium
        scenarios:
          - name: Cheap
            stages:
              Pilot and scale-up:
                cost_usd:
                  low: 5.0e6
                  mode: 8.0e6
                  high: 12.0e6
                  source: {grade: assumed, detail: x}
          - name: Default
        """,
    )
    cheap, default = ScenarioSet.from_file(path).run(profile(), draws=6_000)
    assert cheap.result.success_rate == pytest.approx(default.result.success_rate)


def test_scenarios_report_their_assumption_share(tmp_path):
    path = write(tmp_path, MINIMAL)
    outcome = ScenarioSet.from_file(path).run(profile(), draws=4_000)[0]
    assert outcome.assumption_share == pytest.approx(1.0, abs=1e-9)


def test_comparison_renders_every_scenario(tmp_path):
    path = write(tmp_path, MINIMAL)
    outcomes = ScenarioSet.from_file(path).run(profile(), draws=2_000)
    markdown = render_comparison(outcomes, "Sodium")
    assert "Sodium" in markdown and "Base" in markdown
    assert "Assumption-driven" in markdown


def test_the_shipped_example_file_loads_and_runs():
    """The worked example must stay runnable, or it is documentation that lies."""
    scenario_set = ScenarioSet.from_file("examples/sodium-battery-scenarios.yaml")
    assert scenario_set.track_id == "solid-state-sodium"
    assert len(scenario_set.scenarios) == 3
    outcomes = scenario_set.run(profile(), draws=2_000)
    assert len(outcomes) == 3
    # The conservative case must not out-earn the breakthrough case.
    by_name = {o.scenario.name: o.result.expected_moic() for o in outcomes}
    assert by_name["Conservative"] < by_name["Breakthrough"]
