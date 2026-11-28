"""The forecast ledger: what it must refuse, and that its scores are right."""

import datetime as dt
import json
import math

import pytest
import yaml

from nexus import ledger
from nexus.ledger import LedgerError
from sciscout.provenance import Grade

D = dt.date


def raw_binary(**changes):
    entry = {
        "id": "t-binary", "statement": "It happens", "project": "test",
        "source": {"file": "README.md", "locator": "top", "anchor": "sciscout"},
        "made": "2026-01-01", "resolves": "2026-12-31", "kind": "binary", "probability": 0.7,
        "provenance": {"grade": "assumed", "detail": "a guess"},
        "resolution_rule": "Yes if the registry lists it.",
    }
    entry.update(changes)
    return {k: v for k, v in entry.items() if v is not None}


def raw_quantiles(**changes):
    entry = {"id": "t-band", "kind": "quantiles", "probability": None,
             "quantiles": {0.1: 10.0, 0.5: 15.0, 0.9: 20.0}}
    entry.update(changes)
    return raw_binary(**entry)


# -- the shipped ledger ----------------------------------------------------------


def test_shipped_ledger_loads_and_every_entry_is_complete():
    forecasts = ledger.load_forecasts()
    assert len(forecasts) >= 60
    assert len({f.id for f in forecasts}) == len(forecasts)
    for f in forecasts:
        assert f.resolution_rule and f.statement and f.resolves >= f.made


def test_every_repository_forecast_still_matches_its_source_text():
    forecasts = ledger.load_forecasts()
    in_repo = [f for f in forecasts if f.source.in_repository]
    assert in_repo
    assert [m for m in map(ledger.check_anchor, in_repo) if m] == []


def test_worldsim_entries_equal_the_simulation_output():
    sim = json.loads((ledger.ROOT / "data/world/simulation.json").read_text())
    index = sim["years"].index(2030)
    outcomes = {o["key"]: o["scenarios"]["Current trajectory"] for o in sim["outcomes"]}
    checked = 0
    for f in ledger.load_forecasts():
        if f.project != "worldsim" or f.kind != "quantiles":
            continue
        key = f.id.removeprefix("worldsim-").removesuffix("-2030").replace("-", "_")
        for level, name in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
            assert f.quantiles[level] == outcomes[key][name][index]
        assert f.conditional  # the project calls these projections, not forecasts
        checked += 1
    assert checked == 11


def test_session_imports_are_marked_and_never_graded_above_assumed():
    imported = [f for f in ledger.load_forecasts() if not f.source.in_repository]
    assert imported
    for f in imported:
        assert f.source.note and f.grade is Grade.ASSUMED
        assert ledger.check_anchor(f) is None


def test_shipped_resolutions_are_valid_and_none_is_invented():
    forecasts = ledger.load_forecasts()
    resolutions = ledger.load_resolutions(forecasts)
    for r in resolutions.values():
        assert r.source


# -- what loading refuses --------------------------------------------------------


@pytest.mark.parametrize("field", ["resolution_rule", "source", "provenance", "statement", "resolves"])
def test_forecast_missing_a_required_part_is_rejected(field):
    with pytest.raises(LedgerError):
        ledger.parse_forecast(raw_binary(**{field: None}))


def test_file_source_without_anchor_and_session_source_without_note_are_rejected():
    with pytest.raises(LedgerError):
        ledger.parse_forecast(raw_binary(source={"file": "README.md", "locator": "top"}))
    with pytest.raises(LedgerError):
        ledger.parse_forecast(raw_binary(source={"session": "some chat"}))


@pytest.mark.parametrize("p", [0, 1, 1.2, "likely"])
def test_probability_must_be_strictly_inside_zero_and_one(p):
    with pytest.raises(LedgerError):
        ledger.parse_forecast(raw_binary(probability=p))


def test_sourced_forecast_must_name_a_reference():
    with pytest.raises(LedgerError):
        ledger.parse_forecast(raw_binary(provenance={"grade": "sourced", "detail": "a report"}))


def test_bad_quantiles_are_rejected():
    for q in ({0.1: 5, 0.9: 4}, {0.1: 1, 0.8: 2}, {0.5: 3}, {0.1: 2, 0.9: 2}):
        with pytest.raises(LedgerError):
            ledger.parse_forecast(raw_quantiles(quantiles=q))


def test_changed_source_text_is_reported_as_untraceable(tmp_path):
    (tmp_path / "doc.md").write_text("the milestone has a 70% chance")
    source = {"file": "doc.md", "locator": "line 1", "anchor": "70% chance"}
    f = ledger.parse_forecast(raw_binary(source=source))
    assert ledger.check_anchor(f, tmp_path) is None
    (tmp_path / "doc.md").write_text("the milestone has a 40% chance")
    assert "anchor text not found" in ledger.check_anchor(f, tmp_path)
    (tmp_path / "doc.md").unlink()
    assert "does not exist" in ledger.check_anchor(f, tmp_path)


# -- what resolving refuses ------------------------------------------------------


def test_resolution_without_source_or_date_is_rejected():
    f = ledger.parse_forecast(raw_binary())
    by_id = {f.id: f}
    with pytest.raises(LedgerError, match="where it was read"):
        ledger.parse_resolution({"id": f.id, "outcome": "yes", "resolved_on": "2026-06-01"}, by_id)
    with pytest.raises(LedgerError, match="where it was read"):
        ledger.parse_resolution(
            {"id": f.id, "outcome": "yes", "source": "  ", "resolved_on": "2026-06-01"}, by_id)
    with pytest.raises(LedgerError, match="ISO date"):
        ledger.parse_resolution({"id": f.id, "outcome": "yes", "source": "registry"}, by_id)
    with pytest.raises(LedgerError, match="no such forecast"):
        ledger.parse_resolution(
            {"id": "nope", "outcome": "yes", "source": "registry", "resolved_on": "2026-06-01"}, by_id)


def test_no_cannot_be_recorded_before_the_deadline_but_yes_can():
    f = ledger.parse_forecast(raw_binary())
    by_id = {f.id: f}
    early = {"id": f.id, "source": "registry", "resolved_on": "2026-06-01"}
    with pytest.raises(LedgerError, match="may still happen"):
        ledger.parse_resolution({**early, "outcome": "no"}, by_id)
    assert ledger.parse_resolution({**early, "outcome": "yes"}, by_id).outcome is True
    late = {**early, "resolved_on": "2027-01-02", "outcome": "no"}
    assert ledger.parse_resolution(late, by_id).outcome is False


def test_outcome_type_must_match_the_forecast_kind():
    b = ledger.parse_forecast(raw_binary())
    q = ledger.parse_forecast(raw_quantiles())
    by_id = {b.id: b, q.id: q}
    base = {"source": "registry", "resolved_on": "2027-01-02"}
    with pytest.raises(LedgerError):
        ledger.parse_resolution({**base, "id": b.id, "outcome": 3.2}, by_id)
    with pytest.raises(LedgerError):
        ledger.parse_resolution({**base, "id": q.id, "outcome": "yes"}, by_id)
    with pytest.raises(LedgerError, match="void resolution needs a note"):
        ledger.parse_resolution({**base, "id": b.id, "outcome": "void"}, by_id)


def test_record_resolution_writes_once_and_refuses_a_second(tmp_path):
    f = ledger.parse_forecast(raw_binary())
    path = tmp_path / "resolutions.yaml"
    path.write_text("# kept comment\n[]\n")
    raw = {"id": f.id, "outcome": "yes", "source": "registry page", "resolved_on": "2026-07-01"}
    ledger.record_resolution(raw, [f], path)
    assert path.read_text().startswith("# kept comment\n")
    assert yaml.safe_load(path.read_text())[0]["source"] == "registry page"
    assert ledger.load_resolutions([f], path)[f.id].outcome is True
    with pytest.raises(LedgerError, match="already resolved"):
        ledger.record_resolution(raw, [f], path)


# -- scores, against hand-computed values ----------------------------------------


def test_brier_and_log_score_by_hand():
    assert ledger.brier(0.7, True) == pytest.approx(0.09)
    assert ledger.brier(0.7, False) == pytest.approx(0.49)
    assert ledger.brier(0.5, True) == pytest.approx(0.25)
    assert ledger.log_score(0.7, True) == pytest.approx(-math.log(0.7))
    assert ledger.log_score(0.7, False) == pytest.approx(-math.log(0.3))
    assert ledger.log_score(0.5, False) == pytest.approx(math.log(2))


def test_interval_score_by_hand():
    # 80% band [10, 20]: alpha 0.2, so each unit of miss costs 2 / 0.2 = 10.
    assert ledger.interval_score(10, 20, 0.8, 15) == pytest.approx(10)
    assert ledger.interval_score(10, 20, 0.8, 22) == pytest.approx(10 + 10 * 2)
    assert ledger.interval_score(10, 20, 0.8, 7) == pytest.approx(10 + 10 * 3)
    # A wider band that still misses pays for the width as well.
    assert ledger.interval_score(0, 30, 0.8, 15) > ledger.interval_score(10, 20, 0.8, 15)


def test_pinball_by_hand():
    q = {0.1: 10.0, 0.5: 15.0, 0.9: 20.0}
    # outcome 22: 0.1*12 + 0.5*7 + 0.9*2 = 6.5, mean 6.5/3
    assert ledger.pinball(q, 22.0) == pytest.approx(6.5 / 3)
    # outcome 15: 0.1*5 + 0 + 0.1*5 = 1.0, mean 1/3
    assert ledger.pinball(q, 15.0) == pytest.approx(1.0 / 3)


def test_wilson_matches_known_value_and_is_wide_for_small_samples():
    lo, hi = ledger.wilson(3, 3)
    assert hi == 1.0 and lo == pytest.approx(0.4385, abs=1e-3)
    assert ledger.wilson(0, 0) == (0.0, 1.0)


def test_scorecard_counts_and_means():
    forecasts = [
        ledger.parse_forecast(raw_binary(id="a", probability=0.7)),
        ledger.parse_forecast(raw_binary(id="b", probability=0.2, project="other")),
        ledger.parse_forecast(raw_binary(id="c", probability=0.9)),
        ledger.parse_forecast(raw_quantiles(id="d")),
        ledger.parse_forecast(raw_quantiles(id="e")),
        ledger.parse_forecast(raw_binary(id="f", probability=0.6)),
    ]
    by_id = {f.id: f for f in forecasts}
    base = {"source": "registry", "resolved_on": "2027-01-02"}
    raws = [{"id": "a", "outcome": "yes"}, {"id": "b", "outcome": "yes"},
            {"id": "d", "outcome": 15}, {"id": "e", "outcome": 22},
            {"id": "f", "outcome": "void", "note": "the registry closed"}]
    resolutions = {r["id"]: ledger.parse_resolution({**base, **r}, by_id) for r in raws}
    cards = ledger.score(forecasts, resolutions)

    overall = cards["all"]["all forecasts"]
    assert (overall.open, overall.void, overall.resolved) == (1, 1, 4)
    assert overall.mean_brier == pytest.approx((0.09 + 0.64) / 2)
    hits, n, _ = overall.hit_rate()
    assert (hits, n) == (1, 2)
    covered, n, interval, nominal = overall.coverage()
    assert (covered, n) == (1, 2) and nominal == pytest.approx(0.8)
    assert interval[0] < 0.5 < interval[1]
    assert overall.mean_interval_relative == pytest.approx((1.0 + 3.0) / 2)
    assert overall.thin

    assert cards["project"]["other"].mean_brier == pytest.approx(0.64)
    assert cards["project"]["test"].open == 1
    assert cards["grade"]["assumed"].resolved == 4


def test_unresolved_and_void_forecasts_have_no_score():
    f = ledger.parse_forecast(raw_binary())
    with pytest.raises(LedgerError, match="not resolved"):
        ledger.score_forecast(f, None)
    void = ledger.parse_resolution(
        {"id": f.id, "outcome": "void", "note": "ambiguous", "source": "x",
         "resolved_on": "2027-01-02"}, {f.id: f})
    with pytest.raises(LedgerError, match="void"):
        ledger.score_forecast(f, void)

    cards = ledger.score([f], {})
    card = cards["all"]["all forecasts"]
    assert card.resolved == 0 and card.open == 1
    assert card.mean_brier is None and card.mean_log is None
    assert card.hit_rate() is None and card.coverage() is None


def test_report_says_there_are_no_scores_when_nothing_is_resolved():
    f = ledger.parse_forecast(raw_binary())
    text = ledger.render([f], {}, D(2027, 1, 5))
    assert "No forecast has been resolved, so there are no scores" in text
    assert "nothing to score" in text
    assert "## Resolved" not in text
    assert "| 2026-12-31 | It happens |" in text  # listed as past its deadline


def test_due_lists_only_unresolved_forecasts_past_their_deadline():
    a = ledger.parse_forecast(raw_binary(id="a", resolves="2026-06-30"))
    b = ledger.parse_forecast(raw_binary(id="b", resolves="2026-12-31"))
    c = ledger.parse_forecast(raw_binary(id="c", resolves="2026-03-31"))
    done = ledger.parse_resolution(
        {"id": "c", "outcome": "no", "source": "registry", "resolved_on": "2026-04-02"}, {"c": c})
    assert [f.id for f in ledger.due([a, b, c], {"c": done}, D(2026, 7, 1))] == ["a"]
    assert [f.id for f in ledger.upcoming([a, b, c], {"c": done}, D(2026, 7, 1), 365)] == ["b"]


def test_cli_prints_the_report_offline(capsys):
    from nexus.cli import main

    assert main(["ledger", "--asof", "2026-11-28"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Forecast ledger")
    assert main(["ledger", "score"]) == 0
    assert "nothing to score" in capsys.readouterr().out
    assert main(["ledger", "resolve", "--id", "missing", "--outcome", "yes",
                 "--source", "x", "--date", "2026-12-01"]) == 2
