"""nexus.impact: the claims the impact pricing makes, asserted directly."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import yaml

from nexus.impact import (DEFAULT_MAP, ImpactMap, ImpactPricer, analyse, effective_lever,
                          register, render)
from sciscout.models import Track, Work
from sciscout.pipeline import Pipeline
from sciscout.provenance import Grade
from sciscout.sources.local import LocalCorpus
from worldsim.model import load_calibration, simulate

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "demo_corpus.json"
DRAWS, SEED, ASOF = 300, 7, dt.date(2026, 11, 28)
KEEP = {"solid-state-sodium", "car-t-solid-tumour"}


@pytest.fixture(scope="module")
def calibration():
    return load_calibration()


@pytest.fixture(scope="module")
def reports():
    works = [w for w in LocalCorpus(CORPUS).harvest().works if w.extra.get("track") in KEEP]
    pipeline = Pipeline(asof=ASOF, draws=DRAWS, seed=SEED)
    return {r.track.id: r for r in pipeline.analyse(pipeline.tracks_from_works(works))}


def _map(size) -> ImpactMap:
    raw = yaml.safe_load(DEFAULT_MAP.read_text())
    raw["rules"][0]["size"] = {"value": size, "source": {"grade": "assumed", "detail": "test"}}
    return ImpactMap.from_dict(raw)


def _pricer(calibration, mapping=None, asof=ASOF) -> ImpactPricer:
    return ImpactPricer(calibration, mapping or ImpactMap.load(), DRAWS, SEED, asof)


def test_failed_programme_moves_nothing(calibration, reports):
    result = _pricer(calibration).price(reports["solid-state-sodium"])
    failed = ~result.simulation.reached_market
    assert result.priced and failed.any() and (~failed).any()
    assert np.all(result.effective_lever[failed] == 0.0)
    for impact in result.outcomes.values():
        assert np.all(impact.delta[failed] == 0.0), impact.key
    # ... and a success that arrives before the horizon does move the outcome
    # the lever acts on. (One that arrives too late legitimately moves nothing.)
    arrived = result.effective_lever > 0
    assert arrived.any() and np.all(result.outcomes["emissions"].delta[arrived] < 0.0)


def test_zero_lever_size_gives_exactly_zero(calibration, reports):
    result = _pricer(calibration, _map(0.0)).price(reports["solid-state-sodium"])
    assert result.priced
    assert set(result.outcomes) == {"emissions", "temperature"}  # only the direct ones are listed
    for impact in result.outcomes.values():
        assert np.all(impact.delta == 0.0)


def test_unmapped_sector_is_unpriced_not_zero(calibration, reports):
    result = _pricer(calibration).price(reports["car-t-solid-tumour"])
    assert not result.priced
    assert "health lever" in result.reason
    assert result.outcomes == {} and result.goodness() is None
    text = render(_pricer(calibration), [result], CORPUS)
    assert "| CAR-T for solid tumours | biopharma |" in text and "unpriced" in text


def test_value_without_source_is_rejected():
    raw = yaml.safe_load(DEFAULT_MAP.read_text())
    del raw["rules"][0]["size"]["source"]
    with pytest.raises(ValueError, match="source"):
        ImpactMap.from_dict(raw)
    raw = yaml.safe_load(DEFAULT_MAP.read_text())
    raw["rules"][0]["levers"] = {"warp_drive": {"value": 1.0, "source": {"grade": "assumed", "detail": "x"}}}
    with pytest.raises(ValueError, match="unknown lever"):
        ImpactMap.from_dict(raw)
    raw = yaml.safe_load(DEFAULT_MAP.read_text())
    raw["rules"][0]["size"]["source"] = {"grade": "sourced", "detail": "claims a source, names none"}
    with pytest.raises(ValueError, match="reference"):
        ImpactMap.from_dict(raw)


def test_shipped_map_is_all_assumed():
    ledger = ImpactMap.load().ledger()
    assert ledger.entries and ledger.weakest() is Grade.ASSUMED
    assert ledger.assumption_share() == 1.0


def test_certain_immediate_success_equals_the_worldsim_lever(calibration, reports):
    """With arrival at the base year and no ramp, the pricing must reproduce
    worldsim's own lever run draw for draw -- the grid adds nothing of its own."""
    report = reports["solid-state-sodium"]
    sure = dataclasses.replace(
        report.simulation, reached_market=np.ones(DRAWS, dtype=bool), years_to_market=np.zeros(DRAWS)
    )
    raw = yaml.safe_load(DEFAULT_MAP.read_text())
    fixed = {"grade": "assumed", "detail": "test"}
    raw["rules"][0]["size"] = {"value": 0.4, "source": fixed}
    raw["rules"][0]["ramp_years"] = {"value": 0.0, "source": fixed}
    pricer = _pricer(calibration, ImpactMap.from_dict(raw), asof=dt.date(calibration.base_year, 1, 1))
    result = pricer.price(dataclasses.replace(report, simulation=sure))
    direct = simulate(calibration, {"mitigation": 0.4}, DRAWS, SEED)
    for key in ("emissions", "temperature", "hunger_people"):
        expected = direct.final(key) - pricer.baseline.final(key)
        assert np.allclose(result.outcomes[key].delta, expected, rtol=1e-9, atol=1e-12), key


def test_interpolation_between_grid_levels_is_close(calibration, reports):
    """Draws at different lever levels are read off a grid; check one against a direct run."""
    result = _pricer(calibration).price(reports["solid-state-sodium"])
    index = int(np.argsort(result.effective_lever)[-DRAWS // 10])  # a mid-grid successful draw
    level = float(result.effective_lever[index])
    assert level > 0
    direct = simulate(calibration, {"mitigation": level}, DRAWS, SEED)
    base = simulate(calibration, {}, DRAWS, SEED)
    for key in ("emissions", "temperature"):
        expected = direct.final(key)[index] - base.final(key)[index]
        assert result.outcomes[key].delta[index] == pytest.approx(expected, rel=1e-3)


def test_later_arrival_and_longer_ramp_buy_less():
    reached = np.array([True, True, True, True, False])
    years = np.array([0.0, 10.0, 10.0, 40.0, np.nan])
    ramp = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    lever = effective_lever(reached, years, np.full(5, 0.5), ramp, 2025.0, 2025, 2050)
    assert lever[0] == 0.5                      # there from the start: the full size
    assert lever[1] == pytest.approx(0.5 * 15 / 25)
    assert lever[2] == pytest.approx(0.5 * 10 / 25)  # half the ramp is lost
    assert lever[3] == 0.0                      # arrives after the horizon
    assert lever[4] == 0.0                      # never arrives


def _track(sentence: str) -> Track:
    works = [Work(id=f"t:{i}", title="A study", abstract=sentence) for i in range(4)]
    return Track(id="t", name="t", sector="biopharma", works=works)


def test_term_rule_needs_whole_words():
    mapping = ImpactMap.load()
    assert mapping.rule_for(_track("A base editor corrects a pathogenic point mutation.")) is None
    rule = mapping.rule_for(_track("A new antibiotic class active against resistant pathogens."))
    assert rule is not None and "health" in rule.levers


def test_draw_counts_must_match(calibration, reports):
    pricer = ImpactPricer(calibration, ImpactMap.load(), DRAWS + 100, SEED, ASOF)
    with pytest.raises(ValueError, match="paired draw for draw"):
        pricer.price(reports["solid-state-sodium"])


def test_report_states_its_limits_and_cli_registers(tmp_path):
    import argparse

    pricer, results = analyse(CORPUS, DEFAULT_MAP, draws=200, seed=SEED, asof=ASOF)
    text = render(pricer, results, CORPUS)
    assert "The corpus is synthetic" in text and "The lever mapping is assumed" in text
    assert sum(r.priced for r in results) == 5 and len(results) == 16
    assert "CRISPR base and prime editing | biopharma" in text  # in a table, as unpriced
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    out = tmp_path / "impact.md"
    args = parser.parse_args(["impact", "--draws", "200", "--out", str(out), "--asof", "2026-11-28"])
    assert args.func(args) == 0 and out.read_text().startswith("# Research directions priced")
