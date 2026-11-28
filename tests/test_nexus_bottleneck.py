"""The physical-bottleneck layer: what it measures, and what it refuses to claim."""

import math

import numpy as np
import pytest

from nexus import bottleneck as bn
from sciscout.commercial.profile import PriorLibrary
from sciscout.invest.dist import PERT, LogNormal
from sciscout.invest.model import MarketModel
from sciscout.models import Track
from sciscout.provenance import Grade, Provenance, assumed


def test_wright_fit_recovers_a_planted_learning_rate():
    rng = np.random.default_rng(0)
    cumulative = 2.0 ** np.arange(1, 16)
    slope = math.log2(1 - 0.20)  # 20% cheaper per doubling
    cost = 100 * cumulative**slope * np.exp(rng.normal(0, 0.05, len(cumulative)))
    fit = bn.fit_wright(cumulative, cost)
    assert fit.learning_rate == pytest.approx(0.20, abs=0.02)
    assert fit.decline_after(3) == pytest.approx(0.8**3, rel=0.1)


def test_wright_backtest_fits_only_on_years_up_to_the_cutoff():
    years = np.arange(2000, 2020)
    cumulative = 2.0 ** np.arange(1, 21)
    cost = 100 * cumulative ** math.log2(0.8)
    broken = cost.copy()
    broken[years > 2010] *= 5  # the law fails after the cutoff
    result = bn.backtest_wright(years, cumulative, broken, cutoff=2010)
    assert result["train_points"] == 11 and result["held_out"] == 9
    # The fit never saw the break, so it still reports the planted rate and
    # misses every held-out point by the size of the break.
    assert result["learning_rate"] == pytest.approx(0.20, abs=1e-6)
    assert result["mean_abs_log_error"] == pytest.approx(math.log(5), rel=1e-6)
    assert result["interval_covered"] == 0


def test_wright_refuses_too_few_points_and_no_holdout():
    with pytest.raises(ValueError):
        bn.fit_wright([1, 2, 4], [3, 2, 1])
    with pytest.raises(ValueError):
        bn.backtest_wright([2000, 2001, 2002, 2003], [1, 2, 4, 8], [4, 3, 2, 1], 2005)


def _series(start_share, step, years=range(2000, 2021), grid=100.0):
    return [(y, start_share + step * i, grid) for i, y in enumerate(years)]


def test_ramp_is_interpolated_and_unfinished_ramps_are_censored():
    series = {
        ("solar", "AAA"): _series(0.0, 1.0),  # 1% in 2001, 10% in 2010
        ("solar", "BBB"): _series(0.0, 0.25),  # 1% in 2004, 5% by 2020: unfinished
        ("solar", "CCC"): _series(3.0, 1.0),  # already above 1%: start unknown
        ("solar", "DDD"): _series(0.0, 1.0, grid=2.0),  # grid too small
    }
    found = {r.country: r for r in bn.ramps(series)}
    assert set(found) == {"AAA", "BBB"}
    assert found["AAA"].completed and found["AAA"].duration == pytest.approx(9.0)
    assert not found["BBB"].completed and found["BBB"].duration == pytest.approx(16.0)


def test_kaplan_meier_counts_censored_ramps_instead_of_dropping_them():
    done = [bn.Ramp("solar", f"D{i}", 2000, d, True) for i, d in enumerate([2, 4, 6, 8])]
    waiting = [bn.Ramp("solar", f"W{i}", 2000, 9, False) for i in range(4)]
    assert bn.KaplanMeier.fit(done).quantile(0.5) == 4
    # With four more still running past year 8, half are not done until year 8.
    assert bn.KaplanMeier.fit(done + waiting).quantile(0.5) == 8
    assert bn.KaplanMeier.fit(done + waiting).quantile(0.9) is None


def test_ramp_backtest_never_sees_held_out_years():
    series = bn.load_deployment()
    cutoff = 2015
    assert bn.last_year(bn.truncate(series, cutoff)) == cutoff
    baseline = bn.backtest_ramps(series, cutoff)
    # Rewrite everything after the cutoff: what the fit saw must not change.
    altered = {
        key: [(y, s if y <= cutoff else 0.0, g) for y, s, g in points]
        for key, points in series.items()
    }
    changed = bn.backtest_ramps(altered, cutoff)
    for key in ("train_ramps", "train_completed", "at_risk", "predicted_completions"):
        assert changed[key] == baseline[key]
    assert changed["actual_completions"] == 0 < baseline["actual_completions"]


def _case(sector):
    profile = PriorLibrary().profile(Track(id="t", name="T", sector=sector, works=[]))
    return profile, MarketModel.placeholder()


def test_evidence_replaces_ramp_time_and_leaves_every_stage_alone():
    profile, market = _case("energy_hardware")
    adjusted = bn.apply_overrides(profile, market, bn.evidence_for("energy_hardware"))
    assert adjusted.market.ramp_years.grade is Grade.OBSERVED
    assert market.ramp_years.grade is Grade.ASSUMED  # the original is not mutated
    assert adjusted.market.tam_usd is market.tam_usd
    for before, after in zip(profile.stage_path, adjusted.profile.stage_path):
        assert after.provenance.grade is before.provenance.grade
        for field in bn.STAGE_FIELDS:
            assert getattr(after, field) is getattr(before, field)
    assert "market.tam_usd" in adjusted.still_assumed
    assert "market.ramp_years" not in adjusted.still_assumed
    assert any(item.startswith("stage.") for item in adjusted.still_assumed)


def test_sectors_without_data_get_no_evidence():
    for sector in ("biopharma", "semiconductors", "software_ai", "generic"):
        assert bn.evidence_for(sector) == []


def test_override_without_provenance_is_rejected():
    profile, market = _case("energy_hardware")

    class Bare:
        provenance = None

    with pytest.raises(ValueError, match="no provenance"):
        bn.apply_overrides(profile, market, [bn.Override("market.ramp_years", Bare())])


def test_a_guess_cannot_be_labelled_observed():
    profile, market = _case("energy_hardware")
    claimed = PERT(1, 2, 3, Provenance(Grade.OBSERVED, "trust me"), "ramp (y)")
    with pytest.raises(ValueError, match="names no series"):
        bn.apply_overrides(profile, market, [bn.Override("market.ramp_years", claimed)])
    with pytest.raises(ValueError, match="names no series"):
        bn.apply_overrides(
            profile, market, [bn.Override("market.ramp_years", claimed, "no-such-series")]
        )
    honest = PERT(1, 2, 3, assumed("a guess, labelled as one"), "ramp (y)")
    adjusted = bn.apply_overrides(profile, market, [bn.Override("market.ramp_years", honest)])
    assert "market.ramp_years" in adjusted.still_assumed


def test_stage_override_applies_and_unknown_targets_fail():
    profile, market = _case("energy_hardware")
    cost = LogNormal(1e8, 2.0, assumed("scenario cost"), "cost")
    target = "stage.Pilot and scale-up.cost_usd"
    adjusted = bn.apply_overrides(profile, market, [bn.Override(target, cost)])
    stage = next(s for s in adjusted.profile.stage_path if s.name == "Pilot and scale-up")
    assert stage.cost_usd is cost
    for bad in ("stage.Pilot plant.cost_usd", "market.ramp", "stage.Pilot and scale-up.price"):
        with pytest.raises(ValueError, match="matches nothing"):
            bn.apply_overrides(profile, market, [bn.Override(bad, cost)])


def test_shipped_data_has_a_manifest_entry_and_no_cost_series():
    manifest = bn.load_manifest()
    assert manifest["deployment"]["url"].startswith("https://")
    assert manifest["deployment"]["retrieved"] == "2026-11-28"
    assert bn.load_cost_series() == {}


def test_report_names_what_is_still_assumed(tmp_path):
    text = bn.render(draws=400)
    assert "Still assumed" in text and "No cost series" in text
    assert "biopharma" in text and "weak result" in text
