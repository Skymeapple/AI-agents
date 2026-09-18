"""Investment engine: the modelling claims that matter, asserted directly."""

import numpy as np
import pytest

from sciscout.commercial.profile import PriorLibrary
from sciscout.invest.dist import BetaProb, LogNormal, PERT, Point
from sciscout.invest.engine import MonteCarlo
from sciscout.invest.model import DealTerms, MarketModel
from sciscout.models import Track
from sciscout.provenance import assumed

RNG = np.random.default_rng(1234)


# -- distributions ---------------------------------------------------------


def test_pert_sampled_mean_matches_analytic_mean():
    dist = PERT(1.0, 2.0, 6.0, assumed("t"), "d")
    assert dist.sample(200_000, RNG).mean() == pytest.approx(dist.mean(), rel=0.01)


def test_pert_rejects_an_impossible_range():
    with pytest.raises(ValueError, match="low <= mode <= high"):
        PERT(5.0, 1.0, 2.0, assumed("t"), "d")


def test_beta_probabilities_stay_within_zero_and_one():
    dist = BetaProb(0.05, 0.3, 0.95, assumed("t"), "p")
    draws = dist.sample(50_000, RNG)
    assert draws.min() >= 0.0 and draws.max() <= 1.0


def test_beta_rejects_a_value_outside_zero_and_one():
    with pytest.raises(ValueError, match="not a probability"):
        BetaProb(0.1, 0.5, 1.4, assumed("t"), "p")


def test_lognormal_spread_means_p95_over_median():
    dist = LogNormal(100.0, 3.0, assumed("t"), "m")
    draws = dist.sample(300_000, RNG)
    assert np.percentile(draws, 95) / np.median(draws) == pytest.approx(3.0, rel=0.05)


def test_lognormal_rejects_a_spread_that_is_not_a_ratio():
    with pytest.raises(ValueError, match="must exceed 1.0"):
        LogNormal(100.0, 0.8, assumed("t"), "m")


# -- engine ----------------------------------------------------------------


def profile_for(sector="energy_hardware", trl=4):
    track = Track(id="t", name="Test track", sector=sector, trl_override=trl)
    return PriorLibrary().profile(track)


def run(profile=None, tam=5.0e9, draws=20_000, seed=7, check=5.0e6, ownership=0.15):
    return MonteCarlo(draws=draws, seed=seed).run(
        profile or profile_for(),
        MarketModel.placeholder(tam_usd=tam),
        DealTerms.placeholder(check_usd=check, entry_ownership=ownership),
    )


def test_run_is_reproducible_under_a_fixed_seed():
    a, b = run(seed=42), run(seed=42)
    assert np.array_equal(a.project_npv, b.project_npv)
    assert run(seed=42).success_rate != run(seed=43).success_rate


def test_success_rate_tracks_the_product_of_stage_probabilities():
    profile = profile_for()
    expected = profile.cumulative_probability()
    assert run(profile, draws=40_000).success_rate == pytest.approx(expected, abs=0.02)


def test_abandoned_programmes_stop_spending():
    """The central modelling claim: failing early must be cheap.

    A model that charges full development cost regardless of when a programme
    dies overstates expected spend by a large factor. This asserts the option to
    abandon is actually exercised.

    Note the comparison is against *first-stage* failures specifically. A draw
    that fails at the final gate has already paid for every stage, so it spends
    as much as a success -- that is correct, and comparing all failures against
    all successes would not test anything.
    """
    result = run(draws=30_000)
    died_at_first_gate = result.failure_stage == 0
    assert died_at_first_gate.any() and result.reached_market.any()

    # The exact claim: a draw that died at the first gate paid that stage's cost
    # and nothing downstream. Asserting on totals instead would be weaker, since
    # the stage cost distributions overlap and an unlucky first-stage draw can
    # cost more than a lucky full programme.
    first_stage_cost = result.inputs[f"{result.stage_names[0]}: cost (USD)"]
    assert result.total_spend[died_at_first_gate] == pytest.approx(
        first_stage_cost[died_at_first_gate]
    )

    # On average, abandonment keeps spend far below the full-programme cost.
    full_cost = profile_for().nominal_cost_usd()
    assert result.total_spend.mean() < 0.5 * full_cost
    assert (
        result.total_spend[died_at_first_gate].mean()
        < result.total_spend[result.reached_market].mean()
    )


def test_failures_are_attributed_to_the_stage_that_killed_them():
    result = run(draws=20_000)
    breakdown = result.failure_breakdown()
    assert sum(breakdown.values()) == pytest.approx(1.0, abs=1e-6)
    # Earlier gates kill more programmes than later ones, because fewer reach them.
    assert breakdown[result.stage_names[0]] > breakdown[result.stage_names[-1]]


def test_capital_intensity_dilutes_the_early_investor():
    """A capital-hungry programme must erode an early stake.

    This is the behaviour a fixed-percentage dilution model gets wrong, and
    getting it wrong produces absurd returns on small cheques into programmes
    that consume hundreds of millions.
    """
    hungry = run(profile_for("energy_hardware", trl=4))
    lean = run(profile_for("software_ai", trl=4))
    hungry_stake = hungry.final_ownership[hungry.reached_market].mean()
    lean_stake = lean.final_ownership[lean.reached_market].mean()
    assert hungry_stake < lean_stake
    # And the eroded stake is a small fraction of the entry position.
    assert hungry_stake < 0.15


def test_investor_never_recovers_anything_from_a_failed_programme():
    result = run()
    assert result.investor_moic[~result.reached_market].max() == 0.0
    assert (result.investor_irr[~result.reached_market] == -1.0).all()


def test_bigger_market_raises_value():
    small, large = run(tam=1.0e9), run(tam=50.0e9)
    assert large.expected_moic() > small.expected_moic()
    assert large.project_npv.mean() > small.project_npv.mean()


def test_a_higher_discount_rate_lowers_project_value():
    profile = profile_for()
    market = MarketModel.placeholder(tam_usd=5.0e9)

    def npv_at(rate):
        terms = DealTerms.placeholder()
        terms.discount_rate = Point(rate, assumed("fixed for test"), "discount rate")
        return MonteCarlo(draws=20_000, seed=11).run(profile, market, terms).project_npv.mean()

    assert npv_at(0.35) < npv_at(0.10)


def test_later_stage_technology_carries_a_better_investment_case():
    """Same sector, same market: readiness alone should move the investment case."""
    early = run(profile_for("biopharma", trl=2), tam=8.0e9)
    late = run(profile_for("biopharma", trl=7), tam=8.0e9)
    assert late.success_rate > early.success_rate
    # Value conditional on success, and return on capital, both improve.
    assert (
        late.project_npv[late.reached_market].mean()
        > early.project_npv[early.reached_market].mean()
    )
    assert late.expected_moic() > early.expected_moic()


def test_expected_npv_can_favour_early_stage_because_failure_is_cheap():
    """A real and counterintuitive property of staged models, pinned deliberately.

    A TRL 2 programme mostly dies in a cheap discovery stage; a TRL 7 programme
    that dies has already spent tens of millions on trials. So mean NPV can rank
    the early-stage programme higher even though every other measure prefers the
    late-stage one. This is why the platform reports NPV and MOIC side by side
    rather than collapsing to a single figure of merit -- and the test exists so
    that nobody later 'fixes' this into agreeing with intuition.
    """
    early = run(profile_for("biopharma", trl=2), tam=8.0e9)
    late = run(profile_for("biopharma", trl=7), tam=8.0e9)
    assert early.total_spend.mean() < late.total_spend.mean()
    assert late.expected_moic() > early.expected_moic()


def test_discount_rate_default_is_a_cost_of_capital_not_a_venture_hurdle():
    """Guards against reintroducing the double-count of programme risk.

    Attrition is simulated explicitly, so discounting survivors at a 25-30%
    venture hurdle rate charges for the same risk twice. Over a decade-long
    development path that is enough to make sound programmes look negative.
    """
    assert DealTerms.placeholder().discount_rate.mean() < 0.20


def test_sensitivity_finds_the_input_it_was_given():
    """Pin every input but one; that one must dominate the ranking."""
    profile = profile_for("software_ai", trl=5)
    market = MarketModel.placeholder(tam_usd=5.0e9)
    market.peak_share = Point(0.05, assumed("pinned"), "peak market share")
    market.gross_margin = Point(0.5, assumed("pinned"), "gross margin")
    market.ramp_years = Point(3.0, assumed("pinned"), "ramp to peak (y)")
    market.exit_revenue_multiple = Point(4.0, assumed("pinned"), "exit revenue multiple")
    market.years_to_exit = Point(5.0, assumed("pinned"), "years launch to exit")
    terms = DealTerms.placeholder()
    terms.discount_rate = Point(0.25, assumed("pinned"), "discount rate")

    result = MonteCarlo(draws=20_000, seed=3).run(profile, market, terms)
    top_label, top_corr, _ = result.sensitivity(top=1, conditional_on_success=True)[0]
    assert top_label == "TAM (USD)"
    assert top_corr > 0.5


def test_sensitivity_skips_constant_inputs():
    """A pinned input explains no variance and must not appear as a driver."""
    market = MarketModel.placeholder(tam_usd=5.0e9)
    market.gross_margin = Point(0.5, assumed("pinned"), "gross margin")
    result = MonteCarlo(draws=5_000, seed=5).run(
        profile_for(), market, DealTerms.placeholder()
    )
    labels = [label for label, _, _ in result.sensitivity(top=50)]
    assert "gross margin" not in labels


def test_placeholder_inputs_are_reported_as_assumptions():
    """Running on placeholders must not look like running on evidence."""
    result = run()
    from sciscout.provenance import Grade

    assert result.variance_by_grade()[Grade.ASSUMED] == pytest.approx(1.0, abs=1e-9)
    assert result.ledger.needing_review()


def test_too_few_draws_is_rejected():
    with pytest.raises(ValueError, match="noise"):
        MonteCarlo(draws=10)


def test_deal_terms_reject_impossible_ownership():
    with pytest.raises(ValueError, match="entry_ownership"):
        DealTerms.placeholder(entry_ownership=1.5)
