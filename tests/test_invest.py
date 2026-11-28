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
    # Averaged over the TRL belief, since each draw starts at a sampled TRL.
    expected = profile.expected_success_probability()
    result = run(profile, draws=40_000)
    assert result.success_rate == pytest.approx(expected, abs=0.02)
    assert result.expected_success_rate() == pytest.approx(expected, abs=0.005)


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
    # The first stage a programme at the modal TRL faces. (Draws that start a
    # TRL lower face an earlier stage too; they are excluded here so the claim
    # stays exact.)
    first = result.stage_names.index(profile_for().remaining_stages[0].name)
    died_at_first_gate = (result.failure_stage == first) & (result.starting_trl >= 4)
    assert died_at_first_gate.any() and result.reached_market.any()

    # The exact claim: a draw that died at the first gate paid that stage's cost
    # and nothing downstream. Asserting on totals instead would be weaker, since
    # the stage cost distributions overlap and an unlucky first-stage draw can
    # cost more than a lucky full programme.
    first_stage_cost = result.inputs[f"{result.stage_names[first]}: cost (USD)"]
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
    # Earlier gates kill more programmes than later ones, because fewer reach
    # them. Compared from the modal first stage: the stage before it is only
    # faced by the minority of draws that start a TRL lower.
    modal_first = profile_for().remaining_stages[0].name
    assert breakdown[modal_first] > breakdown[result.stage_names[-1]]


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

    shares = result.variance_by_grade()
    # The only input here that is not a placeholder is the analyst's TRL.
    assert shares[Grade.SOURCED] == 0.0
    assert shares[Grade.ASSUMED] > 0.95
    assert shares[Grade.ASSUMED] + shares[Grade.OBSERVED] == pytest.approx(1.0)
    assert result.ledger.needing_review()


def test_too_few_draws_is_rejected():
    with pytest.raises(ValueError, match="noise"):
        MonteCarlo(draws=10)


def test_deal_terms_reject_impossible_ownership():
    with pytest.raises(ValueError, match="entry_ownership"):
        DealTerms.placeholder(entry_ownership=1.5)


# -- readiness uncertainty, financing, overlap ------------------------------


def _fixed(value, label):
    return Point(value, assumed("fixed for test"), label)


def _stage(name, from_trl, to_trl, years, cost, p, overlap=None):
    from sciscout.commercial.profile import Stage

    return Stage(
        name=name,
        from_trl=from_trl,
        to_trl=to_trl,
        duration_years=_fixed(years, f"{name} duration"),
        cost_usd=_fixed(cost, f"{name} cost"),
        p_success=_fixed(p, f"{name} p"),
        provenance=assumed("fixed for test"),
        overlap=None if overlap is None else _fixed(overlap, f"{name} overlap"),
    )


def _profile(stages, probabilities):
    from sciscout.commercial.profile import CommercialProfile
    from sciscout.scoring.trl import TRLBelief

    belief = TRLBelief(
        {t: probabilities.get(t, 0.0) for t in range(1, 10)}, [], assumed("test belief")
    )
    return CommercialProfile(
        track_id="t",
        track_name="T",
        sector="generic",
        sector_label="Generic",
        trl=belief,
        remaining_stages=[s for s in stages if s.applies_from(belief.mode)],
        constraints=[],
        stage_path=stages,
    )


def _terms(step_up=2.0, rate=0.10, **kwargs):
    terms = DealTerms.placeholder(**kwargs)
    terms.valuation_step_up = _fixed(step_up, "valuation step-up per round")
    terms.discount_rate = _fixed(rate, "discount rate")
    return terms


def _simulate(profile, terms=None, draws=20_000, seed=5):
    return MonteCarlo(draws=draws, seed=seed).run(
        profile, MarketModel.placeholder(tam_usd=5.0e9), terms or _terms()
    )


def test_each_draw_faces_the_stages_its_own_starting_trl_leaves_ahead():
    """The TRL belief reaches the simulation as a distribution, not as its mode.

    A direction that is either at TRL 3 or at TRL 7 is two programmes. Worlds
    starting at 7 must not pay for the pilot stage, and worlds starting at 3
    must, whichever of the two the mode happens to be.
    """
    stages = [
        _stage("Pilot", 1, 7, 4.0, 100.0e6, 1.0),
        _stage("Plant", 7, 9, 2.0, 300.0e6, 1.0),
    ]
    result = _simulate(_profile(stages, {3: 0.4, 7: 0.6}))
    early, late = result.starting_trl == 3, result.starting_trl == 7
    assert early.mean() == pytest.approx(0.4, abs=0.02)
    assert result.total_spend[early] == pytest.approx(400.0e6)
    assert result.total_spend[late] == pytest.approx(300.0e6)
    assert result.years_to_market[early] == pytest.approx(6.0)
    assert result.years_to_market[late] == pytest.approx(2.0)
    # A mode-only model would report one of these for every draw.
    assert set(np.unique(result.total_spend)) == {300.0e6, 400.0e6}


def test_readiness_uncertainty_shows_up_as_a_driver():
    stages = [
        _stage("Pilot", 1, 7, 4.0, 100.0e6, 0.5),
        _stage("Plant", 7, 9, 2.0, 300.0e6, 0.5),
    ]
    result = _simulate(_profile(stages, {3: 0.5, 7: 0.5}))
    assert "starting TRL" in result.variance_decomposition().by_input


def test_a_first_stage_bigger_than_the_entry_round_dilutes():
    """The cheque does not conjure the rest of the first stage's budget.

    A $5M cheque for 15% implies a $33M post-money. If the stage it buys into
    costs $100M, the other $95M is raised afterwards and the stake shrinks
    accordingly -- pre-money 66.7M (2x step-up) against 161.7M post.
    """
    stages = [_stage("Pilot", 1, 9, 3.0, 100.0e6, 1.0)]
    result = _simulate(_profile(stages, {4: 1.0}))
    pre = (5.0e6 / 0.15) * 2.0
    assert result.final_ownership == pytest.approx(0.15 * pre / (pre + 95.0e6))


def test_a_first_stage_the_entry_round_covers_does_not_dilute():
    stages = [_stage("Lab", 1, 9, 3.0, 4.0e6, 1.0)]
    assert _simulate(_profile(stages, {4: 1.0})).final_ownership == pytest.approx(0.15)


def test_a_larger_entry_round_leaves_less_to_raise_later():
    stages = [_stage("Pilot", 1, 9, 3.0, 30.0e6, 1.0)]
    alone = _simulate(_profile(stages, {4: 1.0}))
    terms = _terms()
    terms.round_usd = 30.0e6
    terms.__post_init__()
    syndicated = _simulate(_profile(stages, {4: 1.0}), terms)
    assert alone.final_ownership.mean() < 0.15
    assert syndicated.final_ownership == pytest.approx(0.15)


def test_an_entry_round_cannot_exceed_its_own_post_money():
    with pytest.raises(ValueError, match="post-money"):
        terms = DealTerms.placeholder()
        terms.round_usd = 1.0e9
        terms.__post_init__()


def test_overlap_shortens_the_path_and_puts_the_next_budget_at_risk():
    """Overlap is a trade: time saved on success, money lost on failure."""

    def build(overlap):
        return _profile(
            [
                _stage("Trials", 1, 6, 4.0, 50.0e6, 0.5),
                _stage("Plant", 6, 9, 4.0, 200.0e6, 1.0, overlap=overlap),
            ],
            {3: 1.0},
        )

    sequential, overlapped = _simulate(build(None)), _simulate(build(0.5))
    won = sequential.reached_market
    # Same random numbers, so the same draws succeed in both.
    assert np.array_equal(won, overlapped.reached_market)
    assert sequential.years_to_market[won] == pytest.approx(8.0)
    assert overlapped.years_to_market[won] == pytest.approx(6.0)
    # Failing the first gate now costs half the plant as well.
    assert sequential.total_spend[~won] == pytest.approx(50.0e6)
    assert overlapped.total_spend[~won] == pytest.approx(150.0e6)
    assert overlapped.total_spend[won] == pytest.approx(250.0e6)


# -- where the variance comes from, and how well the mean is known ----------


def test_variance_shares_sum_to_one():
    split = run().variance_decomposition()
    total = sum(split.by_grade.values()) + split.interactions + split.gate_luck
    assert total == pytest.approx(1.0)
    assert sum(split.by_input.values()) == pytest.approx(sum(split.by_grade.values()))


def test_certain_gates_leave_no_gate_luck():
    stages = [_stage("Only", 1, 9, 3.0, 20.0e6, 1.0)]
    split = _simulate(_profile(stages, {4: 1.0})).variance_decomposition()
    assert split.gate_luck == pytest.approx(0.0, abs=1e-12)
    assert split.inputs_total == pytest.approx(1.0)


def test_with_every_input_pinned_the_spread_is_all_gate_luck():
    stages = [_stage("Only", 1, 9, 3.0, 20.0e6, 0.3)]
    market = MarketModel.placeholder()
    for name, value in (
        ("tam_usd", 5.0e9), ("peak_share", 0.05), ("gross_margin", 0.5),
        ("ramp_years", 3.0), ("exit_revenue_multiple", 4.0), ("years_to_exit", 5.0),
    ):
        setattr(market, name, _fixed(value, name))
    result = MonteCarlo(draws=5_000, seed=5).run(_profile(stages, {4: 1.0}), market, _terms())
    split = result.variance_decomposition()
    assert split.gate_luck == pytest.approx(1.0)
    assert split.by_input == {}


def test_the_exact_variance_split_agrees_with_the_simulated_variance():
    result = run(draws=400_000)
    exact = result.npv_given_inputs.var() + result.npv_var_given_inputs.mean()
    assert exact == pytest.approx(result.project_npv.var(), rel=0.05)


def test_integrating_out_gate_luck_agrees_with_the_simulation_and_is_tighter():
    result = run(profile_for("biopharma", trl=2), draws=400_000)
    errors = result.sampling_error()
    assert result.expected_moic() == pytest.approx(
        result.investor_moic.mean(), abs=4 * errors["naive"]
    )
    assert result.expected_npv() == pytest.approx(result.project_npv.mean(), rel=0.05)
    # At a ~2% success rate nearly all the naive estimator's noise is luck.
    assert errors["integrated"] < 0.25 * errors["naive"]


def test_the_sampling_interval_narrows_with_more_draws():
    def width(draws):
        low, high = run(draws=draws).expected_moic_interval()
        return high - low

    assert width(40_000) < 0.6 * width(5_000)


def test_tail_concentration_reports_how_much_the_mean_leans_on_the_tail():
    result = run()
    assert 0.01 < result.tail_concentration(0.01) < 1.0
    assert result.tail_concentration(0.5) > result.tail_concentration(0.01)
