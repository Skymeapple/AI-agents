"""Monte Carlo engine for staged research investments.

The structure that matters here is *staged abandonment*. Research programmes are
not a single bet with one probability attached; they are a sequence of gates, and
a programme that fails at the first gate never spends the money budgeted for the
fifth. Modelling this correctly is the difference between a usable number and a
badly wrong one: multiplying a full development budget by a cumulative success
probability overstates expected cost by a large factor, because it charges the
full programme cost to worlds where the programme died early.

That abandonment option is also most of why early-stage research investment can
be rational at all despite success rates in the single digits. A model without it
will tell you nothing is ever worth funding.

Every sampled input is retained so the run can report which inputs actually drove
the spread, and -- because inputs carry provenance -- how much of that spread
traces to assumption rather than evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..commercial.profile import CommercialProfile
from ..provenance import Grade, ProvenanceLedger
from .dist import Distribution
from .model import DealTerms, MarketModel


@dataclass
class MonteCarloResult:
    """Outcome of a simulation run."""

    track_id: str
    track_name: str
    draws: int
    seed: int
    # Per-draw outcomes.
    project_npv: np.ndarray
    investor_moic: np.ndarray
    investor_irr: np.ndarray
    years_to_market: np.ndarray
    total_spend: np.ndarray
    reached_market: np.ndarray
    failure_stage: np.ndarray
    stage_names: list[str]
    # Investor position at exit, after capital-weighted dilution.
    final_ownership: np.ndarray = field(default_factory=lambda: np.zeros(0))
    exit_valuation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Sampled inputs, keyed by label, for sensitivity analysis.
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    input_grades: dict[str, Grade] = field(default_factory=dict)
    ledger: ProvenanceLedger = field(default_factory=ProvenanceLedger)

    # -- headline statistics --------------------------------------------------

    @property
    def success_rate(self) -> float:
        return float(self.reached_market.mean())

    def percentiles(self, values: np.ndarray, points=(10, 50, 90)) -> dict[int, float]:
        return {p: float(np.percentile(values, p)) for p in points}

    def probability_of_loss(self) -> float:
        """Share of draws where the project destroys value."""
        return float((self.project_npv < 0).mean())

    def expected_moic(self) -> float:
        """Mean investor multiple across all draws, including total losses.

        The mean, not the median, is the decision-relevant statistic for a
        portfolio investor: venture returns are driven by the tail, and the
        median outcome of an early-stage research bet is usually zero.
        """
        return float(self.investor_moic.mean())

    def conditional_moic(self) -> float:
        """Mean investor multiple in the draws that reached market.

        Read alongside :meth:`expected_moic`: the gap between them is the price
        of attrition.
        """
        if not self.reached_market.any():
            return 0.0
        return float(self.investor_moic[self.reached_market].mean())

    def failure_breakdown(self) -> dict[str, float]:
        """Share of draws that died at each stage."""
        result: dict[str, float] = {}
        for index, name in enumerate(self.stage_names):
            result[name] = float((self.failure_stage == index).mean())
        result["reached market"] = self.success_rate
        return result

    # -- sensitivity ----------------------------------------------------------

    def sensitivity(
        self, top: int = 10, conditional_on_success: bool = False
    ) -> list[tuple[str, float, Grade]]:
        """Rank inputs by Spearman rank correlation with project NPV.

        Rank correlation rather than Pearson because the relationships are
        monotonic but strongly non-linear -- a doubling of market size does not
        double NPV once attrition and discounting are applied -- and rank
        correlation measures the strength of that relationship without assuming
        it is a straight line.

        ``conditional_on_success`` restricts the analysis to draws that reached
        market, and you generally want to read both views. Unconditionally, most
        draws are failures whose NPV is simply the money spent before the
        programme died, so development cost dominates and stage *duration* shows
        up with a positive sign -- deferred spending on a doomed programme costs
        less in present value. That is arithmetically true and practically
        useless as guidance. Conditioned on success, the ranking shows what
        actually drives the value of the outcome you are buying.

        Returns ``(label, correlation, grade)`` sorted by absolute correlation.
        """
        mask = self.reached_market if conditional_on_success else slice(None)
        target = self.project_npv[mask]
        if len(target) < 30:
            return []
        results: list[tuple[str, float, Grade]] = []
        npv_ranks = _rank(target)
        for label, values in self.inputs.items():
            selected = values[mask]
            if np.allclose(selected, selected[0]):
                continue  # A constant input explains no variance.
            correlation = _pearson(_rank(selected), npv_ranks)
            if np.isnan(correlation):
                continue
            results.append((label, float(correlation), self.input_grades[label]))
        results.sort(key=lambda row: abs(row[1]), reverse=True)
        return results[:top]

    def variance_by_grade(self) -> dict[Grade, float]:
        """Share of explained spread attributable to each provenance grade.

        Computed from squared rank correlations, normalised to sum to 1. This is
        a heuristic attribution, not a variance decomposition -- inputs are not
        independent and the squares do not truly partition the variance. It
        answers a coarser but more useful question than exact decomposition
        would: is this forecast being driven by things we measured or by things
        we made up?
        """
        weights: dict[Grade, float] = {grade: 0.0 for grade in Grade}
        for label, correlation, grade in self.sensitivity(top=len(self.inputs)):
            weights[grade] += correlation**2
        total = sum(weights.values())
        if total == 0:
            return {grade: 0.0 for grade in Grade}
        return {grade: value / total for grade, value in weights.items()}

    def summary(self) -> str:
        npv = self.percentiles(self.project_npv)
        return (
            f"{self.track_name}: P(reach market) {self.success_rate:.1%}; "
            f"project NPV P10/P50/P90 "
            f"${npv[10] / 1e6:,.0f}M / ${npv[50] / 1e6:,.0f}M / ${npv[90] / 1e6:,.0f}M; "
            f"expected MOIC {self.expected_moic():.2f}x "
            f"({self.conditional_moic():.2f}x if it reaches market)"
        )


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Equivalent to scipy.stats.rankdata."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average tied ranks so that a heavily-tied input (many zeros) does not get
    # an arbitrary ordering baked into its correlation.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < len(values):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _stream(seed: int, label: str) -> np.random.Generator:
    """An independent generator for one named input.

    Each input draws from its own stream rather than from one shared generator.
    This matters for scenario comparison: numpy's beta sampler uses rejection
    sampling, so it consumes a variable number of underlying values depending on
    its shape parameters. With a single shared stream, changing one distribution
    shifts every draw made after it, and two scenarios differing only in a stage
    cost would come back with different success rates -- pure sampling artefact,
    indistinguishable from a real effect.

    Keying each stream by a stable hash of the input's label gives common random
    numbers across scenarios: change one assumption and only that column moves.
    The hash is taken from hashlib rather than ``hash()`` because Python's string
    hashing is randomised per process, which would make runs irreproducible
    across invocations.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence([seed, int.from_bytes(digest, "big")])
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = np.sqrt((a_centred**2).sum() * (b_centred**2).sum())
    if denominator == 0:
        return float("nan")
    return float((a_centred * b_centred).sum() / denominator)


class MonteCarlo:
    """Runs staged-investment simulations over a commercial profile."""

    def __init__(self, draws: int = 20_000, seed: int = 20260918) -> None:
        if draws < 100:
            raise ValueError("draws below 100 gives percentiles that are noise")
        self.draws = draws
        self.seed = seed

    def run(
        self,
        profile: CommercialProfile,
        market: MarketModel,
        terms: DealTerms,
    ) -> MonteCarloResult:
        n = self.draws

        inputs: dict[str, np.ndarray] = {}
        grades: dict[str, Grade] = {}
        ledger = ProvenanceLedger()

        def draw(distribution: Distribution, label: str) -> np.ndarray:
            values = distribution.sample(n, _stream(self.seed, label))
            inputs[label] = values
            grades[label] = distribution.grade
            ledger.record(distribution.as_estimate().relabel(label))
            return values

        stages = profile.remaining_stages
        stage_names = [s.name for s in stages]

        discount_rate = draw(terms.discount_rate, "discount rate")
        step_up = draw(terms.valuation_step_up, "valuation step-up per round")

        # -- development phase, stage by stage --------------------------------
        #
        # `alive` carries the abandonment option: once a draw fails a gate, all
        # downstream costs and durations stop accruing for that draw. This is
        # the single most consequential piece of the model.
        alive = np.ones(n, dtype=bool)
        elapsed = np.zeros(n)
        discounted_cost = np.zeros(n)
        nominal_cost = np.zeros(n)
        failure_stage = np.full(n, -1, dtype=int)
        rounds_raised = np.zeros(n, dtype=int)

        # The investor's stake, tracked through each subsequent financing round.
        # Each stage is funded by a round raising that stage's cost, at a
        # post-money valuation of (previous post-money x step-up) + the amount
        # raised. Ownership is multiplied by the pre/post ratio each time.
        ownership = np.full(n, terms.entry_ownership)
        post_money = np.full(n, terms.entry_post_money)

        for index, stage in enumerate(stages):
            duration = draw(stage.duration_years, f"{stage.name}: duration (y)")
            cost = draw(stage.cost_usd, f"{stage.name}: cost (USD)")
            p_success = draw(stage.p_success, f"{stage.name}: P(success)")

            # Spend is discounted from the stage midpoint rather than its end:
            # cost accrues throughout a stage, not in a lump at the gate.
            midpoint = elapsed + duration / 2.0
            stage_cost = np.where(alive, cost, 0.0)
            discounted_cost += stage_cost / (1.0 + discount_rate) ** midpoint
            nominal_cost += stage_cost
            elapsed = np.where(alive, elapsed + duration, elapsed)
            rounds_raised += alive.astype(int)

            # The first remaining stage is the one the entry cheque buys into,
            # so it does not dilute; every stage after it does.
            if index > 0:
                pre_money = post_money * step_up
                new_post = pre_money + cost
                retained = np.where(alive, pre_money / np.maximum(new_post, 1e-9), 1.0)
                ownership = ownership * retained
                post_money = np.where(alive, new_post, post_money)

            survived = _stream(self.seed, f"{stage.name}: survival").random(n) < p_success
            newly_failed = alive & ~survived
            failure_stage[newly_failed] = index
            alive &= survived

        reached_market = alive.copy()

        # -- commercial phase --------------------------------------------------
        tam = draw(market.tam_usd, "TAM (USD)")
        peak_share = draw(market.peak_share, "peak market share")
        margin = draw(market.gross_margin, "gross margin")
        ramp = draw(market.ramp_years, "ramp to peak (y)")
        exit_multiple = draw(market.exit_revenue_multiple, "exit revenue multiple")
        years_to_exit = draw(market.years_to_exit, "years launch to exit")

        peak_revenue = tam * peak_share

        # Revenue ramps linearly from launch to peak over `ramp` years, then
        # holds flat. Integrating that profile from launch to exit gives
        # cumulative revenue without simulating year by year.
        ramp_portion = np.minimum(years_to_exit, ramp)
        flat_portion = np.maximum(0.0, years_to_exit - ramp)
        cumulative_revenue = peak_revenue * (
            0.5 * ramp_portion**2 / np.maximum(ramp, 1e-9) + flat_portion
        )
        operating_cash = cumulative_revenue * margin

        # Revenue at exit, and the terminal value it supports.
        revenue_at_exit = peak_revenue * np.minimum(1.0, years_to_exit / np.maximum(ramp, 1e-9))
        terminal_value = revenue_at_exit * exit_multiple

        exit_time = elapsed + years_to_exit
        # Operating cash is discounted from the midpoint of the commercial
        # window, a reasonable approximation to discounting the full stream.
        cash_midpoint = elapsed + years_to_exit / 2.0
        discount_at_exit = (1.0 + discount_rate) ** exit_time
        discounted_operating = np.where(
            reached_market, operating_cash / (1.0 + discount_rate) ** cash_midpoint, 0.0
        )
        discounted_terminal = np.where(
            reached_market, terminal_value / discount_at_exit, 0.0
        )

        project_npv = discounted_operating + discounted_terminal - discounted_cost

        # -- investor position -------------------------------------------------
        #
        # One financing round per development stage entered. The investor's stake
        # is diluted by each round *after* the one they joined, so a long
        # programme erodes the position even when the science succeeds.
        final_ownership = ownership

        exit_equity_value = np.where(
            reached_market, np.maximum(0.0, terminal_value + operating_cash), 0.0
        )
        proceeds = final_ownership * exit_equity_value
        investor_moic = proceeds / terms.check_usd

        # Two-point IRR: the check goes in at t=0, proceeds come out at exit.
        # A total loss is reported as -100%, which is correct and keeps the
        # distribution readable; the alternative (undefined) loses information.
        safe_exit_time = np.maximum(exit_time, 1e-6)
        with np.errstate(invalid="ignore", divide="ignore"):
            investor_irr = np.where(
                proceeds > 0,
                np.power(np.maximum(investor_moic, 1e-12), 1.0 / safe_exit_time) - 1.0,
                -1.0,
            )

        years_to_market = np.where(reached_market, elapsed, np.nan)

        return MonteCarloResult(
            track_id=profile.track_id,
            track_name=profile.track_name,
            draws=n,
            seed=self.seed,
            project_npv=project_npv,
            investor_moic=investor_moic,
            investor_irr=investor_irr,
            years_to_market=years_to_market,
            total_spend=nominal_cost,
            reached_market=reached_market,
            failure_stage=failure_stage,
            stage_names=stage_names,
            final_ownership=final_ownership,
            exit_valuation=exit_equity_value,
            inputs=inputs,
            input_grades=grades,
            ledger=ledger,
        )
