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

Two kinds of uncertainty are kept apart throughout. *Inputs* (costs, durations,
success rates, market size, the starting TRL) are things better information
could pin down. *Gate luck* -- whether a programme with a 30% chance at a gate
turns out to be in the 30% -- is not. Given a draw's inputs, the expectation and
variance of its outcome over gate luck have a closed form, so the engine
computes them exactly per draw instead of estimating them from the simulated
coin flips. That gives an exact split of the outcome variance between the two
kinds, and expected values with far less sampling error than averaging over
draws in which a few percent happened to succeed.
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
    # Starting TRL drawn for each simulated world.
    starting_trl: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Exact per-draw moments over gate luck, conditional on the draw's inputs.
    p_reach_given_inputs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    npv_given_inputs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    npv_var_given_inputs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    moic_given_inputs: np.ndarray = field(default_factory=lambda: np.zeros(0))

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

        Computed from each draw's exact expectation over gate luck rather than
        from the realised outcomes. Both are unbiased; this one does not depend
        on which few draws happened to survive, which at a 3% success rate is
        most of the sampling error.
        """
        if len(self.moic_given_inputs):
            return float(self.moic_given_inputs.mean())
        return float(self.investor_moic.mean())

    def expected_npv(self) -> float:
        """Mean project NPV across all outcomes, with gate luck integrated out."""
        if len(self.npv_given_inputs):
            return float(self.npv_given_inputs.mean())
        return float(self.project_npv.mean())

    def expected_success_rate(self) -> float:
        """Probability of reaching market, with gate luck integrated out."""
        if len(self.p_reach_given_inputs):
            return float(self.p_reach_given_inputs.mean())
        return self.success_rate

    # -- sampling error -------------------------------------------------------

    def expected_moic_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Sampling interval for :meth:`expected_moic` at this draw count.

        This is Monte Carlo error only: how far the reported mean could move if
        the same model were run on another seed. It says nothing about whether
        the inputs are right. A normal approximation, which is adequate here
        because integrating gate luck out removes the point mass at zero; read
        :meth:`tail_concentration` alongside it, since a mean resting on a
        handful of draws is less settled than its interval suggests.
        """
        values = self.moic_given_inputs if len(self.moic_given_inputs) else self.investor_moic
        return _mean_interval(values, level)

    def expected_npv_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Sampling interval for :meth:`expected_npv`. See the MOIC interval."""
        values = self.npv_given_inputs if len(self.npv_given_inputs) else self.project_npv
        return _mean_interval(values, level)

    def sampling_error(self) -> dict[str, float]:
        """Standard errors of the expected multiple, both ways of estimating it.

        ``naive`` averages realised outcomes; ``integrated`` averages each
        draw's exact expectation over gate luck. Their ratio squared is how many
        times more draws the naive estimate would need for the same precision.
        """
        n = max(1, self.draws)
        naive = float(self.investor_moic.std(ddof=1) / np.sqrt(n))
        integrated = (
            float(self.moic_given_inputs.std(ddof=1) / np.sqrt(n))
            if len(self.moic_given_inputs)
            else naive
        )
        return {"naive": naive, "integrated": integrated}

    def tail_concentration(self, share: float = 0.01) -> float:
        """Fraction of the expected multiple contributed by the top ``share`` of draws.

        A check on how much the headline mean leans on the extreme tail of the
        input distributions. If the top 1% of draws carry a third of the
        expected value, the expected value is mostly a statement about the tail
        you assumed.
        """
        values = self.moic_given_inputs if len(self.moic_given_inputs) else self.investor_moic
        total = float(values.sum())
        if total <= 0:
            return 0.0
        k = max(1, int(round(share * len(values))))
        return float(np.sort(values)[-k:].sum() / total)

    def conditional_moic(self) -> float:
        """Mean investor multiple in the draws that reached market.

        Read alongside :meth:`expected_moic`: the gap between them is the price
        of attrition.
        """
        if len(self.moic_given_inputs):
            reach = float(self.p_reach_given_inputs.sum())
            return float(self.moic_given_inputs.sum() / reach) if reach > 0 else 0.0
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

    def variance_decomposition(self, bins: int = 20) -> "VarianceDecomposition":
        """Split the variance of project NPV by where it comes from.

        The first cut is exact, by the law of total variance::

            Var(NPV) = Var( E[NPV | inputs] )  +  E[ Var(NPV | inputs) ]
                       ---- inputs ----------     ---- gate luck ------

        Both conditional moments are computed in closed form per draw, so the
        split between "what better information could remove" and "what it could
        not" carries no estimation error beyond ordinary Monte Carlo error.

        The input part is then attributed to individual inputs by their
        first-order (main-effect) index: the share of ``Var(E[NPV | inputs])``
        that remains when everything except that one input is averaged out,
        estimated by binning on the input. Main effects do not sum to the whole
        -- the outcome is a product of market size, share, margin and survival,
        so much of the variance lives in combinations -- and the remainder is
        reported as ``interactions`` rather than being quietly spread across the
        inputs. Because the inputs are sampled independently, main effects and
        interactions are a true partition (the Sobol-Hoeffding decomposition).
        """
        if not len(self.npv_given_inputs):
            raise ValueError("this result carries no per-draw conditional moments")
        g = self.npv_given_inputs
        between = float(g.var())
        within = float(self.npv_var_given_inputs.mean())
        total = between + within
        if total <= 0:
            return VarianceDecomposition({}, {grade: 0.0 for grade in Grade}, 0.0, 0.0)

        explained = between / total
        by_input: dict[str, float] = {}
        by_grade: dict[Grade, float] = {grade: 0.0 for grade in Grade}
        for label, values in self.inputs.items():
            index = _first_order_index(values, g, bins) * explained
            if index <= 0:
                continue
            by_input[label] = index
            by_grade[self.input_grades[label]] += index
        main_effects = sum(by_input.values())
        if main_effects > explained:
            # Binning noise can push the sum of main effects past the total it
            # partitions; scale back so the shares still add to one.
            scale = explained / main_effects
            by_input = {k: v * scale for k, v in by_input.items()}
            by_grade = {k: v * scale for k, v in by_grade.items()}
            main_effects = explained
        return VarianceDecomposition(
            by_input=dict(sorted(by_input.items(), key=lambda kv: -kv[1])),
            by_grade=by_grade,
            interactions=explained - main_effects,
            gate_luck=within / total,
        )

    def variance_by_grade(self) -> dict[Grade, float]:
        """Share of the input-attributable spread carried by each provenance grade.

        Main-effect variance shares from :meth:`variance_decomposition`,
        normalised over the grades so they sum to 1. It answers: of the spread
        that individual inputs account for, how much comes from things we
        measured and how much from things we made up? Gate luck and
        interactions are left out of the denominator; read
        :meth:`variance_decomposition` for those.
        """
        if not len(self.npv_given_inputs):
            return {grade: 0.0 for grade in Grade}
        weights = self.variance_decomposition().by_grade
        total = sum(weights.values())
        if total <= 0:
            return {grade: 0.0 for grade in Grade}
        return {grade: value / total for grade, value in weights.items()}

    def summary(self) -> str:
        npv = self.percentiles(self.project_npv)
        return (
            f"{self.track_name}: P(reach market) {self.expected_success_rate():.1%}; "
            f"project NPV P10/P50/P90 "
            f"${npv[10] / 1e6:,.0f}M / ${npv[50] / 1e6:,.0f}M / ${npv[90] / 1e6:,.0f}M; "
            f"expected MOIC {self.expected_moic():.2f}x "
            f"({self.conditional_moic():.2f}x if it reaches market)"
        )


@dataclass
class VarianceDecomposition:
    """Where the variance of project NPV comes from. Shares sum to 1."""

    # Main-effect share per input, largest first.
    by_input: dict[str, float]
    # The same main effects, totalled by the provenance grade of the input.
    by_grade: dict[Grade, float]
    # Variance that only combinations of inputs explain.
    interactions: float
    # Variance that remains when every input is known: which side of each gate
    # the programme lands on.
    gate_luck: float

    @property
    def inputs_total(self) -> float:
        """Everything better information about the inputs could remove."""
        return sum(self.by_grade.values()) + self.interactions


def _mean_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Normal-approximation interval for the mean of ``values``."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return (mean, mean)
    # Two-sided normal quantile by rational approximation (Acklam); avoids a
    # scipy dependency for one number.
    z = _normal_quantile(0.5 + level / 2.0)
    half = z * float(values.std(ddof=1)) / np.sqrt(n)
    return (mean - half, mean + half)


def _normal_quantile(p: float) -> float:
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low = 0.02425
    if p < low:
        q = np.sqrt(-2 * np.log(p))
        return float((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                     / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > 1 - low:
        return -_normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return float((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                 / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def _first_order_index(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Estimate ``Var(E[y | x]) / Var(y)`` from paired samples.

    Sorts by ``x``, cuts into equal-count bins (or one bin per distinct value
    when ``x`` is discrete), and takes the variance of the bin means of ``y``.
    Bin means of pure noise still vary, by about ``(bins - 1) / n`` of the total
    variance, so that much is subtracted: an input with no effect should score
    zero rather than a small positive number that then gets summed over thirty
    inputs.
    """
    n = len(x)
    total = float(y.var())
    if n < 4 * bins or total <= 0 or np.allclose(x, x[0]):
        return 0.0
    unique, inverse = np.unique(x, return_inverse=True)
    if len(unique) <= bins:
        groups = inverse
        k = len(unique)
    else:
        order = np.argsort(x, kind="stable")
        groups = np.empty(n, dtype=np.int64)
        groups[order] = (np.arange(n) * bins) // n
        k = bins
    counts = np.bincount(groups, minlength=k).astype(float)
    sums = np.bincount(groups, weights=y, minlength=k)
    means = sums / np.maximum(counts, 1.0)
    between = float((counts * (means - y.mean()) ** 2).sum() / n)
    raw = between / total
    noise = (k - 1) / (n - 1)
    return max(0.0, (raw - noise) / (1.0 - noise))


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

        # -- readiness ---------------------------------------------------------
        #
        # Each simulated world starts at a TRL drawn from the belief, and faces
        # the stages that TRL leaves ahead of it. A direction that might be at
        # TRL 3 or at TRL 6 is two different programmes, and averaging them into
        # one path describes neither.
        if profile.stage_path:
            stages = profile.possible_stages()
            trl = np.asarray(
                profile.trl.sample(_stream(self.seed, "starting TRL"), size=n),
                dtype=float,
            )
            inputs["starting TRL"] = trl
            grades["starting TRL"] = profile.trl.grade_note.grade
            ledger.record(profile.trl.as_estimate().relabel("starting TRL"))
            applicability = [trl < stage.to_trl for stage in stages]
        else:
            stages = list(profile.remaining_stages)
            trl = np.full(n, float(profile.trl.mode))
            applicability = [np.ones(n, dtype=bool) for _ in stages]
        stage_names = [s.name for s in stages]

        discount_rate = draw(terms.discount_rate, "discount rate")
        step_up = draw(terms.valuation_step_up, "valuation step-up per round")
        growth = 1.0 + discount_rate

        # -- development phase, stage by stage --------------------------------
        #
        # Two things are tracked side by side. The *success path* -- timeline,
        # cumulative cost and ownership assuming every gate is passed -- is
        # computed for every draw, because it is a function of the inputs alone.
        # The *realised* outcome then picks a point on that path: the gate at
        # which the draw's luck ran out, or the end. `alive` carries the
        # abandonment option: once a draw fails a gate, nothing downstream
        # accrues to it. This is the single most consequential piece of the
        # model.
        alive = np.ones(n, dtype=bool)
        failure_stage = np.full(n, -1, dtype=int)

        path_time = np.zeros(n)          # time of the latest gate on the path
        path_disc_cost = np.zeros(n)     # discounted cost through that gate
        path_nominal_cost = np.zeros(n)
        post_gate_duration = np.zeros(n)  # how long the latest stage ran alone
        entered = np.zeros(n, dtype=bool)
        reach = np.ones(n)               # P(alive at this point | inputs)

        realised_disc_cost = np.zeros(n)
        realised_nominal_cost = np.zeros(n)
        realised_elapsed = np.zeros(n)

        # Exact first and second moments of NPV over gate luck, given inputs.
        npv_mean = np.zeros(n)
        npv_second = np.zeros(n)

        # The investor's stake, tracked through each subsequent financing round.
        # Each stage is funded by a round raising that stage's cost, at a
        # post-money valuation of (previous post-money x step-up) + the amount
        # raised. Ownership is multiplied by the pre/post ratio each time.
        ownership = np.full(n, terms.entry_ownership)
        post_money = np.full(n, terms.entry_post_money)
        entry_round = terms.entry_round_usd

        # A failure at a gate is settled when the *next* stage is processed,
        # because an overlapping next stage has already spent money by then.
        pending_mask = np.zeros(n, dtype=bool)
        pending_prob = np.zeros(n)

        def settle(extra_disc: np.ndarray | float, extra_nominal: np.ndarray | float) -> None:
            disc = path_disc_cost + extra_disc
            realised_disc_cost[pending_mask] = disc[pending_mask]
            realised_nominal_cost[pending_mask] = (path_nominal_cost + extra_nominal)[pending_mask]
            realised_elapsed[pending_mask] = path_time[pending_mask]
            npv_mean[:] -= pending_prob * disc
            npv_second[:] += pending_prob * disc**2

        for index, stage in enumerate(stages):
            applies = applicability[index]
            duration = draw(stage.duration_years, f"{stage.name}: duration (y)")
            cost = draw(stage.cost_usd, f"{stage.name}: cost (USD)")
            p_success = draw(stage.p_success, f"{stage.name}: P(success)")

            # Overlap: this stage starts `lead` years before the previous gate.
            # That buys time on the success path, and costs the share of this
            # stage's budget already spent if the previous gate then fails.
            if stage.overlap is not None:
                overlap = draw(stage.overlap, f"{stage.name}: overlap with previous stage")
                lead = np.where(
                    entered & applies,
                    np.minimum(overlap * duration, post_gate_duration),
                    0.0,
                )
            else:
                lead = np.zeros(n)
            at_risk = np.where(duration > 0, lead / np.maximum(duration, 1e-12), 0.0) * cost
            settle(at_risk / growth ** (path_time - lead / 2.0), at_risk)

            # Spend is discounted from the stage midpoint rather than its end:
            # cost accrues throughout a stage, not in a lump at the gate.
            start = path_time - lead
            midpoint = start + duration / 2.0
            path_disc_cost = path_disc_cost + np.where(applies, cost / growth**midpoint, 0.0)
            path_nominal_cost = path_nominal_cost + np.where(applies, cost, 0.0)
            path_time = np.where(applies, start + duration, path_time)
            post_gate_duration = np.where(applies, duration - lead, post_gate_duration)

            # Financing. The entry round pays toward the first stage a draw
            # faces; only what that stage costs beyond the round has to be
            # raised afresh. Every later stage is a new round for its full cost.
            raised = np.where(entered, cost, np.maximum(0.0, cost - entry_round))
            dilutes = applies & (raised > 0)
            pre_money = post_money * step_up
            new_post = pre_money + raised
            ownership = np.where(
                dilutes, ownership * pre_money / np.maximum(new_post, 1e-9), ownership
            )
            post_money = np.where(dilutes, new_post, post_money)
            entered = entered | applies

            survived = _stream(self.seed, f"{stage.name}: survival").random(n) < p_success
            pending_mask = alive & applies & ~survived
            pending_prob = np.where(applies, reach * (1.0 - p_success), 0.0)
            failure_stage[pending_mask] = index
            reach = np.where(applies, reach * p_success, reach)
            alive &= ~pending_mask

        settle(0.0, 0.0)
        reached_market = alive.copy()
        elapsed = path_time
        realised_disc_cost[reached_market] = path_disc_cost[reached_market]
        realised_nominal_cost[reached_market] = path_nominal_cost[reached_market]
        realised_elapsed[reached_market] = path_time[reached_market]
        discounted_cost = realised_disc_cost
        nominal_cost = realised_nominal_cost

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

        # The success outcome completes the conditional moments.
        success_value = (
            operating_cash / growth**cash_midpoint
            + terminal_value / discount_at_exit
            - path_disc_cost
        )
        npv_mean += reach * success_value
        npv_second += reach * success_value**2
        npv_variance = np.maximum(0.0, npv_second - npv_mean**2)

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
        moic_if_success = (
            final_ownership * np.maximum(0.0, terminal_value + operating_cash) / terms.check_usd
        )

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
            starting_trl=trl.astype(int),
            p_reach_given_inputs=reach,
            npv_given_inputs=npv_mean,
            npv_var_given_inputs=npv_variance,
            moic_given_inputs=reach * moic_if_success,
        )
