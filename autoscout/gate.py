"""The acceptance gate: when is a change to the agent an improvement?

This is the most important module in autoscout. A self-iterating system is only
as good as its criterion for keeping a change, and the failure it must guard
against is not a bad change -- those are easy to reject -- but a lucky one. Run
enough candidates through a noisy benchmark and some will look better by
chance; accept them and the agent drifts, cycle by cycle, towards whatever
quirks its benchmark happens to have, while reporting steady progress.

So a candidate is promoted only if all of these hold:

1. **Improvement is real on tune.** On at least one benchmark, the mean paired
   per-item gain is at least ``min_effect`` *and* the one-sided lower bound of
   a paired bootstrap interval on that gain is above zero.
2. **Nothing else got worse on tune.** Every other benchmark's mean paired
   change is above ``-tolerance``.
3. **Holdout agrees.** The same non-inferiority check passes on the holdout
   split. Holdout is consulted for one finalist per cycle, not every candidate
   (see :mod:`autoscout.bench`).

A second route exists for efficiency: a candidate that is no worse (lower
bound above ``-tolerance`` everywhere) but cuts reading load by at least
``efficiency_gain`` is promoted too. Getting the same results for less reading
is an improvement, and a gate that only rewarded more-of-everything would
never find it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bench import BenchResult


@dataclass(frozen=True)
class GatePolicy:
    min_effect: float = 0.004  # mean per-item gain worth having
    tolerance: float = 0.002  # largest mean per-item loss tolerated elsewhere
    confidence: float = 0.95
    resamples: int = 2000
    efficiency_gain: float = 0.05  # relative cut in reading load
    seed: int = 0
    # Ablation only: accept any candidate whose summed mean gain on tune is
    # positive, with no interval and no holdout. This is plain hill-climbing,
    # kept so experiments can measure what the statistics above are worth.
    naive: bool = False


@dataclass
class Comparison:
    """Champion vs candidate on one benchmark and split."""

    benchmark: str
    split: str
    delta_mean: float
    lower: float
    upper: float
    n: int

    def to_dict(self) -> dict:
        return {"benchmark": self.benchmark, "split": self.split,
                "delta": round(self.delta_mean, 5), "ci": [round(self.lower, 5), round(self.upper, 5)],
                "n": self.n}


@dataclass
class Decision:
    accepted: bool
    reason: str
    comparisons: list[Comparison] = field(default_factory=list)


def paired_bootstrap(champion: BenchResult, candidate: BenchResult,
                     policy: GatePolicy) -> Comparison:
    """Bootstrap interval on the mean per-item difference, candidate - champion."""
    if champion.item_ids != candidate.item_ids:
        raise ValueError(f"{champion.name}/{champion.split}: item order differs; "
                         "paired comparison is meaningless")
    delta = candidate.scores - champion.scores
    n = len(delta)
    if n == 0:
        return Comparison(champion.name, champion.split, 0.0, 0.0, 0.0, 0)
    # Seeded from the policy so a decision is reproducible from the logs.
    rng = np.random.default_rng(policy.seed)
    idx = rng.integers(0, n, size=(policy.resamples, n))
    means = delta[idx].mean(axis=1)
    alpha = 1.0 - policy.confidence
    return Comparison(
        benchmark=champion.name,
        split=champion.split,
        delta_mean=float(delta.mean()),
        lower=float(np.quantile(means, alpha)),  # one-sided lower bound
        upper=float(np.quantile(means, 1 - alpha)),
        n=n,
    )


def _reading_load(results: dict[str, BenchResult]) -> float:
    loads = [r.metrics["reading_load"] for r in results.values() if "reading_load" in r.metrics]
    return float(np.mean(loads)) if loads else 0.0


def tune_verdict(champion: dict[str, BenchResult], candidate: dict[str, BenchResult],
                 policy: GatePolicy) -> tuple[bool, str, list[Comparison], float]:
    """Stage one: does the candidate beat the champion on tune?

    Returns (passes, reason, comparisons, selection score). The selection score
    ranks passing candidates against each other when several pass in one cycle.
    """
    comps = [paired_bootstrap(champion[name], candidate[name], policy) for name in champion]
    if policy.naive:
        gain = sum(c.delta_mean for c in comps)
        return gain > 0, f"naive: summed gain {gain:+.4f}", comps, gain
    improved = [c for c in comps if c.delta_mean >= policy.min_effect and c.lower > 0]
    regressed = [c for c in comps if c.delta_mean < -policy.tolerance]
    if regressed:
        worst = min(regressed, key=lambda c: c.delta_mean)
        return False, f"regresses {worst.benchmark} by {worst.delta_mean:+.4f}", comps, -1.0
    if improved:
        gain = sum(c.delta_mean for c in improved)
        names = ", ".join(f"{c.benchmark} {c.delta_mean:+.4f}" for c in improved)
        return True, f"improves {names}", comps, gain

    before, after = _reading_load(champion), _reading_load(candidate)
    non_inferior = all(c.lower > -policy.tolerance for c in comps)
    if non_inferior and before > 0 and after <= before * (1 - policy.efficiency_gain):
        return True, f"same quality, reading load {before:.3f} -> {after:.3f}", comps, 0.0
    best = max(comps, key=lambda c: c.delta_mean)
    return False, (f"no significant gain (best {best.benchmark} {best.delta_mean:+.4f}, "
                   f"lower bound {best.lower:+.4f})"), comps, -1.0


def holdout_verdict(champion: dict[str, BenchResult], candidate: dict[str, BenchResult],
                    policy: GatePolicy) -> tuple[bool, str, list[Comparison]]:
    """Stage two, for the cycle's finalist only: non-inferior on holdout?"""
    comps = [paired_bootstrap(champion[name], candidate[name], policy) for name in champion]
    if policy.naive:
        return True, "naive: holdout not consulted", comps
    failed = [c for c in comps if c.delta_mean < -policy.tolerance]
    if failed:
        worst = min(failed, key=lambda c: c.delta_mean)
        return False, (f"holdout disagrees: {worst.benchmark} {worst.delta_mean:+.4f} "
                       "(likely overfit to tune)"), comps
    return True, "holdout non-inferior", comps
