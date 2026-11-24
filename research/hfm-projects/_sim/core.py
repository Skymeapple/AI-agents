"""Shared simulation core for the operations projects.

Three small pieces that every replay simulator here uses:

  Simulator      base class for an event-driven replay with a policy hook. A
                 simulator must be copyable (`clone`) so a policy can test a
                 decision by simulating ahead ("rollout").
  RolloutPolicy  turns any simple base policy into a look-ahead policy: for each
                 option at a decision point, copy the simulator, take the option,
                 finish the run with the base policy, and keep the cheapest option.
  compare        paired evaluation of several policies on the same instances, with
                 a hindsight bound where the project can compute one.

Design rules, taken from what the public-data experiments in this repository showed:
  * policies see only what would be known at decision time (the simulator exposes a
    causal "belief" copy for rollouts, with future surprises replaced by the plan);
  * every comparison is paired on identical instances and reports how many
    instances improved, not only the mean;
  * a perfect-information bound is reported next to the policies, so the remaining
    headroom is visible.
"""
from __future__ import annotations

import copy
import statistics
from typing import Callable, Iterable, Sequence


class Simulator:
    """Event-driven replay. Subclasses implement the four methods below."""

    def done(self) -> bool:
        raise NotImplementedError

    def decisions(self) -> list:
        """Advance to the next time a choice is needed and return the decision points there.

        Each decision point is (key, options). An option of None means "wait".
        Returns [] when the run is over.
        """
        raise NotImplementedError

    def apply(self, key, option) -> None:
        raise NotImplementedError

    def cost(self) -> float:
        raise NotImplementedError

    def clone(self) -> "Simulator":
        return copy.deepcopy(self)

    def belief(self) -> "Simulator":
        """A copy holding only what is known now. Default: a plain copy (no hidden future)."""
        return self.clone()


def run(sim: Simulator, policy: Callable) -> float:
    """Play a simulator to the end under a policy and return its cost."""
    while not sim.done():
        points = sim.decisions()
        if not points:
            break
        for key, options in points:
            sim.apply(key, policy(sim, key, options))
    return sim.cost()


class RolloutPolicy:
    """One-step look-ahead on top of a base policy."""

    def __init__(self, base: Callable, max_options: int = 4):
        self.base = base
        self.max_options = max_options

    def __call__(self, sim: Simulator, key, options):
        if len(options) <= 1:
            return options[0]
        best, best_cost = None, None
        for opt in options[: self.max_options]:
            trial = sim.belief()
            if not trial.can_apply(key, opt):
                continue
            trial.apply(key, opt)
            c = run(trial, self.base)
            if best_cost is None or c < best_cost - 1e-9:
                best, best_cost = opt, c
        return best if best_cost is not None else self.base(sim, key, options)


def compare(instances: Iterable, make_sim: Callable, policies: dict, reference: str, bound: Callable | None = None, label: str = "") -> dict:
    """Run every policy on every instance. Prints a paired table and returns the raw costs."""
    costs = {name: [] for name in policies}
    bounds = []
    for inst in instances:
        for name, pol in policies.items():
            costs[name].append(run(make_sim(inst), pol))
        if bound is not None:
            bounds.append(bound(inst))
    n = len(costs[reference])
    ref = costs[reference]
    print(f"\n{label} ({n} instances); cost: lower is better")
    print(f"  {'policy':<28} {'mean cost':>10} {'vs ' + reference:>22} {'instances better':>17} {'worse':>6}")
    for name, c in costs.items():
        better = sum(a < b - 1e-9 for a, b in zip(c, ref))
        worse = sum(a > b + 1e-9 for a, b in zip(c, ref))
        change = statistics.mean(c) / statistics.mean(ref) - 1 if statistics.mean(ref) else 0.0
        print(f"  {name:<28} {statistics.mean(c):>10.1f} {change:>+22.1%} {better:>13}/{n} {worse:>6}")
    if bounds:
        change = statistics.mean(bounds) / statistics.mean(ref) - 1 if statistics.mean(ref) else 0.0
        print(f"  {'hindsight bound':<28} {statistics.mean(bounds):>10.1f} {change:>+22.1%}")
        costs["hindsight bound"] = bounds
    return costs
