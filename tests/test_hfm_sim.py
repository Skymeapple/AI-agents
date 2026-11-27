"""Checks for the HFM simulation core and the simulators built on it."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "research", "hfm-projects")
sys.path.insert(0, os.path.join(ROOT, "_sim"))


def load(rel):
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def two_opposing_trains(late=0.0):
    # train 1 runs sections 0,1,2; train 2 runs 2,1,0; both planned so that they meet in the middle
    a = dict(id=1, w=3.0, legs=[(0, 10.0, -1e9), (1, 10.0, -1e9), (2, 10.0, -1e9)], plan_in=0.0, actual_in=0.0 + late, plan_out=30.0)
    b = dict(id=2, w=1.0, legs=[(2, 10.0, -1e9), (1, 10.0, -1e9), (0, 10.0, -1e9)], plan_in=0.0, actual_in=0.0, plan_out=30.0)
    return [a, b]


def test_rail_sections_never_hold_two_trains_and_all_trains_finish():
    pytest.importorskip("scipy")
    rail = load("optimization/rail-dispatch/rail_sim.py")
    from core import run
    sim = rail.make_sim(two_opposing_trains())
    cost = run(sim, rail.priority)
    assert sim.done() and all(f is not None for f in sim.finish)
    # opposing trains on a single track cannot both be on time: someone waits
    assert cost > 0


def test_rail_hindsight_bound_is_not_above_any_policy():
    pytest.importorskip("scipy")
    rail = load("optimization/rail-dispatch/rail_sim.py")
    from core import RolloutPolicy, run
    trains = two_opposing_trains(late=4.0)
    bound = rail.hindsight(trains, time_limit=10)
    for pol in (rail.first_come, rail.priority, RolloutPolicy(rail.priority)):
        assert run(rail.make_sim(trains), pol) >= bound - 1e-6


def test_rollout_is_never_worse_than_its_base_when_the_future_is_known():
    pytest.importorskip("scipy")
    rail = load("optimization/rail-dispatch/rail_sim.py")
    from core import RolloutPolicy, run
    trains = two_opposing_trains()          # no hidden future: belief equals the truth
    base = run(rail.make_sim(trains), rail.priority)
    assert run(rail.make_sim(trains), RolloutPolicy(rail.priority)) <= base + 1e-9


def test_bus_simulation_is_deterministic_and_holding_costs_riders():
    pytest.importorskip("numpy")
    bus = load("optimization/bus-holding/bus_sim.py")
    from core import run
    route = json.load(open(os.path.join(ROOT, "optimization", "bus-holding", "hsl_trunk_routes.json")))["500"]
    inst = bus.make_instance(route, seed=1)
    a = run(bus.BusSim(inst), bus.no_holding)
    assert a == run(bus.BusSim(inst), bus.no_holding)
    always_hold = lambda sim, k, options: options[-1]
    sim = bus.BusSim(inst)
    run(sim, always_hold)
    assert sim.cost() != a


def test_belief_copy_does_not_change_the_real_simulation():
    pytest.importorskip("numpy")
    bus = load("optimization/bus-holding/bus_sim.py")
    from core import run
    route = json.load(open(os.path.join(ROOT, "optimization", "bus-holding", "hsl_trunk_routes.json")))["570"]
    inst = bus.make_instance(route, seed=3)
    sim = bus.BusSim(inst)
    sim.decisions()
    before = (sim.cost(), list(sim.events), list(sim.last_dep))
    trial = sim.belief()
    trial.apply(trial.pending[0], 60.0)
    run(trial, bus.no_holding)
    assert (sim.cost(), list(sim.events), list(sim.last_dep)) == before
