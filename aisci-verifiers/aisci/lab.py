"""I3: an experiment protocol language, a simulated lab behind a Lab API, and
safety interlocks (tasks 3.2 to 3.4).

The lab is SIMULATED: a hidden response surface stands in for a thin-film
materials system (3 composition fractions + an anneal temperature -> a
measured property, with noise and occasional failed runs). The API, schema and
interlocks are written so a real instrument driver can replace `SimulatedLab`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jsonschema
import numpy as np

PROTOCOL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ExperimentProtocol v1",
    "type": "object",
    "required": ["protocol_id", "steps", "measure"],
    "properties": {
        "protocol_id": {"type": "string"},
        "hypothesis_id": {"type": ["string", "null"]},
        "steps": {"type": "array", "minItems": 1, "items": {"oneOf": [
            {"type": "object", "required": ["op", "reagent", "fraction"], "additionalProperties": False,
             "properties": {"op": {"const": "dispense"}, "reagent": {"type": "string"},
                            "fraction": {"type": "number", "minimum": 0, "maximum": 1}}},
            {"type": "object", "required": ["op", "temp_c", "minutes"], "additionalProperties": False,
             "properties": {"op": {"const": "anneal"}, "temp_c": {"type": "number"},
                            "minutes": {"type": "number", "exclusiveMinimum": 0}}},
            {"type": "object", "required": ["op"], "additionalProperties": False,
             "properties": {"op": {"const": "mix"}, "seconds": {"type": "number"}}},
        ]}},
        "measure": {"enum": ["conductivity", "bandgap", "xrd"]},
        "signoff": {"type": ["string", "null"]},
    },
}

REAGENTS = {
    "A": {"hazard": "low"}, "B": {"hazard": "low"}, "C": {"hazard": "low"},
    "D_toxic": {"hazard": "high"},  # requires human sign-off
}
LIMITS = {"max_temp_c": 600.0, "min_temp_c": 20.0}
FORBIDDEN_PAIRS = {frozenset({"C", "D_toxic"})}  # e.g. a combination that releases a toxic gas


class SafetyBlock(Exception):
    pass


def safety_check(p: dict, screen=None) -> list[str]:
    """Task 3.4 interlocks. Returns a list of violations (empty = safe)."""
    v = []
    reagents = {s["reagent"] for s in p["steps"] if s["op"] == "dispense"}
    for r in reagents:
        if r not in REAGENTS:
            v.append(f"unknown reagent {r}")
        elif REAGENTS[r]["hazard"] == "high" and not p.get("signoff"):
            v.append(f"{r} needs human sign-off")
    for pair in FORBIDDEN_PAIRS:
        if pair <= reagents:
            v.append(f"forbidden combination {sorted(pair)}")
    for s in p["steps"]:
        if s["op"] == "anneal" and not (LIMITS["min_temp_c"] <= s["temp_c"] <= LIMITS["max_temp_c"]):
            v.append(f"anneal {s['temp_c']} C outside limits")
    total = sum(s["fraction"] for s in p["steps"] if s["op"] == "dispense")
    if abs(total - 1) > 1e-6:
        v.append(f"fractions sum to {total:.3f}, not 1")
    if screen is not None:
        v += screen(p)
    return v


def hidden_property(x: np.ndarray, anomaly=True, center=(0.15, 0.15, 0.9)) -> float:
    """Ground truth the agent never sees. x = (fA, fB, fC, T_norm)."""
    fa, fb, fc, t = x
    smooth = (3.0 * math.exp(-((fa - 0.55) ** 2 + (fb - 0.25) ** 2) / 0.03)
              + 1.5 * math.exp(-((fc - 0.6) ** 2) / 0.02) * t
              + 0.8 * t - 0.6 * t * t)
    # a narrow, unexpected high-property phase: the "anomaly" RP4 should notice
    ca, cb, ct = center
    spike = 4.0 * math.exp(-((fa - ca) ** 2 + (fb - cb) ** 2 + (t - ct) ** 2) / 0.015) if anomaly else 0.0
    return smooth + spike


@dataclass
class SimulatedLab:
    seed: int = 0
    noise: float = 0.05
    fail_rate: float = 0.03
    anomaly: bool = True
    center: tuple = (0.15, 0.15, 0.9)
    runs: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def execute(self, p: dict) -> dict:
        self.runs += 1
        if self.rng.random() < self.fail_rate:
            return {"outcome": "failed_run", "result": None}
        f = {s["reagent"]: s["fraction"] for s in p["steps"] if s["op"] == "dispense"}
        temp = next((s["temp_c"] for s in p["steps"] if s["op"] == "anneal"), 20.0)
        x = np.array([f.get("A", 0), f.get("B", 0), f.get("C", 0), (temp - 20) / 580])
        y = hidden_property(x, self.anomaly, self.center) + self.rng.normal(0, self.noise)
        return {"outcome": "success", "result": {"value": float(y), "units": "S/cm (sim)"}}


@dataclass
class LabAPI:
    """Task 3.3: the only way an agent touches the lab. Validates, screens, logs."""
    lab: SimulatedLab
    store: object  # DataStore
    campaign: str = "camp"
    agent: str = "agent-v1"
    screen: object = None
    human_steps: int = 0
    cycle: int = 0

    def run(self, p: dict, hypothesis_id: str | None = None) -> dict:
        self.cycle += 1
        rid = self.store.issue_id(self.campaign)
        jsonschema.validate(p, PROTOCOL_SCHEMA)
        problems = safety_check(p, self.screen)
        if problems:
            out = {"outcome": "blocked_by_safety", "result": {"violations": problems}}
        else:
            out = self.lab.execute(p)
        self.store.append({"record_id": rid, "campaign_id": self.campaign, "hypothesis_id": hypothesis_id,
                           "protocol": p, "outcome": out["outcome"], "result": out["result"],
                           "agent": self.agent, "instrument": "sim-thin-film-1",
                           "timestamp_cycle": self.cycle, "units": {"value": "S/cm (sim)"}})
        return out


def make_protocol(fa, fb, temp_c, pid):
    fc = max(0.0, 1 - fa - fb)
    return {"protocol_id": pid, "hypothesis_id": None, "measure": "conductivity", "signoff": None,
            "steps": [{"op": "dispense", "reagent": "A", "fraction": fa},
                      {"op": "dispense", "reagent": "B", "fraction": fb},
                      {"op": "dispense", "reagent": "C", "fraction": fc},
                      {"op": "mix", "seconds": 30},
                      {"op": "anneal", "temp_c": temp_c, "minutes": 60}]}
