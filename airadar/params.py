"""Scoring parameters, and where each value came from.

The growth score has knobs -- how much to weigh relative against absolute
growth, how far to trust a number read off a web page against one from an API.
Every knob starts as an ``assumed`` judgement call. The self-tuner
(:mod:`airadar.iterate`) may later move one, but only on backtest evidence, and
when it does the parameter's provenance changes to say so: a value the radar
learned from its own track record is a different kind of thing from one
somebody guessed, and the report should be able to tell them apart.

Parameters live in ``params.json`` so a tuning step is a one-line diff in git.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Knob:
    """A tunable parameter: its default, the range the tuner may explore, and
    the step it moves by. Bounds stop a run of noisy evaluations from walking a
    weight somewhere absurd one plausible-looking step at a time."""

    default: float
    low: float
    high: float
    step: float
    rationale: str


KNOBS: dict[str, Knob] = {
    "relative_weight": Knob(
        0.5, 0.0, 1.0, 0.1,
        "even split between relative growth (scale-free, favours small apps) "
        "and absolute velocity (favours big ones) until evidence says otherwise",
    ),
    "trust.api": Knob(1.0, 0.2, 1.0, 0.1, "a number read verbatim from an API is the best evidence held"),
    "trust.page_extract": Knob(
        0.7, 0.1, 1.0, 0.1,
        "page reads can be mangled by the extraction step, so discounted against API reads",
    ),
    "trust.reported": Knob(
        0.5, 0.1, 1.0, 0.1,
        "company-reported figures are selectively disclosed and unaudited",
    ),
    "weight.delta": Knob(1.0, 0.2, 1.0, 0.1, "growth measured between two of our own snapshots"),
    "weight.period": Knob(1.0, 0.2, 1.0, 0.1, "growth over a period as stated by the source"),
    "weight.reported_growth": Knob(
        0.6, 0.1, 1.0, 0.1,
        "a stated growth rate usually picks its own flattering baseline",
    ),
    "shrinkage": Knob(
        0.5, 0.0, 3.0, 0.25,
        "pseudo-evidence pulling thinly-evidenced apps toward the median, so one "
        "lucky signal cannot top the ranking",
    ),
}

# Not tuned: these are definitions, not estimates. Changing them changes what
# the radar is measuring, which is a decision for a person, not for a backtest.
FIXED = {
    "lookback_days": 7,
    # Company-reported figures arrive sparsely and late -- a quarter's numbers
    # surface weeks after the quarter -- so they get a longer window, and their
    # weight halves every staleness_half_life_days of age.
    "reported_lookback_days": 120,
    "reported_max_span_days": 365,
    "staleness_half_life_days": 45,
    "horizon_days": 7,
    "top_k": 15,
    # Below these levels relative growth is noise (1 -> 10 stars is +900%), so it
    # is not computed; absolute velocity still counts.
    "min_base": {
        "stars": 200, "likes": 20, "downloads": 1000, "users": 10000,
        "visits": 100000, "revenue": 1.0e6, "points": 20, "reported_growth": 0,
    },
    # A gain of more than this share of the total level within a week is flagged
    # as a suspected extraction error or inorganic growth (star farming), and
    # its signal weight multiplied by anomaly_penalty. Flagged and discounted,
    # never dropped: a genuine viral launch looks exactly like this too, and
    # the report is where a person decides which it was. Longer windows are not
    # checked, because a repo created this month legitimately gained most of
    # its stars this month.
    "anomaly_share": 0.5,
    "anomaly_max_period_days": 7,
    "anomaly_penalty": 0.5,
}


class Params:
    def __init__(self, values: dict[str, float], provenance: dict[str, str], history: list[dict]):
        self.values = values
        self.provenance = provenance
        self.history = history

    @classmethod
    def defaults(cls) -> "Params":
        return cls(
            {name: knob.default for name, knob in KNOBS.items()},
            {name: f"assumed: {knob.rationale}" for name, knob in KNOBS.items()},
            [],
        )

    @classmethod
    def load(cls, path: Path) -> "Params":
        params = cls.defaults()
        if not path.exists():
            return params
        raw = json.loads(path.read_text())
        for name, value in raw.get("values", {}).items():
            if name not in KNOBS:
                raise ValueError(f"{path}: unknown parameter {name!r}")
            knob = KNOBS[name]
            if not knob.low <= value <= knob.high:
                raise ValueError(f"{path}: {name}={value} outside [{knob.low}, {knob.high}]")
            params.values[name] = float(value)
        params.provenance.update(raw.get("provenance", {}))
        params.history = list(raw.get("history", []))
        return params

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"values": self.values, "provenance": self.provenance, "history": self.history}
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    def with_value(self, name: str, value: float) -> "Params":
        other = Params(dict(self.values), dict(self.provenance), copy.deepcopy(self.history))
        knob = KNOBS[name]
        other.values[name] = round(min(knob.high, max(knob.low, value)), 6)
        return other

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def trust(self, method: str) -> float:
        return self.values[f"trust.{method}"]
