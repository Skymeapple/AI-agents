"""The agent's configuration: the part of itself it is allowed to change.

autoscout iterates on this and nothing else. It does not rewrite its own code.
That is a deliberate boundary rather than a missing feature: a change to a
bounded, validated configuration can be evaluated, gated, logged and reverted
mechanically, whereas a change to code needs a reviewer. Ideas that cannot be
expressed as configuration are written to an ideas backlog for a human (see
:mod:`autoscout.proposers`), which is where structural change belongs.

The configuration has three parts:

* numeric parameters with declared bounds (:data:`SPACE`) -- triage thresholds,
  discovery settings, harvest budget;
* topic profiles -- the terms the agent believes signal each agenda topic, with
  weights. These start from ``agenda.yaml`` and grow from the literature;
* nothing else. Anything a proposer returns outside this is rejected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sciscout.discovery import STOPWORDS

AGENDA_PATH = Path(__file__).resolve().parent / "agenda.yaml"
TERM_RE = re.compile(r"^[a-z][a-z-]{2,}$")
MAX_TERMS_PER_TOPIC = 60


@dataclass(frozen=True)
class Param:
    """One tunable number, with the range a proposer may move it within."""

    low: float
    high: float
    integer: bool = False
    log: bool = False  # mutate on a log scale (for scale-free quantities)
    doc: str = ""

    def clip(self, value: float) -> float:
        value = min(self.high, max(self.low, float(value)))
        return float(round(value)) if self.integer else value


SPACE: dict[str, Param] = {
    "triage.threshold": Param(0.02, 5.0, log=True,
                              doc="relevance score a paper needs to be flagged for reading"),
    "triage.length_norm": Param(0.0, 1.0,
                                doc="exponent dividing score by document length"),
    "triage.title_boost": Param(1.0, 5.0,
                                doc="weight of a title hit relative to an abstract hit"),
    "triage.min_term_hits": Param(1, 5, integer=True,
                                  doc="distinct profile terms a paper must contain"),
    "discovery.threshold": Param(0.05, 0.45,
                                 doc="average-linkage similarity floor"),
    "discovery.min_df": Param(1, 6, integer=True, doc="minimum document frequency"),
    "discovery.max_df_ratio": Param(0.15, 0.9, doc="maximum document-frequency share"),
    "discovery.min_cluster_size": Param(2, 10, integer=True,
                                        doc="smallest cluster reported as a direction"),
    "scout.terms_per_query": Param(2, 10, integer=True,
                                   doc="profile terms OR-ed into each live query"),
    "scout.limit_per_query": Param(20, 200, integer=True,
                                   doc="papers requested per topic per live harvest"),
}

DEFAULTS: dict[str, float] = {
    "triage.threshold": 0.6,
    "triage.length_norm": 0.5,
    "triage.title_boost": 2.0,
    "triage.min_term_hits": 2,
    "discovery.threshold": 0.16,
    "discovery.min_df": 2,
    "discovery.max_df_ratio": 0.5,
    "discovery.min_cluster_size": 3,
    "scout.terms_per_query": 5,
    "scout.limit_per_query": 60,
}


def valid_term(term: str) -> bool:
    """Whether ``term`` is something the tokenizer could ever produce."""
    return bool(TERM_RE.match(term)) and term not in STOPWORDS


@dataclass
class AgentConfig:
    params: dict[str, float] = field(default_factory=lambda: dict(DEFAULTS))
    # topic -> term -> weight
    profiles: dict[str, dict[str, float]] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.params[name]

    def topics(self) -> list[str]:
        return sorted(self.profiles)

    def copy(self) -> "AgentConfig":
        return AgentConfig(params=dict(self.params), profiles=copy.deepcopy(self.profiles))

    def fingerprint(self) -> str:
        """Stable short hash, so identical configs are recognised as such."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "params": {k: float(self.params[k]) for k in sorted(self.params)},
            "profiles": {t: dict(sorted(p.items())) for t, p in sorted(self.profiles.items())},
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "AgentConfig":
        params = dict(DEFAULTS)
        for name, value in raw.get("params", {}).items():
            if name not in SPACE:
                raise ValueError(f"unknown parameter {name!r} in config")
            params[name] = SPACE[name].clip(value)
        return cls(params=params, profiles={t: dict(p) for t, p in raw.get("profiles", {}).items()})

    @classmethod
    def from_agenda(cls, path: Path = AGENDA_PATH) -> "AgentConfig":
        agenda = yaml.safe_load(Path(path).read_text())
        profiles = {}
        for topic, spec in agenda["topics"].items():
            terms = [t for t in spec["terms"] if valid_term(t)]
            profiles[topic] = {t: 1.0 for t in terms}
        return cls(params=dict(DEFAULTS), profiles=profiles)

    def diff(self, other: "AgentConfig") -> list[str]:
        """Human-readable list of what ``other`` changes relative to self."""
        lines = []
        for name in sorted(SPACE):
            a, b = self.params[name], other.params[name]
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                lines.append(f"{name}: {a:.4g} -> {b:.4g}")
        for topic in sorted(set(self.profiles) | set(other.profiles)):
            mine, theirs = self.profiles.get(topic, {}), other.profiles.get(topic, {})
            added = sorted(set(theirs) - set(mine))
            removed = sorted(set(mine) - set(theirs))
            reweighted = sorted(t for t in set(mine) & set(theirs)
                                if not math.isclose(mine[t], theirs[t]))
            if added:
                lines.append(f"{topic}: + {', '.join(added)}")
            if removed:
                lines.append(f"{topic}: - {', '.join(removed)}")
            if reweighted:
                lines.append(f"{topic}: reweighted {', '.join(reweighted)}")
        return lines


@dataclass
class Change:
    """A proposed edit to a config. Applying it always validates."""

    set_params: dict[str, float] = field(default_factory=dict)
    add_terms: dict[str, dict[str, float]] = field(default_factory=dict)
    remove_terms: dict[str, list[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.set_params or any(self.add_terms.values())
                    or any(self.remove_terms.values()))

    def apply(self, base: AgentConfig) -> tuple[AgentConfig, list[str]]:
        """Apply to a copy of ``base``; return it and any parts that were rejected.

        Rejection is itemised rather than all-or-nothing so that one malformed
        suggestion from a language model doesn't discard the good ones -- but
        each rejection is reported, never silently dropped.
        """
        config = base.copy()
        rejected: list[str] = []
        for name, value in self.set_params.items():
            if name not in SPACE:
                rejected.append(f"unknown parameter {name!r}")
                continue
            try:
                config.params[name] = SPACE[name].clip(float(value))
            except (TypeError, ValueError):
                rejected.append(f"non-numeric value for {name!r}")
        for topic, terms in self.add_terms.items():
            if topic not in config.profiles:
                rejected.append(f"unknown topic {topic!r}")
                continue
            for term, weight in terms.items():
                term = term.strip().lower()
                if not valid_term(term):
                    rejected.append(f"{term!r} is not a valid term")
                elif len(config.profiles[topic]) >= MAX_TERMS_PER_TOPIC and term not in config.profiles[topic]:
                    rejected.append(f"{topic} profile full ({MAX_TERMS_PER_TOPIC} terms)")
                else:
                    config.profiles[topic][term] = min(3.0, max(0.05, float(weight)))
        for topic, terms in self.remove_terms.items():
            profile = config.profiles.get(topic)
            if profile is None:
                rejected.append(f"unknown topic {topic!r}")
                continue
            for term in terms:
                if term in profile and len(profile) > 1:
                    del profile[term]
                else:
                    rejected.append(f"cannot remove {term!r} from {topic}")
        return config, rejected

    def to_dict(self) -> dict:
        return {"set_params": self.set_params, "add_terms": self.add_terms,
                "remove_terms": self.remove_terms}
