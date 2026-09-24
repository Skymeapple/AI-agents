"""Triage: decide which new papers are worth reading, and under which topic.

This is the step whose quality the agent most directly controls, and the one
where its effectiveness and its efficiency trade against each other. Flag too
little and the agent misses advances; flag too much and whoever reads the
digest -- a person or a model, both of which cost money -- wades through noise.

Scoring is deliberately simple and fully inspectable: a paper's score for a
topic is the weighted count of that topic's profile terms it contains (title
hits boosted), divided by a power of its length. There is no learned model to
drift silently. What changes over time is the profile -- which is exactly the
thing the agent iterates on, under measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from sciscout.discovery import tokenize
from sciscout.models import Work

from .config import AgentConfig


@dataclass
class Verdict:
    work: Work
    topic: str | None  # best-scoring topic, or None if nothing matched
    score: float  # score for that topic
    runner_up: float  # second-best topic score
    flagged: bool

    @property
    def margin(self) -> float:
        """How clearly the best topic beats the runner-up (>= 1)."""
        if self.score <= 0:
            return 1.0
        return self.score / self.runner_up if self.runner_up > 0 else float("inf")


def score_work(work: Work, config: AgentConfig) -> dict[str, float]:
    """Score ``work`` against every topic profile."""
    title = tokenize(work.title)
    abstract = tokenize(work.abstract)
    length = max(1, len(title) + len(abstract))
    boost = config.get("triage.title_boost")
    min_hits = int(config.get("triage.min_term_hits"))
    norm = length ** config.get("triage.length_norm")

    scores: dict[str, float] = {}
    for topic, profile in config.profiles.items():
        raw = 0.0
        hits: set[str] = set()
        for token in abstract:
            weight = profile.get(token)
            if weight:
                raw += weight
                hits.add(token)
        for token in title:
            weight = profile.get(token)
            if weight:
                raw += boost * weight
                hits.add(token)
        scores[topic] = raw / norm if len(hits) >= min_hits else 0.0
    return scores


def triage(works: list[Work], config: AgentConfig) -> list[Verdict]:
    """Verdicts for ``works``, most relevant first."""
    threshold = config.get("triage.threshold")
    verdicts = []
    for work in works:
        scores = score_work(work, config)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_topic, best = ranked[0] if ranked else (None, 0.0)
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        verdicts.append(Verdict(
            work=work,
            topic=best_topic if best > 0 else None,
            score=best,
            runner_up=runner_up,
            flagged=best >= threshold and best > 0,
        ))
    verdicts.sort(key=lambda v: v.score, reverse=True)
    return verdicts
