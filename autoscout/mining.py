"""Learning new vocabulary from the latest papers.

Fields rename things constantly. A research agent that looks for last year's
words gets steadily worse at spotting this year's work, without any error to
tell it so -- recall just quietly falls. This module is how autoscout keeps up:
among papers it has confidently placed under a topic, it finds terms that are
markedly over-represented relative to everything else it has read, and
nominates them as candidate additions to that topic's profile.

Nomination is not adoption. Mined terms go to :mod:`autoscout.proposers`, which
turns them into proposed changes, and those face the same gate as any other
change. That separation is what stops the obvious failure of learning from your
own output: mining from papers the agent itself selected is a feedback loop,
and without an independent check it would happily amplify its own mistakes.

The statistic is a log odds ratio of document frequency, inside-topic against
outside, with add-one smoothing and its standard error. Statistics accumulate
across cycles, so a term that is rare in any one week can still earn a
nomination as evidence builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sciscout.discovery import tokenize

from .config import AgentConfig, valid_term
from .state import MAX_EVIDENCE, Knowledge
from .triage import Verdict

CONFIDENT_MARGIN = 1.5  # best topic must beat runner-up by this factor
MIN_TOPIC_DOCS = 3
MIN_Z = 2.5
MAX_BACKGROUND_SHARE = 0.25  # a term in a quarter of everything is not topical


@dataclass
class MinedTerm:
    topic: str
    term: str
    z: float
    log_odds: float
    topic_docs: int
    evidence: list[str]


def update_statistics(knowledge: Knowledge, verdicts: list[Verdict]) -> int:
    """Fold a batch of triaged papers into the running term statistics.

    Returns how many papers counted as confident topic examples.
    """
    confident = 0
    for verdict in verdicts:
        terms = set(tokenize(verdict.work.text()))
        knowledge.n_docs += 1
        for term in terms:
            knowledge.background_df[term] = knowledge.background_df.get(term, 0) + 1
        if not (verdict.flagged and verdict.topic and verdict.margin >= CONFIDENT_MARGIN):
            continue
        confident += 1
        topic = verdict.topic
        knowledge.topic_docs[topic] = knowledge.topic_docs.get(topic, 0) + 1
        topic_df = knowledge.topic_df.setdefault(topic, {})
        evidence = knowledge.term_evidence.setdefault(topic, {})
        for term in terms:
            topic_df[term] = topic_df.get(term, 0) + 1
            ids = evidence.setdefault(term, [])
            if len(ids) < MAX_EVIDENCE:
                ids.append(verdict.work.id)
    return confident


def mine(knowledge: Knowledge, config: AgentConfig, limit_per_topic: int = 10) -> list[MinedTerm]:
    """Candidate new terms per topic, strongest evidence first."""
    known = {term for profile in config.profiles.values() for term in profile}
    total = knowledge.n_docs
    out: list[MinedTerm] = []
    if total == 0:
        return out
    for topic in config.topics():
        n_in = knowledge.topic_docs.get(topic, 0)
        n_out = total - n_in
        if n_in < MIN_TOPIC_DOCS or n_out <= 0:
            continue
        rejected = knowledge.rejected_terms.get(topic, {})
        candidates = []
        for term, y_in in knowledge.topic_df.get(topic, {}).items():
            if y_in < MIN_TOPIC_DOCS or term in known or term in rejected or not valid_term(term):
                continue
            background = knowledge.background_df.get(term, 0)
            if background / total > MAX_BACKGROUND_SHARE:
                continue
            y_out = max(0, background - y_in)
            log_odds = (math.log((y_in + 1) / (n_in - y_in + 1))
                        - math.log((y_out + 1) / (n_out - y_out + 1)))
            se = math.sqrt(1 / (y_in + 1) + 1 / (n_in - y_in + 1)
                           + 1 / (y_out + 1) + 1 / (n_out - y_out + 1))
            z = log_odds / se
            if z >= MIN_Z:
                candidates.append(MinedTerm(
                    topic, term, z, log_odds, y_in,
                    knowledge.term_evidence.get(topic, {}).get(term, []),
                ))
        candidates.sort(key=lambda m: m.z, reverse=True)
        out.extend(candidates[:limit_per_topic])
    return out
