"""Technology Readiness Level inference.

TRL (1 = basic principles observed, 9 = proven in operation) is the hinge between
the science half of this platform and the money half. Everything commercial --
how long to market, how much it costs to get there, which constraints bite --
follows from where a direction currently sits on that scale.

Inference is deliberately Bayesian and deliberately returns a *distribution*, not
a point. Reading TRL off abstracts is a weak signal, and a model that answers
"TRL 4" hides that while one answering "most likely 4, but 3-6 is all consistent
with the evidence" does not. The downstream Monte Carlo consumes the full
distribution, so that uncertainty propagates into the ROI spread rather than
being quietly discarded at the first step.

The prior is deliberately flat-ish over the low-to-middle range, because a
research corpus is overwhelmingly made of work below TRL 6; evidence has to
argue a direction upward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Maturity, Track
from ..provenance import Estimate, Provenance, assumed, observed

TRL_NAMES: dict[int, str] = {
    1: "Basic principles observed",
    2: "Technology concept formulated",
    3: "Experimental proof of concept",
    4: "Validated in laboratory",
    5: "Validated in relevant environment",
    6: "Demonstrated in relevant environment",
    7: "Prototype demonstrated in operational environment",
    8: "System complete and qualified",
    9: "Proven in operational environment",
}

# Prior mass by TRL. Weighted toward the lower half: that is what a research
# corpus is mostly made of, and a direction should have to earn a high reading.
_PRIOR: dict[int, float] = {
    1: 0.14, 2: 0.16, 3: 0.18, 4: 0.16, 5: 0.12,
    6: 0.09, 7: 0.07, 8: 0.05, 9: 0.03,
}

# Lexical evidence. Each pattern, when it appears in a meaningful share of a
# track's works, multiplies the likelihood of the TRL band it indicates.
# These are indicative phrases, not a validated classifier -- the provenance on
# every TRL estimate says so.
_SIGNALS: tuple[tuple[str, tuple[int, ...], float], ...] = (
    (r"\b(?:first principles|theoretical|ab initio|we propose a theory)\b", (1, 2), 2.0),
    (r"\b(?:simulation|numerical study|computational model|in silico)\b", (2, 3), 1.6),
    (r"\b(?:proof of concept|proof-of-concept|we demonstrate for the first time)\b", (3, 4), 2.2),
    (r"\b(?:in vitro|bench(?:-| )scale|laboratory (?:validation|demonstration))\b", (4, 5), 2.0),
    (r"\b(?:in vivo|animal model|mouse model|preclinical)\b", (4, 5, 6), 1.8),
    (r"\b(?:pilot (?:plant|line|scale)|scale[- ]up|relevant environment)\b", (5, 6, 7), 2.2),
    (r"\b(?:prototype|field (?:trial|test)|operational environment)\b", (6, 7, 8), 2.2),
    (r"\b(?:phase (?:i|1)\b|first[- ]in[- ]human)\b", (6, 7), 2.4),
    (r"\b(?:phase (?:ii|2)\b)", (7,), 2.4),
    (r"\b(?:phase (?:iii|3)\b|pivotal trial)\b", (8,), 2.6),
    (r"\b(?:commercial(?:ly)? (?:deployed|available)|in production|fda[- ]approved|ce[- ]marked)\b", (8, 9), 3.0),
    (r"\b(?:manufactur(?:ing|ability)|yield|throughput|cost per (?:unit|watt|kg))\b", (6, 7, 8), 1.5),
)


@dataclass
class TRLBelief:
    """A probability distribution over TRL 1-9."""

    probabilities: dict[int, float]
    evidence: list[str]
    grade_note: Provenance

    @property
    def mode(self) -> int:
        return max(self.probabilities, key=lambda k: self.probabilities[k])

    @property
    def mean(self) -> float:
        return sum(level * p for level, p in self.probabilities.items())

    def credible_range(self, mass: float = 0.8) -> tuple[int, int]:
        """Narrowest contiguous TRL band holding at least ``mass`` probability."""
        levels = sorted(self.probabilities)
        best: tuple[int, int] | None = None
        for i, low in enumerate(levels):
            cumulative = 0.0
            for j in range(i, len(levels)):
                cumulative += self.probabilities[levels[j]]
                if cumulative >= mass:
                    if best is None or (levels[j] - low) < (best[1] - best[0]):
                        best = (low, levels[j])
                    break
        return best or (levels[0], levels[-1])

    def describe(self) -> str:
        low, high = self.credible_range()
        return (
            f"TRL {self.mode} ({TRL_NAMES[self.mode]}); "
            f"80% credible range {low}-{high}, mean {self.mean:.1f}"
        )

    def as_estimate(self) -> Estimate[int]:
        return Estimate(self.mode, self.grade_note, label="TRL (modal)")

    def sample(self, rng) -> int:
        """Draw one TRL from the belief. Used by the Monte Carlo engine."""
        levels = sorted(self.probabilities)
        weights = [self.probabilities[level] for level in levels]
        return int(rng.choice(levels, p=weights))


def infer_trl(track: Track) -> TRLBelief:
    """Infer a TRL distribution for ``track``.

    An analyst-supplied ``trl_override`` short-circuits inference, producing a
    tight distribution around the stated level -- tight but not a spike, because
    even a domain expert's single number carries some uncertainty and the
    downstream model should see it.
    """
    if track.trl_override is not None:
        return _override_belief(track)

    if not track.works:
        return TRLBelief(
            dict(_PRIOR),
            ["no works; prior only"],
            assumed(
                "track carries no works, so TRL is the untouched prior and "
                "conveys no information about this direction"
            ),
        )

    posterior = dict(_PRIOR)
    n_works = len(track.works)
    evidence = _apply_lexical_signals(posterior, track)
    evidence += _apply_structural_signals(posterior, track)

    if not evidence:
        note = assumed(
            f"no readiness signals found in {n_works} works; TRL is the prior, "
            "which reflects where research corpora generally sit rather than "
            "anything about this direction"
        )
    else:
        note = Provenance(
            observed("x").grade,
            f"inferred from {len(evidence)} readiness signals across {n_works} "
            "works using indicative-phrase matching, which is a weak classifier: "
            "treat the distribution's width as real and override it where a "
            "domain expert knows better",
        )
    return TRLBelief(_normalise(posterior), evidence, note)


def _normalise(weights: dict[int, float]) -> dict[int, float]:
    total = sum(weights.values())
    return {level: weight / total for level, weight in weights.items()}


def _override_belief(track: Track) -> TRLBelief:
    """Tight distribution around an analyst-supplied TRL, clamped to 1-9."""
    level = max(1, min(9, track.trl_override))
    probabilities = {t: 0.0 for t in TRL_NAMES}
    probabilities[level] = 0.7
    for neighbour in (level - 1, level + 1):
        if neighbour in probabilities:
            probabilities[neighbour] = 0.15
    return TRLBelief(
        _normalise(probabilities),
        [f"analyst override: TRL {level}"],
        observed(f"analyst-supplied TRL {level} for track {track.id}"),
    )


def _boost(posterior: dict[int, float], levels, multiplier: float) -> None:
    for level in levels:
        posterior[level] *= multiplier


def _apply_lexical_signals(posterior: dict[int, float], track: Track) -> list[str]:
    """Reweight ``posterior`` in place by indicative phrases; return the evidence."""
    evidence: list[str] = []
    corpus = " \n ".join(w.text() for w in track.works)
    n_works = len(track.works)

    for pattern, levels, strength in _SIGNALS:
        matches = len(re.findall(pattern, corpus, re.IGNORECASE))
        if matches == 0:
            continue
        # Scale by prevalence, not raw count: one mention of "prototype" across
        # 50 papers is noise, the same word in half of them is a signal.
        prevalence = min(1.0, matches / max(1, n_works))
        multiplier = 1.0 + (strength - 1.0) * prevalence
        _boost(posterior, levels, multiplier)
        phrase = pattern.replace("\\b", "").replace("\\", "")
        bands = "/".join(str(level) for level in levels)
        evidence.append(
            f"{matches} mention(s) matching {phrase!r} -> TRL {bands} x{multiplier:.2f}"
        )
    return evidence


def _apply_structural_signals(posterior: dict[int, float], track: Track) -> list[str]:
    """Reweight ``posterior`` in place by record types and industry share."""
    evidence: list[str] = []
    n_works = len(track.works)

    # Registered clinical trials are strong structural evidence, not lexical.
    clinical = sum(1 for w in track.works if w.maturity is Maturity.CLINICAL_REGISTRY)
    if clinical:
        _boost(posterior, (6, 7, 8), 1.0 + 1.5 * min(1.0, clinical / n_works))
        evidence.append(f"{clinical} registered clinical trial record(s) -> TRL 6-8")

    patents = sum(1 for w in track.works if w.maturity is Maturity.PATENT)
    if patents:
        _boost(posterior, (4, 5, 6, 7), 1.0 + 1.0 * min(1.0, patents / n_works))
        evidence.append(f"{patents} patent record(s) -> TRL 4-7")

    industry = track.industry_fraction()
    if industry > 0.2:
        _boost(posterior, (5, 6, 7, 8), 1.0 + industry)
        evidence.append(f"{industry:.0%} of works industry-affiliated -> TRL 5-8")
    return evidence
