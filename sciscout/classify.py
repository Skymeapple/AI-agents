"""Routing a research direction to a commercialisation model.

A discovered direction arrives with no sector attached, and the sector decides
everything downstream: which development stages lie ahead, what they cost, how
long they take, which constraints bite. Getting it wrong does not produce a
slightly-off answer, it produces an answer about a different kind of technology.

Classification is lexical and rule-based: discipline labels and abstract
vocabulary, scored against keyword sets per sector. That is a weak method and it
is graded honestly -- a routed sector is ``assumed``, never ``observed``.

The important design choice is what happens when nothing fits. The temptation is
to pick the best-scoring sector regardless; a direction always lands somewhere
and the pipeline never visibly fails. This module does the opposite: below a
confidence floor it routes to ``generic`` and says so, and
:func:`coverage_report` lists exactly which directions fell through. On a corpus
spanning all of science most directions will not match any of the five modelled
archetypes, and a platform that hides that is lying about its coverage.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .models import Track
from .provenance import Estimate, assumed, observed

# Per sector: discipline fragments and abstract vocabulary. Discipline matches
# weigh more because a discipline label is a curator's judgement about the whole
# work, while a keyword may be one incidental sentence.
DISCIPLINE_WEIGHT = 3.0
KEYWORD_WEIGHT = 1.0

# A sector matched only by a discipline label, with none of its vocabulary
# anywhere in the corpus, is discounted to this fraction of its score. Discipline
# labels are broad -- "electrical engineering" covers battery packs and lithography
# alike -- so a discipline hit with no corroborating words is a hint, not a claim.
# Without this, one incidental discipline label scores as highly as a direction
# saturated in a sector's actual vocabulary, and suppresses the margin test.
UNCORROBORATED_DISCIPLINE_DISCOUNT = 0.4

SECTOR_RULES: dict[str, dict[str, list[str]]] = {
    "biopharma": {
        "disciplines": [
            "medicine", "pharmacology", "biochemistry", "immunology", "oncology",
            "genetics", "molecular biology", "microbiology", "neuroscience",
            "virology", "toxicology",
        ],
        "keywords": [
            "patient", "clinical", "therapeutic", "in vivo", "in vitro", "dosing",
            "efficacy", "phase i", "phase ii", "phase iii", "preclinical",
            "pharmacokinetic", "clinical trial", "antibody", "tumour", "tumor",
            "disease", "treatment", "receptor", "first-in-human",
        ],
    },
    "energy_hardware": {
        "disciplines": [
            # Deliberately no "condensed matter physics": it is a basic-physics
            # discipline covering quantum devices as much as power hardware, and
            # listing it here routed qubit work toward energy. Superconductivity
            # for the grid still matches via materials science and vocabulary.
            "materials science", "chemical engineering", "environmental science",
            "energy",
        ],
        "keywords": [
            "battery", "electrolyte", "photovoltaic", "solar", "hydrogen",
            "electrolysis", "catalyst", "carbon capture", "emission", "grid",
            "kwh", "energy density", "sorbent", "superconduct", "turbine",
            "electrode", "anode", "cathode", "levelised cost",
        ],
    },
    "semiconductors": {
        "disciplines": [
            "quantum physics", "electrical engineering", "applied physics",
            "optics", "nanotechnology",
        ],
        "keywords": [
            "qubit", "wafer", "transistor", "photonic", "lithography",
            "semiconductor", "silicon", "nanowire", "logic gate", "fab",
            "process integration", "chip", "on-chip", "coherence",
        ],
    },
    "software_ai": {
        "disciplines": [
            "computer science", "statistics", "mathematics", "information science",
        ],
        "keywords": [
            "neural", "training data", "algorithm", "dataset", "benchmark",
            "inference", "transformer", "software", "gpu", "latency",
            "machine learning", "deep learning", "fine-tun", "pretrain",
        ],
    },
    "agriculture_food": {
        "disciplines": [
            "plant science", "agronomy", "food science", "biotechnology",
            "soil science", "veterinary",
        ],
        "keywords": [
            "crop", "field trial", "seed", "harvest", "soil", "fermentation",
            "livestock", "farmer", "cultivar", "germplasm", "agronomic",
            "novel food", "titre", "bioreactor",
        ],
    },
}

# Deliberately NOT used as keywords, despite looking discriminative:
#
#   "model"   -- "mouse model", "theoretical model", "manufacturing model"
#   "cell"    -- a battery cell and a biological cell
#   "yield"   -- fab yield and crop yield
#   "strain"  -- a bacterial strain and mechanical strain
#   "device", "architecture", "node", "gate", "prediction", "compute"
#
# Each of these matches strongly in fields it has nothing to do with. Including
# them is how a lexical classifier quietly routes half of biology into software.

# Two conditions must both hold for a routing to be trusted. With five sectors,
# a uniform split gives each 20%, so a share-based floor alone is a weak test --
# 34% of a fragmented vote is not evidence of anything. The margin requirement
# does the real work: the leading sector must clearly beat the runner-up, not
# merely come first in a tie.
CONFIDENCE_FLOOR = 0.34
MARGIN_OVER_RUNNER_UP = 1.5


@dataclass
class SectorMatch:
    """How a direction was routed, and how well."""

    sector: str
    confidence: float
    scores: dict[str, float]
    evidence: list[str]

    @property
    def is_confident(self) -> bool:
        return self.sector != "generic"

    def as_estimate(self) -> Estimate[str]:
        if self.sector == "generic":
            return Estimate(
                "generic",
                assumed(
                    f"no commercialisation archetype matched this direction "
                    f"clearly enough (best sector took {self.confidence:.0%} of "
                    f"the score, needing {CONFIDENCE_FLOOR:.0%} and a "
                    f"{MARGIN_OVER_RUNNER_UP:g}x margin over the runner-up); "
                    f"routed to the generic model, whose stage costs and "
                    f"timelines describe no particular technology"
                ),
                label="sector",
            )
        return Estimate(
            self.sector,
            assumed(
                f"routed by keyword and discipline matching at {self.confidence:.0%} "
                f"of total score ({'; '.join(self.evidence[:3])}); lexical routing "
                f"is a weak classifier, so override it where you know the field"
            ),
            label="sector",
        )


def classify(track: Track, sectors_available: set[str] | None = None) -> SectorMatch:
    """Route ``track`` to a commercialisation sector.

    Args:
        track: The direction to route.
        sectors_available: Restrict to sectors the prior library actually
            defines. A sector we can name but cannot model is worse than
            ``generic``, because it implies a specificity the numbers do not have.
    """
    corpus = " ".join(w.text() for w in track.works).lower()
    disciplines = {d.lower() for d in track.disciplines()}

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for sector, rules in SECTOR_RULES.items():
        if sectors_available is not None and sector not in sectors_available:
            continue
        discipline_score = 0.0
        keyword_score = 0.0
        hits: list[str] = []
        for fragment in rules["disciplines"]:
            matched = [d for d in disciplines if fragment in d]
            if matched:
                discipline_score += DISCIPLINE_WEIGHT * len(matched)
                hits.append(f"discipline '{matched[0]}'")
        for keyword in rules["keywords"]:
            count = len(re.findall(re.escape(keyword), corpus))
            if count:
                # Normalise by corpus size so a large cluster does not outscore a
                # small one purely by having more text.
                keyword_score += KEYWORD_WEIGHT * min(
                    3.0, count / max(1, len(track.works))
                )
                hits.append(f"'{keyword}'")

        if keyword_score == 0.0:
            score = discipline_score * UNCORROBORATED_DISCIPLINE_DISCOUNT
            if discipline_score:
                hits.append("no sector vocabulary found, discipline match discounted")
        else:
            score = discipline_score + keyword_score

        if score > 0:
            scores[sector] = score
            evidence[sector] = hits

    if not scores:
        return SectorMatch("generic", 0.0, {}, ["no sector vocabulary matched"])

    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = best_score / total if total else 0.0
    margin = best_score / runner_up_score if runner_up_score > 0 else float("inf")

    if confidence < CONFIDENCE_FLOOR or margin < MARGIN_OVER_RUNNER_UP:
        return SectorMatch("generic", confidence, scores, evidence.get(best, []))
    return SectorMatch(best, confidence, scores, evidence[best])


def coverage_report(matches: dict[str, SectorMatch]) -> str:
    """Which directions we can model, and which we cannot.

    The honest half of breadth. Anything routed to ``generic`` is a direction the
    platform can rank on importance but cannot cost, and listing them is how that
    gap stays visible instead of dissolving into an average.
    """
    by_sector: Counter[str] = Counter(m.sector for m in matches.values())
    unmodelled = sorted(name for name, m in matches.items() if not m.is_confident)

    lines = ["| Sector | Directions |", "| --- | --- |"]
    for sector, count in by_sector.most_common():
        lines.append(f"| {sector} | {count} |")
    lines.append("")

    if unmodelled:
        lines.append(
            f"**{len(unmodelled)} of {len(matches)} directions "
            f"({len(unmodelled) / len(matches):.0%}) have no commercialisation "
            "model.** They are ranked on importance and urgency, which need only "
            "the research record, but their costs and timelines come from the "
            "generic fallback and describe no particular technology. Treat their "
            "investment cases as placeholders until a sector model exists:"
        )
        lines.append("")
        for name in unmodelled:
            lines.append(f"- {name}")
        lines.append("")
    else:
        lines.append(
            "Every direction routed to a modelled sector. Note that this says the "
            "vocabulary matched, not that the routing is right -- lexical "
            "classification is a weak method and the sector drives every cost and "
            "timeline downstream, so spot-check the assignments."
        )
        lines.append("")
    return "\n".join(lines)
