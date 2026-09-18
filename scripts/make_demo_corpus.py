"""Generate a clearly-labelled SYNTHETIC corpus for demos and tests.

This exists because the real sources (arXiv, OpenAlex) are not always reachable
-- managed environments frequently block outbound egress -- and a pipeline you
cannot run is a pipeline you cannot evaluate.

Every record here is machine-generated. Identifiers are prefixed ``synthetic:``,
author names are placeholders, and institutions are letters. Nothing in this file
is a real paper, a real citation count or a real research group, and it must
never be presented as evidence about the actual state of any field. It exercises
the pipeline; it does not inform anyone about batteries.

Replace it by pointing the CLI at a live source as soon as you have egress:

    python -m sciscout.cli harvest --source openalex --query "..." --out data/corpus.json

Run:
    python scripts/make_demo_corpus.py
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.json"
SEED = 20260918

# Each track gets a shape: how activity grows, how cited it is, how far along it
# is, and how much industry has moved in. These shapes are what the scoring
# engine is meant to discriminate between.
TRACKS = [
    {
        "track": "solid-state-sodium",
        "name": "Solid-state sodium batteries",
        "sector": "energy_hardware",
        "disciplines": ["materials science", "chemistry", "electrical engineering"],
        "yearly": {2021: 3, 2022: 6, 2023: 11, 2024: 19, 2025: 28, 2026: 22},
        "citation_scale": 14,
        "industry_rate": 0.35,
        "n_institutions": 11,
        "phrases": [
            "We report a solid electrolyte with room-temperature ionic conductivity",
            "Pilot line scale-up of sodium cell manufacturing demonstrates yield improvements",
            "A prototype pouch cell is validated in a relevant environment over 1000 cycles",
            "Interfacial degradation is characterised by operando spectroscopy",
            "Cost per kWh is projected from a bottom-up manufacturing model",
            "Proof of concept for a chloride-based superionic conductor",
        ],
    },
    {
        "track": "targeted-degradation",
        "name": "Targeted protein degradation",
        "sector": "biopharma",
        "disciplines": ["biochemistry", "medicine", "pharmacology"],
        "yearly": {2021: 8, 2022: 12, 2023: 15, 2024: 21, 2025: 26, 2026: 17},
        "citation_scale": 26,
        "industry_rate": 0.55,
        "n_institutions": 14,
        "phrases": [
            "A molecular glue degrader shows selective substrate recruitment in vitro",
            "Preclinical pharmacokinetics in a mouse model support once-daily dosing",
            "Phase I first-in-human dose escalation establishes a tolerable range",
            "Phase 2 expansion cohort results indicate objective responses",
            "Structure-guided optimisation improves ternary complex cooperativity",
            "Resistance mechanisms are mapped by genome-wide screening",
        ],
    },
    {
        "track": "neuromorphic-photonics",
        "name": "Neuromorphic photonic computing",
        "sector": "semiconductors",
        "disciplines": ["physics", "computer science", "electrical engineering"],
        "yearly": {2021: 4, 2022: 5, 2023: 7, 2024: 8, 2025: 9, 2026: 6},
        "citation_scale": 9,
        "industry_rate": 0.18,
        "n_institutions": 8,
        "phrases": [
            "We propose a theoretical framework for coherent optical matrix multiplication",
            "A simulation study predicts energy per operation below electronic baselines",
            "Experimental proof of concept on a silicon photonic platform",
            "Wafer-scale process integration remains limited by yield",
            "A scalable architecture for optical recurrent networks is described",
            "First demonstration of on-chip nonlinear activation",
        ],
    },
    {
        "track": "room-temp-superconductivity",
        "name": "Ambient-pressure high-Tc superconductivity",
        "sector": "energy_hardware",
        "disciplines": ["condensed matter physics", "materials science"],
        "yearly": {2021: 2, 2022: 3, 2023: 14, 2024: 6, 2025: 4, 2026: 3},
        "citation_scale": 18,
        "industry_rate": 0.05,
        "n_institutions": 9,
        "phrases": [
            "We report evidence of a superconducting transition from first principles",
            "An ab initio study identifies candidate hydride stoichiometries",
            "Attempts to replicate the reported transition are described",
            "Magnetic susceptibility measurements are inconsistent with bulk superconductivity",
            "A theoretical model is proposed for the observed resistivity drop",
        ],
    },
]


def build() -> list[dict]:
    rng = random.Random(SEED)
    asof = dt.date(2026, 9, 18)
    records: list[dict] = []

    for spec in TRACKS:
        institutions = [f"Institution {chr(65 + i)}" for i in range(spec["n_institutions"])]
        counter = 0
        for year, count in spec["yearly"].items():
            for _ in range(count):
                counter += 1
                day = rng.randint(1, 28)
                month = rng.randint(1, 12)
                if year == 2026 and month > 9:
                    month = rng.randint(1, 9)
                published = dt.date(year, month, day)
                age_years = max(0.05, (asof - published).days / 365.25)

                # Citations accumulate roughly linearly with age, with a
                # heavy-tailed multiplier so a few works dominate -- which is how
                # real citation distributions behave.
                tail = rng.paretovariate(1.6)
                citations = int(spec["citation_scale"] * age_years * min(tail, 12) * rng.uniform(0.3, 1.4))

                n_authors = rng.randint(2, 7)
                authors = []
                chosen = rng.sample(institutions, k=min(n_authors, len(institutions)))
                for index, institution in enumerate(chosen):
                    is_industry = rng.random() < spec["industry_rate"]
                    authors.append(
                        {
                            "name": f"Author {counter}-{index}",
                            "affiliation": (
                                f"{institution} (industry)" if is_industry else institution
                            ),
                            "is_industry": is_industry,
                        }
                    )

                if rng.random() < 0.55:
                    maturity = "peer_reviewed"
                elif rng.random() < 0.85:
                    maturity = "preprint"
                elif spec["sector"] == "biopharma":
                    maturity = "clinical_registry"
                else:
                    maturity = "patent"

                abstract = " ".join(rng.sample(spec["phrases"], k=min(3, len(spec["phrases"]))))
                records.append(
                    {
                        "id": f"synthetic:{spec['track']}:{counter:04d}",
                        "title": f"{spec['name']}: study {counter}",
                        "abstract": abstract,
                        "authors": authors,
                        "published": published.isoformat(),
                        "venue": "Synthetic Journal of Demonstration Data",
                        "maturity": maturity,
                        "disciplines": rng.sample(
                            spec["disciplines"], k=rng.randint(1, len(spec["disciplines"]))
                        ),
                        "citations": citations,
                        "url": None,
                        "source": "synthetic",
                        "extra": {
                            "track": spec["track"],
                            "track_name": spec["name"],
                            "sector": spec["sector"],
                            "SYNTHETIC": True,
                        },
                    }
                )
    return records


if __name__ == "__main__":
    records = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} SYNTHETIC records to {OUT}")
