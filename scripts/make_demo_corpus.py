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
            "Interfacial dendrite growth is characterised by operando spectroscopy",
            "Cost per kWh is projected from a bottom-up manufacturing model",
            "Proof of concept for a chloride-based superionic sodium conductor",
        ],
    },
    {
        "track": "perovskite-tandem-pv",
        "name": "Perovskite tandem photovoltaics",
        "sector": "energy_hardware",
        "disciplines": ["materials science", "physics", "chemistry"],
        "yearly": {2021: 5, 2022: 9, 2023: 14, 2024: 20, 2025: 24, 2026: 16},
        "citation_scale": 17,
        "industry_rate": 0.30,
        "n_institutions": 12,
        "phrases": [
            "A perovskite silicon tandem cell achieves certified power conversion efficiency",
            "Photostability under damp heat remains the barrier to module qualification",
            "Scale-up to large-area modules is demonstrated on a pilot coating line",
            "Lead sequestration strategies address the toxicity constraint",
            "Bandgap tuning of the wide-gap absorber reduces open-circuit voltage deficit",
            "Encapsulation extends operational lifetime under accelerated ageing",
        ],
    },
    {
        "track": "green-hydrogen-electrolysis",
        "name": "Green hydrogen electrolysis catalysts",
        "sector": "energy_hardware",
        "disciplines": ["chemistry", "materials science", "chemical engineering"],
        "yearly": {2021: 6, 2022: 9, 2023: 13, 2024: 17, 2025: 21, 2026: 14},
        "citation_scale": 13,
        "industry_rate": 0.28,
        "n_institutions": 10,
        "phrases": [
            "An iridium-free anode catalyst sustains current density in acidic electrolysis",
            "Membrane electrode assembly durability is tested over 2000 hours",
            "Levelised cost of hydrogen is modelled against electricity price",
            "A pilot stack demonstrates operation in a relevant industrial environment",
            "Catalyst degradation is traced to metal dissolution at the interface",
            "Proof of concept for anion exchange membrane water splitting",
        ],
    },
    {
        "track": "direct-air-capture",
        "name": "Direct air carbon capture",
        "sector": "energy_hardware",
        "disciplines": ["chemical engineering", "chemistry", "environmental science"],
        "yearly": {2021: 4, 2022: 7, 2023: 10, 2024: 15, 2025: 19, 2026: 13},
        "citation_scale": 11,
        "industry_rate": 0.40,
        "n_institutions": 9,
        "phrases": [
            "An amine sorbent achieves working capacity under ambient humidity",
            "Regeneration energy dominates the cost per tonne of carbon dioxide removed",
            "A pilot plant demonstrates continuous contactor operation",
            "Sorbent degradation over thermal cycling limits lifetime",
            "Techno-economic analysis projects cost at scale against policy credits",
            "Proof of concept for electrochemical carbon dioxide separation",
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
        "track": "crispr-base-editing",
        "name": "CRISPR base and prime editing",
        "sector": "biopharma",
        "disciplines": ["genetics", "molecular biology", "medicine"],
        "yearly": {2021: 9, 2022: 13, 2023: 17, 2024: 22, 2025: 25, 2026: 16},
        "citation_scale": 30,
        "industry_rate": 0.45,
        "n_institutions": 13,
        "phrases": [
            "A base editor corrects a pathogenic point mutation with minimal bystander editing",
            "Off-target editing is quantified by unbiased genome-wide sequencing",
            "Lipid nanoparticle delivery achieves hepatic editing in a mouse model",
            "Phase I first-in-human data show durable protein reduction",
            "Prime editing efficiency is improved by engineered reverse transcriptase",
            "Preclinical toxicology supports an investigational new drug filing",
        ],
    },
    {
        "track": "car-t-solid-tumour",
        "name": "CAR-T for solid tumours",
        "sector": "biopharma",
        "disciplines": ["immunology", "oncology", "medicine"],
        "yearly": {2021: 10, 2022: 12, 2023: 14, 2024: 16, 2025: 18, 2026: 11},
        "citation_scale": 24,
        "industry_rate": 0.50,
        "n_institutions": 12,
        "phrases": [
            "An armoured chimeric antigen receptor resists the immunosuppressive microenvironment",
            "Phase 2 results show objective responses in a refractory solid tumour cohort",
            "Cytokine release syndrome is managed with a tunable safety switch",
            "Preclinical models demonstrate trafficking into the tumour",
            "Manufacturing vein-to-vein time constrains commercial delivery",
            "Antigen escape drives relapse in treated patients",
        ],
    },
    {
        "track": "microbiome-therapeutics",
        "name": "Gut microbiome therapeutics",
        "sector": "biopharma",
        "disciplines": ["microbiology", "medicine", "immunology"],
        "yearly": {2021: 11, 2022: 12, 2023: 12, 2024: 11, 2025: 10, 2026: 6},
        "citation_scale": 12,
        "industry_rate": 0.35,
        "n_institutions": 11,
        "phrases": [
            "A defined bacterial consortium restores colonisation resistance",
            "Phase 2 trial of a live biotherapeutic misses its primary endpoint",
            "Metagenomic sequencing associates strain abundance with response",
            "Preclinical gnotobiotic models establish a causal mechanism",
            "Regulatory classification of live biotherapeutics remains unsettled",
            "Donor variability confounds replication across sites",
        ],
    },
    {
        "track": "quantum-error-correction",
        "name": "Quantum error correction",
        "sector": "semiconductors",
        "disciplines": ["physics", "computer science", "quantum physics"],
        "yearly": {2021: 6, 2022: 9, 2023: 13, 2024: 18, 2025: 23, 2026: 15},
        "citation_scale": 19,
        "industry_rate": 0.42,
        "n_institutions": 10,
        "phrases": [
            "A surface code logical qubit is operated below the error threshold",
            "Syndrome extraction is demonstrated with real-time decoding",
            "Logical error rate is suppressed as code distance increases",
            "A theoretical framework bounds the overhead for fault tolerance",
            "Prototype control electronics scale to thousands of physical qubits",
            "Proof of concept for magic state distillation on hardware",
        ],
    },
    {
        "track": "topological-qubits",
        "name": "Topological qubits",
        "sector": "semiconductors",
        "disciplines": ["condensed matter physics", "quantum physics", "physics"],
        "yearly": {2021: 5, 2022: 6, 2023: 8, 2024: 7, 2025: 6, 2026: 4},
        "citation_scale": 15,
        "industry_rate": 0.30,
        "n_institutions": 7,
        "phrases": [
            "Evidence for Majorana zero modes in a semiconductor nanowire is reported",
            "Attempts to replicate the reported conductance quantisation are described",
            "An ab initio study identifies candidate topological superconductor materials",
            "Disorder is shown to produce signatures mimicking the topological phase",
            "A theoretical model proposes braiding operations for fault-tolerant gates",
            "Measurement of the topological gap remains inconsistent between groups",
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
    {
        "track": "protein-structure-prediction",
        "name": "Protein structure and design models",
        "sector": "software_ai",
        "disciplines": ["computer science", "structural biology", "biochemistry"],
        "yearly": {2021: 7, 2022: 12, 2023: 18, 2024: 24, 2025: 29, 2026: 19},
        "citation_scale": 28,
        "industry_rate": 0.48,
        "n_institutions": 12,
        "phrases": [
            "A diffusion model generates de novo binders with experimentally validated affinity",
            "Structure prediction accuracy is benchmarked against held-out targets",
            "A general framework for conditioning on functional constraints is described",
            "Wet-lab validation confirms a minority of computationally designed candidates",
            "Training data scarcity limits generalisation to novel folds",
            "The platform enables rapid iteration between design and assay",
        ],
    },
    {
        "track": "climate-ml-emulators",
        "name": "Machine learning climate emulators",
        "sector": "software_ai",
        "disciplines": ["computer science", "atmospheric science", "environmental science"],
        "yearly": {2021: 4, 2022: 8, 2023: 13, 2024: 19, 2025: 24, 2026: 17},
        "citation_scale": 16,
        "industry_rate": 0.33,
        "n_institutions": 11,
        "phrases": [
            "A neural weather model matches numerical forecast skill at lower compute cost",
            "The emulator generalises poorly to out-of-distribution extreme events",
            "A scalable architecture for autoregressive atmospheric rollout is described",
            "Physical conservation constraints are imposed during training",
            "Operational deployment requires integration with existing forecast pipelines",
            "A simulation study compares resolution against numerical baselines",
        ],
    },
    {
        "track": "precision-fermentation",
        "name": "Precision fermentation proteins",
        "sector": "agriculture_food",
        "disciplines": ["biotechnology", "food science", "chemical engineering"],
        "yearly": {2021: 5, 2022: 8, 2023: 11, 2024: 13, 2025: 15, 2026: 9},
        "citation_scale": 8,
        "industry_rate": 0.52,
        "n_institutions": 9,
        "phrases": [
            "An engineered yeast strain expresses a functional dairy protein at pilot scale",
            "Titre and downstream recovery drive cost per kilogram",
            "Scale-up from bench bioreactor to a relevant production environment is described",
            "Regulatory approval as a novel food ingredient remains a gating step",
            "Sensory panels compare the product against the animal-derived equivalent",
            "Feedstock cost dominates the manufacturing model",
        ],
    },
    {
        "track": "gene-edited-crops",
        "name": "Gene-edited climate-resilient crops",
        "sector": "agriculture_food",
        "disciplines": ["plant science", "genetics", "agronomy"],
        "yearly": {2021: 6, 2022: 8, 2023: 10, 2024: 12, 2025: 14, 2026: 9},
        "citation_scale": 10,
        "industry_rate": 0.38,
        "n_institutions": 10,
        "phrases": [
            "An edited allele confers drought tolerance without a yield penalty",
            "Multi-season field trials validate performance in a relevant environment",
            "Regulatory treatment of edited crops differs sharply between jurisdictions",
            "Greenhouse proof of concept precedes field deployment",
            "Introgression into elite germplasm is the rate-limiting step",
            "Farmer adoption depends on seed cost and agronomic fit",
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
