# Data gap map (task 4.1)

Researched 1 November 2026 by web search of dataset pages; the full per-dataset tables with sources are below. Figures marked **unconfirmed** were not seen on an opened page (several stats pages were blocked by captchas). Gap rankings are judgements, not measurements.

## Ranked gaps by expected discovery value

| Domain | Gap | Value (1-5) | Why |
| --- | --- | --- | --- |
| Materials | Synthesis conditions and failed syntheses | 5 | Millions of predicted-stable phases already exist, and knowing which ones can actually be made is now the main bottleneck for closed-loop labs |
| Protein | Conformational ensembles, dynamics, ligand complexes | 5 | Function and druggability depend on states that static PDB and AFDB entries do not capture |
| Small molecule | Standardised negative results across the proteome | 5 | Publication bias stops models from learning selectivity and false-positive rates |
| Single-cell | Combinatorial, dose and time perturbations across many cell types | 5 | Causal "virtual cell" models need interventions, not atlases |
| Catalysis | Measured activity, selectivity and stability linked to DFT | 5 | Adsorption energies are only proxies for turnover |
| Materials | Measured properties linked to computed entries | 4 | Without measured values, models learn DFT's systematic errors |
| Protein | Uniform function and affinity data with negatives | 4 | Curated function covers well under 1% of known sequences |
| Small molecule | Consistent ADMET, in vivo and dose-response data | 4 | Most candidates fail on ADMET, which open data barely covers |
| Catalysis | Barriers, solvent effects, realistic surfaces | 4 | Kinetics and the real active site decide performance |
| Single-cell | Donor diversity, disease, spatial and protein layers | 3 | Adds observational, not causal, information |

**Cross-cutting finding.** Every domain's top gap is either negative or failed results, or interventional data. This is the data a closed-loop lab produces by default when it logs everything (the I4 store does), which supports making I3 and I4 one program.

## Datasets audited (selected figures)

| Domain | Dataset | Size (as of) | Main gap | Source |
| --- | --- | --- | --- | --- |
| Materials | Materials Project | DB v2026.09.28; 117k GNoME r2SCAN entries added Apr 2026 | Computed only; no synthesis routes or failures | [docs](https://docs.materialsproject.org/changes/database-versions) |
| Materials | OQMD | >1M compounds (2022) | T=0 K DFT energies only | [TRI](https://www.tri.global/research/reflections-one-million-compounds-open-quantum-materials-database-oqmd) |
| Materials | Alexandria | ~5.8M structures, ~175k on hull | Ordered stoichiometric compounds only, biased to oxides and H-containing compounds | [arXiv 2512.09169](https://arxiv.org/html/2512.09169v1) |
| Materials | ICSD | 347,985 structures (30 Oct 2026) | Subscription access; synthesis not structured, no failures | [FIZ](https://icsd.products.fiz-karlsruhe.de/aktuelles/icsd-news) |
| Protein | UniProtKB/Swiss-Prot | 575,748 entries (2026_03) | Curated function covers a tiny share of sequences | [Expasy](https://web.expasy.org/docs/relnotes/relstat.html) |
| Protein | AlphaFold DB | >214M predictions (unconfirmed) | Predicted monomers; few complexes or dynamics | [download](https://alphafold.com/download) |
| Small molecule | ChEMBL 37 | ~3M compounds, >9.55M activities (May 2026) | Inactive results under-reported | [ChEMBL blog](https://chembl.blogspot.com/2026/05/chembl-37-is-here-bigger-cleaner-and.html) |
| Small molecule | BindingDB | 2.9M measurements (Sep 2024) | Patent bias toward actives | [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/66e360db12ff75c3a12b3a00) |
| Single-cell | HCA Data Portal | 70.9M cells, 11.3k donors (Nov 2026) | Mostly healthy adults | [portal](https://data.humancellatlas.org/) |
| Single-cell | JUMP Cell Painting | ~116k compounds, ~1.81B cells | One cell line; no dose series | [JUMP hub](https://broadinstitute.github.io/jump_hub/explanations/data_description.html) |
| Single-cell | Replogle Perturb-seq | >2.5M cells (2022) | Few lines; single knockdowns only | [HMS](https://ntc.hms.harvard.edu/publications/mapping-information-rich-genotype-phenotype-landscapes-genome-scale-perturb-seq) |
| Catalysis | OC20 | ~1.28M relaxations (2020) | Adsorption energies only; no kinetics | [arXiv 2010.09990](https://arxiv.org/abs/2010.09990v4) |
| Catalysis | OC22 | 62,331 relaxations (2022) | Idealised oxide slabs | [arXiv 2206.08917](https://arxiv.org/abs/2206.08917) |
| Catalysis | Catalysis-Hub | >50 datasets | Non-reconstructed surfaces, DFT only | [Sci. Data 2019](https://link.springer.com/content/pdf/10.1038/s41597-019-0080-z.pdf) |

**Unresolved:** the current totals for PDB, PubChem BioAssay, CELLxGENE Census, NOMAD and Materials Project could not be confirmed, because the pages were blocked or showed no numbers.
