# References: Optical Coating Design Under Manufacturing Variation

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Tikhonravov, Trubetskov, DeBell (1996).** Application of the needle optimization technique to the design of optical coatings. Applied Optics 35(28), 5493-5508. https://doi.org/10.1364/AO.35.005493
  - The layer-insertion method named in the README's next step; it widens the candidate pool by changing the number of layers.
- **Birge, Kärtner, Nohadani (year not confirmed).** Improving thin-film manufacturing yield with robust optimization. https://bib-pubdb1.desy.de/record/97205/files/32DE6A27-C2C7-A869-C92E4A7D3539AB91_206412.pdf
  - Min-max optimisation over layer-thickness errors, tested on an anti-reflection coating with Monte Carlo; the closest published match to this project's task and the method to compare against.

## Recent advances and breakthroughs (2023-2026)

- **Lawrence Livermore National Laboratory (2023).** Robustly Designed Optical Coatings for Cutting-Edge Science. LDRD annual report. https://ldrd-annual.llnl.gov/ldrd-annual-2023/project-highlights/lasers-and-optical-science-and-technology/robustly-designed-optical-coatings-cutting-edge-science
  - A current programme that designs coatings to tolerate thickness and index errors instead of tightening tolerances; reports that this is feasible.
- **Authors not captured (2025).** Surrogate sensitivity analysis of facet optical coatings produced without and with in situ design reoptimization. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11833107/
  - Compares applying all thickness errors at once with applying them layer by layer and re-optimising the remaining layers. If the HFM process model allows sequential correction, this changes which design is most robust.
- **Authors not captured (2025).** Sensitivity and tolerance analysis of single-layer MgF2 and SiO2 anti-reflection coatings. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12749774/
  - Finds reflectance most sensitive to thickness reduction; the asymmetric response is a reason to bias nominal thicknesses.
- **Ma, Wang, Guo (2023).** OptoGPT: A Foundation Model for Inverse Design in Optical Multilayer Thin Film Structures. Opto-Electronic Advances 7, 240062 (2024). https://arxiv.org/abs/2304.10294
  - A transformer trained on 10 million designs that proposes material and thickness sequences of up to 20 layers from a target spectrum; a fast generator of varied candidate topologies for the selection step.
- **Schaible, Özdemir, Debus, Burger et al. (2026).** Inverse Design of Optical Multilayer Thin Films using Robust Masked Diffusion Models. arXiv:2604.01106. https://arxiv.org/abs/2604.01106
  - OptoLlama, a masked diffusion model that generates the whole layer stack at once; a second candidate generator.

## Open-source code and tools

- **Luce, Mahdavi, Marquardt, Wankerl (2021).** TMM-Fast: A Transfer Matrix Computation Package for Multilayer Thin-Film Optimization. arXiv:2111.13667. https://arxiv.org/abs/2111.13667 (code: https://github.com/MLResearchAtOSRAM/tmm_fast)
  - Vectorised transfer-matrix code over wavelengths, angles and many stacks, with PyTorch gradients; makes Monte Carlo scoring of thousands of candidates cheap. It builds on Byrnes's `tmm` package.
- **`tmmax`.** JAX transfer-matrix library with JIT and GPU support. https://github.com/bahremsd/tmmax
  - Differentiable alternative for gradient-based optimisation of the expected-reflectance objective.

## Breakthroughs in related fields that could transfer

- **Pierro, Granata, Michel, Pinard et al. (2026).** Experimental Realization of Optimized Ternary Mirror Coatings. arXiv:2601.17886. https://arxiv.org/abs/2601.17886
  - Gravitational-wave mirror coatings designed by optimisation and then fabricated; shows the gap between designed and realised stacks.

## Published results to beat

- No published robust-AR benchmark with comparable scoring was found. The numbers to beat are this folder's: 14.5% lower expected reflectance from selecting by robust score (README), and HFM's 0.2523.

## Unverified leads

- UNVERIFIED: Larouche and Martinu, OpenFilters open-source coating design software (GPL), described in a secondary listing; primary article and repository not confirmed.
- UNVERIFIED: Trubetskov and co-authors (2012), Applied Optics 51, 7319, a stochastic optimisation procedure that returns several candidate designs (PDF seen at attoworld.de; title not captured).
- UNVERIFIED: a Monte Carlo tolerance framework in Applied Optics 65(24), 8191, finding that thickness errors dominate over index errors (https://opg.optica.org/abstract.cfm?URI=ao-65-24-8191; title not captured).
- UNVERIFIED: Tikhonravov and Trubetskov (2007), Applied Optics 46(5), 704-710, on the role of total optical thickness in design.
- UNVERIFIED: reinforcement-learning approaches to multilayer design; one search returned none.
