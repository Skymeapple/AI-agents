# References: Battery Voltage Model for Cold Operating Conditions

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Geng, Wang, Lacey, Brandell et al. (2021).** Bridging physics-based and equivalent circuit models for lithium-ion batteries. Electrochimica Acta 372, 137829. https://arxiv.org/abs/2005.09746
  - Derives equivalent-circuit elements from a physics-based model, including the charge-transfer resistance RT/(j0 F) that holds only near equilibrium; this is the term that stops being linear in the cold.
- **Sandia National Laboratories (1999).** Electrical Characteristics of 18650 Li-Ion Cells at Low Temperatures. OSTI report. https://www.osti.gov/biblio/9500
  - Reports that the ohmic resistance stays nearly constant from 25 °C to −20 °C, which matches the README's finding of a near-zero activation energy for one resistance and points to charge transfer and diffusion as the terms that grow.
- **Sun, Qi, Song, Ruan (2023).** Review of Low-Temperature Performance, Modeling and Heating for Lithium-Ion Batteries. Energies 16, 7142. https://mdpi.com/1996-1073/16/20/7142
  - Tabulates models that include low-temperature effects (down to −40 °C); a starting list of model structures beyond a constant-resistance 2-RC circuit.

## Recent advances and breakthroughs (2023-2026)

- **Fan, Zhou, Meng, Pang et al. (2024).** Development of a Comprehensive Physics-Based Battery Model and Its Multidimensional Comparison with an Equivalent-Circuit Model: Accuracy, Complexity, and Real-World Performance under Varying Conditions. arXiv:2411.12152. https://arxiv.org/abs/2411.12152
  - Physics-based model with hysteresis, concentration-dependent diffusivity and Arrhenius temperature dependence, compared with an equivalent-circuit model over C-rates and driving cycles; a direct template for the "one activation energy per process" route.
- **Barat, Wilson, Kim, Fang (2026).** System Identification of Lithium-Ion Battery Equivalent Circuit Models Using Ensemble Kalman Inversion. American Control Conference 2026. https://arxiv.org/abs/2604.10813
  - A derivative-free fitting method for equivalent-circuit parameters; relevant to the fitting-procedure part of the task.
- **Liang, Emadi, Vidal, Kollmeyer et al. (year not confirmed).** A Comparative Study between Physics, Electrical and Data Driven Lithium-Ion Battery Voltage Modeling Approaches. https://bohrium.dp.tech/paper/arxiv/66b24e7e01d2a3fbfcaad281
  - From the group that made the LG and Panasonic datasets: a third-order equivalent circuit and an LSTM were both more accurate than a single-particle model for voltage on drive cycles.

## Datasets and benchmarks

- **Kollmeyer et al. (2020).** LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script. Mendeley Data, v3. https://data.mendeley.com/datasets/cp3473x7xv/3
  - The dataset used in this folder's experiment. The raw-data companion record (Kollmeyer, with Naguib and Skells, McMaster University) is https://data.mendeley.com/datasets/b5mj79w5w9/2.
- **Kollmeyer (year not confirmed).** Panasonic 18650PF Li-ion Battery Data. Mendeley Data. https://data.mendeley.com/datasets/wykht8y7tg/1
  - Second cell, tested at the University of Wisconsin-Madison; the donor for a prior on activation energies and the cell named in HFM's current best idea.
- **Kollmeyer et al. (year not confirmed).** Samsung INR21700 30T 3Ah Li-ion Battery Data. Mendeley Data. https://data.mendeley.com/datasets/9xyvy2njj3/2
  - Third cell with the same test plan at 40, 25, 10, 0, −10 and −20 °C; not mentioned in the README, and it gives a second donor chemistry for cross-cell priors.
- **Mendeley Data record (2026).** Multi-Temperature ECM–EKF Modeling Artifacts for the Panasonic 18650PF Cell. https://data.mendeley.com/datasets/fnz4rrr986/1
  - Published equivalent-circuit parameters per temperature for the Panasonic cell; a ready check on fitted resistances and their temperature trend.

## Open-source code and tools

- **Sulzer, Marquis, Timms, Robinson, Chapman (2021).** Python Battery Mathematical Modelling (PyBaMM). Journal of Open Research Software 9(1), 14. https://doi.org/10.5334/jors.309 (code: https://github.com/pybamm-team/PyBaMM)
  - Single-particle, DFN and equivalent-circuit models with thermal coupling; use it to test whether a physics model extrapolates to cold better than the 2-RC circuit.
- **Planden, Courtier, Robinson, Khetarpal, Brosa Planella, Howey (2025).** PyBOP: A Python package for battery model optimisation and parameterisation. Journal of Open Source Software. https://joss.theoj.org/papers/10.21105/joss.07874 (preprint https://arxiv.org/abs/2412.15859)
  - Parameter identification for PyBaMM electrochemical and equivalent-circuit models, with deterministic and Bayesian methods.

## Published results to beat

- No published cold-extrapolation voltage error for the LG HG2 cell was found in this session. The numbers to beat are this folder's own: 120.1 mV RMS (Arrhenius form) against an 82.4 mV oracle (README).

## Unverified leads

- UNVERIFIED: Tripathy, University of Warwick PhD thesis on parameterising cell models for low-temperature applications (a search result pointed to https://core.ac.uk/download/478783965.pdf; exact title not captured).
- UNVERIFIED: an IJEECS paper giving 2-RC parameters for the Panasonic 18650PF from −20 °C to +25 °C (search result at ijeecs.iaescore.com; title not captured).
- UNVERIFIED: Plett's enhanced self-correcting (ESC) cell model and its toolbox, with hysteresis and temperature-dependent parameters.
- UNVERIFIED: equivalent-circuit models with a Butler-Volmer (current-dependent) charge-transfer resistance for low temperature; a search summary supported the idea but named no peer-reviewed paper.
- UNVERIFIED: the Kollmeyer/Vidal journal papers that report voltage or SOC error per temperature on these datasets.
