# References: Predict Cyclic Steel Stress Response from Limited Material Tests

Status: the shared web-search budget ran out before this project was searched. The entries below were confirmed on 2026-11-25 by fetching the Zenodo record named in the README and its reference file. Everything else is under "Unverified leads" and must be checked before use.

## Theory and established methods

- **Hartloper, de Castro e Sousa, Lignos (2021).** Constitutive Modeling of Structural Steels: Nonlinear Isotropic/Kinematic Hardening Material Model and Its Calibration. Journal of Structural Engineering 147(4), 04021031. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002964
  - Defines the updated Voce-Chaboche model and how it is calibrated from cyclic coupon tests; the model class behind this folder's 0.087 oracle error and the first reference of the database.

## Datasets and benchmarks

- **Hartloper, Ozden, de Castro e Sousa, Lignos (2022).** Database of Uniaxial Cyclic and Tensile Coupon Tests for Structural Metallic Materials, v1.0.0. Zenodo, CC-BY 4.0. https://doi.org/10.5281/zenodo.6965147
  - The data used here. `Clean_Data_v1-0-0.zip` (63 MB) is the downsampled set; `Overall_Summary` lists every specimen; `Summarized_Mechanical_Props_Campaign` gives mean yield stress and modulus per campaign, which are ready-made donor features.
- **Suzuki, Lignos (2021).** Experimental Evaluation of Steel Columns under Seismic Hazard-Consistent Collapse Loading Protocols. Journal of Structural Engineering 147(4), 04021020. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002963
  - Source campaign for part of the database; describes the collapse-consistent loading protocols that the coupon protocols mirror.
- **Inamasu, de Castro e Sousa, Lignos (2022).** Development and Experimental Validation of Dissipative Embedded Column Base Connections for Enhanced Seismic Performance of Steel Moment-Resisting Frames. Journal of Structural Engineering 148(3), 04021280. https://doi.org/10.1061/(ASCE)ST.1943-541X.0003259
  - Source campaign for further steels in the database; needed to group donors by campaign correctly.
- **Heredia Rosa, Hartloper, de Castro e Sousa, Lignos, Motavalli, Ghafoori (2021).** Experimental behavior of iron-based shape memory alloys under cyclic loading histories. Construction and Building Materials 272, 121712. https://doi.org/10.1016/j.conbuildmat.2020.121712
  - Source of the one non-steel material in the database (an iron-based shape memory alloy); exclude it from the donor library or treat it as an outlier.

## Open-source code and tools

- **Hartloper.** `rlmtp` post-processing library and protocol definitions. https://github.com/ahartloper/rlmtp/tree/master/protocols
  - Linked from the Zenodo record; documents the file format and the load protocols (LP1 to LP10).

## Published results to beat

- No published score for predicting cyclic response from a tension test alone was found. The numbers to beat are this folder's: donor median 0.145, conditioned prior refined on the tension test 0.138, oracle 0.087 (README), and HFM's 0.5453.

## Unverified leads

- UNVERIFIED: RESSPyLab, the Python library by de Castro e Sousa and Hartloper for calibrating Voce-Chaboche and updated Voce-Chaboche parameters.
- UNVERIFIED: de Castro e Sousa, Suzuki, Lignos, a paper on consistency in solving the inverse problem of the Voce-Chaboche model (parameter non-uniqueness, which limits donor averaging in parameter space).
- UNVERIFIED: Chaboche's review of cyclic plasticity constitutive theories (International Journal of Plasticity, about 2008) and the Armstrong-Frederick kinematic hardening rule.
- UNVERIFIED: OpenSees uniaxial materials implementing the updated Voce-Chaboche model, for checking a stress integrator.
- UNVERIFIED: 2023-2026 work on neural or Bayesian constitutive models that infer cyclic hardening from monotonic tests or from hardness, chemistry and grade.
- UNVERIFIED: other EPFL theses and reports listed in `Database_References_v1-0-0.bib` in the Zenodo record (Suzuki 2018, Garcia 2020, Ozden 2021, El Jisr 2022, Skiadopoulos 2022) as sources of per-campaign metadata.

## Second pass (confirmed by web search, 25 November 2026)

- **Hartloper, de Castro e Sousa, Lignos (2021). Constitutive Modeling of Structural Steels: Nonlinear Isotropic/Kinematic Hardening Material Model and Its Calibration. Journal of Structural Engineering.** https://ascelibrary.com/doi/10.1061/%28ASCE%29ST.1943-541X.0002964
  - Defines the updated Voce-Chaboche model, its parameter constraints and calibration, with parameters for nine steels; the natural richer model to lower this project's oracle error (it adds the yield plateau).
- **de Castro e Sousa, Suzuki, Lignos (2020). Consistency in Solving the Inverse Problem of the Voce-Chaboche Constitutive Model for Plastic Straining.** https://infoscience.epfl.ch/record/278992?ln=en
  - Shows how to calibrate the model consistently across cyclic load histories, and reports that the classic model underestimates initial yield stress by 10-30% when fitted for overall error.
- **Hartloper, Ozden, de Castro e Sousa, Lignos (2023). Uniaxial Cyclic and Tensile Tests on Structural Metallic Materials. Journal of Structural Engineering 149(5). Listed on the RESSLab publication page.** https://epfl.ch/labs/resslab/publications
  - The paper describing the coupon database used here; the search confirmed the citation only, not its contents.

