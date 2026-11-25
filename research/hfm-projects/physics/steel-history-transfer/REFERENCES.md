# References: Steel Cyclic Loading History Transfer

Status: the shared web-search budget ran out before this project was searched. The entries below were confirmed on 2026-11-25 by fetching the Zenodo record named in the README and its reference file. Everything else is under "Unverified leads" and must be checked before use.

## Theory and established methods

- **Hartloper, de Castro e Sousa, Lignos (2021).** Constitutive Modeling of Structural Steels: Nonlinear Isotropic/Kinematic Hardening Material Model and Its Calibration. Journal of Structural Engineering 147(4), 04021031. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002964
  - Defines the updated Voce-Chaboche model and its calibration from cyclic coupon tests; the model whose kinematic-hardening terms a tension coupon cannot identify.

## Datasets and benchmarks

- **Hartloper, Ozden, de Castro e Sousa, Lignos (2022).** Database of Uniaxial Cyclic and Tensile Coupon Tests for Structural Metallic Materials, v1.0.0. Zenodo, CC-BY 4.0. https://doi.org/10.5281/zenodo.6965147
  - The data for both steel tracks. Use `Clean_Data_v1-0-0.zip` (63 MB) and `Overall_Summary` to pick, per steel, the protocols held out for scoring (HFM's notes mention independent LP9 and LP10).
- **Suzuki, Lignos (2021).** Experimental Evaluation of Steel Columns under Seismic Hazard-Consistent Collapse Loading Protocols. Journal of Structural Engineering 147(4), 04021020. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002963
  - Describes the earthquake-consistent loading histories behind the coupon protocols; relevant because this track scores stress and energy under earthquake-like strain histories.
- **Inamasu, de Castro e Sousa, Lignos (2022).** Development and Experimental Validation of Dissipative Embedded Column Base Connections for Enhanced Seismic Performance of Steel Moment-Resisting Frames. Journal of Structural Engineering 148(3), 04021280. https://doi.org/10.1061/(ASCE)ST.1943-541X.0003259
  - Source campaign for part of the database; needed to keep donor and target steels from the same campaign apart.
- **Heredia Rosa, Hartloper, de Castro e Sousa, Lignos, Motavalli, Ghafoori (2021).** Experimental behavior of iron-based shape memory alloys under cyclic loading histories. Construction and Building Materials 272, 121712. https://doi.org/10.1016/j.conbuildmat.2020.121712
  - The one non-steel material in the database; exclude it when borrowing cyclic parameters.

## Open-source code and tools

- **Hartloper.** `rlmtp` post-processing library and protocol definitions. https://github.com/ahartloper/rlmtp/tree/master/protocols
  - Linked from the Zenodo record; defines the load protocols, which is what is needed to design the "one cheap reversal" add-on test.

## Published results to beat

- No published score for history transfer from a tension coupon was found. The numbers to beat are the shared experiment's: donor median 0.145, oracle 0.087, one reversal closing 8% of the gap and four closing 30% (README), and HFM's 0.6677.

## Unverified leads

- UNVERIFIED: RESSPyLab, the Python library by de Castro e Sousa and Hartloper for Voce-Chaboche and updated Voce-Chaboche calibration (named in the README; not confirmed in this session).
- UNVERIFIED: de Castro e Sousa, Suzuki, Lignos, a paper on consistency in solving the inverse problem of the Voce-Chaboche model.
- UNVERIFIED: Chaboche's review of cyclic plasticity constitutive theories and the Armstrong-Frederick rule that HFM's fast/slow backstress idea builds on.
- UNVERIFIED: OpenSees uniaxial materials for the updated Voce-Chaboche model, for response-history checks at frame level.
- UNVERIFIED: optimal experiment design studies on the shortest cyclic protocol that identifies kinematic hardening.
- UNVERIFIED: EPFL theses and reports listed in `Database_References_v1-0-0.bib` in the Zenodo record (Suzuki 2018, Garcia 2020, Ozden 2021, El Jisr 2022, Skiadopoulos 2022).

## Second pass (confirmed by web search, 25 November 2026)

- **Hartloper, de Castro e Sousa, Lignos (2021). Constitutive Modeling of Structural Steels: Nonlinear Isotropic/Kinematic Hardening Material Model and Its Calibration. Journal of Structural Engineering.** https://ascelibrary.com/doi/10.1061/%28ASCE%29ST.1943-541X.0002964
  - Defines the updated Voce-Chaboche model, its parameter constraints and calibration, with parameters for nine steels; the natural richer model to lower this project's oracle error (it adds the yield plateau).
- **de Castro e Sousa, Suzuki, Lignos (2020). Consistency in Solving the Inverse Problem of the Voce-Chaboche Constitutive Model for Plastic Straining.** https://infoscience.epfl.ch/record/278992?ln=en
  - Shows how to calibrate the model consistently across cyclic load histories, and reports that the classic model underestimates initial yield stress by 10-30% when fitted for overall error.
- **Hartloper, Ozden, de Castro e Sousa, Lignos (2023). Uniaxial Cyclic and Tensile Tests on Structural Metallic Materials. Journal of Structural Engineering 149(5). Listed on the RESSLab publication page.** https://epfl.ch/labs/resslab/publications
  - The paper describing the coupon database used here; the search confirmed the citation only, not its contents.

