# References: Predicting Optical Constants Beyond the Measured Wavelength Band

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Kuzmenko (2005).** Kramers-Kronig constrained variational analysis of optical spectra. Review of Scientific Instruments 76, 083108. https://arxiv.org/abs/cond-mat/0503565
  - The causality-constrained fit used in this folder's experiment: ε2 is free at each mesh node and ε1 follows from the Kramers-Kronig transform.
- **Jellison, Modine (1996).** Parameterization of the optical functions of amorphous materials in the interband region. Applied Physics Letters 69(3), 371-373. Summary: https://en.wikipedia.org/wiki/Tauc%E2%80%93Lorentz_model
  - The Tauc-Lorentz oscillator: zero absorption below the gap, Lorentz shape above; the natural prior for dielectrics and amorphous semiconductors.
- **Rakić, Djurišić, Elazar, Majewski (1998).** Optical properties of metallic films for vertical-cavity optoelectronic devices. Applied Optics 37(22), 5271-5283. https://pubmed.ncbi.nlm.nih.gov/18286006/
  - Lorentz-Drude and Brendel-Bormann fits for 11 metals from 0.1 to 6 eV; a ready-made class prior for metals.
- **Milton, Eyre, Mantese (1997).** Finite Frequency Range Kramers-Kronig Relations: Bounds on the Dispersion. Physical Review Letters. https://ouci.dntb.gov.ua/works/4Orz0rz7
  - Bounds on what a finite measured band can say about dispersion; sets expectations for how far extrapolation can go.
- **Grabovsky, Hovsepyan (2020).** On feasibility of extrapolation of the complex electromagnetic permittivity function using Kramer-Kronig relations. arXiv:2009.12441. https://arxiv.org/abs/2009.12441
  - Gives a sharp upper bound on the worst-case extrapolation error; this is the formal statement of "causality alone is not enough".
- **Kuzmenko, van der Marel, Carbone, Marsiglio (2007).** Model-Independent Sum Rule Analysis Based on Limited-Range Spectral Data. New Journal of Physics 9, 229. https://arxiv.org/abs/cond-mat/0701593
  - Shows that with both n and k measured in a band, spectral weight hidden outside the band can be estimated without a model; a possible feature for detecting an out-of-band resonance.
- **Tanner (2014).** Use of X-ray scattering functions in Kramers-Kronig analysis of reflectance. arXiv:1411.2061. https://arxiv.org/abs/1411.2061
  - Builds the high-frequency extension from tabulated atomic scattering functions; a physics-based prior for the short-wavelength end that needs only the chemical formula.
- **Dantas, Rezende, Pinto (2015).** A Self-Consistent Extrapolation Method for the Complex Permittivity and Permeability Based on Finite Frequency Data. J. Comp. Int. Sci. 6(1), 3-20. https://arxiv.org/abs/1402.3526
  - A truncated Kramers-Kronig extrapolation scheme with stated assumptions on the loss tangent.
- **Nakatsukasa, Sète, Trefethen (2018).** The AAA algorithm for rational approximation. SIAM J. Sci. Comput. 40(3), A1494-A1522. https://arxiv.org/abs/1612.00337
  - Rational (pole) fitting that picks its own degree and needs no initial poles; relevant because HFM's best method selects rational poles.

## Recent advances and breakthroughs (2023-2026)

- **Polyanskiy (2024).** Refractiveindex.info database of optical constants. Scientific Data 11, 94. https://doi.org/10.1038/s41597-023-02898-2
  - The data descriptor for the database used here: YAML structure, shelves, and data provenance.
- **Leibl, Sytenko, Demircioglu, Beliaev et al. (2026).** AAA rational approximation for scatterometry of nanoscale gratings. arXiv:2608.28185. https://arxiv.org/abs/2608.28185
  - Applies AAA to silicon's permittivity and gets a four-pole generalized Lorentz model at under 5% relative error (figure from the search summary of the paper); a current example of automatic pole selection.
- **Grunert, Großmann, Runge (2024).** Deep learning of spectra: Predicting the dielectric function of semiconductors. Physical Review Materials 8, L122201. https://arxiv.org/abs/2406.08191
  - A database of 9,915 ab initio dielectric tensors and a graph network that predicts the full spectrum from structure; a source of material-class priors outside the measured band.
- **Ibrahim, Ataca (2024).** Prediction of Frequency-Dependent Optical Spectrum for Solid Materials: A Multi-Output & Multi-Fidelity Machine Learning Approach. ACS Applied Materials & Interfaces. https://arxiv.org/abs/2407.07736
  - Predicts the complex dielectric function from crystal structure with transfer between low- and high-fidelity data.
- **Hung, Okabe, Chotrattanapituk, Li (2024).** Ensemble-Embedding Graph Neural Network for Direct Prediction of Optical Spectra from Crystal Structure. arXiv:2406.16654. https://arxiv.org/abs/2406.16654
  - GNNOpt predicts ε1, ε2, n, absorption and reflectance directly; could supply the out-of-band shape that in-band data cannot.

## Datasets and benchmarks

- **Polyanskiy (ongoing).** refractiveindex.info database, CC0. https://github.com/polyanskiy/refractiveindex.info-database
  - The 620 records used in this folder come from here; the repository is the bulk-download route.

## Open-source code and tools

- **Toftul.** `refractiveindex` Python package. https://pypi.org/project/refractiveindex/
  - Loads any shelf/book/page of the database and downloads it on first use.
- **`riip` (RII_Pandas).** Python 3 + Pandas wrapper for the refractiveindex.info database. https://pypi.org/project/riip/
  - Gives the whole catalogue as data frames, which suits building class priors.

## Published results to beat

- No published benchmark for out-of-band extrapolation of tabulated n and k was found. The numbers to beat are this folder's +0.11 held-out skill and HFM's 0.7907 (private scoring, not comparable).

## Unverified leads

- UNVERIFIED: Kuzmenko's RefFIT software and its guide (cited inside the 2005 paper; no working URL confirmed).
- UNVERIFIED: vector fitting with passivity enforcement applied to optical dispersion; a search found only an electrical-insulation example.
- UNVERIFIED: Wemple-DiDomenico single-oscillator model and Sellmeier-coefficient compilations for glasses as dielectric priors.
- UNVERIFIED: Palik, Handbook of Optical Constants of Solids, as an independent test source.
