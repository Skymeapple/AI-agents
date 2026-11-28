# References: Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions

Checked 2026-11-25. Each entry was confirmed in this session by a web search result, by the arXiv record, or by the project's own README on GitHub. Nothing is cited from memory.

## Theory and established methods

- **Salvato, Ilbert, Hoyle (2018).** The many flavours of photometric redshifts. Nature Astronomy. https://arxiv.org/abs/1805.12574
  - Review of template and machine-learning methods and of the standard metrics (scatter, bias, outlier fraction).
- **Newman, Gruen (2022).** Photometric Redshifts for Next-Generation Surveys. Annual Review of Astronomy and Astrophysics 60:363-414. https://arxiv.org/abs/2206.13633
  - Review of the limits set by incomplete and unrepresentative spectroscopic training samples.
- **Beck, Dobos, Budavári et al. (2016).** Photometric redshifts for the SDSS Data Release 12. MNRAS 460:1371-1381. https://arxiv.org/abs/1603.09708
  - The catalogue-based SDSS photo-z that image-based papers compare against.
- **Pasquet, Bertin, Treyer et al. (2019).** Photometric redshifts from SDSS images using a Convolutional Neural Network. A&A 621, A26. https://arxiv.org/abs/1806.06607
  - The reference CNN on image cutouts. Uses the network as a classifier and outputs a redshift probability distribution. Argues that precision is limited by image signal-to-noise, not by the method.
- **D'Isanto, Polsterer (2018).** Photometric redshift estimation via deep learning. A&A 609, A111. https://arxiv.org/abs/1706.02467
  - CNN with a mixture-density output; the standard reference for mixture-density photo-z.
- **Lima, Cunha, Oyaizu et al. (2008).** Estimating the Redshift Distribution of Faint Galaxy Samples. MNRAS 390:118. https://arxiv.org/abs/0801.3822
  - Nearest-neighbour reweighting of the training sample to match the target sample; the classic fix for covariate shift.
- **Schlegel, Finkbeiner, Davis (1998).** Maps of Dust IR Emission for Use in Estimation of Reddening and CMBR Foregrounds. ApJ 500:525. https://arxiv.org/abs/astro-ph/9710327
  - Source of the extinction values that gave the 1.29% gain in this folder's experiment.
- **Schlafly, Finkbeiner (2011).** Measuring Reddening with SDSS Stellar Spectra and Recalibrating SFD. ApJ 737:103. https://arxiv.org/abs/1012.4804
  - Recalibration of the dust map; check which calibration the inputs use.

## Recent advances and breakthroughs (2023-2026)

- **Parker, Lanusse, Shen et al. (2025).** AION-1: Omnimodal Foundation Model for Astronomical Sciences. NeurIPS 2025. https://arxiv.org/abs/2510.17960
  - The model named in the task. Pretrained on Legacy Survey, HSC, SDSS, DESI and Gaia. Has a redshift token (`tok_z`), so it can predict redshift directly as well as give embeddings.
- **Treyer, Ait-Ouahmed, Pasquet et al. (2023).** CNN photometric redshifts in the SDSS at r ≤ 20. https://arxiv.org/abs/2310.02173
  - Trained on SDSS, GAMA and BOSS spectra, the same label sources as this task. Builds a representative training set to stop luminous red galaxies dominating.
- **Ait-Ouahmed, Arnouts, Pasquet et al. (2023).** Multimodality for improved CNN photometric redshifts. A&A. https://arxiv.org/abs/2310.02185
  - Processes subsets of bands in separate branches before fusion; a low-cost change to an image encoder.
- **Zhou, Li, Zou et al. (2024).** Estimating Photometric Redshifts for Galaxies from the DESI Legacy Imaging Surveys with Bayesian Neural Networks Trained by DESI EDR. MNRAS. https://arxiv.org/abs/2412.02390
  - Uses g, r, z images plus WISE W1 and W2; shows the mid-infrared bands are worth adding to Legacy Survey inputs.
- **Wei, Li, Zhang et al. (2025).** Photometric redshift estimation for emission line galaxies of DESI Legacy Imaging Surveys by CNN-MLP. PASA. https://arxiv.org/abs/2505.24175
  - Image branch plus catalogue branch, the same two-view design as HFM's current best.
- **Tian, Wen, Xia (2026).** Photometric Redshift PDFs via Neural Network Classification for DESI Legacy Imaging Surveys and Pan-STARRS. ApJS. https://arxiv.org/abs/2602.01548
  - Bins redshift and optimises CRPS. Trains on DESI DR1 plus SDSS DR19. Data at https://doi.org/10.5281/zenodo.18410731.
- **Luo, Li, Chen et al. (2026).** Beyond Colors: Probing Redshifts from Galaxy Morphology in Single-band Images with ViT-MDNz. https://arxiv.org/abs/2602.22711
  - Vision transformer with a mixture-density head on Legacy Survey images; shows how much signal is in morphology alone.
- **Dey, Andrews, Newman et al. (2022).** Photometric Redshifts from SDSS Images with an Interpretable Deep Capsule Network. MNRAS 515:5285-5305. https://arxiv.org/abs/2112.03939
  - Matches CNN accuracy with fewer parameters and less data; code and catalogues at https://biprateep.github.io/encapZulate-1.

## Datasets and benchmarks

- **Dey, Schlegel, Lang et al. (2019).** Overview of the DESI Legacy Imaging Surveys. AJ. https://arxiv.org/abs/1804.08657
  - Describes the imaging. A search result confirms it combines three surveys (BASS, MzLS, DECaLS); differences between them are a candidate cause of region shift.
- **Zhou, Ferraro, White et al. (2023).** DESI luminous red galaxy samples for cross-correlations. JCAP 11, 097. https://arxiv.org/abs/2309.06443
  - Releases general-purpose photo-z for every DR9 galaxy, trained with DESI spectra: https://www.legacysurvey.org/dr9/files/#photo-z-sweeps. Also publishes imaging-systematics weights.
- **Zhou, Newman, Mao et al. (2021).** The Clustering of DESI-like Luminous Red Galaxies Using Photometric Redshifts. MNRAS 501:3309-3331. https://arxiv.org/abs/2001.06018
  - Method paper for the Legacy Survey photo-z built on DECam and WISE photometry; released as the PRLS catalogue for DR8.
- **DESI Collaboration (2025).** Data Release 1 of the Dark Energy Spectroscopic Instrument. AJ. https://arxiv.org/abs/2503.14745
  - Large public spectroscopic sample over the Legacy Survey footprint; extra labels beyond SDSS.
- **DESI Collaboration (2023).** The Early Data Release of the Dark Energy Spectroscopic Instrument. AJ 168:58. https://arxiv.org/abs/2306.06308
  - Smaller, deeper sample used by several of the papers above.
- **Abdurro'uf, Accetta, Aerts et al. (2021).** The Seventeenth Data Release of the Sloan Digital Sky Surveys. ApJS. https://arxiv.org/abs/2112.02026
  - The catalogue used by the experiment in this folder.
- **The Multimodal Universe Collaboration (2024).** The Multimodal Universe: Enabling Large-Scale Machine Learning with 100TB of Astronomical Scientific Data. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2412.02527
  - Images and spectra from many surveys, including DESI and SDSS, in one format; previews at https://huggingface.co/MultimodalUniverse, code at https://github.com/MultimodalUniverse/MultimodalUniverse.

## Open-source code and tools

- **AION (Polymathic AI).** https://github.com/PolymathicAI/AION ; weights https://huggingface.co/polymathic-ai/aion-base
  - `model.encode` gives embeddings for a light regressor; the README shows redshift prediction from photometry plus image. The base model has 300M parameters. MIT licence (per the model card, as reported by a search result).
- **AstroCLIP (Polymathic AI).** https://github.com/PolymathicAI/AstroCLIP ; paper https://arxiv.org/abs/2310.03024 (Parker, Lanusse, Golkar et al., MNRAS 2024)
  - Shared embedding space for multi-band galaxy images and optical spectra; the README lists photometric redshift as a downstream task. A second image view to blend with AION-1.
- **AstroPT.** https://arxiv.org/abs/2405.14930 (Smith, Roberts, Angeloudi et al., 2024); code https://github.com/Smith42/astroPT
  - Large pretrained model for astronomical observations; another possible source of frozen embeddings.
- **Pasquet CNN code.** https://github.com/jpasquet/Photoz
  - Reference architecture (listed in the arXiv comment of Pasquet et al.).

## Breakthroughs in related fields that could transfer

- **Hayat, Stein, Harrington et al. (2021).** Self-Supervised Representation Learning for Astronomical Images. ApJL 911:L33. https://arxiv.org/abs/2012.13083
  - Self-supervised pretraining on unlabelled SDSS images, then fine-tuning for photo-z. Pretraining on unlabelled galaxies from the held-out sky regions is allowed in principle and addresses the shift directly.
- **Autenrieth, van Dyk, Trotta et al. (2021/2024).** Stratified Learning: A General-Purpose Statistical Method for Improved Learning under Covariate Shift. Statistical Analysis and Data Mining 17, e11643. https://arxiv.org/abs/2106.11211
  - Stratifies on a propensity score for being in the training set; the abstract reports improved SDSS redshift density estimates.
- **Moskowitz, Gawiser, Crenshaw et al. (2024).** Improving Photometric Redshift Estimates with Training Sample Augmentation. ApJL 967:L6. https://arxiv.org/abs/2402.15551
  - Adds simulated galaxies outside the training range; abstract reports outlier fraction cut by nearly 50%. Relevant because outliers dominate this task's RMSE.

## Published results to beat

All numbers below are from the papers' arXiv abstracts. Samples and magnitude limits differ, so they are not directly comparable with each other or with HFM's NRMSE.

- Pasquet et al. 2019: σ_MAD < 0.01 on SDSS main-sample galaxies (r ≤ 17.8, z < 0.4).
- Treyer et al. 2023: σ_MAD = 0.014, mean bias 0.0015, outlier rate (|Δz_norm| > 0.05) 4% at r < 19.8.
- Tian, Wen, Xia 2026: σ_NMAD = 0.0153 and outlier fraction 0.50% on Legacy Surveys DR10.
- Wei et al. 2025: σ_NMAD = 0.0140, outlier fraction 2.57% (emission-line galaxies, DR10).
- Zhou et al. 2021: σ_NMAD = 0.02 for luminous red galaxies.
- Luo et al. 2026: σ_NMAD = 0.034, outliers 2.6% from a single r-band image.
- This folder's catalogue-only baseline: NMAD 0.0225, outliers 1.21%. HFM best NRMSE: 0.02434.

## Cheapest credible way to run experiments

- CPU only: keep the SDSS catalogue experiment and add (a) WISE W1 and W2 magnitudes, (b) the published DR9 photo-z from the Zhou et al. sweeps as an input feature or blend member, (c) per-object depth and seeing columns from the Legacy Survey catalogue. All are table columns.
- Images: embed cutouts once with AION-1 base on a GPU, save the embeddings, then run every region-grouped experiment on CPU.

## Unverified leads

- "Photometric redshift estimation of galaxies in the DESI Legacy Imaging Surveys" (title seen in a search result at https://ouci.dntb.gov.ua/works/45xQ8GO7); authors and year not confirmed.
- A σ_NMAD of about 0.013 and outlier rate of about 0.8% for the Legacy Survey random-forest photo-z was quoted in a search summary without a confirmed source paper.
- Published AION-1 redshift accuracy figures: the abstract gives none; check the paper body.
- Dedicated studies of photo-z transfer between sky regions (depth, seeing, extinction): the search for these did not run, so none is listed.
