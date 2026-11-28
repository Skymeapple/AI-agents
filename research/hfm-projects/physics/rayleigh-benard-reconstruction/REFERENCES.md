# References: Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Willcox (2005).** Unsteady Flow Sensing and Estimation via the Gappy Proper Orthogonal Decomposition. Computers & Fluids. https://kiwi.ices.utexas.edu/papers/Gappy-POD-sensing-Willcox.pdf
  - Extends gappy POD (Everson and Sirovich, 1995) to unsteady flow and gives a condition-number rule for sensor placement; the task's starting method.
- **Fukami, Maulik, Ramachandra, Fukagata, Taira (2021).** Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning. Nature Machine Intelligence 3, 945-951. https://arxiv.org/abs/2101.00554
  - Turns point sensors into a Voronoi image so a CNN can be used; the standard learned baseline. Sample code is at kfukami/Voronoi-CNN (stated by the authors).
- **de Silva, Manohar, Clark, Brunton et al. (2021).** PySensors: A Python Package for Sparse Sensor Placement. arXiv:2102.13476. https://arxiv.org/abs/2102.13476
  - Implements the QR-pivot placement and reconstruction of Manohar et al. (2018, IEEE Control Systems Magazine 38(3), 63-86); useful for conditioning analysis of the fixed sensor set.
- **Raissi, Perdikaris, Karniadakis (2017).** Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. arXiv:1711.10561. https://arxiv.org/abs/1711.10561
  - Basis of the physics-informed reconstructions below, which recover unmeasured fields from the governing equations.

## Recent advances and breakthroughs (2023-2026)

- **Clark Di Leoni, Agasthya, Buzzicotti, Biferale (2023).** Reconstructing Rayleigh-Benard flows out of temperature-only measurements using Physics-Informed Neural Networks. arXiv:2301.07769. https://arxiv.org/abs/2301.07769
  - Same flow, partial observations: PINNs match nudging at low Rayleigh number and degrade as Ra rises; expect the same Ra trend here.
- **Mommert, Barta, Bauer, Volk et al. (2024).** Periodically activated physics-informed neural networks for assimilation tasks for three-dimensional Rayleigh-Bénard convection. Computers and Fluids 283, 106419. https://arxiv.org/abs/2403.02970
  - Sine activations improve recovery of unmeasured fields in Rayleigh-Bénard convection.
- **Barta, Volk, Bauer, Wagner et al. (2025).** Temperature and pressure reconstruction in turbulent Rayleigh-Bénard convection by Lagrangian velocities using PINN. arXiv:2505.02580. https://arxiv.org/abs/2505.02580
  - Recovers pressure, which is never measured in this task, from velocity data and the equations.
- **Volk, Sergent, Lucor, Mommert et al. (2025).** A PINN Methodology for Temperature Field Reconstruction in the PIV Measurement Plane: Case of Rayleigh-Bénard Convection. arXiv:2503.23801. https://arxiv.org/abs/2503.23801
  - Tests how much 2D data is enough to recover an unmeasured field in this flow.
- **Toscano, Käufer, Wang, Maxey et al. (2024).** Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks. arXiv:2407.15727. https://arxiv.org/abs/2407.15727 (published as AIVT, Science Advances 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12057682/)
  - Sparse-to-full inference in thermal convection validated against experiment.
- **Amorós-Trepat, Medrano-Navarro, Liu, Guastoni et al. (2025).** Guiding diffusion models to reconstruct flow fields from sparse data. Physics of Fluids 38, 015112 (2026). https://arxiv.org/abs/2510.19971
  - Guides reverse diffusion with sparse measurements and reports better spectra and pixel accuracy than other diffusion methods on 2D and 3D turbulence; the closest recent match to the 256-sensor setting.
- **Huang, Yang, Wang, Park (2024).** DiffusionPDE: Generative PDE-Solving Under Partial Observation. arXiv:2406.17763. https://arxiv.org/abs/2406.17763
  - Samples full fields from sparse observations with both data and PDE-residual guidance; the PDE term is how pressure could be filled in.
- **Shan, Wang, Chen, Liu et al. (2024).** PiRD: Physics-informed Residual Diffusion for Flow Field Reconstruction. arXiv:2404.08412. https://arxiv.org/abs/2404.08412
  - Diffusion reconstruction with a physics residual in training.
- **Du, Parikh, Fan, Liu et al. (2024).** CoNFiLD: Conditional Neural Field Latent Diffusion Model Generating Spatiotemporal Turbulence. arXiv:2403.05940. https://arxiv.org/abs/2403.05940
  - Zero-shot conditioning of a latent diffusion model on sparse sensors, with ensemble spread as uncertainty.
- **Williams, Zahn, Kutz (2023).** Sensing with shallow recurrent decoder networks. arXiv:2301.12011. https://arxiv.org/abs/2301.12011
  - SHRED uses the time history of a few sensors to reconstruct the field; fits this task's eight-frame sensor window and trains on a CPU-scale budget.
- **Los Alamos National Laboratory authors (2023).** Development of the Senseiver for efficient field reconstruction from sparse observations. Nature Machine Intelligence. https://www.nature.com/articles/s42256-023-00746-x (code: https://github.com/rohitvuppala/Senseiver)
  - Attention encoder for arbitrary sensor sets with sparse decoding; a stronger learned baseline than Voronoi-CNN on three datasets. A NeurIPS 2024 follow-up adds time-varying sensors: https://neurips.cc/virtual/2024/100017
- **Karnik, Bhangale, Abdo, Klishin et al. (2025).** PySensors 2.0: A Python Package for Sparse Sensor Placement. arXiv:2509.08017. https://arxiv.org/abs/2509.08017
  - Adds noise-propagation uncertainty metrics, which give a per-mode error estimate for a fixed sensor layout.

## Datasets and benchmarks

- **Ohana, McCabe, Meyer, Morel et al. (2024).** The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. NeurIPS 2024. https://arxiv.org/abs/2412.00568
  - Source of the simulations.
- **Polymathic AI.** `rayleigh_benard` dataset card. https://huggingface.co/datasets/polymathic-ai/rayleigh_benard
  - 1,750 simulations over 35 (Ra, Pr) settings, 358.4 GB; loader at https://github.com/PolymathicAI/the_well

## Breakthroughs in related fields that could transfer

- **Rozet, Louppe (2023).** Score-based Data Assimilation. arXiv:2306.10574. https://arxiv.org/abs/2306.10574
  - Trains a score model on short trajectory segments and conditions on observations at inference; suits an eight-frame window with fixed sensors.
- **Rybchuk, Hassanaly, Hamilton, Doubrawa et al. (2023).** Ensemble flow reconstruction in the atmospheric boundary layer from spatially limited measurements through latent diffusion models. Physics of Fluids 35, 126604. https://arxiv.org/abs/2303.00836
  - Treats reconstruction as inpainting and returns an ensemble, which separates recoverable large scales from unrecoverable small ones.
- **Shu, Li, Barati Farimani (2022).** A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction. arXiv:2211.14680. https://arxiv.org/abs/2211.14680
  - Early diffusion reconstruction from sparse or low-resolution data with PDE-residual conditioning.

## Published results to beat

- No published score for 256-sensor reconstruction of The Well's Rayleigh-Bénard fields was found. HFM: gappy-POD baseline 0.2918, best 0.2342 (README).

## Unverified leads

- UNVERIFIED: Everson and Sirovich (1995), the original gappy POD paper (cited by Willcox; no direct URL confirmed).
- UNVERIFIED: Manohar, Brunton, Kutz, Brunton (2018), "Data-driven sparse sensor placement for reconstruction", IEEE Control Systems Magazine (cited by PySensors; no direct URL confirmed).
- UNVERIFIED: "GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting" (arXiv:2603.26023 appeared in a search result; record not fetched).
