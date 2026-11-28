# References: 64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Li, Kovachki, Azizzadenesheli, Liu et al. (2020).** Fourier Neural Operator for Parametric Partial Differential Equations. arXiv:2010.08895. https://arxiv.org/abs/2010.08895
  - The FNO baseline reported on The Well's Rayleigh-Bénard card.
- **Gupta, Brandstetter (2022).** Towards Multi-spatiotemporal-scale Generalized PDE Modeling. arXiv:2209.15616. https://arxiv.org/abs/2209.15616
  - Systematic comparison of U-Net and operator architectures with parameter conditioning; relevant because the task spans 35 (Ra, Pr) settings.
- **Lippe, Veeling, Perdikaris, Turner, Brandstetter (2023).** PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers. NeurIPS 2023. https://arxiv.org/abs/2308.05732
  - Shows that one-step-trained models neglect low-amplitude high-frequency content and that this limits rollout length; fixes it with a few denoising refinement steps.
- **Burns, Vasil, Oishi, Lecoanet et al. (2020).** Dedalus: A Flexible Framework for Numerical Simulations with Spectral Methods. Physical Review Research 2, 023068. https://arxiv.org/abs/1905.10388
  - Open spectral solver with a 2D Rayleigh-Bénard example; the tool for the README's twin-simulation predictability test.

## Recent advances and breakthroughs (2023-2026)

- **McCabe, Mukhopadhyay, Marwah, Régaldo-Saint Blancard et al. (2025).** Walrus: A Cross-Domain Foundation Model for Continuum Dynamics. ICML 2026. https://arxiv.org/abs/2511.15684
  - The 1.3B-parameter model named in the task; describes its stabilisation method and adaptive tokenization, which matter when fine-tuning for 64-frame rollouts.
- **Rozet, Ohana, McCabe, Louppe, Lanusse, Ho (2025).** Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation. NeurIPS 2025. https://arxiv.org/abs/2507.02608
  - Tests latent diffusion emulators on The Well's Rayleigh-Bénard data; latent models match pixel-space ones at up to 1000x compression and diffusion beats deterministic emulators. Code and weights: https://github.com/polymathicai/lola
- **Diaconu, Cranmer, Turner, Marwah et al. (2026).** Probabilistic Retrofitting of Learned Simulators. arXiv:2603.01949. https://arxiv.org/abs/2603.01949
  - Fine-tunes an already trained deterministic emulator with a CRPS loss to make it an ensemble model; this is the README's "ensembles scored with CRPS" route applied to pretrained models. Code: https://github.com/cddcam/lola_crps
- **Greenbury, Jersakova, Conti, Famili et al. (2026).** Reliability of Probabilistic Emulation of Physical Systems. arXiv:2606.12997. https://arxiv.org/abs/2606.12997
  - Compares CRPS-trained ensembles with generative models for calibration and coverage; helps choose between the two probabilistic routes.
- **McCabe, Régaldo-Saint Blancard, Parker, Ohana et al. (2023).** Multiple Physics Pretraining for Physical Surrogate Models. arXiv:2310.02994. https://arxiv.org/abs/2310.02994
  - The predecessor of Walrus; explains the shared-embedding pretraining idea.
- **Herde, Raonić, Rohner, Käppeli et al. (2024).** Poseidon: Efficient Foundation Models for PDEs. NeurIPS 2024. https://arxiv.org/abs/2405.19101
  - An alternative pretrained fluid model with continuous-in-time conditioning; open weights for a second fine-tuning baseline.
- **Hao, Su, Liu, Berner et al. (2024).** DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training. arXiv:2403.03542. https://arxiv.org/abs/2403.03542
  - Pretraining with noise-corrupted inputs for rollout stability; a cheap trick to add when fine-tuning.
- **Shokar, Kerswell, Haynes (2025).** Conditioning on PDE Parameters to Generalise Deep Learning Emulation of Stochastic and Chaotic Dynamics. arXiv:2509.09599. https://arxiv.org/abs/2509.09599
  - Parameter-conditioned probabilistic emulation of chaotic flows; relevant to conditioning on Ra and Pr.
- **Nathaniel, Gentine (2025).** Generative emulation of chaotic dynamics with coherent prior. arXiv:2504.14264. https://arxiv.org/abs/2504.14264
  - Diffusion emulator guided by a coarse deterministic prior; a way to keep large-scale structure while generating small scales.
- **Saupe, Lunet, Bode, Speck et al. (2026).** Space-Time Parallel Direct Numerical Simulation of Rayleigh-Bénard Convection. arXiv:2610.17942. https://arxiv.org/abs/2610.17942
  - Recent solver work for generating reference Rayleigh-Bénard runs faster.

## Datasets and benchmarks

- **Ohana, McCabe, Meyer, Morel et al. (2024).** The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2412.00568
  - Source of the task data and of the baseline protocol.
- **Polymathic AI.** `rayleigh_benard` dataset card. https://huggingface.co/datasets/polymathic-ai/rayleigh_benard
  - 1,750 simulations (35 parameter sets × 50 initial conditions), 358.4 GB.

## Open-source code and tools

- **Polymathic AI.** `the_well` loader and benchmark code. https://github.com/PolymathicAI/the_well
  - `WellDataModule` loads `rayleigh_benard` directly.
- **Polymathic AI.** Walrus weights and fine-tuned variants (MIT licence). https://huggingface.co/polymathic-ai/walrus
  - Starting checkpoint for this task.

## Breakthroughs in related fields that could transfer

- **Price, Sanchez-Gonzalez, Alet, Andersson et al. (2023).** GenCast: Diffusion-based ensemble forecasting for medium-range weather. arXiv:2312.15796. https://arxiv.org/abs/2312.15796
  - Weather forecasting moved from point RMSE to diffusion ensembles scored with CRPS for the same reason this README gives: point forecasts blur past the predictability time.

## Published results to beat

- The Well's dataset card reports a relative error on `rayleigh_benard` of FNO 0.8395, TFNO 0.6566, U-Net 1.4860, CNextU-net 0.6699 (source: Hugging Face card above; the card's metric and horizon differ from HFM's 64-frame `rollout_nrmse`, so compare with care).
- Diaconu et al. (2026) report 20-54% lower rollout CRPS and up to 30% lower variance-normalised RMSE after CRPS fine-tuning of single-system emulators (source: arXiv:2603.01949 abstract).
- HFM: baseline 0.3693, best 0.3129 (README).

## Unverified leads

- UNVERIFIED: a published predictability-time (Lyapunov) study of 2D Rayleigh-Bénard convection as a function of Rayleigh number.
- UNVERIFIED: the Walrus GitHub repository under the PolymathicAI organization (a search summary said it exists; the only repository URL returned was a third-party copy).
