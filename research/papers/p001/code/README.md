# Code for "Deceleration Without a Break" (p001)

Requirements: Python 3, numpy, scipy, matplotlib. No GPU. Revision 1 added Exps. 8-12; a full rerun takes roughly an hour of wall-clock time on 4 cores (see `../results/run_logs/run_all_r1.sh` and `timing.txt`). Exps. 8, 9 and 11 use `multiprocessing`.

Data: `../data/chinchilla.csv` is Epoch AI's reconstruction of the Hoffmann et al. (2022) Figure 4 runs.
It was downloaded from
https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv

Run the scripts in this order from this directory. Each one writes JSON, figures and logs to `../results/`.

| script | what it does | runtime |
|---|---|---|
| `exp1_fit_and_mechanical_slowdown.py` | Refits the Chinchilla law: Huber loss with delta=1e-3 on log-loss, 4,500-point initialisation grid, Epoch's rule of excluding the 5 highest-loss runs. Runs 300 bootstrap resamples. Writes the frontier, the local exponents and Fig. 1. | ~3 min |
| `exp2_break_test_real.py` | Fits the nested break model to the real data with a profiled breakpoint and 300 bootstrap resamples. Also runs the compute-split backtest. Writes Fig. 2. | ~3 min |
| `exp3_simulation_power.py` | Simulation study measuring the size and power of the naive curvature test against the break likelihood-ratio test (4,000 synthetic datasets). Writes Fig. 3. | ~6 min |
| `exp4_data_wall.py` | Computes the compute-optimal frontier under a cap on unique tokens, using the Muennighoff et al. (2023) law. Reports compute efficiency, the shadow price of data and a calendar illustration. Writes Fig. 4. | <5 s |
| `exp5_wall_sensitivity.py` | Numerical check of the self-similarity result, plus sensitivity to R_D*, R_N* and to the base law. | ~10 s |
| `exp6_confound_and_wall_detection.py` | Measures how much a fitted E absorbs an unmodelled break, and when the break test detects a data wall. | ~4 min |
| `exp7_signal_shrinkage.py` | Measures break-test power as the design window moves to higher compute, and the paired data-doubling sample sizes. Writes Fig. 5. | ~3 min |

| `exp8_detectability_realistic.py` | (rev. 1) Power of the exact real-data procedure (Huber, profiled breakpoint, calibrated sup statistic) at per-run noise 1e-4 to 7.5e-3, windows ending 1e22-1e28, per-budget correlated noise, small design; Wilson MC intervals; fitted-null size check. Writes Fig. 6. | ~25 min |
| `exp9_real_inference.py` | (rev. 1) Support counts, digitisation/noise decomposition, wide-grid profiled fit, parametric and wild-cluster calibrated tests, cluster bootstrap, power on the actual design, all-245 sensitivity. | ~13 min |
| `exp9b_two_break.py` | (rev. 1) Two-break model with cluster bootstrap. | ~3 min |
| `fig7_real_profile.py` | (rev. 1) Fig. 7 from the Exp. 9 output. | <5 s |
| `exp10_wall_generalised.py` | (rev. 1) Generalised self-similarity (consistent U_N rule), transfer-exponent uncertainty, Lovelace et al. (2026) functional forms, economics and calendar with the 4.6e26 anchor. | ~1 min |
| `exp11_competing_mechanisms.py` | (rev. 1) Unique-data cap vs quality decline vs non-data exponent break: efficiency, value of the data lever, matched-doubling gap, loss-curve detectability. | ~5 min |
| `exp12_noise_budget.py` | (rev. 1) Assumption-driven evaluation-noise budget. | <1 s |

`breaktest.py` holds the break test used identically on real and simulated data (rev. 1).
`common.py` holds the shared loading, fitting (closed-form gradient, BFGS) and frontier code.
All random seeds are fixed.
