# Code for "Deceleration Without a Break" (p001)

Requirements: Python 3, numpy, scipy, matplotlib. No GPU. Total CPU time is about 20 minutes.

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

`common.py` holds the shared loading, fitting (closed-form gradient, BFGS) and frontier code.
All random seeds are fixed.
