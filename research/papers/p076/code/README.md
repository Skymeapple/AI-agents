# TB-MHC: throughput-bounded meaningful human control — code

Requirements: Python 3, numpy, scipy, matplotlib.

Run everything (about 10 s of CPU; deterministic seed 20261122):

    cd code && python3 experiments.py

Files:
- `model.py`: closed-form model (speed-accuracy function, quota and Bayes reviewer regimes, d* thresholds, throughput bound lambda_max, veto rate).
- `experiments.py`: experiments E1-E8 (the paper's Section 5). Writes to `../results/`:
  - `summary.json`: every number quoted in the paper
  - `e1_validation.json`: Monte Carlo vs. closed form, 40 configurations x 1e6 draws
  - `e3_lambda_max_grid.csv`: sensitivity grid for t* and lambda_max
  - `fig1_catch_vs_time.png` ... `fig5_nonmonotone_region.png`

JSON key convention: `q0p9` means q = 0.9.
