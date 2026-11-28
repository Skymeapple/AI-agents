# Code for "What Can a Finite Evaluation Rule Out?" (p037)

Requirements: Python 3, numpy, scipy, matplotlib, PyYAML (all CPU). Total runtime about 30 min on 4 cores.

Data: `../data/metr_th11_runs.jsonl` and `../data/release_dates.yaml` from METR's public
repository `METR/eval-analysis-public` (commit recorded in `../data/SOURCE.txt`), Time Horizon 1.1 runs.

Run from this directory, in order:

```
python3 exp1_clustering.py        # clustering: naive / runs-as-replicates / task / family SEs, CR1, design effects
python3 exp2_retry_metr.py        # pass@K on METR runs: held-out sanity check + K up to 1000 (14-point grid)
                                  #   (`python3 exp2_retry_metr.py b` regenerates only the extrapolation part)
python3 exp2_supplement.py        # rev.1: slack-relaxation usage, grid-floor sensitivity, per-model aggregation
python3 exp3_theory_synthetic.py  # theory checks: Prop. 3(a) coverage, vanishing identification, misspecification
python3 exp3_supplement.py        # rev.1: corrected Prop. 3(b) with power beta (+ LP check), tightness of Prop. 2 rate
python3 exp4_floor_metr.py        # value of a declared resolution-floor assumption on real data
python3 exp5_suite_sizes.py       # suite sizes (Cybench; WMDP chance-corrected) and cyber attempt counts (analytic)
python3 exp6_clustered_coverage.py # rev.1: coverage of confidence boxes under cluster sampling (~7 min, 4 processes)
python3 make_figures.py           # figures -> ../results/fig*.png
```

`common.py` holds the estimators: the unbiased pass@k estimator, the sharp identified-set
linear program for pass@K given the Binomial(m) count distribution (with optional
resolution floor and minimal-slack relaxation when the empirical pmf lies outside the mixture
moment space), cluster-bootstrap / exact hybrid simultaneous confidence boxes, and the
beta-binomial extrapolation baseline. All random seeds are fixed in each script.
