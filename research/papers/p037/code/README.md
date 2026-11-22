# Code for "What Can a Finite Evaluation Rule Out?" (p037)

Requirements: Python 3, numpy, scipy, matplotlib, PyYAML (all CPU). Total runtime about 15-20 min on 4 cores.

Data: `../data/metr_th11_runs.jsonl` and `../data/release_dates.yaml` from METR's public
repository `METR/eval-analysis-public` (commit recorded in `../data/SOURCE.txt`), Time Horizon 1.1 runs.

Run from this directory, in order:

```
python3 exp1_clustering.py        # clustering / design effect of suite-level success rates
python3 exp2_retry_metr.py        # pass@K extrapolation on METR runs: validation (held-out runs) + K up to 1000
python3 exp3_theory_synthetic.py  # theory checks: attacker rule of three, vanishing identification, misspecification
python3 exp4_floor_metr.py        # value of a declared resolution-floor assumption on real data
python3 exp5_suite_sizes.py       # what public benchmark sizes can rule out (analytic, uses exp1 deff)
python3 make_figures.py           # figures -> ../results/fig*.png
```

`common.py` holds the estimators: the unbiased pass@k estimator, the sharp identified-set
linear program for pass@K given the Binomial(m) count distribution (with optional
resolution floor and minimal-slack relaxation when the empirical pmf lies outside the mixture
moment space), cluster-bootstrap / exact hybrid simultaneous confidence boxes, and the
beta-binomial extrapolation baseline. All random seeds are fixed in each script.
