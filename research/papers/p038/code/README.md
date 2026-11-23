# Code for "Odds, Not Scores" (p038)

Requirements: Python 3, numpy, scipy, matplotlib. No GPU, no network.

Run everything (about 1 minute on one CPU core):

    bash run_all.sh

Outputs (JSON + PNG) go to `../results/`; `../results/run_log.txt` is the console log of the run
that the paper's numbers come from.

| Script | Experiments | Main outputs |
|---|---|---|
| `model.py` | closed forms (Thm 1, Prop 3-4), exact offspring pmf by DP, PGF extinction, attempt-level Monte Carlo | (library) |
| `exp1_validate.py` | E1 theory-vs-simulation check; E2a attempt-budget monotonicity; E2b phase diagram | `e1_validation.json`, `e2a_bangbang.json`, `e2b_*`, `fig_e1*`, `fig_e2a*`, `fig_e2b*` |
| `exp3_aggregation.py` | E3 score aggregation vs R; E4 illustrative 4-stage pipeline, elasticities, defense allocation | `e3_e4_aggregation.json`, `fig_e3*`, `fig_e4*` |
| `exp5_dynamics.py` | E5 valley crossing under improvement; E6 mutational escape; E7 sweeps vs uniform; E8 certification sample size | `e5_e8_dynamics.json`, `fig_e5*`..`fig_e8*` |

All random seeds are fixed. The 4-stage parameter profile in E4 is ILLUSTRATIVE and is not an
estimate of any real model.
