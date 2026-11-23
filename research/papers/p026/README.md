# p026 — Identity Half-Life: persistence of agent memory/identity under recursive consolidation

Paper: `paper.md`. All numbers in the paper come from the scripts below (outputs in `results/`).

Run everything (CPU only, ~2 minutes on 4 cores; Python 3.13, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2):

    bash code/run_all.sh

| script | what it does | outputs |
|---|---|---|
| `code/e1_contraction.py` | exact (matrix-power) check of Theorem 1 bounds on random consolidation kernels | `results/e1_contraction.json`, `fig1_contraction.png` |
| `code/e2_redundancy.py` | self-reconciled replicas with common-mode errors; half-life vs R and gamma (Theorem 2) | `results/e2_redundancy.json`, `fig2_redundancy.png` |
| `code/e3_policies.py` | Monte Carlo of 7 persistence policies vs closed forms (Prop. 1, Theorem 3) | `results/e3_policies.json`, `fig3_policies.png` |
| `code/e4_eviction.py` | budgeted recency-decayed salience memory; eviction age and recall-refresh lifetime (Prop. 4) | `results/e4_eviction.json`, `fig4_eviction.png` |
| `code/e5_estimator.py` | estimating long-horizon persistence from 5-step probe chains; 4 estimators x 3 heterogeneity scenarios | `results/e5_estimator.json`, `fig5_estimator.png` |

`code/common.py` holds closed-form expressions. All random seeds are fixed; logs of the reference run are in `results/*_log.txt`.
No LLMs were run (no GPU / API access in the build environment); Section 8 of the paper gives a pre-registered protocol for testing the predictions on LLM agents.
