# Code for "Beyond the Chain: Redundancy, Correlated Errors, and Collusion in Nested Scalable Oversight"

Requirements: Python 3, numpy, scipy, matplotlib (tested with numpy 2.5.3, scipy 1.18.1).

    cd code
    python3 experiments.py      # ~1 minute on one CPU core

* `nso.py` - the model: Elo-based detection probabilities, oversight topologies
  (window-k, anchored, full), exact dynamic program over trust states
  (`exact_dp`), closed-form strict success (`strict_success`), direct Monte Carlo
  simulator (`monte_carlo`, also supports cross-level correlation), and the
  leading-order formula of Proposition 3.
* `experiments.py` - experiments E1-E8 of the paper. Every number in the paper
  is read from `../results/*.json` / `../results/e6_budget_table.csv`.

Seeds are fixed; Monte Carlo uses 200,000 runs per configuration.
The game parameters (`REGIMES`) are stylized, not fitted to any real model.
