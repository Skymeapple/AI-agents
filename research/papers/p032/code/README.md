# Code for "Beyond the Chain: Propagation Blocking, Correlated Errors, and Collusion in Nested Scalable Oversight"

Requirements: Python 3, numpy, scipy, matplotlib, mpmath (tested with numpy 2.5.3, scipy 1.18.1).

    cd code
    python3 experiments.py      # round-1 experiments E1-E8, ~1 minute on one core
    python3 experiments_r1.py   # revision experiments R0-R7, ~4 minutes on 4 cores
    python3 analyze_r1.py       # summary tables -> ../results/r1_tables.json

* `nso.py` - the model. Elo detection probabilities; window/anchored/full topologies;
  exact DP (`exact_dp`, round 1) and the extended vectorised DP `dp_fast`
  (same-generation copies r, copy correlation rho_s, threshold t, correction
  probability c); Monte Carlo simulators (`monte_carlo`, `monte_carlo_r`, the latter
  with arbitrary across-generation correlation matrices); false-alarm accounting;
  quasi-stationary transition matrix and R_k for Theorem 3.
* `experiments.py` - E1-E8 (unchanged except that the E3 figure draws the
  leading-order curve only where it is <= 0.5).
* `experiments_r1.py` - R0 validation of the extended DP; R1 design study with games
  calibrated to the published Engels et al. (2025) NSO success rates; R2 sensitivity
  over (m_G, m_H, beta); R3 robustness (c, t, false-alarm cost/constraint);
  R4 distance-dependent correlation; R5 Theorem 3 bounds and 60-digit scaling test;
  R6 audit of round-1 E6 claims; R7 same-generation correlation crossover.
* `analyze_r1.py` - aggregates used in the paper.

Seeds are fixed; Monte Carlo uses 200,000 runs per configuration.
Game parameters are stylised (E1-E8, R3, R7) or calibrated to one published number per
game (R1); they are not the Engels et al. fitted curves.
