# p052 code: welfare-indicator testbed

Requirements: Python 3, numpy, torch (CPU), matplotlib.

    cd code
    python3 run_all.py        # all experiments, ~10-15 min on 4 CPU cores
    python3 make_figures.py   # tables.md and fig1-fig5 from results/results.json

Files
- testbed.py      reference agent A (affect-like integrator), mimic architectures, behavioural cloning
- run_all.py      all tests: fidelity / Prop. 1 check, trade-off, persistence, steering-coupling,
                  report-suppression fine-tuning, training-horizon coverage sweep, bottleneck sweep
- make_figures.py tables (results/tables.md) and figures (results/fig*.png)

All randomness is seeded; results/results.json is the single source for every number in the paper.
- diag_bottleneck.py  diagnostic behind Sec. 5.6 (why global normalisation gives rho > 1); writes results/diag_bottleneck.json

Runtime of run_all.py in this environment: 473 s with 2 torch threads (set NTHREADS to change).
