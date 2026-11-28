# Code for "Weakest-Link Geography" (p084)

Requirements: Python 3, numpy, scipy, matplotlib (no GPU, no network).

```
cd code
python3 run_all.py      # ~3 min on 4 CPUs; writes ../results/results.json, E*.csv, fig1-6
python3 shapley.py      # Shapley decomposition of input hardening (reads ../results/E3.csv)
```

Files
- `params.py`  all parameters; each tagged [S:source], [A] assumption, or [P] scenario parameter.
- `model.py`   Monte Carlo simulator (weekly, Leontief in logic/CoWoS/HBM, site/regional hazards,
               priority vs proportional allocation, inventory buffer, foundry-porting lag) and the
               closed-form expressions used in Propositions 1-3.
- `run_all.py` experiments E1-E7 of the paper (seed 12345, 100,000 simulated years per configuration,
               common random numbers across configurations).
- `shapley.py` Shapley values / interaction term from E3.

Logs of the exact runs reported in the paper: `../results/run_log.txt`, `../results/shapley_log.txt`.
