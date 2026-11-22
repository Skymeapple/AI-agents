# Observational twins: code

Requirements: Python 3, numpy, torch (CPU), matplotlib.

```
bash run_all.sh        # all results (12 training runs, 4 in parallel; ~30-35 min on 4 cores)
```

- `scm.py` — linear SCM sampler, covered-edge twin construction (Prop. 3), exact log-likelihoods, MEC table.
- `verify_theory.py` — exhaustive n=4 checks -> `results/theory_checks.json`.
- `twins.py` — trains one DeepSets causal-query model for a regime and evaluates i.i.d. and twin accuracy -> `results/<name>_s<seed>.json`.
- `summarize.py` — aggregates JSONs -> `results/summary_table.md`, `results/fig1_iid_vs_twin.png`, `fig2_interventions.png`, `fig3_regimes.png`.

The four `obs_p*` seed-0 runs were produced in a first pass with identical code (only a logging warning was later silenced); `rest.sh` produced the other eight runs.
