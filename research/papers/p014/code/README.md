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

## Revision round 1

```
bash rev_runs.sh          # only its first wave was run (laplace s0, int32 s0, int128 s0/s1); obs_p05 s1 and obs_p09 s2 were dropped
bash rev_runs2.sh         # remaining revision runs: laplace s1, int32 s1/s2, int128 s2, obs_p09 s1, obs_p1 s2
bash strong_runs.sh       # pooled-moment amortized models (m2/m4 features) -> results/strong/
python3 bayes_oracle.py   # exact Bayes-optimal observational baselines -> results/bayes_oracles.json
python3 rev_checks.py     # generalised refit exactness, out-of-prior fractions -> results/rev_checks.json
python3 rev_baselines.py  # prior-agnostic LiNGAM / CI-test reasoners on twins -> results/rev_baselines.json
python3 eval_saved.py     # cue-removal probe for saved label-order models -> results/cue_removal.json
python3 summarize.py      # tables + figures 1-4
```
All original JSON fields of retrained runs are bit-identical to `results/backup_r0/` (new probes use separate RNG streams).
`summarize.py` now globs `*_s[0-9].json` and no longer references the never-run `int8_p0` regime.

`results/network_check.txt` records that model hubs (Hugging Face and mirrors, Ollama registry, GitHub release assets) were unreachable, which is why no LLM / AVICI checkpoint was evaluated.
