# p033 code: Neyman-Pearson limits on conditional defection

CPU-only, numpy/scipy/matplotlib. All outputs go to `../results/`.

```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 exp1_frontier.py      # learned conditional defectors vs Bayes frontier (~5 min, 4 procs)
python3 exp2_honeypots.py     # honeypot mixtures + realism-response curve (~5 min, 4 procs)
python3 exp3_scaling.py       # analytic N-scaling, literature AUC -> ceilings (seconds)
python3 exp4_measurement.py   # AUC vs tail; finite-sample identifiability (~1 min)
python3 exp0_fixed_lambda.py  # baseline: fixed-lambda training collapses (~10 min, optional)
```

Files: `common.py` (environments, numpy MLP, Adam, trainers), `theory.py`
(exact mixture frontiers, Renyi/Hoelder bound, KL bound, binormal helpers).
Seeds are fixed in each script. Literature AUCs used by exp3 are hard-coded from
Needham et al. (2025), arXiv:2505.23836, and are labelled as such.
python3 check_renyi.py      # Thm 3 bound vs exact frontier (seconds)
