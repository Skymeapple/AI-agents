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

Added in revision 1:
python3 exp0b_ablation.py   # clean annealing ablation (same trainer, fixed vs annealed lambda; ~2 min)
python3 exp4b_misspec.py    # tail estimation under misspecified ROCs; sample-size targets (~6 min)
python3 exp5_certify.py     # probe 'ceilings' are lower bounds; certified bound via deployment resampling (~5 min)
python3 exp6_lowalpha.py    # learned defectors at alpha in [1e-7, 1e-3] with importance sampling (~7 min)
python3 exp7_diagnostic.py  # realism-response curve: units, confounds, factorial control (seconds)
python3 analysis_r1.py      # Table 1 seed uncertainty, per-seed examples, exact collapse rates (seconds)
exp3_scaling.py was extended (d'=2.0 row and ln N_req decomposition); all previous fields are unchanged.
