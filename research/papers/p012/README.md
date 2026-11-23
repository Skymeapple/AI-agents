# p012 — Familiar Is Not Known

Code and results for the paper `paper.md`.

Requirements: Python 3, numpy, scipy, matplotlib (no GPU, no internet).

```
cd code
python3 theory_check.py        # Monte-Carlo checks of Propositions 1-2  -> results/theory_check.json, fig_theory.png  (~10 s)
python3 run_all.py 3 5 30      # (or per world: python3 run_all.py 3 5 30 <seed>, then python3 run_all.py 3 5 30 --merge)
# 3 synthetic worlds x 5-member ensembles x 30 epochs -> results/main_results.json (reviewer measured ~3.2 h/world with concurrent multithreaded-BLAS runs; set OMP_NUM_THREADS=1)
for s in 0 1 2; do python3 e1_qonly.py $s; done   # E1b: delta-hat and best q-only detector -> results/e1_qonly_w*.json (in our session: ~35 min wall-clock with the three worlds run in parallel on a loaded 4-core machine)
python3 make_figures.py        # -> results/summary.txt, fig_familiarity.png, fig_aleatoric.png, fig_probe.png
```

Revision R1 (reviewer response; see reviews/response_r1.md). Use single-threaded BLAS per process:
```
cd code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 7 regimes: base (worlds 0-4), early, small, comp, matched, matched_early, matched_ls (worlds 0-2); 5 members each
python3 train_cache.py base 0      # -> cache/base_w0.npz  (216-269 s per base world; cache/ is not committed, ~600 MB total)
# all 23 jobs: 43 CPU-min, ~20 min wall on 4 cores with xargs -P 4
python3 analysis_r1.py base 0      # -> results/r1/base_w0.json (~20 s each)
python3 aggregate_r1.py            # -> results/r1/summary_r1.txt, fig_r1_prop1_strata.png, fig_r1_prop2_gap.png
```


Files
- `code/world.py`   synthetic knowledge world (Zipf entity popularity, partial relation coverage, latent types, aleatoric relations)
- `code/mlp.py`     numpy embedding-MLP trained with Adam
- `code/theory.py`  closed forms for Prop. 1 (AUROC ceiling) and Prop. 2 (familiarity shortcut in LDA probes)
- `code/run_all.py` experiments E1-E4
- `code/e1_qonly.py` E1b (is Proposition 1 binding? estimated TV delta, trained q-only detector); `results/e1_qonly_summary.txt` holds the 3-world means
- `results/run_log_w*.txt` timing logs of the main run; `results/e1_qonly_log_w*.txt` E1b logs

- `code/train_cache.py`, `code/analysis_r1.py`, `code/aggregate_r1.py` R1 experiments (regimes, exposure-matched E1, Bayes reference learner, within-stratum AUROC, Prop. 2 decomposition, probe ablations)

All seeds are fixed in code; rerunning reproduces the numbers in `results/summary.txt`.
