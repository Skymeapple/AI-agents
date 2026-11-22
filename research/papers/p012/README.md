# p012 — Familiar Is Not Known

Code and results for the paper `paper.md`.

Requirements: Python 3, numpy, scipy, matplotlib (no GPU, no internet).

```
cd code
python3 theory_check.py        # Monte-Carlo checks of Propositions 1-2  -> results/theory_check.json, fig_theory.png  (~10 s)
python3 run_all.py 3 5 30      # (or per world: python3 run_all.py 3 5 30 <seed>, then python3 run_all.py 3 5 30 --merge)
# 3 synthetic worlds x 5-member ensembles x 30 epochs -> results/main_results.json (~6 min per world, worlds run in parallel on 4 cores)
for s in 0 1 2; do python3 e1_qonly.py $s; done   # E1b: delta-hat and best q-only detector -> results/e1_qonly_w*.json (in our session: ~35 min wall-clock with the three worlds run in parallel on a loaded 4-core machine)
python3 make_figures.py        # -> results/summary.txt, fig_familiarity.png, fig_aleatoric.png, fig_probe.png
```

Files
- `code/world.py`   synthetic knowledge world (Zipf entity popularity, partial relation coverage, latent types, aleatoric relations)
- `code/mlp.py`     numpy embedding-MLP trained with Adam
- `code/theory.py`  closed forms for Prop. 1 (AUROC ceiling) and Prop. 2 (familiarity shortcut in LDA probes)
- `code/run_all.py` experiments E1-E4
- `code/e1_qonly.py` E1b (is Proposition 1 binding? estimated TV delta, trained q-only detector); `results/e1_qonly_summary.txt` holds the 3-world means
- `results/run_log_w*.txt` timing logs of the main run; `results/e1_qonly_log_w*.txt` E1b logs

All seeds are fixed in code; rerunning reproduces the numbers in `results/summary.txt`.
