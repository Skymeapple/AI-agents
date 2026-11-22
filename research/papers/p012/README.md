# p012 — Familiar Is Not Known

Code and results for the paper `paper.md`.

Requirements: Python 3, numpy, scipy, matplotlib (no GPU, no internet).

```
cd code
python3 theory_check.py        # Monte-Carlo checks of Propositions 1-2  -> results/theory_check.json, fig_theory.png  (~10 s)
python3 run_all.py 3 5 30      # 3 synthetic worlds x 5-member ensembles x 30 epochs -> results/main_results.json (~10 min, 4 CPU)
python3 make_figures.py        # -> results/summary.txt, fig_familiarity.png, fig_aleatoric.png, fig_probe.png
```

Files
- `code/world.py`   synthetic knowledge world (Zipf entity popularity, partial relation coverage, latent types, aleatoric relations)
- `code/mlp.py`     numpy embedding-MLP trained with Adam
- `code/theory.py`  closed forms for Prop. 1 (AUROC ceiling) and Prop. 2 (familiarity shortcut in LDA probes)
- `code/run_all.py` experiments E1-E4
- `results/run_log.txt` timing log of the main run

All seeds are fixed in code; rerunning reproduces the numbers in `results/summary.txt`.
