# p035 — Bounding deployment behaviour of evaluation-aware models

Reproduce all numbers and figures (CPU only, ~4 min on 4 cores):

```
cd /home/user/AI-agents/research/papers/p035
python3 -W ignore code/run_experiments.py results/raw.json   # 110 model-organism runs (11 configs x 10 seeds)
python3 code/analyze.py                                      # -> results/summary.{txt,json}
python3 -W ignore code/analyze2.py                           # -> results/diagnostics.json, fig1-3
```

Files
- `code/organism.py`   world generator, model organism (2x32 MLP), estimators and diagnostics
- `code/run_experiments.py`  configuration grid (rho, visibility of non-rewritable cue, subtle cues, kappa)
- `code/analyze.py`, `code/analyze2.py`  summaries, Spearman diagnostics, partial-rewrite budgets, policy-level curve, figures
- `results/raw.json`  per-run outputs; `results/summary.txt` main table; `results/diagnostics.json`
- `results/fig1_bias_vs_rho.png`, `fig2_diagnostics.png`, `fig3_partial_budget.png`

Dependencies: numpy 2.5, scipy, scikit-learn 1.9, matplotlib. Seeds are fixed (seed = 1000*config_index + s).
