# p013 — Knowing what you don't know costs bits

Reproduction (CPU only, Python 3 + numpy, scipy, matplotlib, torch):

```
cd code
python3 theory.py          # Theorem 2 numerics, sensitivity tables  (~10 s)  -> results/theory_*.json, theory_tables.txt, fig_theory.png
python3 exp_a_frontier.py  # Experiment A, 90 MLP runs, 4 processes (~10 min) -> results/exp_a.json, exp_a.log
python3 exp_b_penalty.py   # Experiment B, 36 MLP runs, 4 processes (~10 min) -> results/exp_b.json, exp_b.log
python3 analyze.py         # tables + figures                              -> results/analysis_tables.txt, fig_exp_a.png, fig_exp_b.png
```
`python3 exp_a_frontier.py quick` runs a 7-config smoke test.
All seeds are fixed in the scripts. Paper: `paper.md`.
