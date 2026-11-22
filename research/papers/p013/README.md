# p013 — Knowing what you don't know costs bits

Reproduction (CPU only, Python 3 + numpy, scipy, matplotlib, torch 2.x; 4 worker processes):

```
cd code
python3 theory.py          # Theorem 2 / Prop. 3 numerics, Tables T1-T6 (~10 s) -> results/theory_*.json, theory_tables.txt, fig_theory.png
python3 exp_a_frontier.py  # Experiment A, 90 MLP runs (~8 min)   -> results/exp_a.json, exp_a.log
python3 exp_b_penalty.py   # Experiment B, 36 MLP runs (~6 min)   -> results/exp_b.json, exp_b.log
python3 analyze.py         # tables + figures                      -> results/analysis_tables.txt, exp_b_summary.json, fig_exp_a.png, fig_exp_b.png
```
`python3 exp_a_frontier.py quick` runs a 7-config smoke test (results/exp_a_quick.json).
All seeds are fixed in the scripts. Every number in `paper.md` is traceable to `results/theory_tables.txt`
or `results/analysis_tables.txt`. Paper: `paper.md`.
