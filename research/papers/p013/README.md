# p013 — Knowing what you don't know costs bits

Reproduction (CPU only, Python 3 + numpy, scipy, matplotlib, torch 2.x; 4 worker processes):

```
cd code
python3 theory.py          # Theorem 2 / Prop. 3 numerics, Tables T1-T6 (~10 s) -> results/theory_*.json, theory_tables.txt, fig_theory.png
python3 exp_a_frontier.py  # Experiment A, 90 MLP runs (~8 min)   -> results/exp_a.json, exp_a.log
python3 exp_b_penalty.py   # Experiment B, 36 MLP runs (~6 min)   -> results/exp_b.json, exp_b.log
python3 analyze.py         # tables + figures                      -> results/analysis_tables.txt, exp_b_summary.json, fig_exp_a.png, fig_exp_b.png
# --- revision round 1 ---
python3 theory_relax.py            # Theorem 3 any-model relaxation, Tables R1-R4 (~5 min) -> results/relax_tables.txt, relax.json, fig_relax.png
python3 theory_closed_form.py      # Proposition 4 closed form vs exact          -> results/closed_form.txt
python3 relax_resolution_check.py  # fact-binning check (~1.5 min)                -> results/relax_resolution.txt
python3 exp_c_large_universe.py    # Experiment C, 144 MLP runs (~19 min)         -> results/exp_c.json, exp_c.log
python3 analyze_rev.py             # corrected certificates, Exp B per-seed, Exp C cross-fit + bootstrap (~1 min)
                                   #   -> results/analysis_rev.txt, exp_c_summary.json, fig_exp_c.png
python3 corollary2.py              # any-model floor at Exp C operating points    -> results/corollary2.txt
```
`python3 exp_a_frontier.py quick` runs a 7-config smoke test (results/exp_a_quick.json).
All seeds are fixed in the scripts. Every number in `paper.md` is traceable to a file in `results/` (see the Reproducibility section of the paper). Paper: `paper.md`.
