# p023 — Redundancy or Division of Labor? (reproduction)

Requirements: Python 3, numpy, scipy, matplotlib. CPU only. Total ~3 min on 4 cores.

```
cd research/papers/p023
python3 code/run_theory.py          # E1-E6: exact/MC theory checks -> results/theory_results.json, fig1,3,4,5
python3 code/check_rate.py          # Prop. 2 rate constant check -> results/rate_check.json
for s in 0 1 2 3 4; do SEED=$s python3 code/learned_agents.py > results/learned_log_seed$s.txt; done
python3 code/aggregate_learned.py   # -> results/learned_summary.json, fig7_learned_eps_star.png
python3 code/real_data_test.py      # R1/R2 on MMLU-Pro answers -> results/real_data_results.json, fig8, fig9
(cd code && python3 exp1_correlation.py > ../results/exp1_log.txt)  # pairwise correlation stats used in Sec 6.3
```

Files
- code/theory.py — model: beta-binomial voting, ceilings, strategies, eps*, check allocation
- code/run_theory.py — theory experiments E1–E6
- code/check_rate.py — numerical check of the O(1/n) approach-to-ceiling constant
- code/learned_agents.py — small-MLP agents (generalists vs specialists, 3 data-diversity conditions)
- code/aggregate_learned.py — aggregation over 5 seeds
- code/real_data_test.py — tests on the public MMLU-Pro answer matrix (results/answers.npz)
- code/build_matrix.py, common.py, exp1..exp5, run_all.sh — data build + earlier exploratory analyses of the MMLU-Pro matrix
  (only exp1_correlation.py outputs are used in the paper); see run_all.sh for how to rebuild answers.npz from the public repo
- paper.md — the paper
