# p003 — Rate, Race, or Fate: verifier tails and exploit races in RL beyond math

Paper: `paper.md`. Code: `code/`. Outputs: `results/` (JSON + PNG; every number in the paper is in one of these files).

Requirements: Python 3, numpy, scipy, matplotlib. No GPU. E11 reads the public JudgeBench judgment files (github.com/ScalerLab/JudgeBench, commit e2c52c28), which are not redistributed here.

Run everything: `bash code/run_all.sh` (≈20 CPU-minutes). All experiments are seeded.

| script | what it does | output |
|---|---|---|
| `theory.py` | CGF identity, tilted accuracy, three-mode closed form | – |
| `universe.py` | synthetic response universes | – |
| `exp1_theory_checks.py` | numerical checks of Theorem 1 / Corollary 2 / error-floor law | exp1.json, fig1 |
| `exp2_tabular_grpo.py` | KL-regularised GRPO vs tilt; h0 sweep (theory vs softmax PG vs GRPO) | exp2.json, fig2 |
| `exp2c_race_threshold.py` | exploit-starvation threshold h0* vs race heuristic | exp2c.json |
| `exp4_sequence.py` | GRPO on length-10 binary sequences, 3 policy classes, 3 verifier profiles | exp4.json, fig4 |
| `exp5_race_grid.py` | collapse fraction over (J, h0); offline balanced accuracy; learning speed | exp5.json, fig5 |
| `exp6_kl_paradox.py` | KL penalty vs rare exploit | exp6.json, fig6 |
| `exp7_boundary_earlywarning.py` | critical J*(h0) power law; early-warning audit AUC | exp7.json, fig7 |
| `exp2b_partA_bias.py` | (G-1)/G finite-group correction of the Part A fixed point | exp2b.json |
| `exp8_offline_audits.py` | aggregate / tail-aware / look-ahead offline audits vs on-policy audits, matched budgets, bootstrap CIs | exp8.json |
| `exp9_meanfield_boundary.py` | takeover boundary J*(h0) derived from the population (mean-field) dynamics | exp9.json, fig9 |
| `exp10_robustness.py` | alternative exploit encodings, entropy bonus / natural gradient / Adam, lr & group size, J=0.75 column, t90 vs J | exp10.json |
| `exp11_real_judges.py` | real-judge anchor: discrimination and tail-inversion of 33 judges by domain (JudgeBench outputs) | exp11.json |
| `exp12_metastability.py` | exploit-mass trajectories of KL-regularised GRPO over 32k steps | exp12.json |
| `figs_revision.py` | figures 8 and 11 | fig8, fig11 |
