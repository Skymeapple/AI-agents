# Reproducing p034: "Proxy-only detectors cannot see shared exploits" (revision 1)

Requirements: Python 3, numpy, scipy, matplotlib, scikit-learn (tested: sklearn 1.9.1). CPU only.
Runtimes below were measured on a 4-core machine that was shared with other jobs.

```
cd code
python3 exp_theory.py            # Exp 1-3: closed forms, best-of-n, first-version stopping study   (~1.3 min)
python3 exp_learned_rm.py        # Exp 4: single-seed learned-RM study                              (~1.6 min)
python3 exp_lowerbound.py        # Exp 5 (first version): 1/2 inf-sum and deterministic min-max      (seconds)
python3 exp_lowerbound_v2.py     # Exp 5 (revised): exact randomised game value V, twin sweep       (seconds)
python3 exp_revision.py          # R1-R5: equal-budget gold baselines, proxy/classifier baselines,
                                 #        twin world, label noise/bias, sensitivity + transfer       (~2 min, 4 procs)
python3 exp_ppo.py               # Exp 6: PPO-like (Adam + clipping) non-tilting optimiser           (~2 min, 4 procs)
python3 exp_learned_rm_seeds.py  # Exp 4b: 5 seeds x 8 held-out RM types                            (~1 min, 4 procs)
python3 make_fig_revision.py     # Fig 5, Fig 6, derived_label_efficiency.json (after the above)    (seconds)
```

Outputs are written to `../results/`:

| File | Contents |
|---|---|
| `theory_results.json` | Exps 1-3, including regret SEs and cumulative labels; fig1-3 |
| `learned_rm_results.json` | Exp 4, including regret SEs; fig4 |
| `lowerbound_results.json` | Exp 5 (first version) |
| `lowerbound_v2_results.json` | Exp 5 (revised) |
| `revision_results.json` | R1-R5 |
| `ppo_results.json` | Exp 6 |
| `learned_rm_seeds_results.json` | Exp 4b |
| `fig5_budget_frontier.png`, `fig6_twin_tradeoff.png`, `derived_label_efficiency.json` | revision figures and derived numbers |
| `log_*.txt` | console logs |

Rerunning `exp_theory.py` and `exp_learned_rm.py` after the revision only added fields; every number from the first version was reproduced exactly.

Files:

| File | Contents |
|---|---|
| `model.py` | exploit-mixture model, closed forms, samplers |
| `audit.py` | Hansen-Hurwitz covariance estimator and audit designs |
| `exp3_audit.py` | Exp 3 |
| `rev_common.py` | samplers for K proxies, the twin world W', and noisy/biased labels; estimators |

Exp 1a reports its error only at checkpoints with ESS > 1e4. These are lambda in {0, 0.25, 0.5, 0.75}.

All random seeds are fixed in the scripts.
