# p019 code: shared-bias verification of AI-generated hypotheses

Requirements: Python 3, numpy, scipy, matplotlib (CPU only). Total runtime about 5 minutes on 4 cores.

```
cd code
python3 exp1_theory_check.py        # E1: analytic FDR curves vs Monte Carlo, K* rule  (~2 s)
python3 exp2_sem_world.py 300       # E2-E5: 300 synthetic SEM worlds, sample-size sweeps, depth run (~4-5 min)
python3 analyze_exp2.py | tee ../results/exp2_summary.txt   # tables, bootstrap CIs, figures (~2 s)
```

Files
- `theory.py` - analytic model (Theorems 1-2): acceptance probabilities, FDR, n->inf limits, K* rule, Monte Carlo simulator.
- `exp1_theory_check.py` - writes `results/exp1_theory.json`, `fig1_fdr_vs_n.png`, `fig2_kstar.png`.
- `exp2_sem_world.py` - linear-Gaussian structural causal model worlds with latent confounders, a simulated
  literature, an observational hypothesis generator, and five verifiers. Writes `results/exp2_rows.json`,
  `exp2_sweep_{noiseless,noisy}.json`, `exp2_depth_rows.json`.
- `analyze_exp2.py` - writes `results/exp2_summary.json`, `exp2_summary.txt`, `fig3_sem_sweep.png`,
  `fig4_e3_prediction.png`, `fig5_novelty.png`.

All seeds are fixed (world seeds 1000+, 5000+, 9000+; bootstrap seed 7), so reruns reproduce the numbers in the paper.
