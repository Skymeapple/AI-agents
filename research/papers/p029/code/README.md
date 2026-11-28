# Code for p029 — liability apportionment under noisy failure attribution

Python 3 with numpy, scipy, matplotlib. No GPU, no LLM calls.

```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # keep to one core
sh fetch_data.sh                 # clones Who&When (~17 MB) into ../ext/ww (git-ignored); needed only for exp4
python3 exp1_theory_checks.py    # ~15 s  -> results/exp1_theory_checks.json
python3 exp2_regimes.py          # ~2 min -> results/exp2_regimes.json, fig1_regimes.png
python3 exp3_oversight.py        # ~5 min -> results/exp3_oversight.json, fig2_oversight.png
python3 exp3b_optimal_share.py   # ~2 min -> results/exp3b_optimal_share.json, fig3_share.png
python3 exp4_whowhen.py          # <5 s   -> results/exp4_whowhen.json
```

- `model.py` — channel, incentive budget B(C), rules (channelling, attributed-pays-all, equal, debiased C^-1), closed-form deadweight loss, optimal no-rebate rule (multi-start SLSQP).
- All technology parameters (L, b, tau, sigma, beta, gamma) are assumed, illustrative values. Seeds are fixed in each script.
- exp4 reads the Who&When JSON logs only (commit 1bcbbd96 of ag2ai/Agents_Failure_Attribution at the time of writing); the published accuracies it translates are typed in from Table 1 of Zhang et al. (2025).
