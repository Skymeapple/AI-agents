# Reproducing p034: "Proxy-only detectors cannot see shared exploits"

Requirements: Python 3, numpy, scipy, matplotlib, scikit-learn (tested: sklearn 1.9.1). CPU only.

```
cd code
python3 exp_theory.py       # Exp 1-3: closed forms, best-of-n, audit stopping rules  (~10-15 min on 4 CPUs)
python3 exp_learned_rm.py   # Exp 4: learned MLP/ridge reward models over a finite output pool (~3-6 min)
python3 exp_lowerbound.py   # Exp 5: minimax regret lower bound for proxy-only rules (seconds)
```

Outputs (in ../results/):
- theory_results.json, fig1_closed_form.png, fig2_bon_vs_tilt.png, fig3_stopping_regret.png
- learned_rm_results.json, fig4_learned_rm.png
- lowerbound_results.json
- log_*.txt: console logs of the runs

Files: model.py (exploit-mixture model, closed forms, exact samplers), audit.py (Hansen-Hurwitz
covariance estimator and audit designs), exp3_audit.py (stopping-rule experiments, called by exp_theory.py).
All random seeds are fixed (numpy default_rng(0) / (1) and the seeds listed in the scripts).
