# Code for "Privileged Variance and Steering Confounds" (paper p051)

Requirements: Python 3, numpy, scipy, matplotlib. CPU only. No GPU and no LLMs.

Reproduce everything (about 5 minutes on 4 cores):

```
cd code
python3 theory.py            # Props 2-4 numerical checks  -> results/theory.json
python3 run_experiments.py   # E1-E5 MLP experiments       -> results/E*.json
python3 plots.py             # figures + summary.json      -> results/fig*.png, results/summary.json
```

Files:
- `core.py`: the toy system (trunk, internal state z with privileged fraction alpha, entangled representation H), the numpy MLP reporter, label mixing (lambda), the three steering directions (oracle, difference-of-means, residualized), and the metrics (acc_self, agree_typical, DA, ISG_*).
- `theory.py`: Monte-Carlo checks of kappa(alpha) and the mimic ceiling (Prop. 2), the Bayes-optimal mixed-label reporter (Prop. 3), and the closed-form ridge reporter vs. finite-sample ridge (Props. 4-5).
- `run_experiments.py`: E1 alpha sweep, E2 lambda sweep, E3 self-evidence budget (small n), E4 weight-decay sweep, E5 deterministic idiosyncrasy variant.
- `plots.py`: figures 1-4 and `results/summary.json` (all numbers quoted in the paper).

Seeds are fixed, so reruns give the same results up to BLAS nondeterminism.
