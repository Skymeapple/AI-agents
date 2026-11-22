# p036 — Only unforgeable bits count: reproducible code

Requirements: Python 3, numpy, scipy, matplotlib, torch (CPU). All runs use 4 CPU processes.

Run from this directory (`code/`):

    python3 verify_theory.py    # exact-LP checks of Theorems 1-3 (≈10 s)   -> ../results/theory_checks.json
    python3 exp1_tradeoff.py    # E1: 36 models, noisy provenance flag (≈12 min on 4 cores) -> ../results/e1_tradeoff.json
    python3 exp3_adaptive.py    # E3: 4 systems x 3 seeds, static + adaptive attacks (≈16 min) -> ../results/e3_adaptive.json
    python3 forgery_stats.py    # marker-forgery statistics of adversarial spans -> ../results/e3_forgery_stats.json
    python3 analyze.py          # figures + ../results/summary.json

Files: `toy.py` (task generator, tiny transformer, training loop), `exp1_tradeoff.py`, `exp3_adaptive.py`,
`analyze.py`, `forgery_stats.py`, `verify_theory.py`. Figures land in `../results/figs/`.
Logs of the runs used in the paper: `../results/e1_log.txt`, `../results/e3_log.txt`.
