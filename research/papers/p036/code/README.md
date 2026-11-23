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

## Added in revision 1

    python3 verify_partial.py   # exact-LP checks of Theorems 4, 5 and Proposition 6 (~40 s) -> ../results/partial_theory_checks.json
    python3 exp2_partial.py     # E2: 144 learned policies under eps-contaminated mimicry, exact worst-case eval (~13 min on 4 cores) -> ../results/e2_partial.json
    python3 stats_r1.py         # revised E1/E3 statistics (ddof=1 s.d., t and Wilson intervals, Agresti-Coull z) -> ../results/stats_r1.json
    python3 analyze_r1.py       # E2 summary + fig3_partial.png; redraws fig2_adaptive.png -> ../results/e2_summary.json
    python3 cap_analysis.py     # optimal per-content cap of Theorem 5 on the E2 instance + caps of trained policies -> ../results/e2_cap.json

Note: `analyze.py` (round 0) also writes fig2_adaptive.png; run `analyze_r1.py` after it for the revised figure.
