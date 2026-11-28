# p031 — Common-mode blind spots, recovering lineages and resampling in bootstrapped oversight

Reproduce everything (CPU only, about 20 min total, most of it in toy_models.py):

```bash
pip install numpy scipy matplotlib scikit-learn
cd code
python3 run_theory.py                 # rates, schedules, DP vs MC, retry/rho sweeps (exact any-rho DP), sensitivity -> results/theory_results.json, fig1-5
python3 run_revision.py               # round-1 revision: brute-force validation, ensemble floor tests, anchored rate vs rho-floor,
                                      # optimal retry budget, Table-2 sensitivity, gamma(Delta) -> results/revision_results.json, fig6, fig7
python3 gamma_limit.py                # constant-gamma limit check -> results/gamma_limit.json
python3 toy_rho.py ../results/toy_rho.json   # (secondary) input-level dependence toy -> results/toy_rho.json
python3 toy_models.py 0; python3 toy_models.py 1; python3 toy_models.py 2   # model-level rho toy, ~4.6 min/seed (can run in parallel)
python3 toy_models.py merge           # -> results/toy_models.json
```

Files
- `code/bos_model.py` — model; exact DPs (chain any kappa/gamma/f/R; anchored any rho with gamma=kappa=1); brute-force enumeration (small n, any parameters); Monte Carlo (true R=inf); two-state closed form with gamma(Delta).
- `code/run_theory.py`, `code/run_revision.py`, `code/gamma_limit.py` — model experiments and figures.
- `code/toy_models.py` — meta-classifier overseers judging trained (backdoored vs clean) MLPs; rho over model draws.
- `code/toy_rho.py` — round-0 input-level toy (kept as secondary evidence).
- `paper.md` — the paper; `reviews/` — review and response.
All numbers in the paper come from the JSON files in `results/` (seeds fixed in code).
