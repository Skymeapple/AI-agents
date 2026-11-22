# p031 — Persistence, correlated blind spots and resampling in bootstrapped oversight

Reproduce everything (CPU, ~3 min total):

```bash
pip install numpy scipy matplotlib scikit-learn
cd code
python3 run_theory.py            # BOS model: rates, schedules, DP vs MC validation, retries, rho sweep, sensitivity -> results/theory_results.json, fig1-5
python3 gamma_limit.py           # Proposition 5 check (non-persistent misalignment)  -> results/gamma_limit.json
python3 toy_rho.py ../results/toy_rho.json   # toy measurement of overseer-error dependence rho -> results/toy_rho.json (~2 min)
```

Files
- `code/bos_model.py` — model definition, exact dynamic programs (chain; anchored with comonotone errors), vectorised Monte Carlo (general case).
- `code/run_theory.py` — all model experiments and figures.
- `code/gamma_limit.py` — numerical check of the gamma<1 limit.
- `code/toy_rho.py` — MLP-overseer experiment estimating rho on natural vs adversarially selected defects.
- `paper.md` — the paper.
All numbers in the paper come from the JSON files in `results/` (seeds fixed in code).
