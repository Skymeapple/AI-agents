# p081 — Headroom budgets: why frontier training compute growth is mostly transient

Paper: `paper.md`. Code: `code/`. Outputs: `results/`.

Reproduce everything (Python 3, numpy, matplotlib; ~1 min CPU):

    bash code/run_all.sh

Files
- `code/params.py` — every prior, tagged [S] (from a cited source) or [A] (assumption).
- `code/model.py` — vectorised three-channel model (financial / power / latency) with bounded and secular factors.
- `code/accounting.py` — growth accounting of the 4-5x/yr trend; OOM-budget runway.
- `code/run_mc.py` — Monte Carlo (N=20,000) for exogenous and revenue-limited variants; `--clat-low`, `--no-lat` sensitivity runs.
- `code/sobol.py` — Saltelli/Jansen variance-based global sensitivity, N=8192 base samples, bootstrap CIs.
- `code/revenue_stress.py` — conditional grid over AI revenue growth paths.
- `code/theory_check.py` — numerical checks of Propositions 1-2 and cap-exhaustion times.
- `code/predictions.py` — probabilities for the pre-registered predictions in Sec. 6.
All random seeds are fixed.
