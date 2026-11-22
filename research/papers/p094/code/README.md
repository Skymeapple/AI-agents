# Reproducing "Certifying a model is below a dangerous-capability threshold against a repeated-attempt adversary"

Requirements: Python 3, numpy, scipy, matplotlib (multiprocessing uses 4 workers).

    cd code
    python3 exp1_theory.py        # Thm 2/3 numerics: best-approximation error of x^B, population identified sets (~3 min)
    python3 exp2_budget.py        # Thm 1 evaluator-budget lower bound vs direct probe; Monte-Carlo TV check (<1 min)
    REPS=200 python3 exp3_sim.py  # simulation of 95% UCBs, 4 populations x n x T x B, 200 reps/cell (~10 min)
    python3 exp4_metr.py          # backtest on METR Time Horizon 1.1 public runs (needs ../ext/metr, see below)
    python3 make_figs.py          # fig3-fig5

Data for exp4: clone https://github.com/METR/eval-analysis-public into ../ext/metr
(file used: reports/time-horizon-1-1/data/raw/runs.jsonl).

All outputs are written to ../results/ (JSON + PNG). Seeds are fixed in the scripts.
`core.py` holds the estimators: Chen et al. pass@k, beta-binomial MLE, power-law
extrapolation, the partial-identification LP (`lp_bounds`) and Clopper-Pearson boxes.
