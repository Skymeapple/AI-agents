# p063 — Who gets the AI surplus? Distribution-free incidence bounds and pass-through thresholds

Reproduce everything (about 30 s on one CPU core; Python 3 + numpy, scipy, matplotlib):

    cd code
    python3 make_dfa_csv.py   # rebuilds data/dfa_equity_shares.csv from the FRED values transcribed in the script
    python3 run_all.py        # all experiments E1-E7 -> results/

Files
- `code/model.py` — Gini / concentration coefficient algebra, Theorem 1 bounds, synthetic population, gain channels, B/W decomposition.
- `code/run_all.py` — experiments E1 (baseline + simplex), E2 (bound validation), E3 (within vs between), E4 (Monte Carlo, 400 draws), E5 (Fed DFA history 1989Q3-2026Q2), E6 (dynamics), E7 (broad ownership).
- `data/dfa_equity_shares.csv` — Fed Distributional Financial Accounts equity shares by wealth group (FRED series WFRBST01122, WFRBSN09149, WFRBSN40176, WFRBSB50203; retrieved 2026-11-22; bottom-50% imputed as residual before 2025Q2, flagged).
- `results/summary.json` — every number quoted in the paper; `results/run_log.txt` — console log; `results/fig*.png` — figures.
- `paper.md` — the paper.

The population is synthetic (calibrated, not micro-data); see paper Section 4 and Limitations.
