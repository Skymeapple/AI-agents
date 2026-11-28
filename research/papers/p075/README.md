# p075 — Who can close the window? Substitution and the coordination gap in open-weight AI release

Paper: `paper.md` (target venue: AAAI/ACM Conference on AI, Ethics, and Society, AIES 2027).

## Reproduce (about 5 minutes CPU in total)

```bash
cd /home/user/AI-agents/research/papers/p075/code
python3 analysis.py 1000 > ../results/run_log.txt   # main analysis, 1000 Monte Carlo draws (~4 min)
python3 figures.py                                  # fig1..fig5 in results/
python3 toy_model.py                                # Proposition 3 simulation (~5 s)
```

Requirements: Python 3, numpy, pandas, matplotlib. Seed fixed (20261122).

## Data

`data/epoch_eci_2026-11-18/eci_scores.csv` (Epoch AI, Epoch Capabilities Index, CC BY 4.0) and the
2026-11-11 vintage for a robustness check. See `data/PROVENANCE.md` and `data/SHA256SUMS`.

## Outputs (`results/`)

| file | content |
|---|---|
| `results.json` | all scalar results (lag by window, frontier-holder shares, coalition and developer removals, coordination gap, release rule, Shapley totals, open records summary, robustness) |
| `removal_counterfactuals.csv` | per developer / coalition: frontier setback, vertical gap, counterfactual lag, threshold delays (90% intervals over score noise) |
| `shapley.csv` | Shapley values and leave-one-out marginals per open developer |
| `open_records.csv` | every open-frontier record since 2024 with advancement and replacement time |
| `monthly_lag.csv` | monthly mean horizontal lag with 90% band |
| `frontier_series.csv` | daily closed / open / counterfactual frontiers (point estimates) |
| `toy_model.json` | Proposition 3 simulation |
| `fig1..fig5*.png` | figures |
| `run_log.txt` | console output |
