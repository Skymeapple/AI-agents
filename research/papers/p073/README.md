# p073 — Diffusion-bounded leads and stake-driven racing in US–China AI competition

Paper: `paper.md`

## Reproduce (CPU, < 1 minute total)

```bash
cd /home/user/AI-agents/research/papers/p073
# data (public, MIT licensed JSON conversion of the official LMArena leaderboard pickles)
curl -sL -o data/scores.json https://raw.githubusercontent.com/nakasyou/lmarena-history/main/output/scores.json
python3 code/arena_gap.py   > results/arena_gap_stdout.txt     # empirical module
python3 code/race_model.py  > results/race_model_stdout.txt    # theory check, calibration, scenarios, figures
```

Dependencies: Python 3, numpy, scipy, matplotlib. Seeds are fixed (`arena_gap.py`: default_rng(0);
`race_model.py`: default_rng(20261122)).

The copy of `scores.json` used for the paper is in `data/` (117 snapshots, 2023-05-08 to 2025-05-22).
`data/fb/` is a snapshot (2025-08) of github.com/fboulnois/llm-leaderboard-csv, used only to cross-check
organization labels by hand.

## Outputs (results/)

| file | content |
|---|---|
| arena_gap_series.csv | per snapshot and category: top US model, top PRC model, Elo gap, implied P(US beats PRC) |
| arena_lag_series.csv | time lag (months) of best PRC model behind US on the fixed 2025-05 rating scale |
| arena_fit.json | partial-adjustment fits G' = mu - delta G (block bootstrap CIs), plateau trend test, frontier slope |
| contest_check.json | closed-form vs. numerical equilibrium of the handicapped Tullock contest |
| calibration.json | kappa (Elo per OOM effective compute), compute-only gap, bound psi_max on compute pass-through |
| scenarios_summary.json / scenarios.csv | Monte Carlo (200k draws) policy scenarios S0–S5 |
| prcc.json | partial rank correlation sensitivity |
| racing_history.json, analytic_quantities.json | numbers quoted in the text |
| fig_*.png | figures 1–5 of the paper |
