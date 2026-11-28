# Code for p030: delegation-surplus curve

Run from the paper directory (`research/papers/p030`):

```bash
bash code/fetch_data.sh          # clones METR/eval-analysis-public into data_cache/ (about 43 MB, git-ignored)
python3 code/sim_estimator.py    # seconds; synthetic checks -> results/sim_estimator.json
python3 code/usefulness.py > results/run_log.txt   # about 90 s on one core -> results/usefulness.json, *.csv, fig
```

Requirements: Python 3, numpy; matplotlib only for the optional figure.

Data: `data_cache/metr/reports/time-horizon-1-1/data/raw/runs.jsonl` at commit
`52cb829c7a2efb2d659285c4b1768d191d97f8d2` (public release accompanying arXiv:2503.14499).

Outputs:
- `results/usefulness.json` — everything (per-model accuracy, time-weighted accuracy, usefulness and
  stop-loss curves under four information structures and three value weightings; context grid;
  dominance tests; rho sweep; cluster bootstrap; time-weight concentration; consistency).
- `results/table_main.csv` — Table 1 for all 20 models with bootstrap intervals.
- `results/table_gamma.csv` — value-weighting sensitivity.
- `results/context_grid.csv` — rank agreement in each of the 72 contexts per information structure.
- `results/sim_estimator.json` — Proposition 1 identity check and estimator-bias simulation.

Seeds are fixed (`SEED` in `usefulness.py`, 7 in `sim_estimator.py`); reruns reproduce the numbers.
