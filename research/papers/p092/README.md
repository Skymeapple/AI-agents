# p092 — Delegation Value: pricing verification into AI evaluation

Reproduce (CPU, ~1 minute total):

```bash
cd /home/user/AI-agents/research/papers/p092
git clone https://github.com/METR/eval-analysis-public data/eval-analysis-public   # data (commit 52cb829, Mar 2026)
pip install numpy scipy matplotlib pyyaml
python3 code/analysis.py --boot 500 > results/analysis_stdout.txt   # main analysis, bootstrap, figures
python3 code/extra.py > results/extra_stdout.txt                     # pair disagreements, trends, sensitivity summary
```

Files
- `code/dv.py` — the Delegation-Value model (net saving, logistic fit, horizons, stop-loss transform).
- `code/analysis.py` — loads METR Time-Horizon 1.1 per-run data (20 models, 228 tasks, 23,235 runs),
  computes pass-rate metrics, DVR under blind / length-informed (cross-fit) / family-experience / oracle
  delegation, rho sweep, sensitivity grid, family-clustered bootstrap, retry analysis, figures.
- `code/extra.py` — secondary statistics written to `results/extra.json`.
- `results/main_table.csv`, `results/retry.csv`, `results/sensitivity.csv`, `results/results.json`, `results/extra.json`, `results/fig_*.png`.

Data source: METR, eval-analysis-public, `reports/time-horizon-1-1/data/raw/runs.jsonl` (public).
All parameters that are assumptions (rho, r0, s, delta, token price) are swept in the sensitivity grid.
