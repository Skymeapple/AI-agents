# p010 — The Definition Is the Forecast

Code and outputs for the paper `paper.md`.

## Data
`data/runs_1-0.jsonl`, `data/runs_1-1.jsonl`, `data/release_dates.yaml` were downloaded unchanged from the public METR repository
`https://github.com/METR/eval-analysis-public` (paths `reports/time-horizon-1-{0,1}/data/raw/runs.jsonl`,
`data/external/release_dates.yaml`), on 2026-11-22. CHC domain scores are hard-coded in `code/theory_chc.py` from
Hendrycks et al. (2025), arXiv:2510.18212, Table "AGI score summary".

## Reproduce (CPU only; total ≈ 40 min, dominated by bootstraps)
```bash
cd code
python3 analysis_metr.py --version 1-1 --B 500 --lam 1     # main analysis (~10 min)
python3 analysis_metr.py --version 1-0 --B 300 --lam 1     # replication on TH1.0 (~20 min)
python3 analysis_metr.py --version 1-1 --B 200 --lam 0.25  # shrinkage sensitivity
python3 analysis_metr.py --version 1-1 --B 200 --lam 4
python3 cv_lambda.py --version 1-1 --reps 2                # choose lambda by task-level CV
python3 profiles.py --version 1-1 --splits 10              # reliability / structure of family profiles
python3 robust_dispersion.py --version 1-1                 # ceiling-artefact check of dispersion trend
python3 reliability_tax.py                                 # Prop. 3 closed form vs direct fits
python3 theory_chc.py                                      # Props. 1, 4 numerics + CHC re-analysis
python3 make_figures.py                                    # figures + variance decomposition
```
Dependencies: numpy, scipy, pandas, matplotlib, pyyaml.

## Outputs (results/)
- `metr_{v}_lam{λ}_summary.csv` — doubling time and extrapolated arrival dates (point, 5/50/95% bootstrap) per definition
- `metr_{v}_lam{λ}_point.csv` — per-model log2 horizons under every definition
- `metr_{v}_lam{λ}_slopes.json` — trends in logistic slope and in cross-family dispersion; p80/p50 doubling-time ratio
- `variance_decomposition.json` — between- vs within-definition variance of arrival dates
- `cv_lambda_1-1.json`, `profiles_1-1.json`, `robust_dispersion_1-1.json`, `reliability_tax.json`, `theory_chc.json`
- figures: `fig_trajectories.png`, `fig_arrival_forest.png`, `fig_dispersion_slope.png`, `fig_theory.png`, `fig_chc.png`
- `run_log_metr.txt` — console log of the bootstrap runs
