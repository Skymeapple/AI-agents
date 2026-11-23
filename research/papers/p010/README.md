# p010 — How Much Does the Definition Move the Forecast?

Code and outputs for `paper.md` (round-2 revision). The round-1 response is in `reviews/response_r1.md`.

## Data
`data/runs_1-0.jsonl`, `data/runs_1-1.jsonl` and `data/release_dates.yaml` are unchanged copies from the public METR repository
`https://github.com/METR/eval-analysis-public`, commit `52cb829c7a2efb2d659285c4b1768d191d97f8d2` (2026-03-05). That was the latest
commit on 2026-11-22. The source paths are `reports/time-horizon-1-{0,1}/data/raw/runs.jsonl` and `data/external/release_dates.yaml`.
SHA-256:
```
609f904f4b6ae32129388da89d036e00bac511ad94223d2be1e58fc2b45b55cd  runs_1-1.jsonl   (24,008 rows = 23,235 agent runs + 773 human baselines)
f489d11728909937dad99ba1ed2d7c40efe12a766891e67dbc7c2486a3311236  runs_1-0.jsonl   (41,629 rows = 40,836 agent runs + 793 human; 37,539 analysed, systems released 2023+)
317b92915df5bf935908567a857116bcf6f0c7686ef8b0840897ed04a29232bc  release_dates.yaml
```
CHC domain scores are hard-coded in `code/theory_chc.py` from Hendrycks et al. (2025), arXiv:2510.18212.

## Reproduce (CPU only; all paths are relative to the repository, so run from `code/`)
The full pipeline takes about 45 min on 3–4 cores. Set `OMP_NUM_THREADS=1` when running jobs in parallel.
```bash
cd code
# main grid + replication + shrinkage sensitivity (each ~3-4 min with the collapsed-likelihood fitter)
python3 analysis_metr.py --version 1-1 --B 500 --lam 1
python3 analysis_metr.py --version 1-0 --B 500 --lam 1
python3 analysis_metr.py --version 1-1 --B 500 --lam 0.25
python3 analysis_metr.py --version 1-1 --B 500 --lam 4
python3 cv_lambda.py --version 1-1 --reps 2
python3 profiles.py --version 1-1 --splits 10
python3 robust_dispersion.py --version 1-1     # round-1 model-resampling check (superseded by decompose.py, kept for reference)
python3 reliability_tax.py
python3 theory_chc.py
python3 make_figures.py
python3 robustness.py
# round-2 analyses
for v in 1-1 1-0; do
  python3 ext_bootstrap.py --version $v --mode point
  python3 ext_bootstrap.py --version $v --mode task --B 1000 --procs 3   # B=500 for 1-0
  python3 ext_bootstrap.py --version $v --mode time --B 1000 --procs 3   # B=500 for 1-0
  python3 decompose.py --version $v
done
python3 nested_share.py --version 1-1 --Bout 40 --Bin 100 --procs 2
python3 nested_share.py --version 1-0 --Bout 30 --Bin 80 --procs 3
python3 eb_lambda.py --version 1-1
python3 fig_decomposition.py
```
Dependencies: numpy, scipy, pandas, matplotlib, pyyaml.

## Outputs (results/)
- `metr_{v}_lam{λ}_summary.csv / _point.csv / _slopes.json / _boot.npz`: definition grid, per-model horizons, slope and dispersion trends
- `variance_decomposition.json`: round-1 decomposition (task bootstrap, all 24 definitions)
- `ext_{v}_{task,time,point}_lam1.pkl.gz`: family-level fits per bootstrap replicate (task resampling, plus human-time error)
- `decomp_{v}.json`, `decomp_{v}_table.csv`: variance shares by menu × uncertainty set; paired slope tests; dispersion trends under one bootstrap; family-rate heterogeneity; task-support diagnostics; window and Theil–Sen point variants
- `nested_share_{v}.json`: nested-bootstrap CIs for the definitional share
- `eb_lambda_1-1.json`: per-system empirical-Bayes λ and run-count-scaled ridge, with the resulting grids
- `family_rates_{v}.csv`: per-family relative growth rates
- `cv_lambda_1-1.json`, `profiles_1-1.json`, `robust_dispersion_1-1.json`, `reliability_tax.json`, `theory_chc.json`, `robustness.json`
- figures: `fig_trajectories.png`, `fig_arrival_forest.png`, `fig_dispersion_slope.png`, `fig_theory.png`, `fig_chc.png`, `fig_decomposition.png`
- `log_*.txt`: console logs
