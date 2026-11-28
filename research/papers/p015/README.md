# p015 — Replicates buy hazard, not length (METR time-horizon replicate runs)

Code and outputs for `paper.md`.

* Data: METR public Time Horizon 1.1 (and 1.0) run logs, fetched at pinned commit
  `52cb829c7a2efb2d659285c4b1768d191d97f8d2` of github.com/METR/eval-analysis-public by `code/fetch_data.sh`.
  The data are METR's; this repository only re-analyses them.
* Run everything: `bash code/run_all.sh` (Python 3, numpy, scipy, pandas, matplotlib, pyyaml, torch, cvxpy+clarabel;
  about 3 hours on 4 CPU cores).
* Main new outputs (round 2):
  * `duality_summary.txt`, `duality_npmle_k.csv`, `duality_moment_k.csv`, `duality_forecast.csv`, `duality_bounds.csv` — NPMLE-frailty elasticity, overidentification, short->long forecasts, LR bounds (Sec. 5.3–5.5)
  * `forecast_bootstrap.txt`, `forecast_task_preds.csv`, `forecast_subsets.txt` — task-level uncertainty and audit sensitivity of the forecast test (Sec. 5.4)
  * `hazard_map_summary.txt`, `hazard_map.csv`, `within_family.csv` — piecewise hazard-length map, within-family elasticity (Sec. 5.3–5.4)
  * `npmle_sim_summary.txt`, `misspec_summary.txt` — simulations (Sec. 5.6)
  * `crossed_summary.txt`, `crossed_agents.csv`, `task_audit.csv` — crossed agent x task model and task audit (Sec. 5.1, 5.3)
  * `passk_baselines_summary.txt`, `passk_H_brackets.csv` — retry baselines and Prop. 3 brackets (Sec. 5.2)
  * `fig_duality.png`
* Round-1 outputs (pooled families, H, Gaussian GLMMs, robustness) are retained and still referenced.
