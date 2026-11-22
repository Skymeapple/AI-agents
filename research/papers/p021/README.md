# p021 — Reliability ceilings in agent time-horizon evaluations

Paper: `paper.md`. Code: `code/`. Outputs: `results/`.

Data: METR `eval-analysis-public`, report `time-horizon-1-1`, file `data/raw/runs.jsonl`
(commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2; 20 agents, 228 tasks, 23,235 agent runs).
`code/run_all.sh` clones it into `data/` if missing.

Reproduce: `bash code/run_all.sh` (Python 3, numpy, scipy, matplotlib; ~15-20 CPU-minutes).

| script | produces |
|---|---|
| a1_fits.py | results/fits.csv, fits_bootstrap.json, fig1_curves.png |
| a2_decomp.py | results/decomp_T{4,15,60}.csv, consensus_tasks.csv, fig2_decomp.png, fig3_passk.png |
| a3_trends.py | results/trends.json, paired_tests.csv, fig4_trends.png |
| a4_sim.py | results/sim_S1.csv, sim_S2.csv, sigma_calibration.json, fig5_sim.png |
| a5_cert.py | results/certification.csv |

Random seeds are fixed in each script.
