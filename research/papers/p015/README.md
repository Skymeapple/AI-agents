# p015 — Frailty, not fatigue: heterogeneity in METR time-horizon data

Code and outputs for `paper.md`.

* Data: METR public Time Horizon 1.1 (and 1.0) run logs, fetched at a pinned commit
  (`52cb829c7a2efb2d659285c4b1768d191d97f8d2` of github.com/METR/eval-analysis-public) by
  `code/fetch_data.sh` into `data/`. The data are METR's; this repository only re-analyses them.
* Run everything: `bash code/run_all.sh` (Python 3, numpy, scipy, pandas, matplotlib, pyyaml;
  about 12 minutes on 4 CPU cores).
* Outputs in `results/`:
  * `marginal_fits.csv`, `extrapolation.csv`, `marginal_summary.txt` — pooled survival families (Sec. 5.1)
  * `heterogeneity_index.csv`, `heterogeneity_summary.txt`, `fig_heterogeneity.png` — index H (Sec. 5.2)
  * `frailty_glmm.csv`, `frailty_summary.txt`, `passk_validation.csv`, `projections.csv` — GLMMs (Sec. 5.3, 5.5)
  * `kcond_with_se.csv`, `trend_and_sensitivity.txt`, `projection_sensitivity.csv`, `fig_main.png`, `fig_trend.png`
  * `robustness.csv`, `robustness_summary.txt` — robustness (Sec. 5.4)
  * `simulation_identification.csv`, `simulation_summary.txt`, `fig_identification.png` — simulation (Sec. 5.6)
  * `passk_heldout.csv`, `passk_heldout_summary.txt` — held-out retry test (Sec. 5.5)
