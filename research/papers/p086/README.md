# p086 — Depreciation shape, investment growth, and the AI capex "bubble" question

Paper: `paper.md`. Code: `code/`. Outputs: `results/`.

Reproduce everything (Python 3, numpy, scipy, matplotlib; ~30 s CPU):

    bash code/run_all.sh

| Script | Produces |
|---|---|
| `code/vintage.py` | vintage-capital model (closed-form value, economic depreciation, straight-line accounting) |
| `code/data.py` | all sourced inputs (S) and modelling assumptions (A), with source notes |
| `code/theory_checks.py` | numerical verification of Lemma 1, Propositions 1–2, Corollary 1 → `results/theory_checks.json`, `phi_curves.csv` |
| `code/fleet.py` | fleet application to MSFT/GOOGL/AMZN/META capex, three scenarios → `fleet_*.csv`, `fleet_summary.json`, `calibration.json` |
| `code/montecarlo.py` | 2,000-draw sensitivity analysis (seed 20261122) → `montecarlo.csv`, `montecarlo_summary.json` |
| `code/extras.py` | break-even obsolescence g*, break-even life, heat-map, Proposition 3 → `extras.json`, `gstar_curve.csv`, `heatmap_gap_ratio_2026.csv`, `prop3.json` |
| `code/figures.py` | `fig1`–`fig5` PNGs |

Data provenance: capex figures are 10-K cash purchases of property and equipment as compiled by Capexwave; 2026 figures are company guidance (bases differ by firm). No proprietary data are used.
