# p062 — When does AI productivity show up in GDP? A visibility-ratio framework

Reproduce everything (CPU, < 1 minute total):

```bash
cd code
python3 verify_propositions.py   # numerical check of Propositions 1-3, figs 1-2
python3 calibrate.py             # Monte Carlo 2025-2035, decomposition, detectability, figs 3-5
```

Requires Python 3 with numpy, scipy, matplotlib. Random seeds are fixed in the scripts.

Inputs (`data/`): World Bank WDI downloads (fetched 2026-11-22 from api.worldbank.org):
`wb_us_gdp_growth.json` (NY.GDP.MKTP.KD.ZG, US, 1960-2025), `wb_us_gdp_nominal.json` (NY.GDP.MKTP.CD, 2015-2025).

Outputs (`results/`):
- `verify_summary.json`, `verify_props.csv`, `vhome_global_vs_firstorder.csv` — proposition checks
- `mc_summary.json`, `scenarios.csv`, `mc_draws_sample.csv`, `calibrate_stdout.txt` — calibration
- `fig1_vhome.png` … `fig5_detectability.png`

Code: `code/model.py` (nested-CES home-production economy, chained Fisher real GDP, equivalent variation, analytic visibility coefficients).
Paper: `paper.md`.
