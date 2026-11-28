# p083 — Boundary-dependent carbon–water conflicts in carbon-aware AI scheduling

Reproduce (CPU only, ~4 minutes after download; Python 3, numpy, pandas, scipy, matplotlib):

```
bash code/fetch_data.sh        # public data: gridemissions/EIA-930 2023 + NOAA ISD-Lite 2023
python3 code/load_data.py      # -> results/panel_2023.pkl (13 sites x 8754 h)
python3 code/analysis.py       # -> shifting_summary.csv, frontiers.csv, accounting_bias.csv
python3 code/mc.py             # -> mc_draws.csv, mc_summary.csv (300 draws, ~2.5 min)
python3 code/figures.py        # -> fig_*.png
python3 code/summarize.py      # -> headline.json
```

Files
- `code/factors.py` emission and water factors with sources.
- `code/cooling.py` wet-bulb, PUE and two on-site WUE archetype models.
- `code/analysis.py` greedy LP scheduler (Prop. 3), frontiers, Prop. 1/2 checks.
- `results/tables.md` tables quoted in the paper; `results/headline.json` headline numbers.
- `paper.md` the paper.
