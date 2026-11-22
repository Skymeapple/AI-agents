# p082 — Pricing compute flexibility in clean-power supply for islanded AI data centers

Reproduce (CPU only, ~1.5 h wall-clock on 4 cores; network access to NREL's public AWS buckets is needed for step 1):

```bash
pip install pvlib s3fs h5py pandas pyarrow scipy matplotlib
python3 code/fetch_resource.py 2010 2011 2012 2013   # NSRDB v3.0.6 + WIND Toolkit v1.0.0 point extracts -> data/*.npz (~15 min)
python3 code/capacity_factors.py                     # hourly PV (single-axis) and wind capacity factors -> data/cf_*.npz
# NREL ATB 2024 v3 parquet (cost inputs; values are hard-coded in code/model.py::params with provenance comments):
python3 -c "import s3fs; s3fs.S3FileSystem(anon=True).get('oedi-data-lake/ATB/electricity/parquet/2024/v3.0.0/ATBe.parquet','data/ATBe_2024_v3.parquet')"
python3 code/compute_value.py                        # value-of-lost-compute range -> results/volc_summary.json
python3 code/experiments.py E1 E5 E2 E6 E4 E7 E8     # LP sweeps -> results/E*.jsonl (E1 ~17 min, others 1-4 min each on 4 cores)
python3 code/weather_years.py                        # weather-year out-of-sample test -> results/E3.json
python3 code/fig_dispatch.py                         # Figure 6 dispatch illustration
python3 code/analysis.py                             # tables, figures, proposition checks, break-even brackets -> results/summary.json, fig*.png, table_*.md
```

Files
- `code/model.py` — hourly LP (HiGHS via scipy) with endogenous, VoLC-priced compute curtailment.
- `code/experiments.py` — E1 main sweep (3 sites x 2 mixes x 7 CFE targets x 5 VoLC values), E2 SMR capex sweep, E4 hourly-vs-2h validation, E5 sensitivities, E6 exogenous curtailment caps, E7 fine VoLC grid for break-even brackets, E8 lumpy (module-outage) firm availability.
- `code/weather_years.py` — E3 design-year vs. evaluation-year test.
- `results/` — all raw outputs and figures used in `paper.md`.
- `paper.md` — the paper (target venue: Joule, Analysis). 431 LP instances in total.
