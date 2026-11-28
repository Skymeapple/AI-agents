# Code for "The Rarity Tax" (p042)

Everything is CPU-only numpy/scipy/matplotlib (`pip install numpy scipy matplotlib`).

## Run

```bash
cd code
python3 run_experiments.py all      # 29 training jobs, 2 worker processes, about 45-60 min wall clock in total
                                    # (each job 0.5-5 min CPU); writes ../results/raw/*.npz, skips existing files
python3 analyze.py   > ../results/analysis_stdout.txt       # -> results/summary.json, fig1-fig3   (about 10 min)
python3 robustness.py                                       # -> results/robustness.json, fig4     (about 5 min)
python3 firstpassage.py > ../results/firstpassage_stdout.txt  # -> results/firstpassage.json, fig6 (seconds)
python3 extrapolate.py > ../results/extrapolation_stdout.txt  # -> results/extrapolation.json, fig5 (about 3 min)
```

All random seeds are fixed. Training is bit-for-bit reproducible on this machine with single-threaded BLAS
(the script sets `OMP_NUM_THREADS=1`); see `results/rerun_check.txt` for the comparison of a second run
against the first.

## Files

| file | purpose |
|---|---|
| `sae.py` | Zipfian superposition data generator; TopK sparse autoencoder with AuxK loss, manual gradients, Adam or momentum SGD; recovery metric (max cosine to a decoder row) and FVU |
| `run_experiments.py` | experiment grid E1-E5 (see paper Section 4) |
| `analyze_lib.py` | shared fitting utilities: logistic recovery model `P = q * sigmoid((log p + beta log T - c)/s)`, feature bootstrap |
| `analyze.py` | E1 collapse test and cross-seed prediction, E2 width limit, E3/E5 configuration table, E4 replay |
| `robustness.py` | sensitivity of beta to the recovery threshold and transient cut; Fig. 4 |
| `firstpassage.py` | model-free first-passage estimate of the rarity exponent kappa = 1/beta; Fig. 6 |
| `extrapolate.py` | analytic coverage model (paper Section 3 and 6): Tables A, B, B2, C and scaling exponents; Fig. 5 |
| `pilot2.py` | quick single-run pilot used to choose the grid (not used for any number in the paper) |
| `legacy_v1/` | first version of the experiments (different grid, wider dictionaries); superseded, not used in the paper |

Environment variables: `OUT` (run_experiments output dir), `RAW` (analysis input dir), `THR` (recovery
threshold, default 0.9), `TMIN` (transient cut in tokens, default 100000), `SUF` (output filename suffix).
