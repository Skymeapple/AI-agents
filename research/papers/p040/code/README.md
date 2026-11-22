# Reproducing "Who Writes the Constitution?"

All numbers in `paper.md` come from the scripts below. CPU only; total runtime about 15-25 minutes on 4 cores.

## Data (public)
```
cd ../data
git clone --depth 1 https://github.com/saffronh/ccai.git          # CCAI Polis export + cleaning notebook (Huang et al. 2024)
git clone --depth 1 https://github.com/compdemocracy/openData.git  # public Polis conversations (CC BY 4.0)
```

## Run
Set single-threaded BLAS/OpenMP (k-means thread oversubscription otherwise slows runs >10x):
```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
B=40 BS=25 python3 run_ccai.py        # -> ../results/ccai_results.json, ccai_design_audit.csv, ccai_boot_inclusion.npy, ...
python3 run_ccai_addendum.py          # -> ../results/ccai_addendum.json, ccai_design_audit_corrected_groups.csv
B=20 python3 run_opendata.py          # -> ../results/opendata_decomposition.csv
python3 make_figures.py               # -> ../results/fig*.png
```

## Files
- `constlib.py`: loaders, opinion-group clustering (two PCA conventions), agreement estimators, aggregation rules
  (min-group consensus, Polis-style group-aware consensus product, approval voting, seq-PAV, Method of Equal Shares,
  greedy Chamberlin-Courant, consensus-core + coverage hybrid), the designer multiverse, variance decomposition,
  swap distances, JR audit, Proposition 1 bound.
- `run_ccai.py`: the authorship audit on CCAI (agenda, pipeline integrity, designer multiverse, bootstrap sampling,
  sample-size curve, representation audit, Prop. 1 and Prop. 3 checks).
- `run_ccai_addendum.py`: audit against re-aligned groups, curve fits, exposure-selection correlations.
- `run_opendata.py`: replication of the designer-vs-sample comparison on public Polis conversations.
Seeds are fixed (bootstrap seed 20261122 for CCAI, 7 for openData; k-means `random_state` = bootstrap index).
