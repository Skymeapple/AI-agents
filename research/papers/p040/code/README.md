# Reproducing "The Public Writes a Shortlist"

All numbers in `paper.md` come from the scripts below. CPU only; total runtime about 1.5 h on 4 cores
(the openData designer x sample run and opendata_stats.py dominate).

## Data (public)
```
cd ../data
git clone --depth 1 https://github.com/saffronh/ccai.git          # CCAI Polis export + cleaning notebook (Huang et al. 2024)
git clone --depth 1 https://github.com/compdemocracy/openData.git  # public Polis conversations (CC BY 4.0)
```

## Run (in this order)
```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
B=40 BS=25 NSEED=50 python3 run_ccai.py   # agenda, integrity (deterministic PCA, 50 k-means seeds), multiverse, bootstrap
python3 run_ccai_addendum.py              # audit vs re-aligned groups, exposure-selection correlations
B=20 python3 run_opendata.py              # designer x sample on 17 openData conversations (saves inclusion arrays)
python3 opendata_stats.py                 # cross-conversation stats, Monte-Carlo CIs, sub-multiverse sensitivity
BB=1000 python3 run_certify.py            # certification audit (CCAI + openData), Theorem-1 counts, planning simulation
python3 certify_robust.py                 # exposure-law robustness, where designer disagreements fall
python3 opendata_ratio_exposure.py        # designer/sample ratio vs n and m
python3 run_exposure_rules.py             # exposure-normalised proportional rules
python3 run_routing_sim.py                # uniform vs rank-aware vs cutoff-proximity routing
python3 make_figures.py; python3 make_figures_cert.py
```

## Files
- `constlib.py`: loaders, opinion-group clustering, estimators, aggregation rules, designer multiverse, swap distances, JR audit.
- `certify.py`: participant-bootstrap rank-confidence sets (pairwise differences, marginal/simultaneous; Mogstad et al. 2024),
  bootstrap stability, threshold certification.
- Seeds: CCAI bootstrap 20261122; k-means seed fixed at 0 across bootstrap draws (decoupled from the draw index);
  openData 7; certification 11/12; planning 5; routing 2026.
