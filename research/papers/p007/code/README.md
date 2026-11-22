# p007 code: capacity, data and the limits of unified multimodal models

All experiments are CPU-only (numpy / torch-CPU). Total runtime ~15-25 min on 4 cores.

```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 exp_linear.py      # Exp A-D: shared/private factor model, theory vs Monte Carlo (~20 s)
python3 exp_threshold.py   # Exp G: pooling shifts spectral detection thresholds (~1 min)
python3 exp_powerlaw.py    # Exp E: power-law spectra, scale-invariant capacity tax (~30 s)
python3 exp_crossover.py   # Exp E3: data-limited crossover capacity r_x(n) (~1 min)
python3 exp_gap.py         # Exp C2: alignment tolerance theta_c(n) (~3 min)
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS
python3 exp_mlp.py         # Exp F: ReLU networks, shared trunk vs specialists (~10-15 min)
python3 analyze_mlp.py     # summary + figure for Exp F
```

* `common.py` - task generator, df-model theory (`theory_losses`), Monte Carlo estimator (`sim_losses`).
* Outputs (JSON + PNG) are written to `../results/`.
* Seeds are fixed in each script; numbers in the paper are read from the JSON files.
