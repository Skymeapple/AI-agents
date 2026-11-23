# p007 code: capacity, data and the limits of unified multimodal models

All experiments are CPU-only (numpy / torch-CPU). Seeds are fixed; numbers in the paper are read from the JSON files in `../results/`.

```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# --- linear shared-core model (round 1, unchanged) ---
python3 exp_linear.py          # Exp A-D (~20 s)
python3 exp_powerlaw.py        # Exp E1-E2 (~30 s)
python3 exp_gap.py             # alignment tolerance theta_c(n) (~3 min)
python3 exp_threshold.py       # round-1 threshold study (superseded by exp_threshold_fine.py)
# --- round 2 ---
python3 exp_crossover.py       # Exp E3, 10 reps, capped rows filtered, bootstrap CI (~5 min)
python3 exp_random_design.py   # random Gaussian designs, ALS for the unified model (~10 min)
python3 exp_threshold_fine.py  # fine-grid threshold shifts vs spiked-matrix asymptotics (~15 s)
TF_SCALE=4 python3 exp_threshold_fine.py   # finite-size check at d=256 (~1 min)
python3 analyze_r2.py          # error table, Exp C boundary, ratio definitions, Cor. 2 vs MLP conventions
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS
python3 exp_mlp.py && python3 analyze_mlp.py     # round-1 ReLU study (d=32, 3 seeds)
python3 exp_mlp2.py 1024 & python3 exp_mlp2.py 4096   # round-2 ReLU study (d=64, 4 seeds, 2 LRs)
python3 analyze_mlp2.py
# --- small transformers on real token streams ---
# downloads into ../data_cache (not redistributed):
#   MNIST train-images-idx3-ubyte.gz (ossci-datasets.s3.amazonaws.com/mnist/)
#   Fashion-MNIST train images -> fashion-train-images-idx3-ubyte.gz
#   Tiny Shakespeare -> shakespeare.txt (karpathy/char-rnn)
#   FSDD recordings -> fsdd/*.wav (Jakobovski/free-spoken-digit-dataset, recordings/)
python3 prep_tokens.py
for p in 0 1 2; do python3 exp_transformer.py all $p 3 & done; wait   # (~1-2 h on a shared 4-core machine)
python3 analyze_transformer.py
```

* `common.py` - task generator, df-model theory (`theory_losses`), Monte Carlo estimator (`sim_losses`).
