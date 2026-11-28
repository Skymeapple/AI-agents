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

```
# --- round 3 (response to review_r2) ---
export OMP_NUM_THREADS=1
python3 exp_valrank.py         # oracle vs validation-selected vs fixed rank (~1 min)
python3 exp_headint.py         # direct test of Cor. 3 (~20 s)
python3 exp_hetero.py          # linear analogue of the ReLU teacher with Gaussian heads (~1 min)
MLP_THREADS=1 python3 exp_mlp_narrow.py   # ReLU control: k_p=0 vs k_p=6 at widths 2,4,6 (~10 min)
python3 analyze_r3.py          # E3 slope fragility, ReLU vs Cor. 2 formula
python3 exp_transformer_r3.py check       # must print "bit-identical to round-2 run: True"
for p in 0 1 2 3; do python3 exp_transformer_r3.py seeds $p 4 & done; wait      # seeds 2-4, 84 runs (longest run ~14 min under load)
for p in 0 1 2 3; do python3 exp_transformer_r3.py converge $p 4 & done; wait   # 9000-step, 3-LR control, 27 runs
python3 analyze_transformer_r3.py         # 5-seed summary -> transformer_summary_r3.json, figH_transformer_r3.png
```
Round-3 runs used torch 2.15.0 (CPU); `check` confirmed that this version reproduces a round-2 run exactly.
Do not start two copies of the same worker set at once: they would append duplicate lines to the JSONL file.

* `common.py` - task generator, df-model theory (`theory_losses`), Monte Carlo estimator (`sim_losses`).
