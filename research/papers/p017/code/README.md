# Code for "Decodable Is Not Modelled: Suffix Ceilings for World-Model Probes"

Requirements: Python 3, numpy, matplotlib, torch (CPU build is enough).

```
cd code
# full run (3 worlds x 3 seeds; transformers on seed 0 only). ~1-1.5 h on 4 CPU cores
for m in unique balanced2 landmark; do NTHREADS=1 python3 run_experiment.py --modes $m --seeds 0 1 2 & done; wait
for a in "landmark 0" "landmark 1" "landmark 2" "balanced2 0"; do set -- $a; NTHREADS=1 python3 training_dynamics.py --mode $1 --seed $2 --epochs 40 & done; wait
python3 aggregate.py          # writes results/summary.md, summary.json, bound_tightness.json, cra_vs_probe.json, fig1-5
SMOKE=1 python3 run_experiment.py --modes balanced2 --seeds 0   # 15-second smoke test (writes to ../smoke)
```

Files
- `world.py`   torus gridworld (5x5), three colourings, exact k-suffix Bayes filter `suffix_bayes`.
- `models.py`  GRU LM, echo-state network with spectral norm λ (provable λ-contraction), small causal transformer, training loop, logistic-regression probe with standardisation folded into the weights.
- `run_experiment.py`  per (world, seed): Acc*_k ceiling, next-observation accuracy, linear state probe, k_eff,
  truncation test with data-dependent and a-priori bounds (Theorem 1), counterfactual rollout interventions.
- `aggregate.py`  tables and figures.

All randomness is seeded (world colouring: seed+1000; data: numpy default_rng(seed); weights: torch seeds).

Note: the progress line in results/log_*.txt is labelled `k=1,4,8,16,40` but prints Acc*_k at k = 1, 4, 8, 16, 24 (T = 24); the label is a leftover from an earlier configuration.
- `training_dynamics.py`  GRU trained 40 epochs (constant lr 3e-3), checkpoints at epochs 1,2,5,10,20,30,40.

## Round 2 (revision) — additional runs

```
python3 recompute_exact.py                       # exact finite-time ceilings for all round-1 runs -> results/exact_ceiling.json
for m in balanced2 landmark; do for e in 10 30; do NTHREADS=1 python3 gru_seeds.py --mode $m --epochs $e --seeds 0 1 2 3 4 5 6 7 8 9; done; done   # 40 GRUs -> results/seeds/
for m in balanced2 landmark; do for s in 0 1 2; do python3 reservoir_sweep.py --mode $m --seed $s; done; done   # results/sweep/
for m in balanced2 landmark; do python3 tf_window.py --mode $m --seeds 0 1 2; done                               # results/tf/
python3 calibrate_window_decoder.py              # learned window decoders vs exact ceiling -> results/calib_window.json
python3 aggregate_r2.py                          # results/summary_r2.md, fig6
# Othello (code/othello/)
for i in 1 2 3 4; do python3 gen_data.py lm$i 40000 10$i 0; python3 gen_data.py lab$i 10000 20$i 1; done
./run_train.sh 2                                 # 6000 steps -> results/othello/gpt.pt (copied to gpt_6k.pt)
./run_train_long.sh                              # continuation, 4000 steps -> gpt_long.pt
CKPT=gpt_6k.pt AOUT=analysis_6k.json python3 analyze.py ; CKPT=gpt_long.pt AOUT=analysis_long.json python3 analyze.py
CKPT=gpt_6k.pt python3 mlp_probe.py ; CKPT=gpt_long.pt python3 mlp_probe.py
python3 window_decoders.py ; python3 placement_decoders.py 0 ; python3 placement_decoders.py 8
python3 aggregate_othello.py                     # results/summary_othello.md, fig7
```
Notes. `results/othello/data_*.npz` (generated games) are deterministic from their seeds and are not committed.
The reservoir sweep replaced an earlier configuration list after we found that a Gaussian matrix scaled to
spectral norm 0.99 has spectral radius ~0.5 (see paper Sec. 5.1); the first, partial sweep was discarded.
`placement_decoders.py 0` was run first (output renamed placement_k0.json); the k=8 run writes placement.json.
