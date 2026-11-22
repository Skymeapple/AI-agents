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
