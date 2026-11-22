# p006: Long Context Is Not Long-Term Memory (recency–retention bound)

Paper: `paper.md`. Every number in the paper comes from a file in `results/`.

## Reproduce (CPU only; Python 3, numpy, scipy, matplotlib, torch)
```
cd code
python3 theory_sim.py      # E1: Thm-1 bound vs Prop-2 construction, frontier, delta-rule capacity -> results/theory_sim.json (~6 min)
python3 make_fig1.py       # -> results/fig1_theory.png
# E2: trained models (as run for the paper; runs were split across processes because the CPU was shared)
python3 train_toy.py --steps 4000 --seeds 0 1 2 --models tf1                                  # -> results/train_toy.json
python3 train_toy.py --steps 4000 --seeds 0 --models delta --threads 1 --out tt_delta0.json
python3 train_toy.py --steps 4000 --seeds 1 --models delta --threads 1 --out tt_delta1.json
python3 train_toy.py --steps 4000 --seeds 0 1 --models tf2 --threads 2 --out tt_tf.json      # the paper run also produced tf4_s0 into tt_tf.json
python3 train_toy.py --steps 4000 --seeds 0 --models tf4 --threads 2 --out tt_tf.json
python3 train_toy.py --steps 4000 --seeds 1 --models tf4 --threads 2 --out tt_tf4b.json
python3 analyze_toy.py     # merges the JSONs -> results/train_toy_summary.json, *_table.md, fig2_trained.png
```
Seeds are fixed, so reruns reproduce the results up to BLAS nondeterminism. Trained weights are in `results/models/`. Note that `results/train_toy.log` also holds output from aborted earlier attempts; the final runs are the ones recorded in the JSON files.
