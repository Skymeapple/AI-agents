# p006: Revision Costs Logit Range (recency–retention bound for fixed-query softmax readout)

Paper: `paper.md`. Response to round-1 review: `reviews/response_r1.md`.
Every number in the paper comes from a file in `results/` that is written by the scripts below (no hand edits).

## Reproduce (CPU only; Python 3, numpy, scipy, matplotlib, torch)
```
cd code
python3 theory_sim.py      # E1a bound vs Prop-2 / Prop-2' constructions, E1c delta capacity -> results/theory_sim.json (~10 s)
python3 frontier_lp.py     # E1b exact recency-retention frontier (LP)                       -> results/frontier_lp.json (~40 s)
python3 make_fig1.py       # -> results/fig1_theory.png
# E3: trained single heads with known logit budget (each run ~4-9 min on one core; grid in run_e3.sh)
./run_e3.sh                # -> results/e3/*.json, results/models/e3_*.pt
python3 e3_prop4_check.py  # Prop-4 age-0 check on saved clamped models -> results/e3_prop4.json
python3 e3_lp_reference.py # best achievable worst-case weight for age-score heads (LP) -> results/e3_lp_reference.json
python3 analyze_e3.py      # -> results/e3_tables.md, results/e3_summary.json, results/fig3_single_head.png
# E2: submitted multi-layer models (runs were split across processes because the CPU was shared)
python3 train_toy.py --steps 4000 --seeds 0 1 2 --models tf1                                  # -> results/train_toy.json
python3 train_toy.py --steps 4000 --seeds 0 --models delta --threads 1 --out tt_delta0.json
python3 train_toy.py --steps 4000 --seeds 1 --models delta --threads 1 --out tt_delta1.json
python3 train_toy.py --steps 4000 --seeds 0 1 --models tf2 --threads 2 --out tt_tf.json
python3 train_toy.py --steps 4000 --seeds 0 --models tf4 --threads 2 --out tt_tf.json
python3 train_toy.py --steps 4000 --seeds 1 --models tf4 --threads 2 --out tt_tf4b.json
python3 train_toy.py --steps 4000 --seeds 0 1 --models tf1 --cover --threads 2 --out tt_tf1c.json     # r1 control (R-covering training; 2 seeds)
python3 analyze_toy.py     # submitted tables + fig2_trained.png
python3 analyze_toy_r1.py  # r1 re-analysis: pooled Wilson CIs, trend tests, tf1c control -> results/e2_r1_tables.md, e2_r1_stats.json
```
Seeds are fixed; reruns reproduce up to BLAS/thread nondeterminism. `results/train_toy.log` also holds output from aborted
earlier attempts; the final runs are those recorded in the JSON files. E3 and the tf1c control used a reduced grid because the
CPU was shared with other jobs (see paper, Limitations).
