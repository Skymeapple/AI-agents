# p006 — Long Context Is Not Long-Term Memory (recency–retention bound)

Paper: `paper.md`. All numbers in the paper come from files in `results/`.

## Reproduce (CPU only; Python 3, numpy, scipy, matplotlib, torch)
```
cd code
python3 theory_sim.py      # E1: Thm-1 bound vs Prop-2 construction, frontier, delta-rule capacity -> results/theory_sim.json (~6 min)
python3 make_fig1.py       # -> results/fig1_theory.png
python3 train_toy.py --steps 4000 --seeds 0 1 2   # E2: trained 1/2/4-layer RoPE Transformers + delta-rule model -> results/train_toy.json, results/models/*.pt (~40-60 min on 4 cores)
python3 analyze_toy.py     # -> results/train_toy_summary.json, train_toy_table.md, train_toy_age_table.md, fig2_trained.png
```
Seeds are fixed (numpy default_rng / torch.manual_seed), so reruns reproduce up to BLAS nondeterminism.
