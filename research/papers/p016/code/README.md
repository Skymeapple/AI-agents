# p016 — Exact theory of the stability gap under replay (code)

Requirements: Python 3, numpy, scipy, matplotlib, torch (CPU), scikit-learn (for the bundled `digits` dataset; no download).

Run everything: `bash run_all.sh` (≈35 min on 4 cores). Individual pieces:

| script | what it produces |
|---|---|
| `linear_theory.py` | E1 (2-D closed form vs. numerics), E2 (d=100 random tasks, 10 seeds; commuting control; initial-curvature law; recovery times), discrete preconditioning study → `results/linear_results.json`, `fig1_linear2d.png`, `fig2_linear_highdim.png` |
| `linear_theory_lib.py` | helper functions shared with `linear_extras.py` |
| `linear_extras.py` | numerical check of Thm 4 lower bound and Thm 6(b) upper bound; noisy under-parameterised case → `results/linear_extras.json` |
| `mlp_experiments.py <scen> <seed>` | 64-16-10 tanh MLP on sklearn digits, squared loss; scenarios `perm` (domain-incremental) and `split` (class-incremental); GD / Adam / damped Gauss–Newton under replay weights ρ → `results/mlp_<scen>_<seed>.json` |
| `run_mlp.sh` | runs the 8 MLP jobs (seeds 0–3 × 2 scenarios) in parallel |
| `mlp_damping.py <scen> <seed>` | damping-scaling test (δ ∈ {1e-3,1e-4,1e-5}, ρ ∈ {0.01,0.1}) → `results/mlpdamp_*.json` |
| `analyze_mlp.py`, `analyze_damping.py` | aggregate tables and figures `fig3_mlp_curves.png`, `fig4_mlp_scaling.png` |

All randomness is seeded. Every number in `paper.md` comes from these outputs.
