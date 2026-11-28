# p016 — Exact theory of the stability gap under replay (code)

Requirements: Python 3, numpy, scipy, matplotlib, torch (CPU is enough), scikit-learn (for the bundled `digits` dataset; no download).

Run everything: `bash run_all.sh`. On 4 otherwise idle cores this takes roughly 1.5 h (the round-1 matched comparison accounts for about 1 h of it). Individual pieces:

| script | what it produces |
|---|---|
| `linear_theory.py` | E1 (2-D closed form on a time grid), E2 (d=100 random tasks, 10 seeds; commuting control; first-step law; recovery-to-10%-of-peak times), E3 discrete preconditioning study → `results/linear_results.json`, `fig1_linear2d.png`, `fig2_linear_highdim.png` |
| `linear_theory_lib.py` | helper functions shared by the linear scripts |
| `linear_extras.py` | numerical check of the Thm 4 lower bound and Thm 6(b) upper bound; noisy under-parameterised case (E4) → `results/linear_extras.json` |
| `linear_revision.py [parts]` | **R1.** Parts `A` (E1 with a real ODE integrator), `B` (E5: recovery time, Thm 7 bounds, Prop 8 constant), `C` (E6: SGD second-moment recursion + Monte-Carlo check, Thm 9), `D` (E7: finite buffer, Prop 10), `E` (audit of the prose numbers flagged in review R1), `FIG` (merge → `results/linear_revision.json`, `fig5_linear_revision.png`). Each part writes `results/linear_revision_<part>.json`. Run as `python3 linear_revision.py A,B,D,E`, then `C`, then `FIG`. |
| `linear_projection.py` | **R1.** Straight-path / projection baselines (OGD-style, GEM-style, exact GN) for Table 2 → `results/linear_projection.json` |
| `mlp_experiments.py <scen> <seed>` | main sweep: 64-16-10 tanh MLP on sklearn digits, squared loss; scenarios `perm` and `split`; GD / Adam / damped Gauss–Newton under replay weights ρ → `results/mlp_<scen>_<seed>.json` |
| `run_mlp.sh` | runs the 8 main-sweep jobs (seeds 0–3 × 2 scenarios) in parallel |
| `mlp_damping.py <scen> <seed>` | damping-scaling test (δ ∈ {1e-3,1e-4,1e-5}, ρ ∈ {0.01,0.1}) → `results/mlpdamp_*.json` |
| `analyze_mlp.py`, `analyze_damping.py` | aggregate tables and figures `fig3_mlp_curves.png`, `fig4_mlp_scaling.png` |
| `audit_mlp_r1.py` | **R1.** From the main-sweep JSONs: per-run half-recovery steps, paired relative gap change vs GD with 95% CIs, multi-step normalised short-time curves → `results/mlp_audit_r1.json` |
| `mlp_matched.py <scen> <seed>`, `run_matched.sh`, `analyze_matched.py` | **R1.** Progress-matched optimiser comparison (8 seeds requested; seeds 0–5 completed and are what the paper reports — the seed 6–7 jobs were killed by a machine restart), common-step multi-step short-time test, comparison with the network linearised at θ0 → `results/mlpm_<scen>_<seed>.json`, `mlpm_summary.json`, `mlpm_summary.txt`, `fig6_mlp_pareto.png`, `fig7_mlp_short_lin.png` |
| `exploratory_unfinished/` | left over from an interrupted session; **not used by the paper** (see the README inside) |

All randomness is seeded. Every number in `paper.md` comes from these outputs. `results/mlp_<scen>_<seed>.json` (main sweep) and `results/mlpdamp_*` were produced for the first submission and were not regenerated in round 1; phase 1 of the main sweep was re-run and reproduces the stored base losses to all printed digits.
