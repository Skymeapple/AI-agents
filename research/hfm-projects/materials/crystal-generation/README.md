# Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim

- **Field:** materials
- **HFM track:** `discovery_live_ai4x_materials_gen` (active on HFM)
- **Project status:** Blocked

## The challenge

Crystal generation targets thermodynamically stable materials in chemical systems unseen by the starting model, each containing two or three elements (e.g., Cd-Fe-P). Up to 64 structures per system are allowed; a structure counts only when stable, novel, and unique after relaxation by the frozen MatterSim potential.

- **Metric:** `unique_novel_stable_fraction` (maximize)
- **HFM baseline:** 0.1364
- **HFM best so far:** 0.4395
- **HFM effort:** 23 proposals, 20 scored, 11 accepted improvements, about $1,521 of compute
- **HFM's current best idea:** 在完整保留 p4034 的稳定、新颖、唯一结构前缀后，对同一原始父体按确定性哈希顺序尝试第七和第八个尚未使用的 reduced_formula，会提高独立完整评估的 unique_novel_stable_fraction。

## Existing theory and methods

Diffusion generators, prototype substitution, universal interatomic potentials, convex-hull stability screening.

## Public data

MatterGen and MatterSim code and weights are public. Reference hulls come from the Materials Project and Alexandria.

## What limits progress

The 64-structure budget is spent on generation and not on selection.

## Route to a step change

Over-generate, pre-screen with cheap single-point energies and duplicate filters, then spend the budget on the best.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. MatterGen sampling and MatterSim relaxation need one.

## How far this can go

HFM is at 0.44. The ceiling is set by how many of the unseen systems have any undiscovered stable phase, which is unknown; it is also exposed to errors in MatterSim itself.

## Next steps

Check a sample of 'stable' outputs against DFT to see whether the score rewards potential errors.
