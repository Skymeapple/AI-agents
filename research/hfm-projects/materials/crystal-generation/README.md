# Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim

- **Field:** materials
- **HFM track:** `discovery_live_ai4x_materials_gen` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU stand-in for the selection step has run. Generation and MatterSim relaxation wait for a GPU.
- **Data access:** MatterGen weights: Verified reachable from the free compute on 2026-11-25. MatterSim weights are not on Hugging Face (401) and come from its GitHub release.
- **Compute:** A GPU for sampling and relaxation.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

A CPU stand-in for the selection half of the proposed route ("over-generate, pre-screen cheaply, spend the budget on the best"), on public DFT data. Script: [`screen_floor.py`](screen_floor.py); log in `research/compute/results/crystal-screen-floor/`. It does not generate structures, does not test novelty, and is not comparable to HFM's score.

- **Data:** JARVIS-DFT 3D, 2022 release (NIST, public): 62,807 binary and ternary entries in 20,902 chemical systems, each with DFT energy above the convex hull.
- **Setup:** each chemical system with at least 8 entries (1,468 systems) is treated as an unseen target; its entries stand for a pool of generated candidates. A gradient-boosted model trained only on other systems predicts energy above hull from composition and four structure numbers (space group, density, atoms per cell, volume per atom). Five folds over systems. A stricter "family" split also removes every training entry whose element set is a subset or superset of a test system.
- **Selection rules for a budget of k per system:** random; best predicted ("screen"); best predicted with one pick per formula until all formulas are used ("diverse"); true best ("oracle").

Share of picks that are on or near the hull (family split; k = 1, 2, 4 over 1,468 systems, k = 8 over the 304 systems with at least 16 entries):

| Stable means | k | Random | Screen | Diverse | Oracle |
|---|---|---|---|---|---|
| on the hull (≤ 0) | 1 | 0.228 | **0.391** | 0.391 | 0.864 |
| on the hull (≤ 0) | 4 | 0.227 | **0.330** | 0.315 | 0.560 |
| within 0.05 eV/atom | 1 | 0.467 | **0.704** | 0.704 | 0.929 |
| within 0.05 eV/atom | 4 | 0.471 | **0.644** | 0.573 | 0.818 |
| within 0.05 eV/atom | 8 | 0.406 | **0.537** | 0.458 | 0.782 |
| within 0.1 eV/atom | 4 | 0.574 | **0.754** | 0.680 | 0.877 |

- **A cheap screen raises the stable share of a fixed budget by 1.4 to 1.7 times.** For a single pick per system, on-hull picks rise from 23% to 39%; within 0.05 eV/atom, from 47% to 70%. That closes 26% and 51% of the gap between random and oracle choice.
- **The screen transfers to unseen chemical families.** Removing every related subsystem and supersystem from training costs at most 0.016 in stable share and raises the screen's error only from 0.119 to 0.126 eV/atom.
- **Formula diversity buys unique compositions at the cost of stable share.** Within 0.05 eV/atom at k = 4, the diverse rule finds 2.22 distinct stable formulas per system against 1.84 for the plain screen (+21%), while its stable share falls from 0.644 to 0.573. Which wins depends on whether HFM's uniqueness counts formulas or structures. HFM's current best idea (trying unused formulas) is the diverse rule.
- **Limits:** the candidate pools are known, already relaxed structures, so their stable share (23% on the hull) is far above that of raw generator output. The screen uses relaxed structure features, which a generator's unrelaxed candidates only approximate. The value of screening on real generator output is likely larger, because the pool is worse, but this was not measured.

## How far this can go

HFM is at 0.44. The ceiling is set by how many of the unseen systems have any undiscovered stable phase, which is unknown; it is also exposed to errors in MatterSim itself.

## Next steps

Check a sample of 'stable' outputs against DFT to see whether the score rewards potential errors. With a GPU: over-generate 4–8 times the budget with MatterGen, screen with a composition-and-structure model of the kind tested here (or MatterSim single-point energies), and compare the stable fraction with HFM's 0.44.
