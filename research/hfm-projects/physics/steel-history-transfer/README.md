# Steel Cyclic Loading History Transfer

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_steel_cyclic_transfer_auto` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Hartloper et al. database on Zenodo (shared experiment).
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

A nonlinear response-history analysis of a steel frame uses a characterized material model from an acceptance tension coupon qualification campaign. The model defines stress and energy absorption as strain cycles through earthquake-imposed history.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.6677
- **HFM effort:** 100 proposals, 96 scored, 23 accepted improvements, about $5,586 of compute
- **HFM's current best idea:** 在 p3538 中以共享锚点切线归一的拉压初始屈服端点联合偏移替代专属 y0/k 坐标收缩，可提高独立 LP9/LP10 的 skill_mean。

## Existing theory and methods

Voce-Chaboche plasticity, updated Voce-Chaboche (RESSPyLab), parameter identification from cyclic coupon tests.

## Public data

Hartloper et al. coupon-test database, Zenodo 6965147 (CC-BY 4.0).

## What limits progress

Tension coupons do not identify kinematic hardening.

## Route to a step change

Borrow cyclic parameters from other steels; add one cheap load reversal per steel.

## Experiment and results

This track shares its experiment with [`steel-cyclic-from-tension`](../steel-cyclic-from-tension/README.md). Both HFM steel tracks use the same coupon tests and the same transfer question, so one experiment covers both.

## How far this can go

See the shared experiment: the model class reaches 0.087 error when fitted to the steel's own cyclic tests, against 0.145 from donors.

## Next steps

One reversal closes 8% of the gap and four close 30% (see the shared experiment). Find the shortest multi-cycle protocol that closes most of it.
