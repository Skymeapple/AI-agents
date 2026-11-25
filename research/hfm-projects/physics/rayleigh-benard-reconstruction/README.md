# Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors

- **Field:** physics
- **HFM track:** `discovery_live_ai4x_physics_rb_sparse` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** The Well rayleigh_benard dataset: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU is enough for gappy POD and a small network on a subset of files; a GPU for larger networks.
- **Still missing:** Nothing for the linear methods. A GPU for learned decoders.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Full-field reconstruction of two-dimensional Rayleigh–Bénard convection on a 512 × 128 grid uses 256 fixed point sensors measuring buoyancy and both velocity components, but not pressure, over the last eight frames, starting from gappy POD. Data are The Well rayleigh_benard simulations across 35 Rayleigh-number (10⁶–10¹⁰) and Prandtl-number (0.1–10) settings.

- **Metric:** `field_nrmse` (minimize)
- **HFM baseline:** 0.2918
- **HFM best so far:** 0.2342
- **HFM effort:** 27 proposals, 24 scored, 11 accepted improvements, about $3,714 of compute
- **HFM's current best idea:** Equalizing the four temporal feature-block kernels before the centered linear/quadratic pressure mixture reduces field_nrmse relative to p3971's feature-wise standardized kernel.

## Existing theory and methods

Gappy POD, Voronoi-tessellation CNNs, physics-informed reconstruction, pressure-Poisson solves.

## Public data

The Well, `polymathic-ai/rayleigh_benard` on Hugging Face (CC-BY 4.0). Test and valid files are 1.08 GB per flow setting (5 runs × 200 frames); train files are 8.4 GB (40 runs).

## What limits progress

A linear mode basis fitted on few runs cannot represent new turbulent states, and pressure is never measured.

## Route to a step change

A learned reconstruction network, with pressure computed from the Poisson equation.

## Experiment and results

Code: [`reconstruct.py`](reconstruct.py) in this folder. It ran on free GitHub Actions compute in about four minutes; the log is in [`research/compute/results/rb-reconstruction/`](../../../compute/results/rb-reconstruction/run.log).

- **Setup:** The Well's public Rayleigh–Bénard simulations (512 × 128 grid), three flow settings (Rayleigh number 10⁶, 10⁸, 10¹⁰ at Prandtl 1). For each setting one file of 5 runs is used for fitting and a different file of 5 runs for testing (865 test frames each). 256 fixed sensors report buoyancy and both velocity components over the last eight frames; all four fields, pressure included, must be reconstructed at the final frame. Score is RMSE of each field divided by its spread, averaged over the four fields.
- **Result (mean over the three settings):**

  | Method | Field error | Versus gappy POD | Settings better |
  | --- | --- | --- | --- |
  | Gappy POD on the final frame (HFM's starting point) | 0.381 | — | — |
  | Ridge regression from the eight-frame sensor history | 0.444 | +16.7% | 0/3 |
  | Kernel regression from the sensor history | 0.485 | +27.4% | 0/3 |
  | Gappy POD velocity, history-map buoyancy, pressure from the flow equation | **0.340** | **−10.8%** | 2/3 |
  | Best possible with these POD modes | 0.320 | −15.9% | 3/3 |

  | Pressure only | Error |
  | --- | --- |
  | Reconstructed directly (gappy POD) | 0.415 |
  | Computed from the pressure equation using the reconstructed flow | **0.265** |

- **With twice the fitting data** (10 runs, 300 modes, Rayleigh 10⁸): gappy POD improves from 0.407 to 0.352, the best possible with the modes drops to 0.237, and the combined method is again 9.6% better than gappy POD.
- **What this shows:**
  - **Computing pressure from the flow equation works.** Pressure is never measured, and solving for it from the reconstructed velocity and buoyancy cuts its error by 36%. This is route 2 of section 7.9 in the overview, supported.
  - **Regressing on the sensor history does not work for velocity.** Both maps are worse than gappy POD in every setting, although they are slightly better for buoyancy. Route 1 in its simple form is not supported.
  - **The mode basis is the limit, and data moves it.** Even a perfect fit to 150 modes leaves 0.32 error; doubling the fitting data lowers that to 0.24 at one setting.
- **Open issue:** the pressure solver reproduces analytic test flows to 0.01%, but on the dataset's own true fields it differs from the stored pressure by 0.21. The equation or wall condition used by the simulation differs in some detail from the one assumed here. Resolving that should improve the pressure result further.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Reached here:** 10.8% below gappy POD with no learning beyond a linear basis, on CPU.
- **Ceiling of a linear basis:** 0.32 with 5 fitting runs and 0.24 with 10. The full training split has 40 runs per setting, so more of this is available from data alone.
- **Hard floor:** 256 sensors on a 512 × 128 grid cannot resolve structures smaller than the sensor spacing. At Rayleigh 10¹⁰ the plumes are thin, which is why errors rise with Rayleigh number (0.25, 0.41, 0.49 for gappy POD).

## Next steps

- Find why the stored pressure differs from the pressure equation by 0.21 on true fields (the dataset's exact equations and non-dimensionalisation).
- Fit on the full 40-run training split, on a machine with more memory or with an incremental SVD.
- Replace the linear basis with a learned nonlinear decoder (needs a GPU), keeping the equation-based pressure.
- Choose sensor positions instead of drawing them at random, if the task allows.
