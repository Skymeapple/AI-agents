# Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors

- **Field:** physics
- **HFM track:** `discovery_live_ai4x_physics_rb_sparse` (active on HFM)
- **Project status:** Blocked

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

The Well rayleigh_benard dataset is public.

## What limits progress

Linear maps from sparse points cannot recover small plumes; pressure is never measured.

## Route to a step change

A learned reconstruction network, with pressure computed from the Poisson equation.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. The Well dataset is hundreds of GB.

## How far this can go

256 sensors on a 512 by 128 grid cannot resolve scales below the sensor spacing; that energy fraction is the floor.

## Next steps

Compute the energy fraction below the sensor Nyquist scale per Rayleigh number.
