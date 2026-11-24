# Optical Coating Design Under Manufacturing Variation

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_robust_ar_coating` (paused on HFM)
- **Project status:** Tested on public data

## The challenge

This work develops a reusable design procedure for anti-reflection coatings. Given a catalog specification, substrate, measured optical constants, and a process-variation model, the procedure returns a layer stack that maximizes the fixed score under that model.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.2523
- **HFM effort:** 31 proposals, 29 scored, 25 accepted improvements, about $982 of compute
- **HFM's current best idea:** Fixed-topology thickness polishing on a fresh independent Sobol batch after p1464's complete refined-split trajectory increases full-practice skill_mean relative to unchanged p1464.

## Existing theory and methods

Needle optimisation, robust and stochastic optimisation, sensitivity analysis, Monte Carlo tolerance analysis.

## Public data

No external data needed. Transfer-matrix calculation with textbook indices.

## What limits progress

Designs are chosen on nominal reflectance and then polished.

## Route to a step change

Choose among candidates by their score under the variation model.

## Experiment and results

Code: [`robust_ar_coating.py`](robust_ar_coating.py) in this folder. Run instructions are in its header.

- **Setup:** standard thin-film reflectance calculation, six
  specifications (three substrates, two wavelength bands), a shared pool
  of candidate designs with 4, 6 or 8 layers, and thickness and index
  errors on every layer. Final designs are scored on 4,000 fresh
  variation samples.
- **Result:** against the usual practice of picking the design with the
  lowest nominal reflectance:
  - **Choosing among the same candidates by their score under
    variation** cut expected reflectance by **14.5% on average**. It cut
    it by 33–50% in two specifications and 5% in a third, and changed it
    by under 0.5% in the other three.
  - **Re-optimising thicknesses on the robust score added nothing more**
    (−13.8% on average). It cut the 90th-percentile reflectance by 18%.
- **What this shows:** the gain comes from picking a less sensitive
  design, usually one with fewer layers, not from polishing thicknesses.
  HFM's agents spend most of their effort on polishing. Section 7.6 is
  supported on the selection point.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Selection by robust score cut expected reflectance by 14.5%; polishing added nothing. Further gains need a wider candidate pool (global topology search), not more polishing.

## Next steps

Add needle-style layer insertion to widen the pool. Add oblique incidence and dispersion.
