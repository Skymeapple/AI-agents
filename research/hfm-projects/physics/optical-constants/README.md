# Predicting Optical Constants Beyond the Measured Wavelength Band

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_optical_extrapolation` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** refractiveindex.info database on GitHub.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable method for predicting a material sample's refractive index n and extinction coefficient k outside the wavelength band supplied to the method. The method is evaluated by agreement with withheld measurements from the same sample. The representation, model family, and estimation procedure are chosen freely; no dispersion formula or fitting strategy is prescribed.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.7907
- **HFM effort:** 100 proposals, 95 scored, 23 accepted improvements, about $8,376 of compute
- **HFM's current best idea:** For semiconductor rational-pole selection on irregular or duplicated wavelength tables, weighting each held-out wavelength group by its observed log-wavelength Voronoi cell width while averaging duplicate rows within a group increases audited skill_mean relative to p4040 and the tied equal-group p4061 selector.

## Existing theory and methods

Kramers-Kronig relations, Drude-Lorentz and Tauc-Lorentz oscillator models, variational KK-constrained fitting (Kuzmenko), Sellmeier and Cauchy dispersion.

## Public data

refractiveindex.info database (CC0), 620 tabulated n and k records.

## What limits progress

Resonances outside the measured band leave little trace inside it.

## Route to a step change

Material-class priors over oscillators; causality alone is not enough.

## Experiment and results

Code: [`optical_extrapolation.py`](optical_extrapolation.py) in this folder. Run instructions are in its header.

- **Setup:** 620 tabulated (n, k) records from the public
  refractiveindex.info database. The middle of each record's wavelength
  range is treated as measured, and both ends are withheld. Settings are
  chosen on one half of the materials and results reported on the other
  half. Skill is 1 − error ÷ the error of holding the last measured value.
- **Method tested:** the Kramers–Kronig-consistent fit from Section 7.5,
  blended part-way toward the hold-last reference.
- **Result: a modest and fragile gain.**

  | Settings search | Skill on the tuning half | Mean skill, held-out | Median, held-out | Materials improved |
  | --- | --- | --- | --- | --- |
  | First search | +0.25 | **+0.11** | +0.18 | 77% |
  | Wider search | +0.31 | **+0.02** | +0.13 | 65% |

  - Naive linear extension scored −0.20, and the unblended fit scored
    about −0.06.
  - The wider search did better on the tuning half and *worse* on the
    held-out half. That is the practice-set overfitting described in
    Section 14.2, reproduced at small scale.
  - The held-out half was looked at twice, so even +0.11 is slightly
    optimistic.
- **What this shows:** causality alone helps the typical material but
  fails badly on a minority, mainly where a resonance lies outside the
  band and nothing inside the band reveals it. Route 2 in Section 7.5
  (a prior over oscillators by material class) is therefore the part
  that matters, and it is what HFM's agents have built by hand. Section
  7.5's ordering of the two routes should be read with that in mind.
  The result is not comparable to HFM's 0.791, which uses private samples
  and scoring.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Causality-only fitting gives +0.02 to +0.11 skill on held-out materials and is fragile. HFM's 0.79 uses material-class models, which is where the information is.

## Next steps

Build class priors (metals, semiconductors, dielectrics) from the database and re-test.
