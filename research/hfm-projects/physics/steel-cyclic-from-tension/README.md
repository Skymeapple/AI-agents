# Predict Cyclic Steel Stress Response from Limited Material Tests

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_steel_cyclic_transfer` (paused on HFM)
- **Project status:** Tested on public data

## The challenge

This work develops a reusable model that predicts a steel coupon's stress response to a specified strain history using only an acceptance tension test for the target steel, supplemented by a reference library of tests on other steels. The model and inference procedure are chosen to maximize a fixed evaluation score, without prescribing a specific constitutive law, fitting method, or transfer rule.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.5453
- **HFM effort:** 71 proposals, 67 scored, 24 accepted improvements, about $2,147 of compute
- **HFM's current best idea:** Extending the fast/slow Armstrong-Frederick linear-association contrast below p1709's selected boundary delta=-1 at fixed curvature q=4 increases campaign-held-out practice skill_mean while conserving component saturation, initial plastic modulus and aggregate yield-modulus covariance.

## Existing theory and methods

Voce-Chaboche plasticity, hierarchical Bayesian parameter transfer, donor averaging.

## Public data

Hartloper et al. coupon-test database, Zenodo 6965147 (CC-BY 4.0): 374 tests, protocols LP1 to LP10.

## What limits progress

A monotonic tension test barely constrains cyclic hardening.

## Route to a step change

A conditioned prior was tested and did not help reliably. The remaining gain needs an extra measurement.

## Experiment and results

Code: [`steel_cyclic_transfer.py`](steel_cyclic_transfer.py) in this folder. Run instructions are in its header.

- **Setup:** the public Hartloper et al. coupon-test database (Zenodo
  6965147), which uses the LP1–LP10 loading protocols named in HFM's logs
  and is very likely the data behind both steel tracks. 38 steels from 21
  test campaigns. For each target steel only its tension test is known;
  steels from other campaigns are the donor library. Error is the RMS
  stress error on the target's cyclic tests as a share of yield stress.
- **Result:**

  | Method | Error | Versus donor median | Campaigns better than donor median |
  | --- | --- | --- | --- |
  | Tension test alone (reference) | 0.545 | +275% | 1/21 |
  | Donor median | 0.145 | — | — |
  | Donor median, refined on the tension test | 0.146 | +0.6% | 8/21 |
  | Prior conditioned on tension-test features | 0.169 | +16% | 7/21 |
  | Conditioned prior, refined on the tension test | 0.138 | −5.2% | 10/21 |
  | Oracle (fitted to the target's own cyclic tests) | 0.087 | −40% | 21/21 |

- **What this shows:**
  - **The donor library does almost all the work.** Borrowing the median
    cyclic parameters of other steels cuts the error by 73% against using
    the tension test alone.
  - **The conditioned prior proposed in Section 7.3 is not supported.**
    Its best variant is 5% better on average but wins in only 10 of 21
    campaigns, which is not distinguishable from chance.
  - **40% of the error is still there to remove,** and a tension test
    does not carry the information to remove it. The step change would
    come from one cheap extra measurement per steel (for example a
    single load reversal), not from a better prior.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Oracle error is 0.087 of yield stress against 0.145 for the donor median: 40% headroom that a tension test cannot reach. Below 0.087 needs a richer constitutive model.

## Next steps

Simulate adding one reversal cycle to the target's data and measure the error. Try a richer model (yield plateau, strain-range memory) to lower the oracle.
