# Predict Cyclic Steel Stress Response from Limited Material Tests

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_steel_cyclic_transfer` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Hartloper et al. database on Zenodo.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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
    come from an extra measurement per steel, not from a better prior.
    The follow-up below shows it has to be several load cycles.

### Follow-up: does an extra measurement close the gap?

The first cyclic test of each steel donates its first few load reversals to
the fit. All methods are scored on that steel's other cyclic tests.

| Data on the target steel | Error | Versus donor median | Share of oracle gap closed | Campaigns improved |
| --- | --- | --- | --- | --- |
| Donor median only | 0.147 | — | 0% | — |
| Tension test + 1 reversal | 0.142 | −3.2% | 8% | 10/21 |
| Tension test + 2 reversals | 0.137 | −6.8% | 16% | 11/21 |
| Tension test + 4 reversals | 0.128 | −12.6% | 30% | 14/21 |
| Oracle (all own cyclic tests) | 0.086 | −41.3% | 100% | 21/21 |

- **One extra reversal is not enough.** It closes 8% of the gap. Four
  reversals close 30%.
- **Cyclic hardening shows itself over many cycles,** so the earlier
  suggestion of "one cheap load reversal" was too optimistic. A short
  multi-cycle test is the smallest measurement that helps materially.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Oracle error is 0.087 of yield stress against 0.145 for the donor median: 40% headroom that a tension test cannot reach. Below 0.087 needs a richer constitutive model.

## Next steps

Find the shortest multi-cycle protocol that closes most of the gap (four reversals close 30%). Try a richer model (yield plateau, strain-range memory) to lower the oracle error itself.
