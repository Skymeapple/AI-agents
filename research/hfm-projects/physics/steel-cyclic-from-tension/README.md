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

### Follow-up: which single extra test is worth most?

For each steel, one of its cyclic tests was added in full to the tension test, and the model was scored on the steel's other cyclic tests. Results are grouped by loading protocol.

| Protocol added | Steels | Peak strain | Reversals | Error (donor median → with the test) | Gap to oracle closed | Steels improved |
| --- | --- | --- | --- | --- | --- | --- |
| LP5 (constant amplitude) | 39 | 2.0% | 101 | 0.143 → 0.104 | **71%** | 92% |
| LP10 | 27 | 12.3% | 28 | 0.137 → 0.100 | 69% | 81% |
| LP9 (random history) | 44 | 3.0% | 17 | 0.143 → 0.105 | 66% | 77% |
| LP6 | 38 | 4.8% | 25 | 0.141 → 0.108 | 62% | 84% |
| LP7 | 30 | 5.0% | 12 | 0.140 → 0.108 | 61% | 83% |
| LP8 | 33 | 6.0% | 8 | 0.143 → 0.115 | 50% | 64% |
| LP4 (constant amplitude) | 27 | 1.0% | 101 | 0.146 → 0.123 | 38% | 48% |
| LP3 | 28 | 12.3% | 4 | 0.133 → 0.125 | 16% | 50% |
| LP2 | 30 | 12.3% | 4 | 0.143 → 0.136 | 13% | 57% |

- **One well-chosen cyclic test removes most of the remaining error.** A constant-amplitude test at ±2% strain cuts the error by 27% and closes 71% of the gap, improving 92% of steels.
- **The choice of test matters more than its length.** A random history with only 17 reversals closes 66%. A few very large reversals (LP2, LP3) close 13–16%, and a small-amplitude test (LP4) closes 38%. Moderate amplitude with repeated cycling is what identifies the hardening parameters.
- **This is the route past the limit for both steel tracks.** It changes the measurement, not the model. Under HFM's rule that only a tension test is available, the donor median remains the practical ceiling.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Oracle error is 0.087 of yield stress against 0.145 for the donor median: 40% headroom that a tension test cannot reach. Below 0.087 needs a richer constitutive model.

## Next steps

Find the shortest version of a ±2% constant-amplitude test that keeps most of its benefit (the full test closes 71% of the gap; the first four reversals of a mixed test close 30%). Try a richer model (yield plateau, strain-range memory) to lower the oracle error itself.
