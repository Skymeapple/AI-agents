# Online Cloud Resource Allocator for Azure Trace Replay

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_cloud_placement` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Azure Packing Trace 2020 on GitHub.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops an online placement policy for an Azure trace replay. Each VM request is assigned without knowledge of its future lifetime, while respecting constraints on cores, memory, HDD, SSD, network, and machine compatibility. The allocation algorithm and internal representation are chosen to optimize weighted server-hours and specified failure and eviction costs.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.7615
- **HFM effort:** 56 proposals, 47 scored, 11 accepted improvements, about $1,625 of compute
- **HFM's current best idea:** Backing off p2186's busy-objective survival queries unless the selected curve has at least 50 observations at risk at the last queried age increases full-practice mean_instance_score over unchanged p2186.

## Existing theory and methods

Online vector bin packing (first fit, best fit), lifetime-aware packing (Protean, Lifetime Alignment), survival models for VM lifetimes.

## Public data

Azure Packing Trace 2020 (Microsoft, CC-BY 4.0), GitHub release 'dataset-packing-2020'.

## What limits progress

VM lifetimes are unknown at placement time.

## Route to a step change

Predict lifetimes from metadata and co-locate VMs that should leave together.

## Experiment and results

Code: [`cloud_placement.py`](cloud_placement.py) in this folder. Run instructions are in its header.

- **Setup:** Microsoft's public Azure Packing Trace 2020. Four disjoint
  tenant samples of about 100,000 VMs each are replayed over 14 days.
  Cost is server-days. There are no evictions, unlike HFM's objective.
- **Result:**

  | Policy | Server-days versus best fit |
  | --- | --- |
  | First fit | about −3% (better in all four samples) |
  | Lifetime-aware, lifetimes predicted from finished VMs | **+1.3%** (better in 1/4) |
  | Lifetime-aware, true lifetimes | **−13.8%** (better in 4/4) |

- **What this shows:** packing VMs that leave together is worth about
  14% of server time, and it gets within a few percent of the lower
  bound, **but only if lifetimes are known**. With a simple predictor
  (median lifetime of the same tenant and VM type) the gain disappears.
  The limit is lifetime prediction, not the packing rule. This agrees
  with what HFM's agents are doing, which is building survival-curve
  estimates.

### Follow-up: how accurate must the lifetime prediction be?

The lifetime-aware rule was fed true lifetimes multiplied by random noise
of increasing size, and a coarser rule (three lifetime classes kept on
separate servers) was tried.

| Policy | Server-days versus best fit | Samples better |
| --- | --- | --- |
| Lifetime-aware, true lifetimes | −13.8% | 4/4 |
| Lifetime-aware, lifetimes within about a factor of 1.6 (log error 0.5) | −12.2% | 4/4 |
| Lifetime-aware, within about a factor of 2.7 (log error 1.0) | −8.2% | 4/4 |
| Lifetime-aware, within about a factor of 7 (log error 2.0) | −3.0% | 4/4 |
| Lifetime-aware, median predictor (measured log error 2.25) | +1.3% | 1/4 |
| Lifetime classes, predicted | +6.1% | 0/4 |
| Lifetime classes, true | +0.7% | 2/4 |

- **The target is now concrete.** A predictor that is right within a
  factor of about 3 would save about 8% of server time. The simple median
  predictor is off by about a factor of 10 and saves nothing.
- **Coarse classes do not work,** even with true lifetimes. The gain
  comes from aligning exit times, not from separating short and long VMs.
- The noise test is slightly kinder than a real predictor, because its
  errors are unbiased and independent.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

With true lifetimes the rule saves 13.8% of server-days and lands within about 4% of the lower bound. That is the most this family of rules can deliver.

## Next steps

Build a lifetime model that re-predicts remaining lifetime as a VM survives, as in Google's LAVA (arXiv 2412.09840), and measure its log error against the factor-of-3 target. One-shot prediction is what failed here. Add evictions and failures to match HFM's objective.
