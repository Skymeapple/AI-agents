# Agent for Verified Programming with Dafny

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_verified_programming` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Ready to run once a language-model key is connected.
- **Data access:** DafnyBench on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable agent harness that implements methods from Dafny specifications and problem descriptions. The configuration uses a frozen model and Dafny 4.11, with evaluation based on maximizing the fraction of methods that verify against supplied specifications within defined budgets. The harness architecture, internal representations, and research approach are chosen by the developer.

- **Metric:** `verified_rate` (maximize)
- **HFM baseline:** 0.3165
- **HFM best so far:** 0.8289
- **HFM effort:** 66 proposals, 47 scored, 9 accepted improvements, about $2,793 of compute
- **HFM's current best idea:** Ending a first-pass repair sequence after three consecutive eligible failures with the same primary proof-obligation signature and no improvement in error rank increases official verified_rate relative to p2090's fixed eight-attempt first pass.

## Existing theory and methods

Generate-verify-repair loops, invariant synthesis, lemma decomposition.

## Public data

DafnyBench and Dafny 4.11 are public.

## What limits progress

Remaining failures are proof obligations, not code.

## Route to a step change

Generate loop invariants and lemmas separately, verify each, then compose.

## Experiment and results

Code: [`dafny_floor.py`](dafny_floor.py) in this folder. It ran on free GitHub Actions compute (four CPU cores, 15 minutes); the log and per-program records are in [`research/compute/results/dafny-floor/`](../../../compute/results/dafny-floor/).

- **This is a floor with no language model.** HFM's task is to implement methods from specifications with a frozen model, which needs a paid API. The public stand-in is DafnyBench (782 programs with their invariants and assertions removed; restore them so the program verifies), checked with Dafny 4.11. The two tasks differ, so these numbers are not comparable with HFM's 0.3165 or 0.8289.

  | Step | Programs that verify | Share of 782 |
  | --- | --- | --- |
  | Unchanged, no hints added | 225 | 28.8% |
  | + guess loop bounds from the loop guard, delete what the verifier rejects | 232 | 29.7% |
  | + each postcondition as an invariant, with the loop bound replaced by the loop counter | **289** | **37.0%** |

- **Almost three in ten programs need no hints at all,** and nearly all of them have no loop: 209 of 371 programs without loops verify unchanged, against 16 of 411 with loops. Any method's score on this benchmark should be read against 28.8%, not zero.
- **Bounds are not what is missing.** Index-range invariants recover 7 programs. A textbook rule (take the postcondition and replace the loop bound by the loop counter) recovers 57 more. Together the two rules move loop programs from 16 to 80 of 411. No program logic is edited; the verifier decides which guesses survive.
- **A benchmark defect:** 60 hint-stripped files (7.7%) do not parse, because removing a hint left broken syntax. No method can verify those without editing code, so the ceiling for a strict no-edit method is 92.3%.
- **What it means for a model-based agent:** the cheap template step is worth 8 points on its own and costs seconds, so it belongs in front of any model call. The remaining 331 loop programs (some of which are among the files that do not parse) need invariants that are not a rewrite of the postcondition; that is where a language model has to earn its place.
- **Not done:** no language model was run, and HFM's own task (writing implementations) was not attempted.

## How far this can go

HFM is at 0.829. Published DafnyBench results with strong models are in the same range.

## Next steps

Classify failures on DafnyBench by obligation type.
