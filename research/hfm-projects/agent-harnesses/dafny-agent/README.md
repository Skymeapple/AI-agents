# Agent for Verified Programming with Dafny

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_verified_programming` (active on HFM)
- **Project status:** Blocked

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

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

HFM is at 0.829. Published DafnyBench results with strong models are in the same range.

## Next steps

Classify failures on DafnyBench by obligation type.
