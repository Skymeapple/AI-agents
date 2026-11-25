# Agent for Formal Theorem Proving in Lean

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_formal_proofs` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a language-model key is connected.
- **Data access:** miniF2F on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Build a reusable harness that produces accepted Lean proofs for competition-mathematics theorems using a frozen model and Lean 4 with Mathlib. Maximize the fraction of verified theorems under shared budgets. The architecture and effort allocation are free; no specific search breadth, prompt family, or repair schedule is required.

- **Metric:** `proof_rate` (maximize)
- **HFM baseline:** 0.3643
- **HFM best so far:** 0.5661
- **HFM effort:** 57 proposals, 31 scored, 6 accepted improvements, about $2,494 of compute
- **HFM's current best idea:** Increasing p2105's per-theorem cap from two to four independent complete-proof repair requests, each targeting a distinct rejected fresh candidate, increases mean per-set practice proof_rate at its unchanged min(400,12*N) pooled generation ceiling.

## Existing theory and methods

Whole-proof sampling, best-first tactic search with REPL feedback, lemma retrieval from Mathlib.

## Public data

miniF2F and Mathlib are public. Lean 4 runs locally.

## What limits progress

Sampling budget and no reuse of proved lemmas across theorems.

## Route to a step change

Reflective sub-goal decomposition with a lemma store and compiler-guided repair (the design behind the 95.9% result), in place of more independent whole-proof samples.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

HFM is at 0.566. A harness around a frozen general model (Delta Prover, arXiv 2507.15225) reports 95.9% on miniF2F-test using sub-goal decomposition and repair, so headroom is large. HFM's problem set may differ from miniF2F-test.

## Next steps

Run a tactic-search baseline on miniF2F once a model endpoint is available.
