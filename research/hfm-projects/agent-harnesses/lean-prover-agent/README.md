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

Tactic-state search guided by the model, with a lemma cache.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

HFM is at 0.566. Specialised provers exceed this on miniF2F, so headroom is real.

## Next steps

Run a tactic-search baseline on miniF2F once a model endpoint is available.
