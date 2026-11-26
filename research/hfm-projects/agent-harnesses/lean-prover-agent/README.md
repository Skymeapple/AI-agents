# Agent for Formal Theorem Proving in Lean

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_formal_proofs` (paused on HFM)
- **Project status:** Tested on public data (model-free floor; HFM's own task still needs a language-model key)

## Readiness

- **State:** A model-free floor has run. The agent itself is ready to run once a language-model key is connected.
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

**What was run.** [`lean_floor.py`](lean_floor.py) measures how many miniF2F theorems fall to a single built-in Lean/Mathlib tactic, with no language model and no search. This is a floor for the track, not an entry in it: HFM's task uses a frozen language model, which needs a paid key.

- **Data:** the public miniF2F Lean 4 port (github.com/yangky11/miniF2F-lean4, MIT licence; Lean v4.24.0 with Mathlib), 244 validation and 244 test theorems.
- **Method:** each theorem's `sorry` is replaced by one of 19 tactics and checked by Lean under a fixed budget (200,000 heartbeats per theorem). `#print axioms` after each theorem confirms the proof was accepted without `sorry`.
- **Selection:** the tactics to combine were chosen on the validation set (greedy cover) and scored once on the test set.
- **Compute:** free GitHub Actions runners (4 cores). Logs and per-theorem results are in [`../../../compute/results/`](../../../compute/results/) under `lean-floor-test` and `lean-floor-valid-a`, `-b`, `-c`.

**Results.**

| Tactics (chosen on validation) | Validation, of 244 | Test, of 244 |
|---|---|---|
| `grind` alone | 69 (28.3%) | 64 (26.2%) |
| `grind` or `bound` | 85 (34.8%) | 77 (31.6%) |
| best 5 (`grind`, `bound`, `intros; nlinarith [sq_nonneg (0:ℝ)]`, `omega`, `intros; norm_num [*]`) | 92 (37.7%) | 89 (36.5%) |
| any of the 19 (upper bound, selection-biased) | 95 (38.9%) | 94 (38.5%) |

- **One tactic call proves about a third of miniF2F.** Two tactics picked on validation prove 31.6% of the test set; five prove 36.5%.
- **The gains are concentrated in the easy sources.** On validation, 74 of the 88 theorems proved by the second batch of tactics are `mathd` problems; none of the 20 `imo` or 8 `induction` problems is proved by any tactic.
- **The newer automation dominates.** `grind` alone proves more than twice as many test theorems as `norm_num`, `simp`, `linarith` or `omega` alone (64 against 17 to 30).

**Limits of this measurement.**

- These numbers are a lower bound. A few 41-theorem batches were lost whole: when one theorem made a tactic loop, the batch timed out (validation: the last batch for seven simp-based tactics; test: one batch for one tactic) or crashed (test: two `grind` batches; validation: one batch).
- The scores are not comparable with HFM's 0.364 → 0.566: HFM's problem set and budgets are private and may differ from miniF2F.
- The floor depends on the Mathlib version; `grind` and `bound` are recent additions.

**What it means for the track.** A harness should call cheap automation first and spend model budget only on what it leaves. On miniF2F that removes about a third of the theorems before any model call, and the remaining two thirds (almost all `imo`, `aime`, `induction` and most `amc12` problems) are where a model has to supply structure.

## How far this can go

HFM is at 0.566. A harness around a frozen general model (Delta Prover, arXiv 2507.15225) reports 95.9% on miniF2F-test using sub-goal decomposition and repair, so headroom is large. HFM's problem set may differ from miniF2F-test.

## Next steps

Rerun the lost batches one theorem at a time to tighten the floor. Once a model endpoint is available, run a tactic-search baseline that calls `grind` and `bound` at every sub-goal.
