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

**What was run.** [`lean_floor.py`](lean_floor.py) measures how many miniF2F theorems fall to a single short tactic call, with no language model and no search. This is a floor for the track, not an entry in it: HFM's task uses a frozen language model, which needs a paid key.

- **Data:** the public miniF2F Lean 4 port (github.com/yangky11/miniF2F-lean4, MIT licence; Lean v4.24.0 with Mathlib), 244 validation and 244 test theorems.
- **Method:** each theorem's `sorry` is replaced by one tactic call and checked by Lean under a fixed budget (200,000 heartbeats per theorem). `#print axioms` after each theorem confirms the proof was accepted without `sorry`.
- **Tactics:** 34 in two rounds. Round one is 19 built-in tactics and simple pairs. Round two is 15 short compositions around the strongest of them, such as `norm_num at * <;> grind`, and `nlinarith` given the squares of the theorem's real variables and of their pairwise sums and differences (generated from the statement by a fixed rule).
- **Selection:** the tactics to combine were chosen on the validation set (greedy cover) and scored once on the test set.
- **Compute:** free GitHub Actions runners (4 cores). Logs and per-theorem results are in [`../../../compute/results/`](../../../compute/results/) under `lean-floor2-valid-a`, `-valid-b`, `-test-a`, `-test-b`. The `lean-floor-*` folders hold the first round's earlier run, which lost whole batches to timeouts and is superseded.

**Results.**

| Tactics (chosen on validation) | Validation, of 244 | Test, of 244 |
|---|---|---|
| `grind` alone (best of round one) | 69 (28.3%) | 74 (30.3%) |
| `norm_num at * <;> grind` alone | 81 (33.2%) | 84 (34.4%) |
| best 2: adding `bound` | 89 (36.5%) | 88 (36.1%) |
| best 4: adding `nlinarith` with generated squares, `intros; norm_num [*]` | 100 (41.0%) | 95 (38.9%) |
| best 8 | 108 (44.3%) | 99 (40.6%) |
| any of the 34 (upper bound, selection-biased) | 108 (44.3%) | 106 (43.4%) |

- **One short tactic call proves about a third of miniF2F, and a handful prove about 40%.** The eight tactics picked on validation prove 99 of the 244 test theorems.
- **Composition helped.** Normalising numerals before `grind` adds 10 test theorems to `grind` alone (84 against 74). The generated squares add 5 test theorems to plain `nlinarith` (37 against 32).
- **The gains are concentrated in the easy sources.** Of the 106 test theorems proved by any tactic, 86 are `mathd` problems. Two of the 20 test `imo` problems fall (`imo_1964_p2`, `imo_1982_p1`), four `aime` problems and one `induction` problem.
- **Returns flatten quickly.** From four tactics to eight, the test score rises by 4 theorems.

**Limits of this measurement.**

- These numbers are a lower bound. When a theorem makes a tactic hang or crash Lean, that theorem is dropped for that tactic and the rest of its batch is rerun; a handful of theorem-tactic pairs were dropped this way (for example `mathd_numbertheory_543` for the simp-based tactics on validation, and `imo_1977_p6`, `imo_1981_p6`, `mathd_numbertheory_328` for the `grind` family on test).
- The scores are not comparable with HFM's 0.364 → 0.566: HFM's problem set and budgets are private and may differ from miniF2F.
- The floor depends on the Mathlib version; `grind` and `bound` are recent additions.
- The second-round tactics were written after seeing first-round test results per tactic, so the choice of what to compose is not fully blind to the test set. The combination reported was still selected on validation only.

**What it means for the track.** A harness should call cheap automation first and spend model budget only on what it leaves. On miniF2F that removes about 40% of the theorems before any model call, and the remaining 60% (almost all `imo`, most `aime`, `induction` and `amc12` problems) are where a model has to supply structure.

## How far this can go

HFM is at 0.566. A harness around a frozen general model (Delta Prover, arXiv 2507.15225) reports 95.9% on miniF2F-test using sub-goal decomposition and repair, so headroom is large. HFM's problem set may differ from miniF2F-test.

## Next steps

Add a model-free search over two- and three-step tactic sequences to see how far the floor moves without a model. Once a model endpoint is available, run a tactic-search baseline that calls `grind` and `bound` at every sub-goal.
