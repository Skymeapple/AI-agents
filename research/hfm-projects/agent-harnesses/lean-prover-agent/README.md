# Agent for Formal Theorem Proving in Lean

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_formal_proofs` (paused on HFM)
- **Project status:** Tested on public data (model-free floor plus a free open-weight prover model on CPU; HFM's own task uses a paid frozen model)

## Readiness

- **State:** A model-free floor and an open-weight prover model (on free CPU runners) have run. HFM's own setting, a frozen API model, still needs a key.
- **Data access:** miniF2F on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key or a GPU for larger budgets (more attempts, repair rounds). GitHub Models did not accept the workflow's built-in token.
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
- **Tactics:** 52 in three rounds. Round three (18 tactics, run only on theorems the first two rounds left unproved) takes one structural step, such as induction on a natural-number variable, case-splitting a bounded integer, splitting a conjunction or arguing by contradiction, and then tries a fixed portfolio of twelve closing tactics on every goal that remains. Rounds one and two: Round one is 19 built-in tactics and simple pairs. Round two is 15 short compositions around the strongest of them, such as `norm_num at * <;> grind`, and `nlinarith` given the squares of the theorem's real variables and of their pairwise sums and differences (generated from the statement by a fixed rule).
- **Selection:** the tactics to combine were chosen on the validation set (greedy cover) and scored once on the test set.
- **Compute:** free GitHub Actions runners (4 cores). Logs and per-theorem results are in [`../../../compute/results/`](../../../compute/results/) under `lean-floor2-*` (rounds one and two) and `lean-floor3-*` (round three). The `lean-floor-*` folders hold the first round's earlier run, which lost whole batches to timeouts and is superseded.

**Results.**

| Tactics (chosen on validation) | Validation, of 244 | Test, of 244 |
|---|---|---|
| `grind` alone (best of round one) | 69 (28.3%) | 74 (30.3%) |
| `norm_num at * <;> grind` alone | 81 (33.2%) | 84 (34.4%) |
| best 2: adding `bound` | 89 (36.5%) | 88 (36.1%) |
| best 4: adding `nlinarith` with generated squares, `intros; norm_num [*]` | 100 (41.0%) | 95 (38.9%) |
| best 8 | 108 (44.3%) | 99 (40.6%) |
| any of the 34 in rounds one and two (upper bound, selection-biased) | 108 (44.3%) | 106 (43.4%) |
| all three rounds, 14 tactics chosen on validation | 118 (48.4%) | 103 (42.2%) |
| any of the 52 (upper bound, selection-biased) | 118 (48.4%) | 110 (45.1%) |

- **One short tactic call proves about a third of miniF2F, and a handful prove about 40%.** The eight tactics picked on validation prove 99 of the 244 test theorems.
- **Composition helped.** Normalising numerals before `grind` adds 10 test theorems to `grind` alone (84 against 74). The generated squares add 5 test theorems to plain `nlinarith` (37 against 32).
- **The gains are concentrated in the easy sources.** Of the 106 test theorems proved by any tactic, 86 are `mathd` problems. Two of the 20 test `imo` problems fall (`imo_1964_p2`, `imo_1982_p1`), four `aime` problems and one `induction` problem.
- **Returns flatten quickly.** From four tactics to eight, the test score rises by 4 theorems.
- **One structural step adds almost nothing.** Round three proved 10 more validation theorems and 4 more test theorems out of the 136 and 138 left. Induction followed by the portfolio is the only structural step that helped on both sets (3 validation, 2 test). The validation gain did not transfer: the 14 tactics chosen on validation reach 48.4% there and 42.2% on test.
- **Some proved statements may be weaker than the original problems.** Round three's proofs include `imo_1987_p4` and `imo_1977_p6`. Lean accepted them without `sorry`, but short automatic proofs of olympiad problems usually mean the formal statement is easier than the informal one; this was not checked statement by statement.

**Limits of this measurement.**

- These numbers are a lower bound. When a theorem makes a tactic hang or crash Lean, that theorem is dropped for that tactic and the rest of its batch is rerun; a handful of theorem-tactic pairs were dropped this way (for example `mathd_numbertheory_543` for the simp-based tactics on validation, and `imo_1977_p6`, `imo_1981_p6`, `mathd_numbertheory_328` for the `grind` family on test).
- The scores are not comparable with HFM's 0.364 → 0.566: HFM's problem set and budgets are private and may differ from miniF2F.
- The floor depends on the Mathlib version; `grind` and `bound` are recent additions.
- The second-round tactics were written after seeing first-round test results per tactic, so the choice of what to compose is not fully blind to the test set. The combination reported was still selected on validation only.

**Second experiment: an open-weight prover model on the remainder.** [`prover_llm.py`](prover_llm.py) gives DeepSeek-Prover-V1.5-RL (7B, 4-bit GGUF, llama.cpp on the runners' four CPU cores, no GPU) the 134 test theorems that none of the 52 tactics proved. For each theorem it samples 4 whole proofs (temperature 1.0, top-p 0.95, at most 512 tokens, the model's non-chain-of-thought prompt) and Lean checks each one against Mathlib; a proof counts only if `#print axioms` shows no `sorry`. Four shards ran in parallel, about 4.5 hours each. A second round then gave the 116 theorems still unproved 8 fresh samples each (different seeds, eight runners, about 1.5 hours), for 12 samples per theorem in all. Logs and proofs are in [`../../../compute/results/`](../../../compute/results/) under `lean-prover-llm-*` and `lean-prover-llm2-*`.

| | Test, of 244 |
|---|---|
| model, 4 samples, on the 134 left by every tactic | 18 of 134 (13.4%) |
| model, 8 more samples, on the 116 still left | 3 of 116 (2.6%) |
| any of the 52 tactics, then the model (12 samples) | 131 (53.7%) |
| 14 tactics chosen on validation, then the model on what all 52 left (lower bound) | at least 124 (50.8%) |

- **The model proves things the tactics cannot, but few of them.** 18 of 134: 6 `mathd`, 6 hand-written `algebra`, 3 `amc12`, 1 `aime` (`aime_1983_p2`), 1 `induction`, 1 `numbertheory`. No `imo` problem fell. The proofs are short (33–348 characters), mostly a case split or a few `have` steps closed by `nlinarith` or `omega`: one or two structural steps that the fixed round-three recipe did not find. None uses `native_decide`.
- **More samples barely help.** 7 of the first 18 were found by only one of four attempts, which suggested the remainder was budget-limited. It is not: tripling the samples (8 more per theorem) proved only 3 of the 116 left (`mathd_numbertheory_234`, `amc12b_2002_p7`, `mathd_algebra_170`). The remaining theorems need proof structure this model does not produce whole, not more draws.
- **The combination is the useful number.** Cheap tactics first, then the model, reaches 53.7% of miniF2F-test with free CPU and no paid model. The honest variant, where the tactics are those chosen on validation, is at least 50.8% (the model was not run on the 7 theorems proved only by unchosen tactics).
- **This is below HFM's 0.566 (by 3 points), and not comparable with it.** HFM's problem set is private and its model and budgets differ. The prover model may also have seen miniF2F statements during training.

**What it means for the track.** A harness should call cheap automation first and spend model budget only on what it leaves. On miniF2F that removes about 40% of the theorems before any model call. One more automatic step moves that by under two points, so the remaining theorems need several coordinated steps, not a better closing tactic. That remainder (almost all `imo`, most `aime`, `induction` and `amc12` problems) is where a model has to supply structure.

## How far this can go

HFM is at 0.566. A harness around a frozen general model (Delta Prover, arXiv 2507.15225) reports 95.9% on miniF2F-test using sub-goal decomposition and repair, so headroom is large. HFM's problem set may differ from miniF2F-test.

## Next steps

With a GPU or a model key: do not just raise the samples (12 per theorem added 3 theorems over 4); feed Lean's error back for repair rounds, and call the tactic portfolio at every sub-goal the model writes as `have ... := by sorry`.
