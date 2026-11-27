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

> **Audited (2026-11-27).** The first versions of these scripts accepted a theorem when `#print axioms` showed no `sorryAx`. That test is too weak: when Lean's kernel rejects a proof ("declaration has metavariables") it adds the theorem as an axiom of itself, so `#print axioms` lists the theorem's own name and no `sorryAx`, and some failed theorems print "does not depend on any axioms". [`audit.py`](audit.py) re-checked all 2,410 proofs that had been counted, accepting one only if no Lean error points inside the theorem and its axioms are a subset of propext, Classical.choice and Quot.sound. 2,246 passed. **All numbers below are the audited ones** ([`audit_summary.py`](audit_summary.py); logs `lean-audit-*`). Earlier figures in this repository's history (42.2%, 52.5%, 54.9%, 138 and 143 of 244) were too high and are withdrawn. All three scripts now use the strict test.

**What was run.** [`lean_floor.py`](lean_floor.py) measures how many miniF2F theorems fall to a single short tactic call, with no language model and no search. This is a floor for the track, not an entry in it: HFM's task uses a frozen language model, which needs a paid key.

- **Data:** the public miniF2F Lean 4 port (github.com/yangky11/miniF2F-lean4, MIT licence; Lean v4.24.0 with Mathlib), 244 validation and 244 test theorems.
- **Method:** each theorem's `sorry` is replaced by one tactic call and checked by Lean under a fixed budget (200,000 heartbeats per theorem).
- **Tactics:** 52 in three rounds. Round one is 19 built-in tactics and simple pairs. Round two is 15 short compositions around the strongest of them, such as `norm_num at * <;> grind`, and `nlinarith` given the squares of the theorem's real variables and of their pairwise sums and differences (generated from the statement by a fixed rule). Round three (18 tactics, run only on theorems the first two rounds left unproved) takes one structural step, such as induction on a natural-number variable, case-splitting a bounded integer, splitting a conjunction or arguing by contradiction, and then tries a fixed portfolio of twelve closing tactics on every goal that remains.
- **Selection:** the tactics to combine were chosen on the validation set (greedy cover) and scored once on the test set.
- **Compute:** free GitHub Actions runners (4 cores). Logs and per-theorem results are in [`../../../compute/results/`](../../../compute/results/) under `lean-floor2-*` (rounds one and two), `lean-floor3-*` (round three) and `lean-audit-*` (the strict re-check).

**Results (audited).**

| Tactics (chosen on validation) | Validation, of 244 | Test, of 244 |
|---|---|---|
| `grind` alone (best of round one) | 69 (28.3%) | 74 (30.3%) |
| `norm_num at * <;> grind` alone | 80 (32.8%) | 84 (34.4%) |
| best 2: adding `bound` | 88 (36.1%) | 88 (36.1%) |
| best 4: adding `nlinarith` with generated squares, `intros; norm_num [*]` | 97 (39.8%) | 94 (38.5%) |
| best 8 | 107 (43.9%) | 99 (40.6%) |
| any of the 34 in rounds one and two (upper bound, selection-biased) | 106 (43.4%) | 104 (42.6%) |
| all three rounds, 13 tactics chosen on validation | 113 (46.3%) | 101 (41.4%) |
| any of the 52 (upper bound, selection-biased) | 113 (46.3%) | 106 (43.4%) |

- **One short tactic call proves about a third of miniF2F, and a handful prove about 40%.** The eight tactics picked on validation prove 99 of the 244 test theorems.
- **Composition helped.** Normalising numerals before `grind` adds 10 test theorems to `grind` alone (84 against 74). The generated squares add 2 test theorems to plain `nlinarith` (34 against 32).
- **The gains are concentrated in the easy sources.** Of the 106 test theorems proved by any tactic, 87 are `mathd` problems; one `imo` problem (`imo_1964_p2`), four `aime` and two `induction` problems.
- **One structural step adds almost nothing.** Round three adds 7 validation and 2 test theorems.
- **The audit removed the most impressive-looking proofs.** Of the tactic proofs, 5 validation and 4 test theorems were false acceptances, including `imo_1982_p1`, `imo_1977_p6` and `imo_1987_p4`, which earlier versions of this page listed as proved.

**Limits of this measurement.**

- When a theorem makes a tactic hang or crash Lean, that theorem is dropped for that tactic; a handful of pairs were dropped this way.
- The scores are not comparable with HFM's 0.364 → 0.566: HFM's problem set and budgets are private and may differ from miniF2F.
- The floor depends on the Mathlib version; `grind` and `bound` are recent additions.
- The second-round tactics were written after seeing first-round test results per tactic, so the choice of what to compose is not fully blind to the test set. The combination reported was still selected on validation only.

**Second experiment: an open-weight prover model on the remainder.** [`prover_llm.py`](prover_llm.py) gives DeepSeek-Prover-V1.5-RL (7B, 4-bit GGUF, llama.cpp on the runners' four CPU cores, no GPU) the test theorems that none of the 52 tactics had (by the old test) proved. For each theorem it samples whole proofs (temperature 1.0, top-p 0.95, at most 512 tokens, the model's non-chain-of-thought prompt) and Lean checks each one. Round one: 4 samples per theorem; round two: 8 more. Rounds three to five ([`prover_repair.py`](prover_repair.py)) sample 4, 8 and 8 more and also try "repaired" variants: every one-line `have` falls back to the tactic portfolio, and every prefix of the model's steps is followed by `all_goals` portfolio. Logs: `lean-prover-llm-*`, `lean-prover-llm2-*`, `lean-prover-repair-*`, `lean-prover-repair2-*`, `lean-prover-repair3-*`.

| Round (audited) | Claimed | Pass the strict check | New theorems | Running total, of 244 |
|---|---|---|---|---|
| any of the 52 tactics | 110 | 106 | 106 | 106 (43.4%) |
| model, 4 samples | 18 | 16 | 16 | 122 (50.0%) |
| model, 8 more samples | 3 | 3 | 3 | 125 (51.2%) |
| model, 4 more samples with repair | 3 | 0 | 0 | 125 |
| model, 8 more samples with repair | 4 | 3 | 3 | 128 (52.5%) |
| model, 8 more samples with repair | 5 | 1 | 1 | 129 (52.9%) |
| model, 4 fresh samples + 4 rewritten from Lean's error messages (strict check built in) | 2 | 2 | 2 | **131 (53.7%)** |
| 13 tactics chosen on validation, then the model | | | | **126 (51.6%)** |

- **The open model adds about 9 points to the tactic floor:** 23 test theorems over five rounds and 32 samples per theorem, mostly `mathd`, hand-written `algebra` and `amc12` problems, with a case split or a few `have` steps closed by `nlinarith` or `omega`. No `imo` proof survived the audit.
- **More samples help slowly.** 4 samples prove 16; 28 more add 7.
- **Sketch repair did not work.** Every theorem that only a repaired variant had proved (5 of them, among them `aime_1994_p3` and `amc12a_2021_p25`) was a false acceptance: the variants that "closed" the proof had in fact failed. The repair route as built here adds nothing; its earlier write-up is withdrawn.
- **Feedback from Lean's own errors helps a little.** [`prover_feedback.py`](prover_feedback.py) (`lean-feedback-*`) gives every failed fresh sample back to the model with Lean's first error messages and asks for a corrected proof. Of the 115 theorems left, 4 fresh samples proved 1 and the feedback rewrites 1 more (`mathd_algebra_156`, 1 of 4 rewrites). Unlike in Dafny, where one repair turn verified four times as many programs as the first answer, a 7B prover without chat training makes little use of the messages.
- **The total, 0.537 (0.516 with validation-chosen tactics), is clearly below HFM's 0.566,** and not comparable with it: HFM's problem set is private, its model and budgets differ, and the prover model may have seen miniF2F statements during training.

**What it means for the track.** A harness should call cheap automation first and spend model budget only on what it leaves; on miniF2F that removes about 40% of the theorems before any model call. A 7B open prover on CPU adds about 10 points more. The rest needs coordinated multi-step structure. The audit is also a lesson for any agent harness on this track: the acceptance check must look for errors in the theorem itself, not only at `#print axioms`, or the harness will reward its own failures.

## How far this can go

HFM is at 0.566. A harness around a frozen general model (Delta Prover, arXiv 2507.15225) reports 95.9% on miniF2F-test using sub-goal decomposition and repair, so headroom is large. HFM's problem set may differ from miniF2F-test.

## Next steps

With a GPU or a model key: feed Lean's error messages back to the model for repair rounds (the automatic tactic repair tried here added nothing once checked strictly), and decompose into sub-goals the model states as `have ... := by sorry`, each attacked separately.
