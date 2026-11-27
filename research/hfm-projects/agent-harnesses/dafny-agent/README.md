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
- **Not done here:** HFM's own task (writing implementations) was not attempted; see the second experiment for a model.

**Second experiment: an open coding model proposes invariants, the verifier prunes them.** [`dafny_llm.py`](dafny_llm.py) gives Qwen2.5-Coder-7B-Instruct (4-bit GGUF, llama.cpp on four CPU cores, no GPU) each of the 433 parseable programs the rules above leave (22 longer than 6,000 characters skipped, 411 attempted). Logs: `dafny-llm2-*` in [`../../../compute/results/`](../../../compute/results/).

- **Design that failed first.** Asking the model for the whole annotated program and checking that it only added annotations verified 0 of 82 programs before it was stopped (`dafny-llm-2`, `dafny-llm-9`): the model reformatted lines, dropped comments, moved braces and wrote programs that did not parse, so the integrity check or the parser rejected nearly every answer.
- **Design that works.** Only the `invariant` and `decreases` clauses the model writes after each loop header are kept. They are inserted after the matching loop header of the original, untouched program, together with the rule-based candidates. Dafny verifies; every inserted clause it reports as failing is dropped and the rest re-verified, up to 12 rounds (Houdini pruning). If the program still fails, the model gets one repair turn with the pruned program and Dafny's errors, and the clauses from both turns are pooled and pruned again. The program text is never edited and `decreases *` is refused, so the specification cannot be weakened; acceptance is Dafny's own "verified, 0 errors" summary.

| | Programs that verify | Share of 782 |
| --- | --- | --- |
| rules (above) | 289 | 37.0% |
| + model clauses, first turn | 295 | 37.7% |
| + one repair turn with Dafny's errors | **320** | **40.9%** |

- **The repair turn does most of the work:** 6 programs verify on the first turn, 25 more after one turn of feedback. Feedback from the verifier is worth four times the first answer.
- **Pruning matters:** in the verified programs the model proposed 3 to 16 clauses and Dafny kept about two thirds; without pruning, one wrong clause fails the whole program.
- **Still far from published systems.** 31 of 411 remaining programs is 7.5%. DafnyBench reports much higher rates for large models; this is a 7B model on CPU with two turns.

## How far this can go

HFM is at 0.829. Published DafnyBench results with strong models are in the same range.

## Next steps

Classify failures on DafnyBench by obligation type.
