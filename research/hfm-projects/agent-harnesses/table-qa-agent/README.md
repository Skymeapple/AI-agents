# Agent for Questions over Real Tables

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_table_analysis` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A small-scale study with an open model has run on CPU. HFM's frozen model still needs a key.
- **Data access:** DataBench on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable agent harness that answers questions over real tables using a frozen model and installed analysis tools. Tables include lists, manifests, survey exports, and sensor logs. The evaluation maximizes the fraction of correct answers under DataBench's comparator and specified budgets.

- **Metric:** `table_accuracy` (maximize)
- **HFM baseline:** 0.3711
- **HFM best so far:** 0.877
- **HFM effort:** 31 proposals, 30 scored, 12 accepted improvements, about $1,111 of compute
- **HFM's current best idea:** Exposing exact categorical backing dtypes and ordered flags in p1438's initial ranking requests increases table_accuracy over unchanged p1438.

## Existing theory and methods

Program-of-thought (generate pandas code), self-consistency voting, schema profiling.

## Public data

WikiTableQuestions is public (GitHub release). An open 3B model was downloaded from Hugging Face on the free compute.

## What limits progress

Ambiguous questions and type handling.

## Route to a step change

Several programs per question with voting on executed answers.

## Experiment and results

A small-scale harness comparison without any paid key. A small open model (Qwen2.5-3B-Instruct, 4-bit, run with llama.cpp on a free 4-core runner) stands in for HFM's frozen model, on 250 WikiTableQuestions test questions over tables unseen in training. Script: [`tableqa_small_model.py`](tableqa_small_model.py); logs and per-question outputs: [`tableqa-direct`](../../../compute/results/tableqa-direct/), [`tableqa-code`](../../../compute/results/tableqa-code/), [`tableqa-code-notebook`](../../../compute/results/tableqa-code-notebook/).

| Harness, same model | Correct |
| --- | --- |
| Table in the prompt, model answers directly | 28.4% |
| Model writes pandas code; answer is what the code prints | 1.6% |
| The same, but the value of the last expression is printed, as in a notebook | 27.6% |
| Notebook-style code when it ran, otherwise the direct answer | 33.2% |
| Either harness right (the most a perfect selector could reach) | 45.6% |

- **One harness detail moved the code route from 1.6% to 27.6%.** The model writes code that ends in a bare expression and does not call `print`. Telling it "the code printed nothing" did not fix this within two retries. Printing the last value did.
- **The two routes are right on different questions.** Both right on 26, code only on 43, direct only on 45. A simple fallback gains about 5 points over either alone; a perfect selector would gain 17.
- **Code is better on long tables** (35.5% against 25.8% on the 31 tables over 40 rows) and twice as fast (13 against 26 seconds per question).
- **Retries on errors help modestly:** 23.6% on the first attempt, 27.6% after up to two retries. Code still fails to run on 28% of questions.
- **Uncertainty:** with 250 questions, differences under about 5 points are within noise. The 1.6 to 27.6 jump and the complementarity are not.

**Follow-up: a selector, checked on fresh questions.** On the first set, the rule "use the code answer when it looks like an answer (one short line, not empty, zero or a truth value), otherwise the direct answer" scored 40.0%. Because it was chosen on those questions, both harnesses were rerun on a second set of 250 questions with no overlap. Outputs: [`tableqa-direct-set2`](../../../compute/results/tableqa-direct-set2/), [`tableqa-code-notebook-set2`](../../../compute/results/tableqa-code-notebook-set2/).

| Harness, same model | First set | Fresh set |
| --- | --- | --- |
| Direct | 28.4% | 28.0% |
| Notebook-style code | 27.6% | 28.4% |
| Code when it ran, otherwise direct | 33.2% | 36.8% |
| Code when its answer looks clean, otherwise direct | 40.0% | 41.2% |
| Either route right (perfect selector) | 45.6% | 47.6% |

- **The selector holds on fresh questions:** 41.2% against 28.0% for the direct route, a gain of 13 points with the model unchanged. It captures about two thirds of what a perfect selector would add.
- **Long tables on the fresh set:** direct 6.5%, code 29.0% (31 questions).

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

HFM's harness work took its frozen model from 0.37 to 0.88. On this much weaker stand-in, the best simple combination reaches 33% and a perfect choice between two routes 46%. The pattern matches HFM's: most of the gain is in how the model is used, and the remaining headroom is in choosing the route per question and getting code to run.

## Next steps

- Get code to run more often (it fails on 28% of questions) and close the remaining 6 points to a perfect selector.
- Repeat on HFM's frozen model once a language-model key is available.
