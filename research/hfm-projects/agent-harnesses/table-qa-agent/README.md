# Agent for Questions over Real Tables

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_table_analysis` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a language-model key is connected.
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

DataBench is public on Hugging Face.

## What limits progress

Ambiguous questions and type handling.

## Route to a step change

Several programs per question with voting on executed answers.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

HFM is at 0.877. Label noise in the benchmark probably caps accuracy in the low 0.9s.

## Next steps

Audit a sample of failures for label errors to estimate the ceiling.
