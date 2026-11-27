# Agent Harness Evolution Under a Frozen API Model

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_rsi_agent_harness_opus5` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in with a frozen small open model on CPU has run. The track's own frozen API model still needs a key.
- **Data access:** Terminal-Bench on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work examines the evolution of a terminal agent harness under a frozen API model, with the default configuration using openai:kimi-k2.6. The harness is evaluated solely on terminal-bench-2 and includes components such as prompts, planning, memory/notes, tool sets, sub-agents, retry/verification loops, context management, and dependencies on langgraph.json. The evaluation considers configurations that achieve a higher macro-average across four member benchmarks—terminal_bench_2, gaia, simpleqa, and hle_search—on a sealed assessment list. The scope and constraints follow the instruction.md of the harbor task.

- **Metric:** `reward` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.5362
- **HFM effort:** 8 proposals, 6 scored, 0 accepted improvements, about $132 of compute
- **HFM's current best idea:** 起步 harness 一半的 trial（n_timeout=10/20）是被 harbor 的 per-task agent timeout 杀掉的，而 deepagents 完全没有墙钟概念；给 agent 注入「已用/剩余时间」并强制 anytime 契约（先把能被 tests 打分的最小可行产物落到磁盘，预算后段只收尾不探索），能把 n_timeout 显著压低并抬高 practice pass rate。

## Existing theory and methods

Plan-act-verify loops, context compaction, time-aware anytime behaviour.

## Public data

Terminal-Bench 2 is public.

## What limits progress

Half of the starting failures are timeouts. The sealed score has about 6 points of noise.

## Route to a step change

Time awareness with an always-deliverable artefact, then verifier-first loops.

## Experiment and results

A stand-in for the track's question (with the model frozen, how much do harness changes move the held-out pass rate?), using a small open model instead of the paid one. Script: [`shell_harness.py`](shell_harness.py); logs in `research/compute/results/shell-harness-h*`. It is not comparable to HFM's score.

- **Frozen model:** Qwen2.5-3B-Instruct, 4-bit, llama.cpp on four CPU cores, greedy decoding.
- **Tasks:** 12 templates of shell work, each with 4 random instances, run in an empty folder and checked by a Python function (every checker was validated with a correct reference command). Templates 0–5 (count matching lines, sum a CSV column, rename extensions, sort unique, count files recursively, replace a word) are the development set; templates 6–11 (largest file, distinct IPs, a JSON field, write and run a script, make folders, make an archive) are held out, so the test is on new kinds of task, not just new instances. Up to 8 commands per task.
- **Harnesses, cumulative:** h0 minimal (last 400 characters of output); h1 adds exit code, separate stderr and head-and-tail truncation; h2 adds a listing of the folder with the first lines of each file before the first turn; h3 adds one refused DONE with a request to check the result.

| Harness | Development (24 tasks) | Held out (24 tasks) |
|---|---|---|
| h0 minimal | 13 (54%) | **12 (50%)** |
| h1 + structured feedback | 13 (54%) | 12 (50%) |
| h2 + folder listing up front | **16 (67%)** | 9 (38%) |
| h3 + check before finishing | 16 (67%) | 9 (38%) |

- **The change that looks best on development makes the held-out score worse.** Showing the folder up front raises development from 54% to 67% (the "count ERROR lines" template goes from 1 of 4 to 4 of 4) and lowers held-out from 50% to 38% (the "distinct IPs" template goes from 4 of 4 to 0 of 4; the JSON template gains 1). A harness chosen on the development set would have been the wrong one.
- **Two of the three changes did nothing at all.** Structured feedback and the refused DONE leave every one of the 48 outcomes unchanged: with greedy decoding the model finishes most tasks in two turns, so feedback on later turns and a second look rarely come into play.
- **Outcomes move by whole templates.** With greedy decoding, the four instances of a template almost always pass or fail together, so the effective sample is 12 templates, not 48 tasks: one template flipping moves the score by 8 points. This is the same order as the ±6 points of noise the track itself reports.
- **For HFM's track:** accept a harness change only if it helps on held-out task kinds, not only held-out instances, and count the evidence in task kinds. The cheapest harness gains here (the folder listing) also carried the largest risk of overfitting to the development kinds.

## How far this can go

Not estimable without the frozen model. With a small open model, harness changes moved held-out pass rates by up to 12 points in either direction.

## Next steps

Require repeated runs before accepting a change, given the noise, and score on held-out task kinds. With a key: rerun this protocol on the track's frozen model.
