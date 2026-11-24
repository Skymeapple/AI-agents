# Agent Harness Evolution Under a Frozen API Model

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_rsi_agent_harness_opus5` (active on HFM)
- **Project status:** Blocked

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

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

Not estimable without the frozen model.

## Next steps

Require repeated runs before accepting a change, given the noise.
