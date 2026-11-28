# Agent for ARC Grid Transformation Tasks

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_grid_reasoning` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU study has run. The harness itself waits for a language-model key.
- **Data access:** ARC-AGI tasks on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable agent harness for ARC grid transformations. The agent uses example input-output pairs to predict outputs for new inputs, employing a frozen model and local computation to maximize exact-match accuracy under shared budgets. The agent selects its own representations, inference process, and effort allocation.

- **Metric:** `solved_rate` (maximize)
- **HFM baseline:** 0.1213
- **HFM best so far:** 0.7856
- **HFM effort:** 52 proposals, 48 scored, 10 accepted improvements, about $1,236 of compute
- **HFM's current best idea:** Raising p890's fresh-program and repair output-token requests from 3000 to the permitted 4000 increases practice solved_rate at unchanged instance budgets.

## Existing theory and methods

Program synthesis with repair, test-time training, augmentation voting over the eight grid symmetries.

## Public data

ARC-AGI tasks are public on GitHub.

## What limits progress

Search budget per task and the frozen model's ability to induce the rule.

## Route to a step change

Combine test-time adaptation with program search and verified voting.

## Experiment and results

Code: [`arc_classical.py`](arc_classical.py) in this folder. It runs in about a minute on a CPU, on the public ARC-AGI-1 task files (github.com/fchollet/ARC-AGI, Apache 2.0).

- **What was tested:** HFM's harness needs paid access to its frozen language model. With no model at all, a CPU can measure the floor: how many tasks a small, fixed library of grid operations solves by search. The library has 34 operations (rotations, flips, crops, scaling, tiling, mirroring, gravity, hole filling, object selection), composed up to two deep and followed by a learned colour mapping, plus cell-wise lookup rules for same-shape tasks. A program is accepted when it reproduces every example pair. The operations were written looking at training tasks only.
- **Result:**

  | Task set | Tasks | A program fitted all examples | Solved (one or two guesses) |
  | --- | --- | --- | --- |
  | Public training tasks | 400 | 30 (7.5%) | 26 (6.5%) |
  | Public evaluation tasks (held out) | 400 | 6 (1.5%) | 4 (1.0%) |
  | HFM's agent, frozen model (its own task set) | | | 12% at baseline, 79% after harness work |

- **What this shows:**
  - **A fixed library solves about 1% of held-out tasks.** The drop from 6.5% to 1.0% is the point of ARC: each task needs a rule that was not written in advance.
  - **Fitting the examples is nearly sufficient when it happens.** 30 of 36 fitted programs are right on the test input. The difficulty is finding any program that fits, not choosing among programs that do.
  - **So the model's job in HFM's harness is to propose task-specific programs.** The 67-point gain from harness work (12% to 79%) cannot come from a better fixed search; it comes from generating candidates and checking them against the examples, which is the cheap and reliable part.
- **Limits:** a deliberately small library and depth two. Published hand-built searches with far larger libraries score higher. HFM's task set is not stated to be the public evaluation set.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Fixed library, no model:** about 1% on held-out tasks here.
- **With a model proposing programs:** HFM reports 79%. The verification step (run the program on the examples) is exact and free.

## Next steps

- With a language-model key: generate programs per task, keep those that fit all examples, and vote among them; measure how the solved rate grows with the number of candidates.
- On CPU: measure how often two fitted programs disagree on the test input, to size the benefit of a second guess.
