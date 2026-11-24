# Agent for ARC Grid Transformation Tasks

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_grid_reasoning` (paused on HFM)
- **Project status:** Blocked

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

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

HFM is at 0.786 on its own set. Public ARC-AGI-1 results with search are in the same range; the remaining tasks need abstractions the frozen model lacks.

## Next steps

Run the public evaluation set with an available model to locate which task classes remain.
