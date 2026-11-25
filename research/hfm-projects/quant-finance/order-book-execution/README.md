# Execution Policy Development from Ten-Level Order Book Data

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_nasdaq_depth_execution` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Blocked on licensed data.
- **Data access:** Only LOBSTER's free sample day: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** Licensed order-book history.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable execution policy using Nasdaq order book snapshots and trade-flow records. The policy is optimized for net execution cost across replayed parent orders, with open research choices for representations, forecasting methods, and decision logic under fixed market rules.

- **Metric:** `execution_saving_reward` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 1
- **HFM effort:** 11 proposals, 10 scored, 2 accepted improvements, about $261 of compute
- **HFM's current best idea:** Limiting p490's discretionary pre-sweep marketable orders to the observed opposite touch on the cent grid lowers pooled implementation shortfall relative to unchanged p490.

## Existing theory and methods

Almgren-Chriss execution, fill-probability models, reinforcement learning for execution.

## Public data

LOBSTER free samples only.

## What limits progress

A saturated evaluator.

## Route to a step change

Fix the metric first, then optimal-execution control.

## Experiment and results

No experiment has been run for this project.

- **Why:** Nasdaq order-book data is licensed, and the reward is pinned at 1.0.

## How far this can go

Unknown.

## Next steps

Ask HFM to uncap the reward.
