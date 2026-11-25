# Intraday Trading Overlay for A-Share Holdings

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_ashare_intraday` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Blocked on licensed data.
- **Data access:** No free legal source of five-minute A-share bars was found. Daily data are public.
- **Compute:** CPU.
- **Still missing:** Licensed intraday data, or a redefinition on daily data.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable intraday overlay using five-minute and daily records. It explores signal discovery, decision formation, and effort allocation across a stock book to maximize a fixed net-of-cost objective, subject to base holdings, market rules, and end-of-day requirements.

- **Metric:** `reward` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 1
- **HFM effort:** 42 proposals, 38 scored, 3 accepted improvements, about $1,335 of compute
- **HFM's current best idea:** Applying tenfold stronger ridge shrinkage to the common buy/sell payoff slopes while preserving p1504's directional forecast increases the fixed net-of-cost objective relative to unmodified p1504.

## Existing theory and methods

Intraday mean reversion, T+1 constraints, cost models.

## Public data

Daily A-share data are public through Qlib; five-minute bars are not.

## What limits progress

A saturated evaluator.

## Route to a step change

Fix the metric first.

## Experiment and results

No experiment has been run for this project.

- **Why:** Five-minute A-share data are not freely redistributable, and the reward is pinned at 1.0.

## How far this can go

Unknown.

## Next steps

Ask HFM to uncap the reward.
