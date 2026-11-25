# Minute-Level Perpetual Trading Strategy from Price, Flow, and Funding Data

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_perpetual_minute_strategy` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Binance public archive.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable trading policy using minute-level price bars, signed taker flow, and funding records. The research method and portfolio decision process are chosen to maximize a fixed net-of-cost score, without prescribing a specific trading style.

- **Metric:** `reward` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.0776
- **HFM effort:** 14 proposals, 12 scored, 3 accepted improvements, about $234 of compute
- **HFM's current best idea:** Giving post-settlement jump exits exactly one zero-offset passive fill opportunity before taker cleanup increases net-of-cost reward over p801 with taker entries and all funding signals unchanged.

## Existing theory and methods

Funding-rate carry, cross-sectional momentum and reversal, transaction-cost modelling, out-of-sample and deflated-Sharpe evaluation.

## Public data

Binance public data archive: hourly bars with taker flow, and funding rates, 24 contracts, 2022-2025.

## What limits progress

Signals are weak relative to trading costs, and tuned settings do not persist.

## Route to a step change

Fix one hypothesis family and judge it only on a later period.

## Experiment and results

Code: [`perp_funding_carry.py`](perp_funding_carry.py) in this folder. Run instructions are in its header.

- **Setup:** Binance's public archive, 24 large USD-margined perpetual
  contracts, 2022–2025, 8-hour rebalancing. 36 market-neutral settings
  from three signal families (funding carry, taker-flow reversal, price
  reversal) are ranked on 2022–2023 and then scored on 2024–2025, net of
  4 basis points per unit of turnover.
- **Result:**
  - **Families persist.** Funding carry is the only family with
    profitable settings in either period. Flow and price reversal lose
    heavily in both, because trading costs exceed the signal.
  - **6 of the 8 settings that were profitable in 2022–2023 stayed
    profitable in 2024–2025.** The rank correlation between the two
    periods is +0.91 across all settings and +0.63 inside the
    funding-carry family.
  - **The single best setting is not reliable.** The best on development
    went from a Sharpe ratio of 0.79 to 0.63. The second best went from
    0.62 to −0.04.
- **What this shows:** on this small, pre-declared grid the
  later-period check confirms the family and not the fine tuning. That
  is the rule Section 10 proposes. It is a study of evaluation, not a
  trading result.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Funding carry earned a Sharpe ratio of roughly 0.4 to 1.6 across settings in 2024-2025 at 8-hour rebalancing; flow and price reversal lose money after costs. Nothing here suggests a large edge.

## Next steps

Repeat at one-minute resolution with passive execution, which is where HFM's agents report gains.
