# Intraday Trading Overlay for A-Share Holdings

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_ashare_intraday` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in on public crypto minute data has run. The A-share task itself needs licensed data, and its reward is pinned at 1.0.
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

A stand-in on public data with the task's shape: a fixed long-only base book, intraday tilts around it, a cost on every change, and the tilt closed at the end of each day. Script: [`overlay_floor.py`](overlay_floor.py); log in `research/compute/results/overlay-floor/`. It is not comparable to HFM's score.

- **Data:** every public trade of 24 USDT perpetuals on Bybit, 1 April – 21 June 2024, aggregated to minute closes. Fitting on the first 61 days (after one day of history), test on the last 21.
- **Overlay:** every five minutes, a self-financing tilt proportional to a forecast of the next five-minute relative return, capped at the base weight so the book stays long-only. Forecast: ridge regression on the last 5-, 30- and 120-minute relative returns and the volume surprise. HFM's current best idea is ten times stronger shrinkage, so the penalty chosen by cross-validation is compared with 10× and 100× it, and with a plain short-term-reversal sign rule. Scale and a no-trade band are chosen on the fitting days.

| Cost per unit turnover | CV ridge | Ridge ×10 | Ridge ×100 | Reversal sign |
|---|---|---|---|---|
| 0 bp | +73.6 | +75.0 | +74.2 | +79.6 |
| 2 bp | −6.8 | −4.6 | −6.7 | 0 (does not trade) |
| 5 bp | −19.8 | −16.4 | −15.8 | 0 (does not trade) |

(test net P&L of the overlay in basis points of the book per day; standard errors 4.7–12.9; before costs, 95–100% of test days are positive)

- **The signal is real and is all short-term reversal.** Before costs every forecast earns about 75 bp a day, on nearly every test day; the five-minute return carries almost all the weight (−1.12 bp per standard deviation, against −0.22 for 30 minutes).
- **Two basis points of cost wipe it out.** No overlay is profitable on the test days at 2 bp or 5 bp per unit turnover; the best choice is not to trade. The sign rule's fitted band is wide enough that it never trades.
- **Stronger shrinkage does not change the verdict.** Ten times the penalty is 2.2 bp a day better than the cross-validated one at 2 bp cost and 3.4 bp at 5 bp, both inside one standard error, and still negative.
- **For HFM's track:** the question is not the forecast but whether any forecast survives costs at the trading frequency allowed. A reward pinned at 1.0 cannot tell "trades well" from "does not trade"; it needs to be reported in basis points net of costs, with a no-trade benchmark.

## How far this can go

Unknown for A-shares, whose costs (stamp duty, T+1 settlement against the base book) are higher than the ones tested here. On public crypto data, intraday reversal is strong before costs and gone after 2 bp.

## Next steps

Ask HFM to uncap the reward and publish a no-trade benchmark. Before tuning shrinkage, measure the gross-to-net gap: at this frequency it decides the outcome.
