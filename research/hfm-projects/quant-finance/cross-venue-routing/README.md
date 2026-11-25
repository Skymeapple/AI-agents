# Cross-Venue Allocation of Perpetual Orders

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_cross_venue_routing` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU, on trade prints only.
- **Data access:** Binance archive used elsewhere here. Bybit public archive and OKX market-data API: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** Order-book data for a faithful replay.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable router for execution replay across Binance, Bybit, and OKX. It uses historical and current observations to reduce net cost across parent orders, with representations, estimators, and allocation procedure chosen within a fixed single-decision interface.

- **Metric:** `reward` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.5927
- **HFM effort:** 33 proposals, 31 scored, 12 accepted improvements, about $779 of compute
- **HFM's current best idea:** Adding same-side notional intensity and backward-looking flow acceleration to p1528's residual trees lowers equal-day mean execution cost on P148-P209 relative to frozen p1528.

## Existing theory and methods

Smart order routing, venue cost models with adverse selection, convex allocation.

## Public data

Binance USD-M daily aggregate trades (`data.binance.vision`, CC BY-NC-SA 4.0) and Bybit daily trade files (`public.bybit.com/trading/`). No free order-book archive was found for either; OKX has none for trades.

## What limits progress

Free data has no order books, so fills and impact cannot be replayed faithfully.

## Route to a step change

A venue-cost model solved as a small convex allocation per order.

## Experiment and results

Code: [`routing_replay.py`](routing_replay.py) in this folder. It ran on free GitHub Actions compute; the log is in [`research/compute/results/cross-venue-routing/`](../../../compute/results/cross-venue-routing/run.log).

- **Setup:** public trade prints for BTCUSDT perpetuals on Binance and Bybit, 3–7 March 2025: 18 million and 24 million trades. A small order arrives every 30 seconds, alternating buy and sell, and is sent to one venue. The first day chooses one setting; the other four days (11,432 orders) are the test. OKX, HFM's third venue, has no free archive and is left out.
- **Important limit of this replay:** free data has trades only, not order books. The price an order would get is approximated by the next trade in the same direction on that venue at least 200 ms later. That is optimistic, because the print used by the router may no longer be available as a quote. Costs are relative to the average of the two venues' last prices, and fees (several basis points at retail rates) are excluded.
- **Result (cost in basis points, lower is better):**

  | Policy | Mean cost | Versus always Binance | Days better |
  | --- | --- | --- | --- |
  | Always Binance | −0.13 | — | — |
  | Always Bybit | −0.20 | −0.08 | 4/4 |
  | Venue with the cheaper last same-direction trade | **−0.83** | **−0.70** | 4/4 |
  | The same after removing the slow price offset between venues | −0.40 | −0.28 | 4/4 |
  | Hindsight (cheaper venue each time; a bound) | −0.90 | −0.77 | 4/4 |

- **What this shows:**
  - **Short-lived price gaps between the venues are real and persistent enough to route on:** the simplest rule captures 0.70 of the 0.77 basis points available in hindsight, on all four days.
  - **Removing the slow offset makes it worse,** not better. The persistent part of the gap is where the saving is.
  - **The size is small and uncertain.** 0.7 basis points is well below typical taker fees, and the next-trade proxy flatters any rule that chases the last print. With order-book data the saving could be much smaller.
  - This establishes the mechanics and a bound. It is not evidence of a tradable edge.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Hindsight bound in this replay:** 0.77 basis points per order between two venues for a small order.
- **Realistic ceiling:** lower, once real quotes replace trade prints. Larger orders would add market impact, which is where HFM's routing problem really lies and which trade prints cannot show.

## Next steps

- Repeat with order-book snapshots (Tardis offers the first day of each month free for Bybit and OKX) to replace the next-trade proxy.
- Add order size and impact, and a third venue.
- Test over more days and other contracts.
