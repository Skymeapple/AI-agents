# Cross-Venue Allocation of Perpetual Orders

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_cross_venue_routing` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## Readiness

- **State:** Data verified; pipeline or simulator still to build.
- **Data access:** Binance archive used elsewhere here. Bybit public archive and OKX market-data API: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** The three-venue replay and cost model.
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

Binance, Bybit and OKX all publish historical trade and book data.

## What limits progress

Residual-tree tuning with gains about one part in 100,000.

## Route to a step change

A venue-cost model solved as a small convex allocation per order.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs synchronised order-book data from three venues; Binance is public, Bybit and OKX archives were not fetched.

## How far this can go

Bounded by hindsight-best venue per order.

## Next steps

Fetch the three venues' archives and compute the hindsight bound.
