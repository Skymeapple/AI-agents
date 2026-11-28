# Execution Policy Development from Ten-Level Order Book Data

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_nasdaq_depth_execution` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in on public crypto order books has run. The Nasdaq task itself needs licensed data, and HFM's reward is pinned at 1.0.
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

Data: Bybit's public archives for four USDT perpetuals (BTC, ETH, SOL, XRP), 3–5 June 2024: full 500-level order-book snapshots and deltas (quote-saver.bycsi.com) and every public trade (public.bybit.com). The book is rebuilt and sampled once a second at ten levels. Fitting uses 3–4 June; the test is 5 June. Script: [`bybit_book_floor.py`](../order-book-depletion/bybit_book_floor.py); logs in `research/compute/results/bybit-book-*`. Licensed Nasdaq data (Databento) is not used, and LOBSTER's free equity samples now require an application. Crypto perpetuals trade around the clock with no auction and, for BTC and ETH, a tick far below one basis point, so this is an analogue, not a replica, and is not comparable to HFM's score.

Task: buy or sell a fixed notional within five minutes; parents start every five minutes on the test day (568 per instrument and size). Cost is implementation shortfall against the arrival mid, in basis points (positive = cost). Policies: sweep (all at once), TWAP (ten market slices), passive (each slice rests at the near touch for 30 seconds, then crosses for the rest; a resting order fills only from later trades at or through its price beyond the size queued ahead), adaptive (as passive, but crosses at once when the three-level imbalance predicts the price moving away; threshold fitted on the fitting days).

| Instrument, notional | Sweep | TWAP | Passive | Adaptive | Adaptive minus TWAP saving |
|---|---|---|---|---|---|
| BTCUSDT, $20k | +0.05 | +0.01 | +0.86 | +0.25 | −0.24 |
| BTCUSDT, $200k | +0.83 | +0.05 | +0.87 | +0.29 | −0.25 |
| ETHUSDT, $200k | +1.92 | +0.17 | +0.91 | +0.44 | −0.28 |
| SOLUSDT, $200k | +1.36 | +0.45 | +1.14 | +0.68 | −0.23 |
| XRPUSDT, $20k | +1.21 | +0.97 | +1.14 | **+0.90** | **+0.07** (se 0.02) |
| XRPUSDT, $200k | +3.28 | +1.19 | +1.31 | **+1.17** | +0.02 (se 0.02) |

(mean cost in bp; standard errors of the policy differences are 0.01–0.04 bp, of the levels 0.2–0.4 bp because the price moves during each parent)

- **Resting at the touch loses where the spread is below a tick's worth of value.** On BTC, ETH and SOL, the passive policy costs 0.7–0.85 bp more than TWAP and is worse in 74–95% of parents. There is no half-spread to earn, and resting orders fill mainly when the price is about to move through them, while unfilled slices then cross at a worse price. This is adverse selection, and it shows up even with a fill model that gives the resting order every trade beyond its queue.
- **The imbalance rule recovers most of that loss but not all.** Crossing whenever the book leans against the order cuts the passive penalty by about two-thirds on the tight books; the fitted threshold is always zero (cross on any adverse lean).
- **Passive working only pays on the wide-spread instrument.** On XRP (1.9 bp spread) the adaptive policy beats TWAP by 0.07 bp for small parents, a reliable but very small saving, and nothing measurable for large ones.
- **Slicing matters for size.** A $200k sweep costs 0.8–3.3 bp; TWAP brings that to 0.05–1.2 bp.
- **For HFM's track:** savings from order placement are bounded by the spread and by adverse selection, and are a fraction of a basis point here. HFM's capped reward cannot show differences of that size; a reward in basis points with standard errors is needed first.

## How far this can go

Unknown for Nasdaq stocks. On public crypto books, placement logic moves costs by tenths of a basis point; slicing large orders matters far more.

## Next steps

Ask HFM to uncap the reward and report cost in basis points with standard errors. On public data: a queue-position model fitted from the message stream instead of the conservative back-of-queue rule.
