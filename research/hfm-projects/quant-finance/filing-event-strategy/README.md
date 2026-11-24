# Event-Driven Equity Strategy Based on Financial Filing Events

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_filing_event_strategy` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## The challenge

This work develops a reusable event-driven equity policy that uses structured filing events, reported financial values, and market histories. It focuses on extracting information from structured records and XBRL values to generate event positions under fixed execution and risk rules, without access to full report text.

- **Metric:** `reward` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 1
- **HFM effort:** 68 proposals, 65 scored, 3 accepted improvements, about $1,548 of compute
- **HFM's current best idea:** Retaining only the positive part of p1657's bounded NI/assets level contribution raises pooled practice S_fix after costs relative to unchanged p1657.

## Existing theory and methods

Post-earnings-announcement drift, accruals anomaly, event-study methodology.

## Public data

SEC EDGAR financial-statement datasets (public domain); free daily prices.

## What limits progress

A saturated evaluator.

## Route to a step change

Fix the metric first: uncapped, net of costs, on a later window.

## Experiment and results

No experiment has been run for this project.

- **Why:** The reward is pinned at 1.0, so HFM's evaluation cannot rank methods. Public data exist (SEC EDGAR XBRL, free daily prices) but a replay was not built.

## How far this can go

Published anomaly returns have decayed strongly after publication.

## Next steps

Ask HFM to uncap the reward.
