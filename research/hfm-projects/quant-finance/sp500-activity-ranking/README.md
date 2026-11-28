# S&P 500 Minute Trading Activity Ranking

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_sp500_activity_rank_auto` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in on public crypto trades has run. The S&P 500 task itself needs licensed minute bars.
- **Data access:** No free legal source of one-minute consolidated bars for S&P 500 members was found.
- **Compute:** CPU.
- **Still missing:** Licensed minute bars.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Every minute, consolidated tape files a 1-minute bar for each S&P 500 member. This task ranks members by how much more they will trade over the next thirty minutes than over the last thirty—a forward activity surprise.

- **Metric:** `instance_score_mean` (maximize)
- **HFM baseline:** 0.1674
- **HFM best so far:** 0.4625
- **HFM effort:** 100 proposals, 98 scored, 7 accepted improvements, about $9,488 of compute
- **HFM's current best idea:** 在p3839的成交水平PRESS拟合中加入十个相邻收盘收益的累计价格动量协变量，能在保持原图和读出的情况下提高instance_score_mean。

## Existing theory and methods

Intraday volume seasonality, cross-sectional ranking losses, sequence models.

## Public data

No free legal source at one-minute resolution. Daily data would not match the task.

## What limits progress

Hand-made features are saturated (last gains about one part in a million).

## Route to a step change

One cross-sectional sequence model with a ranking loss.

## Experiment and results

A stand-in on public data. Script: [`activity_floor.py`](activity_floor.py); log in `research/compute/results/activity-floor/`. It is not comparable to HFM's score.

- **Data:** every public trade of 24 USDT perpetuals on Bybit (public.bybit.com), 1–21 June 2024, aggregated to one-minute quote volume ($200bn and 117 million trades in all). Fitting on 4–14 June (after three days of history), test on 15–21 June.
- **Task:** every fifth minute, rank the 24 members by log(volume over the next 30 minutes / volume over the last 30). Score: Spearman rank correlation across members per minute, averaged over 2,010 test minutes; the standard error is over the seven test days.

| Predictor | Mean Spearman | Lowest / highest test day |
|---|---|---|
| reversion (minus last 30 / previous 30) | 0.380 | 0.359 / 0.400 |
| seasonal (same time of day, earlier days) | 0.027 | −0.006 / 0.051 |
| boosted trees, nine features | 0.480 | 0.446 / 0.505 |
| boosted trees + ten-minute momentum | 0.493 | 0.465 / 0.523 |
| ridge, eight features | **0.517** | 0.498 / 0.545 |
| ridge + ten-minute momentum | **0.519** | 0.498 / 0.546 |

- **Most of the ranking comes from bursts fading.** Reversion alone gives 0.38 on every test day. Adding the burst within the last five minutes, the level against the trailing day, and absolute returns lifts a linear model to 0.52.
- **The linear model beats boosted trees** by 0.04, on all seven days. Within one instrument's own history, a simple model generalises better here.
- **HFM's current best idea, a price-momentum covariate, adds nothing measurable to the better model:** +0.002 to ridge (−0.001 to +0.005 by day). It helps the trees (+0.013, positive on all seven days) without bringing them level with ridge without it.
- **Time-of-day seasonality is nearly worthless here,** unlike in equities, where the open and close dominate intraday volume. That part of the S&P task is not represented by a market that trades around the clock.

## How far this can go

Unknown for S&P 500 members. On public crypto data, a linear model on burst and level features reaches a rank correlation of about 0.52; news-driven surprises set the rest.

## Next steps

Needs a data licence for the S&P task. Before adding covariates such as momentum, check them against a simple linear model on burst and level features, which here was the strongest method.
