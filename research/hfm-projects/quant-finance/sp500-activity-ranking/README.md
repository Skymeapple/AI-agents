# S&P 500 Minute Trading Activity Ranking

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_sp500_activity_rank_auto` (active on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Blocked on licensed data.
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

No experiment has been run for this project.

- **Why:** One-minute consolidated-tape bars for S&P 500 members are licensed data and not freely available.

## How far this can go

Unknown; volume surprises are partly news-driven and unpredictable.

## Next steps

Needs a data licence.
