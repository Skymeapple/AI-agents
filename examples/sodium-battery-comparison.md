# Scenario comparison: Solid-state sodium batteries

_20,000 draws per scenario, shared seed 20260918 so differences reflect assumptions rather than sampling noise._

| Scenario | P(market) | NPV P10 | NPV P50 | NPV P90 | E[MOIC] | MOIC if it works | P(loss) | Spread from gate luck | Of the rest, from assumptions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conservative | 31% | -$547M | -$141M | -$4M | 1.11x | 3.61x | 95% | 31% | 48% |
| Base case | 31% | -$414M | -$52M | $1.16B | 7.75x | 25.20x | 74% | 50% | 56% |
| Breakthrough | 31% | -$392M | -$44M | $17.41B | 91.13x | 296.42x | 69% | 46% | 48% |

- **Conservative** — Sodium competes only where lithium is supply-constrained, and scale-up costs run at the high end of the prior. This is the case to beat: if the investment still works here, the thesis does not depend on optimism.
- **Base case** — Sodium takes a real but bounded share of stationary storage, and scale-up lands near the middle of the shipped prior.
- **Breakthrough** — Cycle life and cost land well enough that sodium displaces lithium in stationary storage broadly. Note what this case does NOT do: it leaves the technical success probabilities at their priors. Better science does not make scale-up more likely to work, it makes the prize bigger.

Expected multiple ranges from **1.11x** (Conservative) to **91.13x** (Breakthrough). The gap between scenarios is the part of the answer that comes from your assumptions rather than from the science, so it is the first thing to narrow if the decision is close.

Difference between them: +90.02x, 90% sampling interval +86.63x to +93.42x — resolved at this draw count.
