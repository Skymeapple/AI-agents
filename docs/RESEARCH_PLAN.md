# Research plan: a self-iterating research agent

## Aim

Build and study an agent that keeps itself current with a fast-moving field
(research on AI agents), and improves how well and how cheaply it does so,
by iterating on itself automatically, with every change measured.

"Improves itself" is easy to claim and hard to verify. The project's working
rule is the same as sciscout's: **every change is measured, not asserted.**
The agent may change itself only through a bounded configuration, only when a
paired statistical test on a labelled benchmark says the change helps, and
only after a holdout split agrees. A third, sealed split is never read by any
decision, so the gap between what the agent optimises and what it achieves
stays visible.

## System (as built)

```
gather ─▶ triage ─▶ learn ─▶ propose ─▶ evaluate ─▶ gate ─▶ promote ─▶ record
arXiv or   score by   term      mutation /   paired      tune, then   champion   history,
replayed   topic      stats,    literature / bootstrap   holdout      config     digest,
stream     profiles   mining    Claude       on tune     (1/cycle)               report, PR
```

| Component | Module | What it is |
| --- | --- | --- |
| Configuration | `autoscout/config.py` | Bounded numeric parameters plus per-topic term profiles. The only thing the agent may change. |
| Gathering | `autoscout/scout.py` | arXiv queries built from the current profiles, or a replayed stream offline. |
| Triage | `autoscout/triage.py` | Weighted term matching, length-normalised, with a flagging threshold. |
| Learning | `autoscout/mining.py` | Log-odds term statistics over confidently triaged papers, accumulated across cycles. |
| Proposers | `autoscout/proposers.py` | Mutation (1/5-rule step size), literature (mined terms), Claude (optional). A Thompson bandit allocates the budget. |
| Benchmarks | `autoscout/bench.py` | Per-item triage utility and discovery placement; tune/holdout/sealed splits; human-labelled real papers. |
| Gate | `autoscout/gate.py` | Paired bootstrap lower bound on tune, non-inferiority elsewhere, then holdout. |
| Automation | `.github/workflows/autoscout.yml` | Weekly live cycle, opened as a pull request for human review. |

## Research questions

**RQ1: Does learning from the newest literature beat tuning alone?**
Hypothesis: under vocabulary drift, parameter search levels off, while
vocabulary learned from new papers keeps recall up.
*Measure:* sealed utility share and online recall over the last four weeks,
`mutation` vs `literature` vs `full` arms.

**RQ2: What does the statistical gate buy?**
Hypothesis: without the bootstrap bound and holdout, the agent accepts lucky
changes and tune diverges from sealed.
*Measure:* the tune−sealed gap and sealed score, `full` vs `full-naive-gate`,
especially as benchmark size shrinks.

**RQ3: Does learning which proposer to use improve research efficiency?**
Hypothesis: bandit allocation raises promotions per evaluated candidate
compared with uniform allocation.
*Measure:* promotions ÷ candidates evaluated; candidates needed to reach a
target sealed score.

**RQ4: Does an LLM proposer add anything the cheap proposers don't?**
Hypothesis: Claude, reading the digest and experiment history, proposes
changes that combine parameter and vocabulary edits and pass the gate more
often per candidate, at a token cost worth reporting alongside.
*Measure:* pass rate per candidate, promotions per 1M tokens, and the ideas
backlog judged by a human.

**RQ5: Do gains on the synthetic benchmark carry over to real papers?**
The one question synthetic data cannot answer. *Measure:* the `triage-real`
benchmark built from human labels (`autoscout queue` / `label`), once it has
enough items.

## First results (synthetic stream, planted drift)

From `scripts/run_ablation.py`, 12 cycles, 5 seeds. Full table:
[`examples/autoscout-ablation.md`](../examples/autoscout-ablation.md).

| arm | sealed utility share | online recall, last 4 weeks |
| --- | --- | --- |
| frozen (no iteration) | 0.735 | — |
| mutation only | 0.877 | 0.53 |
| literature only | 1.000 | 0.99 |
| full (bandit) | 0.982 | 0.97 |
| full, naive gate | 0.993 | 0.99 |

What this does and does not show:

1. **RQ1 is supported, on synthetic data.** Tuning thresholds recovers some
   ground, but it cannot recognise words the agent has never seen, and online
   recall collapses as the vocabulary drifts. Learning vocabulary from the
   newest papers keeps up.
2. **The full agent is slightly worse than literature-only.** Only one change
   is promoted per cycle, so a mutation win spends a cycle that a vocabulary
   addition could have used. The bottleneck is promotion throughput, not
   proposal quality. Next experiment: stacking compatible winners within a
   cycle, each re-verified against the updated champion.
3. **RQ2 is not supported here, and that is informative.** On a 900-item
   benchmark with a ten-parameter space there is little noise for a naive gate
   to overfit, and the strict gate's caution costs a little. The strict gate
   is designed for the regime the project is heading into: a small,
   human-labelled real benchmark, where each item swings the mean noticeably.
   The next experiment subsamples the benchmark (50–200 items) to find where
   the two gates cross.
4. **Some learned terms landed under the wrong topic** (e.g. `token-budget`
   under self-improvement). The gate lets through changes that don't hurt,
   and a misfiled term that rarely fires doesn't hurt measurably. A topic-purity
   check on mined terms is a candidate fix.
5. **Discovery is at ceiling** on this corpus (≈0.99), so it acts only as a
   regression guard. It needs a harder benchmark before it can say anything.

**Threats to validity.** The generator and the agenda were written by the same
author; the drift was planted; "emerging" terms are cleanly separable in a way
real terminology is not. Every number above tests mechanism, not real-world
performance. The discovery bootstrap treats the clustering as fixed and so
understates uncertainty.

## Roadmap

| Milestone | Work | Exit criterion |
| --- | --- | --- |
| M1: loop | Done: gather → learn → propose → gate → promote, offline and live; ablation harness; CI | Ablation reproducible from one command |
| M2: real benchmark | Run weekly live cycles; label 40+ real papers via the queue; `triage-real` gates live changes | 200 labelled real papers; RQ5 first answer |
| M3: gate under scarcity | Subsample benchmark sizes; stacking of compatible winners; topic-purity check for mined terms | RQ2 answered on small benchmarks |
| M4: LLM proposer study | Enable the Claude proposer in CI; log tokens per promotion; review the ideas backlog | RQ4 answered with cost |
| M5: structural change | Promote the best backlog ideas (e.g. embeddings for triage, citation signals from OpenAlex) into code by human-reviewed PRs, and re-run the ablation | Each structural change justified by the same benchmarks |

## Guardrails

- **Configuration-only self-modification.** Code changes go to `ideas.md` for
  a person. Every value an LLM returns is clipped to the declared space;
  anything outside it is rejected and logged.
- **Human checkpoint.** Scheduled runs open a pull request; nothing reaches
  `main` without a merge.
- **No synthetic steering of the live agent.** In live mode, self-modification
  stays paused until the real-paper benchmark exists.
- **Reproducibility.** Given the same state directory, stream and seed, a cycle
  makes the same decisions; every candidate, its diff and its confidence
  interval are in `history.jsonl`.
