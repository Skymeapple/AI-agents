# Frozen baseline, quality bar and cost indicators for scaling `discover()`

Frozen **before** the held-out verification round. The thresholds below were
chosen after an exploratory pilot on seed-20261025 corpora; the verification
round uses corpora generated with a different seed (4242) that the method was
never run on during development.

## Task

`discover()` clusters works into research directions with exact average-linkage
agglomeration over a dense n-by-n similarity matrix, and refuses corpora above
12,000 works (README limitation 3). The goal is to remove that ceiling without
losing quality and without adding dependencies or network.

**10x target:** 120,000 works (10x the ceiling) inside an 8 GB memory budget, at
the quality bar below.

## Baselines

| Baseline | What it is |
|---|---|
| B0: `discover()` at commit `25d0e47` | Shipped exact method. Run from a git worktree at that commit. |
| B1: `discover_by_discipline()` at `25d0e47` | The README's recommended route above 12k works. |
| B2: exact reference (`bench/exact_reference.py`) | Exact average linkage at any size that fits in memory: ground truth for the sampled path. |

Pilot measurements of B0 on the seed-20261025 stress corpus (4 cores, 15 GB):

| n | wall s | peak RSS MB | clusters | recovered | ARI vs labels |
|---|---|---|---|---|---|
| 1,200 | 0.31 | 83 | 16 | 16/16 | 0.995 |
| 3,000 | 1.32 | 278 | 16 | 16/16 | 0.995 |
| 6,000 | 5.51 | 933 | 24 | 16/16 | 0.992 |
| 12,000 | 20.46 | 3,478 | 24 | 16/16 | 0.992 |
| 30,000 | refuses | (would need ~22 GB for its three n² matrices) | | | |

Profile at 12k: 92% of time in the merge loop, 6% in the similarity product,
2% in TF-IDF.

## Quality requirements (all must hold)

- **Q1 (exact path, n ≤ 12,000):** partition byte-identical to B0. Checked by
  SHA-256 fingerprint of the sorted partition, on every corpus where B0 runs.
- **Q2 (sampled path, n > 12,000):**
  - Q2a: every planted direction recovered, including a rare one planted at
    about 0.02% of the corpus.
  - Q2b: agreement with the exact reference B2 has ARI ≥ 0.97 wherever B2 fits
    in memory (30k).
  - Q2c: ARI against the planted labels is no more than 0.01 below B2's.
  - Q2d: deterministic. Two runs give the same fingerprint.
- **Q3:** full test suite passes, no new dependencies, CLI and public signatures
  stay backwards compatible.

## Full cost indicators

| Indicator | How measured |
|---|---|
| Wall time of `discover()` | `time.perf_counter`, corpus loading excluded |
| CPU time | `time.process_time` |
| Peak resident memory | `ru_maxrss` of a fresh process per run (includes the loaded corpus) |
| Capacity | Largest n that completes inside the 8 GB budget |
| Code cost | Lines changed in `sciscout/`, new dependencies |
| Exactness cost | Whether output is identical (Q1) or approximate (Q2) |
| Engineering time | Reported honestly in the results |
