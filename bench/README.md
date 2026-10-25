# Scaling `discover()`: verification results

The baseline, the quality bar and the cost indicators were frozen in
[`BASELINE.md`](BASELINE.md) (commit `291b852`) before this round. Every run
below uses that commit's code, unchanged, on held-out corpora generated with
seed 4242. No development run used that seed. Reproduce with:

```bash
git worktree add --detach ../sciscout-baseline 25d0e47
bench/verify.sh ../sciscout-baseline /tmp/bench-data
```

All corpora are SYNTHETIC (`make_stress_corpus.py`): 16 planted directions with
word dropout, cross-topic bridges and shared background vocabulary. The 30k,
120k and 240k corpora also plant a rare 17th direction at 0.02% of works. The
machine has 4 cores and 15 GB of RAM.

## What changed

1. **Exact path (≤ 12,000 works): same output, lower cost.**
   - One n² matrix instead of three. Average linkage is computed from
     `pair_sum` and the cluster sizes when a row is scanned, instead of being
     stored, and the input is no longer copied.
   - Lazy row refresh. A cached best partner is an upper bound after any merge,
     so a row is rescanned only when it reaches the top of the queue.
   - Cohesion is computed from member-vector sums instead of the similarity
     block.
2. **Sampled path (> 12,000 works): replaces the refusal.**
   - Cluster a deterministic sample of 8,000 works exactly.
   - Assign every other work through `x · sum(cluster) / size`, which is its
     exact average linkage to that cluster. A work is placed only if that
     clears the merge threshold.
   - Works that matched nothing are clustered in later rounds, so rare
     directions survive.
   - Finish with exact average linkage over the cluster sums.

## Results against the frozen requirements

| Requirement | Result | Pass |
|---|---|---|
| Q1: byte-identical partition to `25d0e47` for n ≤ 12k | Identical fingerprints on demo (1,174), 6,000 and 12,000 works | ✅ |
| Q2a: all planted directions recovered, including the rare one | 17/17 at 30k, 120k and 240k | ✅ |
| Q2b: ARI vs exact reference ≥ 0.97 | 0.984 at 30k | ✅ |
| Q2c: label ARI within 0.01 of exact | 0.9885 sampled vs 0.9883 exact | ✅ |
| Q2d: deterministic | Two 120k runs give the same fingerprint | ✅ |
| Q3: tests, dependencies, compatibility | 81 passed (78 originals + 3 net new); no new dependencies; new keyword arguments have defaults | ✅ |

## Cost

| n | Method | Wall s | Peak RSS MB | Clusters | Label ARI |
|---|---|---|---|---|---|
| 12,000 | B0 shipped exact | 22.4 | 3,478 | 32 | 0.988 |
| 12,000 | new exact path | **7.2** (3.1× faster) | **1,280** (2.7× less) | 32 | 0.988 (identical) |
| 30,006 | B0 | refuses | — | — | — |
| 30,006 | B2 exact reference | 80.6 | 7,656 | 51 | 0.988 |
| 30,006 | B1 by-discipline | 5.7 | 351 | 1,964 | 0.172 |
| 30,006 | new sampled path | **4.7** | **885** | 49 | 0.989 |
| 120,024 | B1 by-discipline | 95.0 | 3,581 | 4,411 | 0.163 |
| 120,024 | new sampled path | **20.7–28.1** | **2,119** | 123 | 0.985 |
| 240,048 | new sampled path | 59.6 | 4,208 | 156 | 0.987 |

**10× target (120k works inside 8 GB at the quality bar): met.** It uses
2.1 GB, and 240k (20×) also fits, in 4.2 GB.

Code cost: one library file changed (`sciscout/discovery.py`), plus tests and
benchmark scripts. No new dependencies.

## What still stands between this and a real 10×

1. **The TF-IDF matrix is dense.** The synthetic vocabulary is about 900 terms.
   Real abstracts with `min_df = 2` will have tens of thousands, and
   120k × 50k float64 is 48 GB. On real data this, not clustering, is the next
   ceiling. It needs a sparse representation: scipy, or a hand-rolled CSR in
   numpy.
2. **Only synthetic data was tested.** OpenAlex and arXiv were blocked by this
   environment's network policy, so the gain has not been shown on real
   abstracts.
3. **Above 12k the result is approximate.** Agreement with exact is ARI 0.98,
   not 1.0. Works are placed against sample clusters as they stand, not as they
   grow.
4. **Fragment clusters grow with n.** There are 49 → 123 → 156 clusters for 16
   planted directions, and the exact reference shows the same trend (51 at 30k).
   This is a property of average linkage at a fixed threshold, not of the
   sampling. Above 100k, extra small clusters would add noise to the ranking.
5. **The exact path is still O(n²) memory**, about 1.2 GB at 12k. Lowering
   `max_exact` trades exactness for memory.

## Side finding: `discover_by_discipline` on this corpus

It splits each direction into thousands of fragments (label ARI 0.16–0.17),
far more than splitting across 2–3 disciplines would explain. The likely cause
is that `max_df_ratio = 0.5` is applied inside each discipline partition. When a
partition holds only a few directions, their defining words appear in more than
half its documents and are dropped. This is not fixed here and should be checked
on real data.
