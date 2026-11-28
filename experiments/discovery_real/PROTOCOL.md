# Pre-registered protocol: AI-assisted research-direction discovery on real titles

Frozen before any AI-arm result was computed. The SHA-256 of this file is
recorded in `FROZEN.txt` together with the data hashes. Any change after
freezing must be logged as a deviation in `RESULTS.md`.

## 1. Task (real-world, from sciscout)

`sciscout discover`: given N unlabelled research works, deliver a map of research
directions (clusters) that an analyst can put in a report. The README states the
open question this answers: discovery is validated only on a synthetic corpus whose
vocabulary was written alongside the classifier, and "embeddings would do better"
is asserted, not measured.

## 2. Data (real, labelled only for scoring)

| Id | Source | Works | Directions | Text |
|---|---|---|---|---|
| BIO | STC2 benchmark `Biomedical` (Xu et al. 2017; PubMed titles, BioASQ MeSH topics), github.com/jacoxu/STC2 | 20,000 | 20 | titles |
| MAG | SciBERT `mag` test split (Microsoft Academic Graph fields), github.com/allenai/scibert | 22,399 | 7 | titles |
| DEMO | sciscout `data/demo_corpus.json` (synthetic) | 1,174 | 16 | title+abstract |

DEMO is a regression check only: an arm must not do worse there than the shipped method.

Sampling: dense agglomerative clustering is capped at 12,000 works, so each
replicate is a stratified random sample of **6,000** works (BIO 300/class, MAG
~857/class). Replicate seeds **0-4**. A disjoint **tuning sample** (BIO 100/class,
MAG 300/class; drawn first with seed 100 and excluded from the replicates) is the
only data on which any hyperparameter may be tuned. Every arm gets the same tuning
grid size (8 values) on the same tuning sample.

## 3. Arms

Strong baselines (frozen at commit `25d0e47`):

- **B0** `sciscout.discovery.discover` as shipped (threshold 0.16, min cluster 3).
- **B1** same code, threshold tuned on the tuning sample over
  {0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25}.
- **B2** classical strong baseline from the short-text clustering literature:
  sciscout's own TF-IDF -> LSA (100 dims, SVD) -> L2 -> k-means (10 restarts),
  k tuned over {5, 7, 10, 15, 20, 25, 30, 40}.

AI arms (WordLlama `l2_supercat` 256-d embeddings, distilled from Llama-2-70B
token embeddings, bundled in the PyPI wheel, run offline):

- **A1** embeddings + sciscout's own average-linkage agglomeration, threshold tuned
  over {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}.
- **A2** embeddings -> k-means, k tuned over the B2 grid.
- **A3** hybrid: [LSA-100 ; embeddings], each block L2-normalised and weighted
  equally -> k-means, k tuned over the B2 grid.

Tuning objective for all arms: minimise the total cost defined in section 5 on the
tuning sample. That is the quantity being compared, so tuning on it is fair to all.

Best strong baseline = whichever of B0-B2 has the lowest mean total cost on the
replicates. It is chosen after the baselines run and before the AI arms run, and
then frozen.

## 4. Quality requirement (fixed, identical for all arms)

The delivered map must have **at least 90% of works correctly assigned** (Q90).
Correct means: in a direction whose majority true label equals the work's label.
Clusters sharing a majority label count as merged by the analyst during review.
Unassigned works count as incorrect until an analyst assigns them.

Secondary: Q95. Also reported, but not used for decisions: NMI, ARI, Hungarian
ACC, coverage, cluster count, and sciscout's own `recovery_rate` / `mean_purity`.
Those two can be gamed by many tiny pure clusters, which is why they are not the
quality bar.

## 5. Full cost indicators

Total cost to deliver a Q90 map = machine cost + analyst cost.

Analyst workflow, modelled with label-based simulation:
1. Review and name every machine cluster: `t_c` = 120 s per cluster.
2. Screen works from least to most confident. Confidence = cosine to the assigned
   cluster centroid in that arm's own space; unassigned works come first. Each
   screened work costs `t_s` = 4 s. Each screened work that is wrong is fixed for
   an extra `t_r` = 10 s. Screening stops as soon as the remaining errors are at or
   below 10% of N.

Cost rates: analyst $60/h. Compute $0.10 per core-hour of measured wall-clock
(this container: 4 vCPU; charged as 4 cores). All time constants are **assumed**
(judgement, not measured), and section 7 tests how sensitive the result is to them.

Also reported, though not part of the total: wall-clock (s), peak RSS (MB), extra
dependency size (MB), network needed at run time (yes/no), labelled data needed.

Human-only reference: an analyst codes every title manually at 12 s/title, plus
naming the true number of directions.

## 6. Decision rules

- **Real gain**: the best AI arm's total cost is below the best strong baseline's
  on all 5 replicates of BIO, and its mean is lower, with a 95% bootstrap CI of the
  paired cost ratio excluding 1.0. The same check is reported for MAG.
- **Gain factor** G = cost(best baseline) / cost(best AI). **Distance to 10x** = 10 / G.
- **DEMO regression**: the AI arm's recovery and purity at its tuned setting must
  not fall below B0 (16/16, 1.00).
- **Restrictions**: the cost breakdown by stage (machine, cluster review, screening,
  fixing) shows which stage caps G (Amdahl). Report the ceiling G would reach if
  each stage were driven to zero.

## 7. Sensitivity

Halve and double each of `t_c`, `t_s` and `t_r` one at a time, and report G under
each.
