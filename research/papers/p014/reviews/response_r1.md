# Response to Round-1 Review

We thank the reviewer for a careful report. The reviewer reran our code bit-exactly and computed independent baselines, and both were decisive for this revision. We accept most criticisms. Where the evidence required it, we retracted or narrowed claims rather than defending them.

Every new number comes from code we wrote and ran in this environment, with outputs saved under `results/`; each item below names its output files. The previous version is preserved as `reviews/paper_r0.md`, and the round-0 result JSONs are in `results/backup_r0/`.

## Summary of changes

| Issue | Verdict | What changed |
|---|---|---|
| M1: the Gaussian twin 50% is a theorem | Accepted | The figure is demoted to a "unit test" (Section 5.1; last column of Table 1) and removed from the abstract and the claimed findings. Its CI is no longer reported as evidence. |
| M2: the "observational limit" was mis-calibrated | Accepted. Claim retracted and redone | Exact Bayes baselines under our prior: MEC-only 86.3% (reviewer: 86.4%); correlation matrix, label-free, 90.1–91.4%; covariance 98.5–99.2%. The shortcut claim is redone against the correct bound, both per seed and on identical items. |
| M3: out-of-prior twins confound the surplus | Accepted. Controls added | Three new probes: direction-balanced, in-prior twins, and any MEC member including non-adjacent queries. Also an m2/m4 feature ablation and prior-agnostic classical reasoners. The surplus is *not* lower on in-prior twins. |
| M4: novelty and missing prior work | Accepted | Props 1–3 are labelled as restatements of known results. Montagna et al. (2024), Reisach et al. (2023), and Hyvärinen & Smith (2013) are now cited, and the relation to Corr2Cause is stated. Contributions are repositioned around calibrated baselines and controls. |
| M5: the research question is not addressed, and "pre-registered" is wrong | Partly addressed. Claims narrowed, terminology fixed | LLM and AVICI weights could not be obtained, because the model hubs are unreachable (evidence below). We added a second, stronger amortized test subject. Scope is stated plainly. "Pre-registered" is now "pre-specified (not registered)". |
| Minor 1: adjacent-only twins | Fixed | General MEC refit (verified to 1.4e-13). Non-adjacent queries are now tested and reported separately. |
| Minor 2: few seeds, no κ CIs | Fixed | 6 new DeepSets runs plus 14 pooled-moment runs. κ gets paired-bootstrap CIs and across-seed s.d. The 32→128 trend claim is withdrawn for DeepSets. |
| Minor 3: no ablations; obs_p0 is weak | Addressed | Second test subject (80.4% vs DeepSets 74.6%). m2/m4 feature ablation. Weakness stated against the Bayes bound. |
| Minor 4: standardization and varsortability | Fixed and quantified | Text corrected, R²-sortability cited. The R-vs-MEC Bayes gap (82–84% vs 75% on unidentified queries) quantifies the scale-invariant signal that remains. |
| Minor 5: "oracle" / "ceiling" wording | Fixed | Now "MEC + label-order reference"; "81% ceiling" removed. |
| Minor 6: smoking/cancer example | Withdrawn | Section 7 explains why. |
| Minor 7: `summarize.py` glob and `int8_p0` | Fixed | Glob is now `*_s[0-9].json`; `int8_p0` removed. |

## Point-by-point

### M1. "The headline empirical result is a theorem, not a finding."

**Agreed.** The 50% Gaussian twin accuracy holds for every function of the input, including an untrained network, so it carries no information about learned reasoners.

Changes:
- The abstract no longer mentions it as a result.
- Section 3.2 adds a "Consequence for reporting" paragraph that says exactly this.
- Section 5.1 is retitled "Unit tests of the generator (not findings)".
- The twin column of Table 1 is labelled "unit test".
- Section 6 now shows the point concretely: two test subjects whose i.i.d. accuracies differ by 6 points both score 1/2 on Gaussian twins. Twins measure evidence use, not inference quality.

The empirical content of the paper now rests on (i) i.i.d. accuracy against calibrated Bayes bounds and (ii) κ under informative evidence with controls.

### M2. "The 'observational limit' oracle is mis-calibrated."

**Agreed; the claim is retracted and redone.**

**What we computed** (`code/bayes_oracle.py` → `results/bayes_oracles.json`). For every i.i.d. test item (regenerated with the same RNG stream, so these are the exact items the networks were scored on) we compute exact posterior-optimal answers under our prior. Three reasoners know, respectively:
- (a) the population MEC;
- (b) the population correlation matrix R;
- (c) the population covariance Σ.

Each is computed label-free and label-aware. The within-MEC density uses the regression parameterization and its Jacobian, with diagonal scales integrated out for R (Section 4.4). Posteriors are calibrated: binned predicted vs. empirical P(yes) agree to within 0.02.

**Results.**
- (a) **86.3%** on the reviewer's items (the reviewer computed 86.4%) and 85.9–87.6% across item sets.
- (b) **90.1–91.4%**.
- (c) **98.5–99.2%**.

The reviewer's suspicion that "the true observational limit ... possibly [exceeds 86.4%] by a lot" is confirmed. Because the per-column rescaling cancels, standardized samples depend on the model only through R. So by data processing, (b) label-free is a valid upper bound for every permutation-equivariant reasoner on standardized data, and for every reasoner when labels are random. It is a population bound, so it is conservative for finite N.

**Redone claim (Section 5.2).** All comparisons are per seed and on identical items.
- **DeepSets, p = 0.9, overall:** does **not** exceed the bound (89.5 vs 90.2; 90.1 vs 91.3). The round-0 claim was wrong here.
- **DeepSets, p = 1, overall:** exceeds it by 1.3–3.1 points (92.3/92.7/93.2 vs 90.6/91.4/90.1). We state that the smallest margin is only about two standard errors.
- **MEC-unidentified queries:** every label-order run exceeds the bound. DeepSets exceeds it by +4 to +11 points; the new pooled-moment models by +6 to +12 points (Table 1).
- **Pooled-moment models, overall:** exceed the bound at both p = 0.9 (+2.1 to +2.6) and p = 1 (+5.4 to +5.5).
- **Cue removal** (new; `results/cue_removal.json`, `results/strong/*.json`):
  - With random labels, label-order models fall to 59.7–64.9%, with 50–52% on unidentified queries.
  - DeepSets with reversed labels falls to 26–27%.
  - Both are below the p = 0 models, so these models depend on the cue.

The phrases "past what any reasoner could achieve from observational data" and "the excess can only come from the label-order cue" are now made only for the unidentified-query comparison, and for overall accuracy only where the per-seed data support it.

**New finding that this enabled.** The simulation prior nearly identifies direction within the MEC: 98.5–99.2% from Σ. Accuracy above the MEC rate on "unidentified" queries is therefore not evidence of a shortcut. This is now Contribution 2 and Section 6. We quantify Montagna et al.'s "implicit prior" point for one standard prior.

### M3. "Out-of-prior twins confound the surplus results."

**Agreed that the confound needed testing.** We tested it, and the data do not support the confound. Code: `twins.py` (`make_general_twins`), `strong_model.py`, `rev_baselines.py`.

**1. In-prior control.**
- We added probes whose refit twin M′ is rejected unless it lies in the prior's support, so both members are plausible training draws.
- If surplus came from detecting out-of-prior parameterizations, κ would *drop* on these twins. Instead κ is higher on in-prior covered twins in 6 of 8 DeepSets runs and all 6 pooled-moment runs. The two exceptions (int32 s1, int128 s2) have overlapping CIs.
- For MEC twins, κ is higher in 13 of 14 runs and equal to within 0.001 in the last.
- Example, DeepSets Laplace: κ = 0.850 ± 0.002 on all covered twins vs **0.964 ± 0.010** on in-prior twins.
- Explanation: out-of-prior members are harder for the network (covariate shift), not a shortcut.

**2. Direction balance.** In round 0 the in-prior member was always the "yes" item. The new probes randomize the queried direction. Accuracy is essentially the same whether the in-prior member is the "yes" or the "no" item (e.g. 92.0/92.6%).

**3. Feature ablation.**
- A pooled network given only second-order statistics (m2) on Laplace data captures κ = −0.01 to +0.01.
- The same architecture with third- and fourth-order cumulants (m4) captures 0.79 on all covered twins and 0.92 on in-prior twins.
- We say explicitly that the m2 result is close to automatic, because twins share Σ exactly. It is a negative control showing that the surplus lives in the higher cumulants on which LiNGAM identifiability rests.

**4. Prior-agnostic classical reasoners.** These know nothing about the parameter prior: pairwise LiNGAM LR on Laplace twins, and a CI test on interventional twins. They reach κ = 0.99–1.00 and 0.90–0.95, so the separating information is available as a causal-direction signal.

**Wording changes.** "The network has learned to exploit LiNGAM identifiability" is replaced. The claim is now that the networks use the non-Gaussianity that identifies direction, and that the surplus is not explained by prior-support detection. We state that this does not show the networks implement a LiNGAM-like algorithm.

We also agree that in non-Gaussian and interventional regimes the pairs are not twins. They are "probes with a high ceiling", and Section 3.3 now says so.

### M4. "Novelty is limited and key prior work is missing."

**Agreed.**
- **Attribution in Section 3.** The heading now says "known results, restated for evaluation". Prop. 1 is labelled as a restatement of non-identifiability and the causal hierarchy theorem, Prop. 2 as Le Cam's bound, and Prop. 3 as a corollary of Chickering (1995). The exhaustive n = 4 check is described as a check of our implementation of a known theorem. We claim no theoretical novelty.
- **Montagna et al. (2024)** (verified: arXiv:2405.16924, TMLR 2025) is cited in the introduction, in related work and in Section 5.4. Our Gaussian→Laplace non-transfer result is presented as an instance of their finding, not as new.
- **Corr2Cause.** Its MEC-based labelling is discussed in Section 2. We also note what our baselines add: once a parameter prior exists, "unidentified by the MEC" is not "50/50".
- **Also cited:** Reisach et al. (2023) on R²-sortability, Hyvärinen & Smith (2013), and Hyvärinen (1998). All were verified by web search.
- **Repositioned contributions.** The paper now rests on calibrated Bayes baselines under the benchmark prior, the shortcut and surplus accounting with controls, and the protocol. These are methodological contributions at small scale, and we say so.

### M5. "The research question is not addressed empirically; 'pre-registered' is a misnomer."

**Partly addressed; we cannot fully fix this here, and we say so in the paper.**

**Why no LLM or AVICI run.** The environment has no GPU and no API key. We also tried to download open-weight models and AVICI checkpoints (`results/network_check.txt`):
- the Hugging Face hub, hf-mirror, ModelScope and the Ollama registry returned no connection;
- GitHub release assets returned 403;
- `avici` 1.0.7 (downloaded from PyPI and inspected) loads its pretrained checkpoints through `huggingface_hub.hf_hub_download`.

Running the reviewer's suggested CPU-scale LLM experiment was therefore impossible. We do not describe any LLM result.

**What we did instead.**
- We added a second, stronger amortized test subject (Section 4.3): a pooled-moment network trained on 200k items. It reaches 80.4% at p = 0 (DeepSets: 74.6%) and κ up to 0.98 on interventional twins without oracle parents, matching the classical CI-test reference.
- This shows that the instruments' conclusions (Bayes-calibrated shortcut detection, the in-prior control, the feature ablation) hold across two architectures. It also shows that ceiling capture depends strongly on architecture and seed: covered-edge κ ranges from 0.49 to 0.98.

**What the paper now says.**
- The abstract, introduction ("Scope"), Section 8 ("Relation to the research question") and Limitations all state that the paper does not answer the question for LLMs.
- "Pre-registered" is replaced by "pre-specified (not registered with any registry)" in the title of Section 7 and everywhere else.
- The protocol now requires Bayes baselines for the instance prior and a cue-removal variant.

### Minor issues

**1. Twin coverage.** Proposition 3 is generalized to a refit on *any* Markov-equivalent DAG and verified numerically over 78,481 refits (max covariance error 1.4e-13, `results/rev_checks.json`). The `rev_mec_*` probes draw any MEC-unidentified query, and 16–20% of their pairs are non-adjacent. Non-adjacent results are reported separately (Section 5.3): the surplus is positive in every run but smaller. The n = 4, linear and unconfounded setting is kept and listed as a limitation.

**2. Seeds and κ CIs.**
- New DeepSets runs: laplace s1, int32 s1 and s2, int128 s2, obs_p09 s1, obs_p1 s2 (`code/rev_runs2.sh`), on top of the first-wave reruns of laplace s0, int32 s0 and int128 s0/s1.
- Two runs listed in an earlier script (obs_p05 s1, obs_p09 s2) were dropped for CPU budget; the paper states the actual seed counts.
- Every κ now has a paired-bootstrap 95% CI per run and an across-seed s.d.
- **The reviewer's concern was right.** With three seeds, DeepSets interventional κ is seed-unstable: 0.70/0.56/0.49 (int32) and 0.83/0.85/0.53 (int128) on balanced covered twins. The round-0 "32 → 128 trend" is withdrawn for DeepSets.
- For the pooled-moment model the trend (0.87 → 0.96) is consistent across seeds but only modestly separated (CIs overlap marginally). We say this.

**3. Ablations and the weak obs_p0 model.**
- The pooled-moment subject and the m2/m4 feature ablation are new.
- The weakness is stated quantitatively: both p = 0 subjects (74.6% and 80.4%) are below the MEC-only Bayes rule and 10–16 points below the population bound.
- Both κ values are described as lower bounds on what amortized models can do.
- We did not compute the finite-sample Bayes rule. We say that the attainable accuracy at N = 256 lies between 80.4% (achieved) and 90.6% (population bound).

**4. Standardization.** Corrected: standardization removes marginal-variance cues but not scale-invariant ones (R²-sortability, Reisach et al. 2023). The Bayes comparison quantifies what is left: on unidentified queries, 82–84% from R vs. 75% from the MEC.

**5. "Oracle" / "ceiling" wording.** The MEC + label-order rule is now called a "reference". It is below the MEC-uniform rule at p = 0 and is not an oracle. The phrase "81% ceiling" is removed.

**6. Smoking/cancer example.** Withdrawn, with an explanation in Section 7.

**7. `summarize.py`.** The glob is now `*_s[0-9].json`, the dead `int8_p0` entry is removed, and the script also aggregates `results/strong/`.

## Answers to the reviewer's questions

1. **Bayes-optimal accuracy under our exact prior.**
   - (a) Given the population MEC: 86.3% (85.9–87.6% across item sets).
   - (b) Given Σ: 98.5–99.2%. Given the correlation matrix (what standardized data reveal): 90.1–91.4%.

   At p = 0.9, DeepSets does not exceed the label-free bound overall; at p = 1 it does by 1.3–3.1 points. On unidentified queries both clearly exceed it.
2. **Constraining twins to the prior.** κ goes *up*, not down (M3 above).
3. **Positioning.** Relative to Montagna et al.: our non-transfer result is an instance of theirs, and our Bayes baselines quantify their implicit-prior point. Relative to Corr2Cause: twins are its distributional analogue, and the Bayes baselines add the prior-driven component.
4. **Running an LLM.** Not possible in this environment (M5 above).
5. **Adjacent-only twins.** That was a convenience of round 0. The general MEC refit is now implemented and evaluated.

## What did not change, and why

- The theory section keeps Propositions 1–3, with attribution. We need them in this exact form to define the test.
- We do not claim results for LLMs, and we do not claim that small networks "reason causally" in any mechanistic sense.
