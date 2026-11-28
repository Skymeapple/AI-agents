# Response to Review Round 1

We thank the reviewer for an unusually thorough review. They reran the full pipeline, read the CCAI notebook and ran additional robustness checks. Several of the issues they found were real. We have fixed all of them. We also took their central point seriously: the theory was thin, and the scale claim was weak.

The paper has been rewritten around a new main contribution, retitled "The Public Writes a Shortlist: Certification Limits and Authorship in Public-Input AI Constitutions". The round-1 audit, corrected, now supports that contribution.

## 0. Diagnosis and the new contribution

**Why round 1 was not a strong accept.** Round 1 had two problems:
- **Measurement without a test.** It measured how much the constitution moves, through designer-swap and sample-swap Jaccard distances, but had no criterion for when a seat is attributable to the public.
- **Thin theory and fragile claims.** Its theory consisted of framing devices. Its headline cross-process claim ("designer dominates at scale") depended on the multiverse and rested on ratios of ±15%.

The pipeline-integrity finding was real but narrow.

**The new question.** Do the public's votes *statistically determine* each seat? We treat each seat as a ranking claim ("statement s is in the top k"). We apply participant-bootstrap rank-confidence sets (Mogstad et al., 2024, *Review of Economic Studies*, verified) and add a procedure-free bound.

**The new results.** All were computed in this round; code and outputs are under `code/` and `results/`.

1. **On CCAI, the votes certify none of the 58 seats** at 95%. This holds for marginal and simultaneous rank intervals, and for all four scores (agree/seen, agree/decided, min-group with released groups, min-group with re-aligned groups). Only 15 statements are selected in ≥95% of participant resamples, and 149 are reliably excluded. The other 43 seats come from a 111-statement contested band (`results/certify_ccai.json`, Figure 1).
2. **Theorem 1 (new): a certification band with matching upper and lower bounds.**
   - *Lower bound.* Any per-statement level-α procedure certifies s with power at most α + (p_s − q_s)·√(m_s/(2q_s(1−q_s))). The proof is a two-point Le Cam argument using Pinsker's inequality and the χ² bound.
   - *Upper bound.* A Hoeffding procedure certifies everything beyond a band of width 4√(log(2M/α)/(2m)).
   - *Consequence.* The certifiable count depends on n only through m, the votes per statement. Applied to CCAI's realized exposure, about 34 of 58 seats are beyond *any* valid procedure at 80% power.
3. **A tested exposure law.** Across CCAI and 17 Polis conversations (n = 100–5,398, now including the 5,398-participant `march-on` that round 1 excluded), contested statements per seat fall with log m (slope −0.41, t = −4.31) and not with log n (t = 1.13). Polis agendas grow as M ∝ n^0.68, which is why headcount does not buy certification.
4. **Where the designer acts.** 89% of the disagreements among the 72 description-consistent consensus designs fall on contested statements.
5. **Design consequences.**
   - A CCAI-calibrated planning table: about 3,200 votes per statement to certify about half the seats.
   - A routing simulation. Rank-aware routing roughly doubles certified seats at 4–8× budget with no false certifications. A naive cutoff-proximity routing looks better on bootstrap stability but produces false certifications.

The last point is a methodological lesson we had not expected.

We do not call this a breakthrough. Theorem 1 is elementary, and we say so in the paper (Section 4.1, Remark 1). The contribution is identifying the governing quantity, testing it across processes, and finding, on the only public AI-constitution vote data, that no seat is certified.

## 1. Fatal issues

**F1 (non-deterministic re-aligned labels). Fixed.**
- PCA now uses `svd_solver='full'`, which is deterministic. The only remaining randomness is k-means initialization, and we report the distribution over 50 seeds (`results/ccai_realignment_by_seed.csv`).
- The numbers the reviewer flagged now read:

| Quantity | Value over 50 seeds |
|---|---|
| Re-aligned τ = 0.72 set size | 175–182 (seed 0: 181) |
| Changes | 16–28 (median 23), out of a union of 192–197 |
| Released vs re-aligned Jaccard | 0.86–0.92 (median 0.88) |
| Top-58 Jaccard | 0.71–0.76 (median 0.73) |
| Re-aligned vs overall-majority Jaccard | 0.83–0.87 |
| Separation | 0.103–0.109 |
| Positional-replica ARI | 0.83–0.96 (median 0.92) |
| Aligned ARI | −0.005 to −0.003 |

- These ranges match what the reviewer observed. The abstract no longer quotes a single label-dependent number.

**F2 (cross-conversation statistics not produced by released code). Fixed.**
- `results/opendata_summary.txt` has been deleted. Every cross-conversation statistic is now produced by released scripts:
  - `code/opendata_stats.py` → `results/opendata_stats.json`, `results/opendata_ratio_ci.csv`;
  - `code/opendata_ratio_exposure.py`;
  - `code/run_certify.py` → `results/certify_opendata.csv`;
  - `code/certify_robust.py` → `results/certify_robust.json`.
- The run order is documented in `code/README.md`.

## 2. Major issues

**M1 (designer-vs-sample comparability depends on the multiverse). Accepted; the claim has been withdrawn.**
- "Designer dominates at scale" is no longer a claim of the paper.
- We now report:
  - Monte-Carlo 95% intervals for every conversation's ratio: above 1 in 8 of 17 conversations and below 1 in 3;
  - the null consensus-family Wilcoxon test (p = 0.49) next to the full-multiverse test (p = 0.004);
  - a composition sensitivity analysis over 100 random sub-multiverses per conversation, whose 5–95% ranges span 0.79–1.87.
- The paper states plainly that multiverse composition moves the ratio as much as the conversation does (Section 6.4).
- The new main results (certification and the exposure law) do not depend on the multiverse at all. The one multiverse-based new claim, "89% of disagreements are contested," concerns *where* disagreement falls, not how large it is.

**M2 (sample-swap absorbs k-means initialization noise). Fixed and quantified.**
- The k-means seed is now fixed across bootstrap draws in both `run_ccai.py` and `run_opendata.py`.
- The CCAI swap distances are unchanged to three decimals: consensus designer 0.440, sample 0.565.
- We measured the seed-only noise floor separately: mean Jaccard distance 0.023 (maximum 0.21) across five seeds on the full data.
- On the bootstrap-duplicate concern: certification resamples participants for score estimation only. It does not re-cluster, except where noted.

**M3 (the scale regularity is weakly interpretable). Accepted.**
- The regularity is demoted and reinterpreted (Section 6.4). After the fixes, the ratio still correlates with n (ρ = 0.89) and with m (ρ = 0.59). Both enter an OLS significantly (t = 5.4 and 4.9).
- The reviewer's observation that neither component alone tracks n is now explained. The *sample* component tracks exposure (ρ = −0.66 with m, p = 0.004; ρ = −0.23 with n, n.s.). The designer component tracks neither.
- Theorem 1 predicts exactly this.
- Round 1's post-hoc exposure carve-out in prediction P1 has been replaced by a prediction stated directly in terms of m (Section 7.3).
- The non-independence of the Austrian conversations is handled in the robustness checks: keeping only one Austrian conversation gives t(log m) = −3.32.
- Our new law has a weak point of its own, which we also report: without `march-on`, and with the disagreement share added as a control, the exposure coefficient becomes non-significant (t = −1.41). See Limitations.

**M4 (theory thin relative to its framing). Accepted.**
- The new Theorem 1 replaces the round-1 framing results as the theoretical core. It gives non-asymptotic, matching-order bounds whose prediction is tested on data.
- Old P1 is kept as Proposition 2, explicitly as *exposure* to the partition and not as an effect, and the 145/210 count is no longer a headline.
- Old P2 and P3 move to Appendix A, with the caveats the reviewer raised:
  - the partition is exogenous in the construction;
  - P3's condition is vacuous on CCAI (|U| = 15);
  - the hybrid loses group parity (gap 0.18).
- The claim that P2 is "borne out on real data" has been removed. Section 6.6 now states that consensus and approval-voting designs violate JR in the *same* 6 conversations, so the violations reflect non-proportional rules generally.

**M5 (the proportional-rule finding is partly a strawman). Accepted, and tested as the reviewer requested.** `code/run_exposure_rules.py` compares count-based PAV and CC with two exposure-normalized variants across the 15 processes that have seeds:

| Variant | Seed over-representation (median ratio to pool share) | Notes |
|---|---|---|
| Count-based PAV / CC | 3.09× / 3.19× | |
| Approval-rate ranking | 1.09× | |
| Imputed-utility PAV | 0.99× | Wilcoxon p = 6×10⁻⁵ vs count-based; median Jaccard 0.95 with approval-rate ranking; violates JR in 3 of 18 |
| Inverse-propensity-weighted PAV | 2.01× | Keeps JR (0 violations) |

- This confirms the P2-exposure prediction: normalization removes the capture.
- It also shows something more useful. Post-hoc normalization cannot deliver exposure-neutrality and proportionality together on non-randomly routed data. This supports Halpern et al.'s (2023) adaptive-elicitation argument.
- The text now presents exposure capture as a property of applying count-based rules to sparse, order-dependent exports, not of proportional rules as such.

**M6 (missing related work). Added and verified by web search.**
- **Gelauff (2026, arXiv:2608.28368)**, verified. We discuss it as close in spirit to our channel framing, including its point that exported clusterings cannot be reproduced.
- **Blair, de Raaij, Procaccia, Rubin-Toles, Shah, Si & Wang, "The Structure of Bridging" (EC 2026)**, verified via the authors' PDF and the EC 2026 accepted-papers list. We discuss it at Proposition 2 as the partition-robust treatment of partition dependence.
- **Revel, Milli, Lu, Watson-Daniels & Nickel (ICML 2025, PMLR 267:51583–51613; arXiv:2503.18962)**, verified. We discuss it as close to the hybrid, which we have demoted.
- We also added and verified **Mogstad, Romano, Shaikh & Wilhelm (2024, REStud 91(1):476–518)** and **Kalyanakrishnan, Tewari, Auer & Stone (ICML 2012, 655–662)**.

**M7 (unit of analysis vs normative conclusion). Partly addressed; partly a standing limitation.**
- The new framing narrows the claim. We ask whether the public's votes determine the *pre-curation* seats, which is the step CCAI describes as public.
- Limitations now states explicitly that merging near-duplicates could make some contested seats moot. Neither the "0 certified seats" result nor the designer-swap percentages should be read as statements about the published constitution's wording.
- We could not obtain the curation step, which is not in the release.

## 3. Minor issues

| Item | Response |
|---|---|
| m1 | The bootstrap-to-full figures are now quoted per design where used: av\|seen 0.60, av\|decided 0.50, consensus mean 0.54. The incorrect "about 0.59" summary has been removed. |
| m2 | The curve-fit extrapolation has been dropped. After the fixes, the consensus fit crosses only at about 8×10⁵ and the full-multiverse fit at about 6×10⁴, which is not informative with 5 points and 3 parameters. The certification planning table replaces it. |
| m3 | Corrected. The notebook's assignment step is complete, and the round-1 "truncation" statement was wrong. We now note that the notebook selects K by silhouette score while our replica fixes K = 2. |
| m4 | Small et al. (2021) has been removed from the references. |
| m5 | Added: the notebook's group-consensus helper would raise an error, rather than return 0, if no group member agreed. |
| m6 | The theory is now stated for certification. Proposition 2 is stated for rates, and the code's strict ">" threshold is used consistently in threshold mode. |
| m7 | B is reported with every table. Swap ratios have Monte-Carlo CIs, and certification uses B = 1,000 participant bootstraps. |

## 4. Answers to the reviewer's questions

1. **PCA seeding and the distribution across seeds.** See F1: over 50 seeds, changes are 16–28 out of 192–197, and the top-58 Jaccard is 0.71–0.76.
2. **Decoupling k-means from the bootstrap.** The ratios are unchanged on CCAI (see M2). For openData, the decoupled-seed ratios with CIs are in `results/opendata_ratio_ci.csv`.
3. **The n-trend with k fixed.** We did not rerun with fixed k. The certification law is stated per seat and uses k = clip(0.2M, 10, 58) throughout. The partial correlation with m given n (−0.60) does not depend on the k rule in an obvious way, but we have not tested this and list it as untested.
4. **Exposure-normalized approvals.** Yes: the over-representation disappears under imputation (0.99×) and halves under inverse-propensity weighting (2.01×). The trade-offs are described under M5.
5. **Contacting the CCAI authors.** Contacting third parties is outside what we can do in this revision. The paper continues to say that the finding should be confirmed by the authors. It also notes that, given Section 6.2, the practical stakes are limited: with either labelling, most seats are uncertified.
6. **Weighting designs.** The new main results do not weight designs. For the designer analysis, we now show the composition sensitivity directly instead of choosing a prior.

## 5. Points where we respectfully disagree, or qualify

- **"The proportional-rule finding is partly a strawman" (M5).** We agree that it is an exposure artefact. However, running count-based rules on raw Polis exports is not purely our choice. It is what off-the-shelf ABC implementations do when given a Polis export, and the reviewer's suggested fix (normalization) turns out not to restore proportionality (imputed PAV violates JR in 3 of 18 processes). We therefore keep the finding, reframed as a caution that has now been tested.
- **"P1's 145/210 mainly restates the agreement distribution" (Section 5 of the review).** We agree. We now present it as exposure only, and Theorem 1 states the mechanism directly: f(q), the density of agreement rates at the cutoff, governs certification.

## 6. What remains limited

These are stated in the paper:
- one AI-constitution process;
- the exposure law rests on 18 processes, with one influential high-exposure point;
- Theorem 1 assumes independence across statements, although the empirical procedures do not;
- routing was simulated, not deployed, and adaptive validity is unproven;
- statements are not principles;
- the integrity finding is unconfirmed by the original authors.
