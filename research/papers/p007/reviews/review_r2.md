# Review (round 2): "Where Unification Stops Paying: A Capacity–Data Model of Competition and Synergy in Unified Multimodal Models, with Small-Scale Tests"

Reviewer role: independent area chair / senior reviewer. Target venue: NeurIPS 2027 main track.
I read the round-1 review and the authors' response. My judgement below is my own.

## 1. Summary

The paper studies a linear "shared-core" model of a unified multimodal network. There are M modalities, each a multi-output linear regression. Each target depends on k_s directions shared by all modalities and k_p private directions, possibly misaligned by an angle θ. A unified model has one rank-ρ ≤ r core and per-modality heads. The baseline is M same-size specialists.

The paper derives:
- the exact population interaction (Prop. 1) and a modality budget M* = ⌊(r−k_s)/k_p⌋ (Cor. 1);
- under a reduced-rank-regression degrees-of-freedom heuristic (Heuristic 1): a bounded oracle-width synergy ratio Γ (Cor. 2), a head-interference penalty (Cor. 3), an alignment tolerance sin²θ_c ∝ 1/n (Cor. 4), and a fixed-width variant (Remark 2);
- a scale-free capacity tax R = (φM^{1/α}+(1−φ)M)^α/M under power-law spectra (Prop. 3), with a predicted crossover r_× ∝ n^{1/α};
- pooling shifts of the spiked-matrix recovery thresholds (Remark 4).

Tests come at three levels:
1. **Monte Carlo inside the model.** Exps. A–E and G, now including random Gaussian designs and a fine-grid threshold study.
2. **One-hidden-layer ReLU teacher–student networks.** Re-run with d = 64, 4 seeds and two learning rates.
3. **New in this round: tiny decoder-only transformers** (12k–442k parameters, 2 layers). They are trained on MNIST pixel tokens, VQ-tokenised spoken digits, Tiny Shakespeare characters and Fashion-MNIST, with 2 seeds per cell.

The transformer results are mostly negative for the theory:
- P1 (rise then fall with size) is not observed.
- P4 (modality budget) is not observed.
- P2, P3 and P5 are untested.
- The image modality shows small, persistent competition of 0.001–0.025 nats/token.

## 2. Reproduction (I re-ran the code)

I copied `code/` and `data_cache/` to a scratch directory, regenerated the token streams, and re-ran the scripts into a fresh `results/` folder with `OMP_NUM_THREADS=1`.

| Script | Status |
|---|---|
| `prep_tokens.py` | `tokens.npz` regenerated; all 19 arrays identical to the shipped file |
| `exp_linear.py` | `linear_results.json` and `figA` byte-identical |
| `exp_powerlaw.py` | `powerlaw_results.json` and `figE1` byte-identical |
| `exp_gap.py` | `gap_results.json` byte-identical |
| `exp_threshold_fine.py` (d = 64 and `TF_SCALE=4`, d = 256) | both JSONs and `figG` byte-identical |
| `exp_crossover.py` | `crossover_results.json` byte-identical (10 reps; slopes 0.567 / 0.750) |
| `exp_random_design.py` | `random_design_results.json` byte-identical |
| `exp_transformer.py` (3 workers, as in README) | 10 of the 56 runs completed in my review window (seed 0, D = 25k, d_model 16–64 cells, plus one more). All 10 are bit-identical to the shipped `transformer_runs.jsonl` (per-modality-stop test loss, joint-checkpoint test loss, stop steps). I verified the remaining runs only through the shipped JSONL → `analyze_transformer.py` → summary chain. |
| `exp_mlp2.py` | not re-run (about 1 CPU-hour); every number in the Sec. 5.7 table was checked against the shipped `mlp2_summary.json` instead |

**Paper vs. JSON.** I checked every number in the abstract, in the error-summary table, and in Secs. 5.1–5.8 against the JSON files. That covers:
- Exp. A windows and ratios;
- Exp. B M* and the jump ratios;
- Exp. C boundary counts (15 near-boundary cells, 15/15 agree) and the θ_c table;
- Exp. D;
- the random-design table and its 95% CIs;
- E2 plateaus and errors;
- E3 slopes, CIs and r_×/ρ_u*;
- all twelve threshold shifts and CIs;
- the full ReLU round-2 table, including CIs, crossover CIs and the LR-selection fractions;
- the transformer table (all 32 Δ ± SE entries);
- the P4 numbers;
- the "33 of 36 image models with best step ≥ 2400" count;
- the round-1 MLP re-analysis (0.84/1.13/1.68 and 1.03/1.03/1.49).

All of them match. I also recomputed several values by hand:
- Γ = 0.9375;
- the closed-form θ_c at n = 1000 (G_n = 0.036, so sin²θ_c = 0.012 and θ_c = 6.29°);
- the fixed-width df ratios 0.527 / 0.609;
- R, κ and M/κ for α = 1.5, 2 and 3.

The round-1 provenance gap (unscripted filtering for the E3 slope) is fixed: `exp_crossover.py` now filters capped rows itself and stores `n_points_used`.

Small presentational inconsistencies remain (none changes a conclusion):
- (a) Sec. 5.4 reports Exp. D errors of the *unified loss* relative to the *simulated* value (17%, 8%, 19%, …). The preamble says the metric is |sim−th|/th of the *ratio* "throughout". The summary table uses the ratio metric (21%, 8.9%, 11%, …). So the two places report different numbers for the same experiment.
- (b) The summary table says "≤0.3%" for n ≥ 3000 in Exp. D. The n = 10⁵ value is 0.34%.
- (c) The prediction-status table supports Cor. 3 with "Exp. G round 1: 1.34 vs 4/3". That experiment is labelled superseded elsewhere in the paper.
- (d) Limitations says "Sec. 5.5 gives the conversion" between same-size, same-parameter and same-compute comparisons. Sec. 5.5 contains only κ and M/κ, which is at most part of such a conversion.

**No unbacked or mismatched results.**

## 3. Reference check (web search)

I prioritised references added in this round.

| Reference | Check | Verdict |
|---|---|---|
| Ildiz, Zhao & Oymak, AISTATS 2024, PMLR 238:4726–4734 | Exists; multitask representation learning with heterogeneous task covariances; multiple-descent risk curves and inverse scaling | Correct |
| Fernandes et al., ICML 2023, PMLR 202:10053–10071 | Exists; abstract: changing language-pair weights "only affect[s] the multiplicative factor of the scaling law" | Correct, and the paper's use of it is accurate |
| Longpre et al., ATLAS, arXiv:2510.22037, ICLR 2026 | Exists; 774 runs; adaptive transfer scaling law; cross-lingual transfer matrix | Correct |
| Paquette, Paquette, Xiao & Pennington, NeurIPS 2024, arXiv:2405.15074 | Exists; solvable compute-optimal phase diagram | Correct |
| Bahri et al., PNAS 121(27):e2311878121, 2024 | Exists; variance-limited and resolution-limited regimes | Correct |
| Gavish & Donoho, IEEE TIT 60(8):5040–5053, 2014 | Exists; the optimal hard threshold; the paper uses the standard keep-if-it-reduces-error criterion | Correct |
| Lin et al., MoMa, arXiv:2407.21770 | Exists; modality-aware experts and FLOP savings | Correct |
| Maloney, Roberts & Sully, arXiv:2210.16859 | Exists; the solvable model that Paquette et al. build on | Correct |

All eight references I checked exist and are described accurately. Round-1 spot checks (Aghajanyan, MoT, Shukor, Huang, Chang, Wu, Gato) were already verified. The corrected description of Wu et al. (they do analyse shared-module capacity) is now right. **No fabricated or misdescribed references.**

## 4. Novelty and missing prior work

The "Novelty, stated plainly" paragraph is honest, and I agree with it:
- Prop. 1 is Eckart–Young / Ky Fan.
- The finite-data analysis is the RRR df count.
- Remark 4 applies BBP / Gavish–Donoho.
- Prop. 3's scale invariance follows from power-law self-similarity.

The genuinely new pieces are the closed-form capacity tax (R, κ), the 1/n alignment tolerance, and the naming of head interference as a separate cost. These are small, tidy corollaries rather than new techniques.

My search found closely related recent work that is not cited. A NeurIPS 2027 submission must discuss it:
- **Han, Tong, Fan, Chen, Torr, Kokkinos & Lewis (arXiv:2608.05000, Aug. 2026), "Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes."**
  - It is a controlled empirical study of exactly the competition/synergy question in unified models, at up to 13.5B-parameter MoE scale.
  - According to its abstract and summaries, whether modalities help or hurt each other depends on data/task complexity, with competition for capacity dominating for complex tasks. (I could read only abstracts and summaries, because arxiv.org was not reachable from my environment.)
  - It reports that shared attention plus modality-specific FFNs promotes synergy.
  - This is the real-scale evidence most relevant to P1, P4 and P6, and it largely pre-empts the paper's architectural discussion.
- **Uni-X (Hao et al., ICLR 2026).** It measures gradient conflict between modalities in shared autoregressive transformers. The conflict is concentrated in the shallow and deep layers, and the remedy is modality-specific ends around a shared core. This is close to the model's "shared core + modality-specific heads", and to its head-interference term.
- **"Capacity and Redundancy Trade-offs in Multi-Task Learning" (arXiv:2607.16554; UAI 2026, PMLR 337).** It gives an information-theoretic bound: shared-bottleneck capacity must grow roughly linearly in the number of tasks unless the tasks are redundant. That is a general-distribution analogue of Cor. 1's modality budget.
- Random-matrix analyses of multitask regression with W_t = W_0 + V_t (NeurIPS 2024). These are relevant to the shared/private decomposition and to the finite-data claims.

Given this literature, novelty is modest. The capacity-competition story is now well established empirically and information-theoretically. The paper's addition is a solvable linear instance with explicit constants.

## 5. Proof check

I re-derived the following line by line.

- **Lemma 1, Prop. 1 (all four cases), Cor. 1 and Lemma 2.** Correct.
- **Cor. 2.**
  - Γ is correct.
  - The infimum claim needs q ≥ k_p. Γ = (ρ_u/Mk)·(d+Mq−ρ_u)/(d+q−k), and the second factor is ≥ 1 iff q ≥ k_p. This always holds when W_m has rank k, and it holds in all the configurations used, including the ReLU one with q = 8 > k_p = 6.
  - The paper should state this condition, because "infimum over q/d" is otherwise false: q → 0 would push Γ below 1−φ(1−1/M).
- **Cor. 3.** Correct.
- **Cor. 4.** Correct to first order. The unified model discards (M−1)k_s·s²sin²θ in total, i.e. per modality the stated amount. The assumption that the gap directions are not retained is now stated honestly.
- **Prop. 3.** I verified the pooled counting function c_u = φM^{1/α}+(1−φ)M, the tail c^α r^{1−α}/(α−1), κ = R^{1/(α−1)}, and M/κ ≥ 1 ⇔ c_u ≤ M.
- **Remark 4.** The rectangular BBP threshold ϑ² > v√(ab) and the √M shifts are correct. The useful-recovery criterion y < 2x·c_u·c_v (keep a component iff it lowers the Frobenius error) is correct.
- **Remark 2 (fixed-width df).** Matches the code. Note that the specialist rank is silently capped at q = 16 for r = 20, 24. That is reasonable, but it should be said.

No errors. Heuristic 1 remains a heuristic, and the paper is now explicit about where it fails: at low SNR, and for the near-threshold tail under power-law spectra.

## 6. Strengths

1. **Exceptional reproducibility and transparency.** Every number I checked reproduces bit-for-bit, including token preparation and the transformer runs I re-ran. Negative results are reported prominently, in the abstract, a prediction-status table and the conclusion.
2. **The round-1 issues are genuinely addressed.**
   - The selective accuracy claim is replaced by a full error table.
   - n < d is relabelled as an SNR model and complemented by real random designs; the n−d−1 correction works well for n ≥ 500.
   - The threshold-shift claim is redone on a fine grid against proper asymptotics, with a finite-size check.
   - The ReLU-vs-Cor. 2 contradiction is resolved convincingly: it was a fixed-width vs oracle-width convention mismatch. Remark 2 explains the fixed-width synergy.
   - The ReLU study is re-run on the theory's footing (d = 64, orthonormal teacher).
   - Related work is corrected (Wu et al., Shukor et al. routing).
3. **Clean theory, correctly stated as limits of a model.** The capacity / statistical / alignment decomposition is a useful conceptual frame.
4. **Real-data contact, albeit tiny.** This was the main round-1 request, and the authors made an honest first attempt.

## 7. Weaknesses

**W1. The research question is still answered only inside the toy model; the new real-data evidence is too small to bear on it and is mostly negative.**

The transformers have at most 442k parameters and 2 layers. At D = 25k tokens per modality they see about 127 MNIST digits (196 tokens each). Most image runs were still improving at the 3000-step cap. At this scale:
- P1 and P4 are *not* observed.
- P2, P3 and P5 are untestable.
- The one positive finding (image competition of 1–5% of the loss) is predicted by almost any capacity- or optimisation-sharing story, not specifically by Cor. 3.

Cor. 3 is also a statement about oracle width with ample capacity and per-modality heads. The transformer has a single shared output layer and fixed widths, and its image competition *shrinks* towards 0 at the largest size with D = 100k (+0.001 ± 0.006). That is not obviously "persistent competition". The audio/image asymmetry is unexplained.

So the paper's answer to "where are the limits of unified multimodal models?" remains: "in a linear model, here; in tiny transformers, we cannot tell." The paper is admirably honest about this, but honesty does not by itself make the contribution significant. Video and action, named in the research question, are absent.

**W2. The theory's quantitative content does not transfer even to the one-hidden-layer ReLU model.**

Under the matched (oracle-width) convention, the ReLU networks are compared with the df model's Γ, not just its d → ∞ bound:
- the df formula gives 0.78, 0.69 and 0.69;
- the networks give 0.80–0.89 for M ≤ 4, and 1.17–1.27 for M = 8 (competition, not synergy).

The paper frames this as "respects the bound". But a lower bound on the ratio that the data sit well above is a weak test: almost any model with less synergy "respects" it. In addition:
- The prediction of a tie below k_s fails badly (1.38–2.05 at h = k_s).
- At M = 8 the unified model is still in competition at h = 64 > k_s+Mk_p.

The surviving qualitative facts (a window, a peak at k_s+k_p, a crossover growing with M) are the generic "capacity is finite" picture. The status table should present Cor. 2 as "quantitative prediction (Γ) fails; only the weak bound holds".

**W3. Statistical treatment of the transformer study.**

- With 2 seeds, the "seed SE" equals |Δ₁−Δ₂|/2. "More than 2 SE in 7/8 cells" corresponds to a t-statistic on 1 degree of freedom, which carries essentially no inferential weight (two-sided 5% critical value 12.7). The SE is described as "crude", but it is still used to make the 7/8 claim. The defensible statistic is a paired sign count: I find Δ_image > 0 in 15 of 16 seed-level comparisons. Report that, or use ≥ 5 seeds.
- Each test set has 160 windows (about 6k tokens for audio). The paired design helps, but no test-sampling uncertainty is reported.
- The P4 sweep has 2 seeds at one size and one budget. Its conclusion ("not supported") is underpowered: differences of 0.003 nats are well inside seed noise. The P4 data also contain oddities that deserve comment:
  - Text competition falls from +0.038 to +0.016 when fashion is added.
  - Image competition falls when Fashion-MNIST, the closest modality, is added.
- The single learning rate (2·10⁻³) for all sizes, together with a step cap that binds for 33/36 image runs, means the size axis mixes capacity with optimisation. That bears directly on the P1 conclusion.

**W4. Missing closely related work (Sec. 4 above).** Han et al. 2026 (Meta, 13.5B-scale controlled study of modality synergy/competition), Uni-X (ICLR 2026) and the capacity–redundancy MTL bound (UAI 2026) need to be discussed. The empirical part of the paper should be positioned relative to Han et al. in particular, which tests P1/P6-type questions at a scale this paper cannot reach.

**W5. Oracle-width convention.** Most theoretical statements are at oracle width. Remark 2 shows that conclusions flip under fixed width (0.53 vs 0.94). Published comparisons, including Aghajanyan et al., are fixed-size. The model's headline "bounded synergy" therefore applies to a comparison that practitioners do not make. Validation-selected rank, which round 1 asked for, was not implemented in the linear model.

## 8. Minor issues

- m1. Cor. 2: state the condition q ≥ k_p for the infimum (see Sec. 5 above).
- m2. Exp. D: the metric differs between the summary table and the Sec. 5.4 text (see Sec. 2a above). There is also the "≤0.3%" vs 0.34% discrepancy.
- m3. The status table cites the superseded Exp. G round-1 result for Cor. 3.
- m4. The dangling "Sec. 5.5 gives the conversion".
- m5. Exp. C (Sec. 5.3) does not restate d and q (d = 64, q = 16 from `gap_results.json`).
- m6. Random designs: the theory value is computed from the last repetition's W_s only. This is harmless because the spectra are identical across repetitions, but say so.
- m7. The α = 1.5 crossover slope rests on only 4 points; say how fragile that is.
- m8. Remark 2: say that the specialist's rank is capped at q.
- m9. "Image and audio tokens share … presumably little structure (φ ≈ 0)" is asserted, not measured. A cheap CCA or probing estimate would make the Cor. 3 link testable.

## 9. Assessment

This is a careful, scrupulously honest, fully reproducible paper. The authors fixed nearly every round-1 issue, and none of my checks found an unbacked number or a bad reference.

The core problem is unchanged: significance. The theory consists of elementary corollaries of standard results, and its quantitative claims fail already in one-hidden-layer ReLU networks. The new real-data study is too small to test the interesting predictions, and where it can test them (P1, P4), they are not confirmed. Meanwhile, recent large-scale empirical work (Han et al. 2026) and information-theoretic capacity bounds (UAI 2026) cover much of the conceptual ground and are not cited.

In its current form this is a good workshop paper, or a candidate for a venue that values solvable toy models (e.g., TMLR). It is below the NeurIPS main-track bar. A path upward needs at least one of:
- (a) a mid-scale test (tens of millions of parameters, more seeds, converged runs) that confirms a non-trivial quantitative prediction such as the scale-free ratio, the 1/√n alignment tolerance, or the modality budget;
- (b) a genuinely new theoretical result beyond the linear df heuristic, for example a rigorous finite-n treatment or a nonlinear extension that explains the competition below k_s.

## 10. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 8 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 10 |
| **Overall** | **5** |

**Decision: major revision (weak reject at NeurIPS main track).** No fatal issues.
