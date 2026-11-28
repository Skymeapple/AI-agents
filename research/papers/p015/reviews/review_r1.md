# Review (Round 1): "Frailty, Not Fatigue: Replicate Runs Show That Between-Task Heterogeneity, Not Declining Hazard, Shapes AI Agents' Success on Long Tasks"

Reviewer role: area chair level, independent. Venue: ICLR 2027 (evaluation methodology / datasets and benchmarks).

## 1. Summary

The paper asks whether the pooled METR time-horizon curves can say anything about agents completing multi-day tasks. It re-reads them as survival curves and argues that the declining pooled hazard behind Weibull k ≈ 0.6 (Hamilton, as reported by Ord, 2026) cannot be told apart from between-task heterogeneity ("frailty").

The paper makes four contributions:

- **Identification theory.**
  - Prop. 1: pooled curves from constant-hazard tasks are exactly the completely monotone (CM) survival functions (Bernstein/Proschan).
  - Prop. 2: frailty strictly attenuates the local pooled Weibull shape.
  - Prop. 3: replicate runs identify an ICC-type heterogeneity index H.
  - Prop. 4: H fixes pass@2 exactly.
- **Measurement on METR TH1.1** (23,235 runs, 20 agents, 228 tasks). Median within-stratum H = 0.61.
- **Frailty GLMMs.** Median task-conditional Weibull shape k = 1.46, against a pooled shape of 0.63. Under alternative specifications the range is 1.18–1.80.
- **Consequences.**
  - Held-out pass@k is predicted well by the frailty model and overstated about 2.4× by a homogeneous model.
  - 40 h success for a random task is not identified (0.01–0.76 across model families).
  - A replicate-run protocol for long tasks is proposed.

The research question is "can models complete tasks spanning days or weeks?". The paper answers it honestly by reframing it: current data cannot identify population-average week-long success. The paper argues that what matters is the lower tail of the difficulty distribution among long tasks.

## 2. Reproduction and number checking

I copied `code/` and `data/` to an isolated scratch directory and re-ran all seven scripts (01–05 and 07 in parallel, then 06). All runs completed with exit code 0. **All 18 CSV/TXT output files are byte-identical to the shipped `results/`**: marginal_summary, marginal_fits, extrapolation, heterogeneity_summary/index, frailty_summary/glmm, passk_validation, projections, robustness_summary/robustness, simulation_summary/identification, passk_heldout_summary/passk_heldout, trend_and_sensitivity, kcond_with_se and projection_sensitivity. I also checked every number in the paper against the shipped `results/*.txt|csv` files, and recomputed the data counts directly from `data/runs_th11.jsonl`.

- **Data counts: all match.** 23,235 runs; 20 agents; 228 tasks; 79 families; T from 0.02 to 1,800 min; 55 tasks ≥ 4 h; 5 tasks ≥ 16 h; 162 baselined and 66 estimated; median 4 runs; 99.96% of pairs with ≥ 2 runs; 98.3% with ≥ 4; 4.3% usage-limit runs.
- **Tables against shipped outputs.**
  - Table 1 matches `marginal_summary.txt`: NLLs, CV wins, extrapolation errors, h99/h50 ratios and shape ranges.
  - Table 2 matches `frailty_summary.txt` and `kcond_with_se.csv`. The ±1.96 s.e. values check out, e.g. GPT-4 0314 1.377 ± 0.367 and Opus 4.6 1.657 ± 0.484. The "12/20 Wald interval above 1" claim also checks out.
  - Table 3 matches `robustness_summary.txt`. Table 4 matches `passk_heldout_summary.txt`: MAE 0.008–0.013 for frailty and 0.068–0.105 for homogeneous; gain ratios 2.56/2.45/2.35; long tasks 0.30/0.30/0.47.
  - Table 5 matches `simulation_summary.txt`. Table 6 matches `projection_sensitivity.csv`, including the 0.615 latent-class mass.
  - Sec. 5.7(b) matches `frailty_summary.txt`: 29% and 34% of tasks with p ≥ 0.8; 54% and 46% with p ≤ 0.05; pass@8 0.45 vs 0.93 and 0.34 vs 0.51.
  - The trend correlations r = 0.68 and 0.87 match `trend_and_sensitivity.txt`.
- **Code inspection.** The `rng.shuffle` warning on a pandas `StringArray` in `01_marginal_fits.py` does not produce duplicates in practice. I checked that each agent's fold assignment covers every task exactly once. There is a stale comment in `common.py` ("see paper Sec. 6.5", a section that does not exist).

I found no reported number that is not backed by the shipped outputs.

## 3. Reference spot-checks

I verified these 11 references by web search:

| Reference | Check |
|---|---|
| Ord (2025) arXiv:2505.05115 | Exists and says what is claimed (constant hazard; T80 ≈ T50/3; T99 ≈ T50/70). |
| Ord (2026) blog | Exists. It reports Hamilton's Weibull fits with k below 1 for all models, "mostly near 0.6", and the factor of about 20 at the 99% horizon. |
| BRIDGE, Liu et al. (2026) arXiv:2602.07267 | Exists. Authors match. 2PL IRT; difficulty roughly linear in log human time. |
| Nguyen (2026) arXiv:2610.12466 | Exists. Authors are Drew T. Nguyen and William Fithian. Splines plus IRT on 228 tasks. The paper's citation "Nguyen, D. ... full author list per arXiv" omits the second author, Fithian, and should be fixed. |
| Sinha et al. (2025) arXiv:2509.09677 | Exists (ICLR 2026). The self-conditioning claim is accurate. |
| METR (2026) TH1.1 blog | Exists. Dated 29 Jan 2026. It says horizons above 16 h are unreliable. |
| Proschan (1963) | *Technometrics* 5(3):375–383. Correct, and on the mixtures-of-exponentials DFR point. |
| Vaupel, Manton and Stallard (1979) | *Demography* 16(3):439–454. Correct. |
| Vending-Bench arXiv:2502.15840 | Correct. |
| LessWrong "(Updated) METR's data can't distinguish between trajectories..." | Exists. It is a Bayesian IRT analysis with typical vs marginal 80% horizons. |
| LessWrong "A(nother) look at METR's time horizon..." | Exists. **See the novelty concern below.** |

No fabricated references were found. Two caveats:

- The Nguyen author attribution is incomplete.
- Two hosts (lesswrong.com and gushamilton.github.io) could not be fetched directly from my environment. Their content was confirmed through search snippets only.

## 4. Novelty and prior work

The core conceptual point has substantial prior art, and the paper understates this.

1. **Prop. 1 and the "Consequence" paragraph are classical.** A mixture of constant hazards gives a falling pooled hazard. This is Proschan (1963) and Vaupel et al. (1979). The paper cites both, which is fine, but it presents Prop. 1 as a contribution.
2. **Undisclosed overlap with the LessWrong "A(nother) look at METR's time horizon" post.** In search results, that post argues directly that pooling tasks with per-task constant hazards and spread difficulty reproduces Hamilton's Weibull shape. Its simulation reportedly gives a pooled shape of about 0.57 against METR's 0.59, and about 1.01 when duration is the only source of difficulty. That is the central "frailty, not Lindy" claim of the paper. The paper cites this post only for "tasks of equal length are solved months apart". It then states that its own contribution is to "frame the problem in survival terms (hazard shape versus frailty)". This must be acknowledged and positioned explicitly.
3. **Possibly related work I could not check.** Search snippets point to further pieces from autumn 2026 that make related points: a Substack post, "The time-horizon curve cannot tell a half-life from recovery", and a preprint, "Three hazards under one horizon" (October 2026). I could not open either. The authors should check them.
4. **Prop. 2 is a standard result.** It is the survival-shape version of the marginal/conditional attenuation in Zeger et al. (1988). It is correct and nicely stated, but not new in substance.
5. **Prop. 3 is standard.** H is the binary ICC, and its replicate-based estimation is standard overdispersion / beta-binomial methodology.

What appears genuinely new for this literature:

- Using METR's replicate runs to separate within-task from between-task variance nonparametrically.
- The held-out-task pass@k test.
- The careful sensitivity analysis showing that week-long projections are not identified.

This is a useful, careful *application and synthesis*. It is not a conceptual advance. On novelty I place it at the workshop-to-solid level.

## 5. Theory: line-by-line check

- **Prop. 1.** Correct.
  - A Laplace transform of a probability measure is CM. The converse (Bernstein) is correct, given S̄(0+) = 1 and allowing an atom at λ = 0.
  - d/dT E[λ | survive] = −Var[λ | survive] is correct.
  - CM ∘ Bernstein is CM (Feller XIII.4), and x^k is Bernstein for k ∈ (0, 1]. Correct.
- **Prop. 2.** Correct.
  - g(x) = −K(−x), where K is the CGF of Z ≥ 0, so g is concave, and strictly so for non-degenerate Z.
  - g(0) = 0 and strict concavity give g(x) > x g′(x).
  - κ = k · x g′/g < k.
  - The proposition implicitly requires Z ⟂ T. This is stated in the setup.
- **Prop. 3.** Correct. One overclaim: the paper calls its estimator a "stratified unbiased estimator of H". The numerator and denominator pieces are unbiased, but the ratio is not. It is consistent, not unbiased.
- **Prop. 4.**
  - The pass@2 identity is correct.
  - The Jensen bound and the k → ∞ limit are correct.
  - The contribution list says H "bounds [retry gains] for all k". The stated bound, 1 − (1 − S̄)^k, does not involve H. An H-dependent bound for k ≥ 3 (for example via Lyapunov's inequality on E(1−p)^k) is not given. Either derive one or soften the claim.
- **Notation.** k denotes both the Weibull shape and the number of attempts in pass@k. This makes Prop. 4 and the contribution list ("exactly for k = 2") confusing.

## 6. Empirical and statistical assessment

### Strengths

- The H estimates are nonparametric and come with task bootstraps. There is a check that within-stratum length variation does not drive H (H_len < 0.03).
- The held-out-task pass@k validation is a real out-of-sample test.
- The simulation on the actual design is a good sanity check.
- The authors are commendably candid about the non-identification of 40 h success.

### Weaknesses

1. **"k_cond > k_marg for 20/20 agents" is not evidence.** Within a Gaussian random-intercept cloglog model with σ > 0, Prop. 2 forces k_cond > k_marg. So once σ̂ > 0 (and the LR test shows σ̂ ≫ 0), 20/20 is mechanical. The paper presents it "as Proposition 2 requires if frailty is real", which reads as a confirmatory test, but it is a model consequence. The real empirical content is H (or σ). k_cond is essentially a model-based transform of the pair (k_marg, σ).

2. **The headline k ≥ 1 depends on the model and is fragile.**
   - The 3-class frailty gives a median of 1.18, with 25% of agents at or below 1. The authors acknowledge this.
   - I ran an additional check with the authors' own `fit_glmm` (task-only, cloglog). 17 tasks are never solved by *any* of the 20 agents, and 13 of them are ≥ 4 h. Dropping these 17 tasks lowers the median conditional k from 1.56 to 1.39. The drop is concentrated in frontier agents:
     - Claude Opus 4.6: 1.88 → 0.96
     - GPT-5.2: 2.12 → 1.41
     - Claude Opus 4.5: 2.23 → 1.36
     - Gemini 3 Pro: 2.02 → 1.52
   - Also dropping the 59 tasks solved in every run by every agent gives a median k of 1.18, with only 11/20 agents above 1.
   - Both filters select on outcomes, so neither is a clean estimate. They do show something important. The frontier-agent "increasing hazard" conclusion, and the frailty-based week-long projections, lean heavily on a handful of long tasks that no system ever solves. Those tasks could be broken, mis-scored (e.g., thresholded RE-Bench tasks) or wrongly timed, rather than "hard".
   - The paper needs a task-validity analysis, and ideally a crossed agent × task model (see point 4).

3. **Frailty and error in human time are confounded.** If log T has error (66 of 228 lengths are estimates, and baselines are noisy), tasks at the same nominal T differ in true length. That shows up as "frailty". The paper only discusses the attenuation of slopes. Yet part of H and σ may be measurement error in difficulty labels, not substantive heterogeneity. This matters for the interpretation in the title. The baseline-only robustness check (k = 1.38) addresses this only partly.

4. **Agents are analysed independently, although they share the same 228 tasks.** A task that is hard for one agent is hard for others. The 20 per-agent fits are therefore strongly dependent, and "20/20" or "15/20" counts overstate the evidence. A joint model with crossed task and agent effects (task difficulty shared across agents, plus an agent × task interaction) would:
   - use the data far better;
   - separate *task-intrinsic* difficulty from agent-specific frailty;
   - directly address the never-solved-task issue.

   This is also the natural link to the IRT work (BRIDGE; Nguyen & Fithian), which the paper positions itself against without comparing to it.

5. **The baselines for the retry test are weak.** The "homogeneous model" (the pooled curve applied to every task with independent retries) is a straw man. Nobody fitting IRT models to METR data would predict pass@k this way. Stronger baselines would be:
   - a beta-binomial with no length dependence;
   - an IRT/2PL model with task difficulty;
   - the 3-class model.

   Also, the frailty model systematically *under*-predicts retry gains at k = 3 and 4: 0.055 vs 0.061, and 0.065 vs 0.074, about 12% low. This is worth reporting, not hiding inside "accurately".

6. **The simulation tests only the well-specified case.** Data are generated from the Gaussian-frailty cloglog model and fitted with the same model, so recovering the truth is expected. The informative test is misspecification. For example:
   - truth k = 0.63 with discrete or heavy-tailed frailty, or with frailty location correlated with T;
   - truth with a share of "impossible" tasks.

   The question is whether the Gaussian GLMM then reports k > 1. The latent-class and drop-never-solved results suggest it might. Scenarios B and C are also not matched on the pooled curve (0.64 vs 0.55). Prop. 1 suggests matching the pooled curves exactly would be a sharper demonstration.

7. **Inference details.**
   - The variance-component LR tests sit on the boundary. This is irrelevant given the magnitude of the statistics, but should be stated.
   - Wald intervals for k, with σ near large values, would benefit from profile-likelihood intervals.
   - The r = 0.87 correlation between log h50 and log σ is plausibly partly mechanical. Agents near ceiling need large σ to fit tasks with p = 0 and p = 1 under a cloglog link. The "more capable agents are more bimodal" reading should be checked against this.

## 7. Does the conclusion follow?

- **Supported.**
  - Large between-task heterogeneity at fixed nominal length (H ≈ 0.6).
  - Pooled curves cannot identify within-task hazard dynamics.
  - Retries help much less than homogeneity implies.
  - Week-long success is not identified by TH1.1.
- **Only conditionally supported.**
  - "For a task of fixed relative difficulty, cumulative hazard grows at least linearly with length" (abstract).
  - "The apparent Lindy decline is a property of the task population."

  Both depend on (a) Gaussian frailty, (b) frailty independent of T, (c) a cross-sectional reading of k, and (d) the inclusion of never-solved long tasks. The abstract and title state them more strongly than the evidence warrants.
- **Title.** "Frailty, Not Fatigue" is confusing. A conditional k > 1 is itself a fatigue-like (increasing) hazard, which the paper also links to self-conditioning. The contrast the authors mean is "frailty, not declining hazard".
- **"Pre-registered" protocol.** The protocol is not registered anywhere. Call it "proposed, with predictions".

## 8. Questions for the authors

1. How do H, k_cond and the projections change if tasks never solved by any agent are audited, and genuinely broken or mis-scored ones removed? Do RE-Bench thresholded tasks behave differently?
2. Can you fit a joint model across agents with crossed task and agent × task effects? How much of σ is shared task difficulty and how much is agent-specific?
3. Under a nonparametric (NPMLE) frailty jointly estimated with k, what is k? The 3-class model is a weak proxy.
4. Can you add a misspecification simulation (discrete or heavy-tailed frailty, or a share of impossible tasks, with true k < 1)?
5. How do you distinguish substantive frailty from error in the log human-time labels?
6. Please acknowledge and position against the LessWrong simulation showing that per-task constant hazards reproduce pooled k ≈ 0.57.

## 9. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 7 |
| Significance | 6 |
| Clarity | 7 |
| Reproducibility | 9 |

**Overall: 6/10 (borderline; workshop level in its current form).**

**Decision: major revision.**

The paper is careful, reproducible and honest about non-identification. The theory is correct but classical. The central conceptual claim was anticipated by public prior analyses, and the paper does not credit them. The headline "conditional k ≥ 1" depends on the model and is sensitive to a small set of never-solved long tasks. Several reported "confirmations" (20/20; the retry test against a straw-man baseline; the well-specified simulation) are weaker evidence than presented.

**The single most valuable change:** a joint crossed agent × task frailty model, with a task-validity audit and a misspecification simulation. It would establish whether k ≥ 1 survives when task-intrinsic impossibility and label noise are separated from agent-specific heterogeneity, and it would credit and go beyond the LessWrong frailty simulation.

### Rerun status

Full rerun completed. All 18 numeric output files are byte-identical to the shipped outputs, and every number checked in the paper matches. The additional robustness check in Section 6 (dropping never-solved tasks) used the authors' own `fit_glmm`, and its full-data baseline reproduced the shipped task-only k for all 20 agents.
