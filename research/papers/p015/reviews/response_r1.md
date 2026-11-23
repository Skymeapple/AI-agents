# Response to Review Round 1

We thank the reviewer for a careful and fully reproduced review. We agree with the main diagnosis. The frailty explanation of the pooled *k* ≈ 0.6 was public before our paper. The theory was classical. Several of our "confirmations" were weaker evidence than we presented. And the headline "conditional *k* ≥ 1" needed to survive a non-Gaussian frailty, a crossed model and a task audit.

We have therefore **re-scoped and retitled the paper**. It is now "Replicates Buy Hazard, Not Length: What Repeated Runs Can and Cannot Tell Us About AI Agents on Multi-Day Tasks". We no longer claim the frailty critique of the Lindy reading, and we credit it to prior public work. The paper was rewritten from the abstract down; it is not a patched version.

## The new contribution

The round-1 paper answered "is the pooled decline frailty?", and that was already known. The revised paper asks what replicate runs *identify*, and whether that is enough to forecast success on longer tasks. It answers with theory and with a test that can fail. On METR TH1.1, the test does fail, and in a specific place.

1. **Theorem 1 (ceiling on all hazard-shape claims).** Even infinite replicates identify only the law of per-task success given length. Within-task hazard shape needs an invariance assumption (frailty law independent of length). Under location drift, only elasticity plus drift is identified. This directly addresses Weakness 2 and the "conditionally supported" points in Sec. 7 of the review. We now call *k* the across-suite *hazard-length elasticity*.
2. **Theorem 2 (replicate–length duality).** Under invariance, pass^*j* at length *T* equals pass@1 at the length with *j* times the cumulative hazard. We credit the identity to Oakes (1989), as the diagonal of the Archimedean copula that a shared frailty induces. Its consequences are new to this literature:
   - (i) the elasticity is identified with the frailty law completely unrestricted, with an overidentifying restriction across *j*;
   - (ii) replicates extend identification in *hazard*, not in *length*, which gives a design rule *n* ≥ (*T*\*/*T*_max)^*k*;
   - (iii) any heterogeneity model becomes a falsifiable forecast with likelihood-ratio bounds.
3. **The decisive empirical test.** We train on short tasks and predict longer ones.
   - Every model calibrated below 1–2 h overpredicts longer tasks, the nonparametric (NPMLE) frailty model included. The NPMLE overpredicts by +0.15 on 1–4 h tasks, with family-bootstrap 95% CI [+0.05, +0.26]; its LR bounds are violated for 6–7 of 20 agents.
   - Calibrated below 4 h, forecasts of 4–30 h tasks are unbiased: +0.01, CI [−0.06, +0.08].
   - A piecewise hazard map, steep between 1 and 4 h, is preferred jointly across agents (p = 2×10⁻⁶).
   - Simulations show that the test is unbiased when invariance holds (−0.09 to +0.02). A kinked hazard map reproduces the observed +0.12. Smooth drift and non-Gaussian frailty do not.
   - Conclusion: on TH1.1 the obstacle to extrapolation is a change of regime in the hazard-length map. It is not between-task heterogeneity, which replicates measure.
4. **Bounds instead of a spread of projections.** Where invariance is supported (4–30 h), the frontier agents' success on a random 40 h task is bounded. For example, Claude Opus 4.6 has [0.09, 0.42] and GPT-5.2 has [0.06, 0.38]. This rules out both the 0.76 (three-class) and the 0.03 (exponential) projections of round 1. At 160 h every lower bound is ≤ 0.007.

All new numbers come from new scripts `code/12_duality.py`, `13_hazard_map.py`, `15_forecast_bootstrap.py`, `16_npmle_sim.py`, `17_forecast_subsets.py` and `14_duality_figures.py`. Scripts `08`–`11` address specific review points. All outputs are in `results/`, and `run_all.sh` has been updated.

---

## Point-by-point

### Sec. 3 of the review: references

- **Nguyen author list.** Fixed. The reference is now Nguyen, D. T. and Fithian, W. (2026), arXiv:2610.12466 (NeurIPS 2026).
- **Unchecked hosts and further prior work.** Following the reviewer's pointer, we searched for "The time-horizon curve cannot tell a half-life from recovery" (Reasoning Under Uncertainty Substack) and "Three Hazards Under One Horizon" (Mahendrakar, 8 October 2026, Zenodo). Both exist. Their hosts are unreachable from our environment, so we cite them only for the framing visible in search snippets, and we say so in Sec. 2 and in the Limitations.
- **New references.** All were verified by web search: Oakes (1989), JASA 84(406):487–493; Kiefer and Wolfowitz (1956), Ann. Math. Stat. 27(4):887–906; Yao et al. (2024), τ-bench, arXiv:2406.12045; Schaeffer et al. (2025), arXiv:2502.17578; METR (2026b), "Impact of modelling assumptions on time horizon results", 20 March 2026.

### Sec. 4 of the review: novelty

1. **Prop. 1 is classical.** Agreed. It is now background fact F1, attributed to Bernstein, Feller and Proschan, and is not a contribution.
2. **Undisclosed overlap with the LessWrong "A(nother) look…" post.** Agreed, and we apologise for the omission. The Introduction now credits that post explicitly with the simulation showing that per-task constant hazards with spread difficulty reproduce a pooled *k* near 0.57. It states that we treat the frailty critique of the Lindy reading as known. Sec. 2 positions our work against the post: no prior piece uses replicates to identify anything, or tests a forecast out of window.
3. **Further related work.** These are now cited, with the caveat above.
4. **Prop. 2 is Zeger et al.** Agreed. It is now background fact F2, labelled as the survival form of their attenuation.
5. **Prop. 3 is the binary ICC.** Agreed. It is now background fact F3. We also corrected "unbiased" to "consistent" for the ratio estimator, as the reviewer noted in Sec. 5.

New theory replaces these items as the contribution: Theorems 1 and 2, and the sharp bracket of Proposition 3, described above.

### Sec. 5 of the review: theory

- **"Stratified unbiased estimator of *H*".** Corrected. F3 now says that ratio estimators of *H* are consistent, not unbiased.
- **Prop. 4 did not give an *H*-dependent bound for *k* ≥ 3.** Done. Proposition 3 now gives a closed-form *sharp* bracket for pass@*m* at every *m* given (*S̄*, *H*). The proof uses quadratic interpolation with the sign of *f*‴, and the bracket is attained by two-point laws. Empirically it contains observed pass@3 in 98.9% and pass@4 in 96.6% of agent-strata, with mean widths 0.033 and 0.065 (`results/passk_H_brackets.csv`).
- **Notation clash between *k* (Weibull shape) and *k* (attempts).** Fixed. Attempts are now *m* (pass@*m*), replicate moments are *j* (pass^*j*), and *k* is reserved for the elasticity.

### Sec. 6 of the review: empirical and statistical weaknesses

**W1. "20/20 *k*_cond > *k*_marg is mechanical."** Agreed. The claim has been removed as evidence. F2 states that it follows from σ > 0. Table 1 now reports, for each estimator, how many agents have *k* > 1 and how many profile CIs lie above or below 1. It no longer compares *k* with *k*_marg.

**W2. "The headline *k* ≥ 1 is model-dependent and fragile; dropping never-solved tasks lowers it."** We addressed this in four ways.

- (a) *Nonparametric frailty jointly with k* (also review question 3). The NPMLE with *G* unrestricted gives a median *k* of 1.77 (IQR 1.45–2.09). It is above 1 for 20 of 20 agents, and the profile CI lies above 1 for 17 and below 1 for none (`results/duality_summary.txt`). By Theorem 2(i), this estimate does not depend on the form of *G*.
- (b) *Never-solved tasks.* With them removed, the NPMLE median is 1.60, with 15 of 20 CIs above 1. For the frontier agents the reviewer highlighted:
  - Claude Opus 4.6: 1.92 → 1.85 (the reviewer's Gaussian refit gave 0.96);
  - GPT-5.2: 2.00 → 1.81;
  - GPT-5.3-Codex: 2.29 → 2.00.

  The large drops the reviewer found are a property of the single-agent Gaussian tail, which is forced to accommodate *p* = 0 tasks. The NPMLE absorbs those tasks with atoms. In the crossed Gaussian model, dropping never-solved tasks gives 1.72. Dropping both never-solved and always-solved tasks lowers the NPMLE median to 1.49 (17/20 above 1, 10/20 CIs above 1, 1 below). We report this, noting that the filter removes all 59 sub-minute tasks and with them most short-length leverage.
- (c) *Task-validity audit* (`results/task_audit.csv`, `crossed_summary.txt`). All 17 never-solved tasks are HCAST, and 16 of the 17 have estimated lengths, against a base rate of 66/228. In 6 of them some run reached a continuous score ≥ 0.8, so they are near the binarisation threshold; in 5 every run scored ≤ 0.02. None is RE-Bench. A latent "invalid task" class in the crossed model receives weight only 0.006–0.011.
- (d) We now say what *k* is (Theorem 1). It is an across-suite elasticity, and a within-run hazard only under invariance.

**W3. "Frailty is confounded with error in human time."** We have:

- (i) shown in the crossed model that label error shared across agents is absorbed into the shared task effect. It enters as *k*·*e*ᵢ, so the agent-specific variance is free of it;
- (ii) simulated label noise of s.d. 0.7 in log *T*. Under homogeneity it produces *H* ≈ 0.08, far below the observed 0.61, and it does not move the Gaussian *k* materially (`misspec_summary.txt`, M8–M10);
- (iii) restricted to baselined tasks: the NPMLE gives 1.79, and the targets in the forecast test give Sec. 5.4's baselined variant.

The H in the title of round 1 is no longer interpreted as purely substantive. Sec. 5.1 reports that about 80% of heterogeneity is shared across agents, which is consistent with task-intrinsic difficulty, label error, or both.

**W4. "Agents analysed independently; fit a crossed model."** Done (`code/08_crossed_model.py`, `results/crossed_summary.txt`). The model has crossed shared task effects (Gaussian or discrete with free weights), agent-specific task effects, and an optional invalid-task class.

- The median agent-specific share of heterogeneity is 0.19 (Gaussian) or 0.16 (discrete).
- The per-agent *k* has median 1.78 (Gaussian) or 1.69 (discrete + invalid), with 20/20 above 1 and 19/20 Wald CIs above 1.
- A common *k* is 1.42 (profile CI 1.39–1.56), but the free-*k* model is preferred (LR 49.6 on 19 df).

The crossed model also enters the retry comparison (W5). For the forecast test we use task-level family-cluster bootstraps, so that shared tasks are not counted 20 times. We note that agent counts such as "18/20" are not independent evidence.

**W5. "Weak retry baselines; frailty underpredicts *m* = 3, 4."** Done (`code/10_passk_baselines.py`). The baselines are:

- beta-binomial without length;
- beta-binomial with logit mean linear in log *T*;
- logit-normal frailty;
- the three-class model;
- the crossed model using *T* only;
- the crossed model using the other agents' outcomes.

All share 5 task folds. The beta-binomial with *T* is as good as or better than the Gaussian frailty model: agent-level MAE 0.0059 / 0.0040 / 0.0037 against 0.0090 / 0.0066 / 0.0078 at *m* = 2–4. We now say explicitly that retry gains are governed by overdispersion (*H*), and that they are not evidence for a particular survival structure. The 12% underprediction at *m* = 4 (0.065 against 0.074) is reported in Sec. 5.2.

**W6. "The simulation only tests the well-specified case; match pooled curves exactly."** Done, in two parts.

- `09_misspecification_sim.py` covers two-point, *t*₃ and skewed frailty; 10% impossible tasks; 30% impossible long tasks; location drift; label noise; and an exactly matched pooled curve (scenario B\*, whose pooled shape is identical to C). Under a declining true hazard (*k* = 0.63), the Gaussian GLMM does not report *k* > 1 except under drift. Two-point frailty gives 0.83, with 10% of replicates above 1. Drift gives 1.44 in every replicate, as Theorem 1(c) now explains.
- `16_npmle_sim.py` shows that the NPMLE recovers 0.65–0.72 for true 0.63 under non-Gaussian frailty. It also calibrates the new forecast test, as summarised above. We report that NPMLE profile CIs undercover in simulation (6–10 of 10), and we therefore lean on task-level bootstraps.

**W7. Inference details.**

- *Boundary LR tests.* These are now stated: the minimum statistic is 133.8, and the maximum boundary-corrected p is 3×10⁻³¹ (`profile_ci_summary.txt`).
- *Profile-likelihood intervals.* These replace Wald intervals for the Gaussian model (15/20 above 1) and are used throughout for the NPMLE.
- *"r = 0.87 may be mechanical".* Checked by simulation. With σ and *k* fixed and capability shifted over 10 units of log hazard, corr(log *ĥ*₅₀, log σ̂) = 0.05. The empirical correlation is therefore not a fitting artefact, though it remains descriptive and confounded with release date. It is no longer a headline.

### Sec. 7 of the review: does the conclusion follow?

- *Abstract and title overstated "cumulative hazard grows at least linearly" and "the Lindy decline is a property of the population".* Agreed. Both claims are now qualified by Theorem 1. The abstract states up front that every hazard-shape statement rests on invariance, and that invariance is partly rejected in the data.
- *Title "Frailty, Not Fatigue" was confusing.* It has been replaced.
- *"Pre-registered" was inaccurate.* The text now says "a protocol with predictions stated in advance; not a registered protocol".

### Sec. 8 of the review: questions

1. *Audit the never-solved tasks; do RE-Bench tasks behave differently?* See W2(c). None of the RE-Bench tasks is among the never-solved, and dropping RE-Bench leaves the NPMLE *k* at 1.78. In the forecast test, removing never-solved targets shrinks the 1–4 h over-prediction from +0.151 to +0.114 (CI [+0.004, +0.227]). Restricting to baselined targets gives +0.090 (CI [−0.018, +0.212]).
2. *Crossed model; how much of σ is shared?* About 80% (W4).
3. *NPMLE frailty jointly with k?* Median 1.77, with 17/20 CIs above 1 (W2a).
4. *Misspecification simulation?* Done (W6).
5. *Substantive frailty versus label error?* See W3. Shared label error and shared task difficulty are not separable from outcome data, and we now say so.
6. *Credit the LessWrong simulation?* Done (Sec. 4, point 2 above).

### Stale comment

The reference in `common.py` to a non-existent "Sec. 6.5" has been removed.

## What we did not do, and why

- We have not run new agents or collected long tasks; we have no model access. The protocol in Sec. 6 is a proposal.
- We could not read the Substack essay or the Zenodo preprint in full. They are cited conservatively.
- The piecewise hazard map's knots (1 h, 4 h) were fixed in advance and not tuned. The weakest agents' segment slopes are unidentified and are reported as such.
