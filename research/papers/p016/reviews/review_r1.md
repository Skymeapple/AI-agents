# Review (Round 1): "The Stability Gap Is a Property of the Path, Not the Objective"

**Reviewer role:** area chair / senior reviewer
**Research question:** Can a model learn continually without catastrophic forgetting?
**Recommendation:** Major revision (overall 5/10, weak reject at ICML main track in its current form)

---

## 1. Summary

The paper studies the *transient* old-task loss (the "stability gap") when a model trained to an interpolator of task 1 is then trained by gradient flow or GD on a replay objective $(1-\rho)L_2+\rho L_1$, in realizable overparameterized linear regression. Because the joint objective has a common zero-loss solution, terminal forgetting is zero for any $\rho>0$, so any rise in $L_1$ is purely along the path. The paper proves:
- Prop. 1 / Thm 1: an exact path formula, plus a closed-form gap depth and peak time for rank-one tasks.
- Thm 2: a short-time/one-step law. Replay reduces initial forgetting only by $(1-\rho)^2$.
- Thm 3: a gap exists for some $e_0$ iff $\Sigma_2$ does not leave $\ker\Sigma_1$ invariant.
- Thm 4: as $\rho\to0$, the gap depth is bounded below by approximately the no-replay forgetting.
- Prop. 5: the smallest nonzero eigenvalue of $H_\rho$ is at most $\rho\|\Sigma_1\|$.
- Thm 6: exact Gauss–Newton (GN) on $H_\rho$ gives zero gap; damped GN gives a gap of $O((\delta/\lambda^+_{\min})^2)$.

Linear simulations check all of these. A 1,210-parameter tanh MLP on sklearn digits (perm and split scenarios, 4 seeds) tests which predictions transfer. The paper reports clearly where they fail: full GN steps, small $\rho$, and the damping-scaling prediction.

## 2. Reproduction

I backed up `results/` and reran every script: `linear_theory.py`, `linear_extras.py`, the 8 MLP jobs, the 4 damping jobs, `analyze_mlp.py` and `analyze_damping.py`.

- **Linear outputs.** `linear_results.json` and `linear_extras.json` came out **byte-identical** to the submitted files.
- **MLP outputs.** See the rerun note in §2a. Phase-1 logs are identical to the submitted log, and all randomness is seeded.
- **Tables 1–4, which carry the main claims, match the outputs** to the stated precision. Examples:
  - Table 1 matches.
  - Table 2 means match (e.g. GD 0.0664/0.0341/0.0182; exact GN about 1.4e-17).
  - Table 3 matches up to third-decimal rounding: split ρ=0.01 GGN-flow is 0.671 vs. 0.672 in the paper; perm ρ=0.01 δ=1e-2 final L2 is 0.034 vs. 0.035.
  - Table 4 matches.
  - The one-step ratio is 0.994 (range 0.979–1.000004) over 56 runs.
  - G(ρ)/G(0) matches.
  - The GGN reduction is 35.7–41.9% (perm) and 51.4–60.1% (split); Adam's is 14.7–22.3%.
  - The accuracy figures (99.3→46.3±6.0→87.5±2.2, etc.) match.
  - Theorem 4 bound values (0.0754 / 0.0861 / 0.0863; 0.0083 / 0.0657) match.
  - Noisy-case overshoots match.
- **Several numbers in the prose do NOT match the outputs.** Each is small, but each is a claim the code does not support:
  1. "Agreement is within $10^{-6}$ relative error" (Table 1 and abstract, "<10^{-6}"). The actual relative errors are 1.66e-6 at ρ=0.3 and 1.05e-6 at ρ=0.5. The error is set by the 0.01 time grid.
  2. "Terminal loss is $<2\times10^{-7}$ for all ρ≥0.003". The maximum over seeds at ρ=0.003 is **4.8e-7**; the mean is 1.5e-7.
  3. "$|L_1(t)|<10^{-17}$" in the commuting control. The maximum is **5.3e-17**.
  4. "log–log slope is −0.79 over ρ≤0.1". A fit on the mean recovery times gives **−0.82** (−0.82 on log-means, −0.87 on medians). The −0.89 for the two smallest ρ is correct.
  5. "GN variants reach $L_2<1\%$ in 4–7 steps, against 15–22 for GD". The per-ρ mean for GD is **18.6–24.4** (individual runs 12–46). For damped GN with δ=0.1, runs take up to 12 steps.
  6. "Half-recovery reached by step 600–1,600 at ρ≥0.1 in split". This range holds for the seed means (627–1,560), but individual runs have sd up to 720 steps, so the range does not hold per run.

  None of these changes a conclusion. Per the venue standard I was asked to apply, though, each one counts as a result not backed by run code. All must be corrected.
- **The E1 check is mislabelled.** The "numeric gradient flow" in E1 is not numerical integration. `L1_traj_flow` evaluates the exact solution through the same eigendecomposition of $H_\rho$ that the closed form uses, sampled on a 0.01 grid. Table 1 therefore checks the algebra of Theorem 1 (the $s$, $r$ reduction and the maximization), not the ODE solution. That check is still useful, but "match numerical integration" overstates it.

### 2a. MLP rerun note
See the appended section at the end of this review.

## 3. Correctness of the theory (line-by-line)

- **Prop. 1.** Correct. For PSD $A,B$ and $\alpha,\beta>0$, $\ker(\alpha A+\beta B)=\ker A\cap\ker B$.
- **Thm 1.** Correct.
  - The coefficients sum to $u^\top u_\perp=0$.
  - At $t^\star$, $e^{-\lambda_1t^\star}=r^{r/(1-r)}$ and $e^{-\lambda_2t^\star}=r\cdot r^{r/(1-r)}$, so the maximum is $r^{r/(1-r)}(1-r)$.
  - As $\rho\to0$, $s^2\to\sin^2\theta\cos^2\theta$, and $F_\infty=\tfrac12ac^2\sin^2\theta\cos^2\theta$ (0.125 at 45°).
  - The limit statement "$G\to F_\infty$" is fine. Note that at $\rho=0$ itself there is no interior peak, which is a discontinuity in definition.
- **Thm 2.** Correct. The GD statement is exact because $\Sigma_1^{1/2}e_0=0$.
- **Thm 3.** Correct. The (a)⇒(b) direction uses Thm 2 for every $e_0\in\ker\Sigma_1$; (b)⇔(c) is standard.
- **Thm 4.** Correct.
  - Duhamel with contractive semigroups gives $\|e^{-(A+B)t}-e^{-At}\|\le t\|B\|$.
  - With $t=\tau/(1-\rho)$ and $\|\Sigma_1^{1/2}\|=\|\Sigma_1\|^{1/2}$, the stated $\|\Sigma_1\|^{3/2}$ factor follows.
  - The second display and the liminf follow.
  - Contribution 4 writes the result as "$G\ge F_\infty-O(\rho\log(1/\rho))$". It is really a bound on $\sqrt{G}$; squaring gives $F_\infty-O(\sqrt{F_\infty}\,\rho\log(1/\rho))$. That is fine, but it should be stated precisely.
- **Prop. 5.** The eigenvalue bound is correct: $z_R\in\ker\Sigma_2\cap\mathrm{range}H_\rho$, so the Rayleigh quotient is $\le\rho\|\Sigma_1\|$. **But the proposition does not support the text built on it.**
  - It shows that *some* eigenmode is slow and gives an *upper* bound on $L_1(t)$.
  - It gives **no lower bound on recovery time**. Nothing in the statement prevents the slow mode from carrying negligible $L_1$ weight.
  - Yet the abstract, the contributions, §4 and the Discussion claim that "recovery takes time of order $1/\rho$" and that the slowest recovery mode has time constant "at least $1/(\rho\|\Sigma_1\|)$", implying $L_1$ stays large.
  - A genuine lower bound follows easily from the Duhamel estimate in Thm 4: $\sqrt{2L_1(t)}\ge\sqrt{2F((1-\rho)t)}-\rho t\|\Sigma_1\|^{3/2}\|e_0\|$, which stays positive until $t\gtrsim\sqrt{F_\infty}/(\rho\|\Sigma_1\|^{3/2}\|e_0\|)$. The authors should prove and state it.
  - Empirically, the measured slope is −0.82 / −0.89, not −1.
- **Thm 6(a).** Correct, but its content is much more modest than the paper suggests.
  - Under exact GN, $e(t)=e_0-(1-e^{-t})P_Re_0$. This is a **straight line** from $\bar w_1$ to a joint interpolator.
  - $L_1$ is a PSD quadratic that is zero at both ends of the segment, so it is zero along it.
  - So Thm 6(a) is the observation that the straight path between two zero-loss points of a convex quadratic has zero loss. *Any* method that moves along a straight line to a point in $\arg\min L_1\cap\arg\min L_2$ has no gap: exact projection, a min-norm solve, or a proximal step onto the joint solution set.
  - Kamath et al. (2024), which the paper cites, already report empirically that a low-loss linear path to the next minimum exists and that SGD does not follow it.
  - The paper calls Thm 6(a) "the sharpest statement in the paper" and rephrases it as "the stability gap is an artefact of Euclidean gradient geometry". Both are overstatements. What the theorem shows is that GD's path in a non-commuting quadratic is curved; it does not show that the joint-curvature metric is privileged.
  - The theorem also hides a severe ill-conditioning. The $\rho\to0^+$ limit of the GN endpoint has $L_1=0$, yet at $\rho=0$ the endpoint has $L_1=F_\infty$. Through $\lambda^+_{\min}\le\rho\|\Sigma_1\|$, any curvature error of order $\rho$ destroys the result. The authors partly acknowledge this through Thm 6(b) and the failed MLP damping test.
- **Thm 6(b).** Correct. The mean-value step $|e^{-\alpha t}-e^{-t}|\le(1-\alpha)te^{-\alpha t}\le(1-\alpha)/(\alpha e)$ is valid, and $\ker H$ modes do not move under either flow. Empirically it is loose by several orders of magnitude.
- **Thm 6(c) / "diagonal does no better than GD".**
  - The first-order statement is correct.
  - "Diagonal preconditioning ... does no better than GD" (Contribution 5) rests on a single experiment with one diagonal preconditioner, $\mathrm{diag}(H)+10^{-3}\lambda_{\max}$, not on a theorem.
  - That preconditioner is a Jacobi preconditioner of the exact Hessian, not Adam. The "(Adam-like)" label is misleading.

**Overall:** the proofs are correct and pleasantly checkable. The Prop. 5 → "recovery $\propto1/\rho$" inference is a real gap between what is proved and what is claimed.

## 4. Novelty and prior work

- I searched for prior theoretical treatments of the stability gap and found none that give an exact linear-model account. The literature I found (De Lange et al. 2023; Hess et al. 2023; Harun & Kanan 2024; Kamath et al. 2024; Łapacz et al. 2025; Rodriguez-Garcia et al. 2025) is empirical or conceptual, and the endpoint theories (Evron et al. 2022; Lin et al. 2023; Goldfarb & Hand; Mahdaviyeh et al. 2025) do not study the within-phase transient. On that basis, the framing as a first exact model of the gap appears novel.
- **The mathematical content is elementary**, and the paper says so. It is a 2×2 eigendecomposition, a Taylor expansion, an invariant-subspace argument, and a Duhamel bound. Non-monotone transients of a quadratic form along $e^{-Ht}$ with non-commuting matrices are classical (two-timescale linear systems).
- **The conceptual messages largely restate existing empirical or conceptual findings in an exactly solvable form.**
  - "Replay's gradient vanishes at the old optimum" is De Lange et al.'s argument.
  - "The gap persists even under joint training" comes from Kamath et al. and Hess et al.
  - "There is a low-loss linear path that SGD does not take" comes from Kamath et al.
  - "Optimize differently, not only a different objective" is Hess et al.'s thesis.
- The genuinely new quantitative items are:
  1. the $(1-\rho)^2$ one-step law;
  2. the existence criterion;
  3. the small-$\rho$ insensitivity bound.
- These are useful but incremental for an ICML main-track theory paper. A theory contribution at this level would ideally include at least one of the following:
  - SGD / finite-buffer analysis (the actual practical replay regime);
  - more than two tasks;
  - a linearized-network (NTK) extension with a proof that the predictions survive feature learning to first order;
  - a matching upper and lower bound on recovery time.

## 5. Experiments and statistics

- **The one-step law check (ratio 0.994, "within 2%") is weak evidence.** For a single small GD step from a minimum of $L_1$, a second-order Taylor expansion is essentially exact. The match mostly verifies Taylor's theorem with the measured Hessian, not the theory's broader predictions. It should not be presented as the headline quantitative confirmation. A multi-step test of the normalized short-time curves (e.g. $L_1(k)/(1-\rho)^2$ for $k=1..10$) would be more informative.
- **Sample sizes and uncertainty.**
  - The study uses 4 seeds (2 for the damping study) and reports no confidence intervals or tests.
  - Some reported effects are within about 1 sd, for example perm ρ=0.3: GD 0.087±0.020 vs. GGN-flow 0.050±0.004.
  - The "36–60%" figure should come with CIs over seeds, or with paired differences, since the seeds are shared across methods.
- **Confounds in the optimizer comparison.**
  - GGN-flow uses lr 0.2 for 150 steps; GD uses $\eta=1/\lambda_{\max}$ for 4,000 steps; Adam uses lr 1e-3.
  - The gap is measured as max over the trajectory, so methods that take smaller effective steps early can appear to have a smaller gap.
  - A comparison at matched new-task loss (as P4 itself proposes) or at matched $L_2$-reduction-per-step is needed. One option is to plot $L_1$ against $L_2$ along each trajectory: a Pareto curve of forgetting vs. learning.
  - Without this, "curvature-aware updates halve the gap" is not cleanly separated from step-size and schedule effects. The paper partly concedes this for Adam.
- **Train vs. test.** $G$ is measured on the old task's training loss and accuracy on its test set. This is fine, but it should be stated consistently.
- **Scale.** One 1,210-parameter network on 8×8 digits, full-batch, squared loss, entire old set replayed. This is far from any regime in which the stability gap is practically relevant (SGD, finite buffers, cross-entropy, deep nets). The authors acknowledge this, but it limits significance.
- **Honesty.** The paper reports its negative results: full-step GN fails, small-ρ elimination fails, damping scaling is unconfirmed, and recovery-time scaling is unmeasured. That candour is a real strength.

## 6. Is the sub-question sharp? Do the conclusions follow?

The sub-question is sharp and well posed: why is there a transient drop under replay or joint training, and what controls its depth and duration? The linear conclusions follow from the theorems, with the Prop. 5 caveat. Three conclusions stated for broader settings are over-reached:
- "The theory reframes the stability gap as an artefact of Euclidean gradient geometry." The paper's own MLP results show that curvature-aware steps do not eliminate the gap outside the local linear regime.
- The Discussion's account of Łapacz et al.'s head localization ("a shared linear head trained on new classes is exactly a linear least-squares problem of the kind analysed here"). This ignores cross-entropy, non-realizability of the head, and the moving features underneath it.
- The pre-registered predictions P2–P4 extrapolate from a regime where P5 (damping) already failed.

## 7. References

I spot-checked these by web search, and all exist with the stated venues:
- Kamath et al. (CVPRW 2024, pp. 4182–4186)
- Rodriguez-Garcia et al. (arXiv:2507.14056; both titles confirmed across versions)
- Mahdaviyeh et al. (arXiv:2506.04377; now also CoLLAs 2025 / PMLR 330, so the citation should be updated)
- Łapacz et al. (WACV 2025, pp. 7551–7560)
- Hess et al. (PMLR 249:37–61)
- Yoo et al. (ICML 2024, PMLR 235:57199–57216)

The content attributed to each matches what the sources say. I found no fabricated or misattributed references.

## 8. Strengths

1. Clean, correct, fully written-out proofs in a well-chosen minimal model, where the transient cannot be blamed on non-convexity or an unreachable optimum.
2. An exact existence criterion and a closed form give the field a precise vocabulary (fast/slow modes, eigenvalue ratio, $(1-\rho)^2$ law).
3. Excellent reproducibility: seeded, CPU-scale, one script; the linear outputs are byte-identical on rerun.
4. Unusually honest reporting of negative transfer results, and explicit falsifiable predictions.

## 9. Weaknesses

1. Several prose numbers do not match the code outputs (§2): the <1e-6 error, the <2e-7 terminal loss, the <1e-17 commuting residual, the −0.79 slope, the 15–22 GD steps, and the 600–1,600 half-recovery range.
2. Prop. 5 does not prove the "recovery takes order $1/\rho$" claim repeated in the abstract, contributions and discussion.
3. The "sharpest" result (Thm 6a) amounts to the observation that a straight path between joint minimizers of a convex quadratic has zero loss. The "Euclidean-geometry artefact" framing overstates it, and Kamath et al. already observed the empirical counterpart.
4. The mathematics is elementary and the conceptual messages mostly restate known empirical or conceptual findings, so novelty is incremental for a main-track venue.
5. The optimizer comparison in the MLP is not matched on step size or learning progress. There are 4 seeds, no CIs, and some differences are within about 1 sd.
6. The one-step "within 2%" confirmation is close to tautological (second-order Taylor at a minimum for one small step).
7. E1's "numerical integration" is the same eigendecomposition evaluated on a grid.
8. The diagonal preconditioner is labelled "Adam-like" but is a Jacobi preconditioner of the exact Hessian. The claim that diagonal preconditioning "does no better than GD" rests on one instance.
9. Practical relevance is limited: full-batch, entire-buffer replay, two tasks, no SGD, a tiny MLP.

## 10. Questions for the authors

1. Can you prove a lower bound on recovery time, e.g. $t_{1/2}\ge c\sqrt{F_\infty}/(\rho\|\Sigma_1\|^{3/2}\|e_0\|)$, so that the $1/\rho$ claim is a theorem?
2. Does any straight-line or projection method, e.g. GD followed by projecting onto the joint solution set, or a GEM-style constrained step, achieve zero gap in the linear model? If so, what is specific to Gauss–Newton beyond producing a straight path?
3. In the MLP, how do GD with a smaller step (e.g. 0.2/λmax), or GD at matched $L_2$ progress, compare with GGN-flow? Please show $L_1$-vs-$L_2$ trajectories.
4. Can the theory be extended to SGD with a finite replay buffer, where replay noise could change the $(1-\rho)^2$ law?
5. Under what conditions on the NTK does Thm 3's commutator criterion predict the observed gap sizes in the MLP? P5 suggests this test, and it is runnable at your scale.

## 11. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 5 |
| Rigor | 6 |
| Correctness | 6 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: Major revision** (weak reject in the current form for ICML main track; a solid workshop paper as is).

The most valuable single change would be to replace the elementary or near-tautological parts with a result that moves beyond them. That means a proven recovery-time lower bound, plus either an SGD/finite-buffer extension or a step-size- and progress-matched optimizer comparison that isolates curvature from step size. The paper should also tone down the "Euclidean-geometry artefact" framing of Thm 6(a) and fix the prose-level numeric mismatches.

---
### Appendix: MLP rerun status (§2a)
The full MLP rerun (`run_mlp.sh` plus the damping jobs) was started on the authors' code. When this review was submitted, the first batch of jobs was still running, and its phase-1 outputs (losses and accuracies for every scenario and seed) were **bit-identical** to the submitted `mlp_log.txt`. I therefore checked every MLP number in the paper against the submitted per-run JSON files using my own aggregation script. The results:
- Table 3, Table 4, the one-step ratios (56 runs), G(ρ)/G(0), the accuracy trajectories and the reduction percentages all match those files (see §2).
- I have not confirmed bit-for-bit regeneration of the final MLP JSONs. Note that the rerun overwrites `results/`; a backup of the submitted files was kept.
