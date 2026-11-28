# Review (Round 1): "Rate, Race, or Fate: Verifier Tails and Exploit Races Limit How Fast RL Spreads Beyond Math and Code"

**Reviewer role:** area chair level, independent
**Research question:** Can RL make similarly fast progress in domains other than math and code? (强化学习能否让模型在数学、代码以外的领域同样快速进步？)
**Recommendation:** Major revision (weak reject in current form). **Overall: 5/10**

---

## 1. Summary

The paper asks what limits RL with imperfect (judge- or rubric-based) verifiers outside math and code. It makes four contributions.

1. **Theorem 1 (CGF identity).** On the exponential-tilt family π_t ∝ π0·e^{tf}, logit A(t) = logit A(0) + K_c(t) − K_w(t). This family contains the KL-regularised optimum and the replicator flow. Corollaries give a rate law dA/dt = A(1−A)·J_t with an on-policy Youden index J_t, a fate law set by the essential suprema of f on correct and wrong responses, an offline non-identifiability statement (Prop. 2), and a three-mode closed form with an error-floor law 1−A* ∝ h0^κ.
2. **Tabular GRPO simulations.** KL-regularised GRPO matches the tilt when λ is large. Unregularised softmax PG "starves" rare exploits, so the outcome is a race with a sharp h0 threshold.
3. **A toy sequence task** (binary strings of length 10) with factorised and AR1 policies. It gives a measured takeover boundary J* ≈ 12.6·h0^0.80 (factorised) and 8.9·h0^0.76 (AR1), t90·J ≈ 80 steps, and an early-warning on-policy audit (AUC 0.98) that beats an "offline audit" (AUC 0.77).
4. **A pre-registered protocol** for LLM-scale tests.

The paper is clearly written and honest about scope ("toy numbers do not transfer quantitatively"). The code is small, seeded and fully reproducible. My concerns are novelty against recent closely related work that the paper omits, a central practical claim (on-policy beats offline auditing) that is an artefact of a weak baseline, and the distance between the evidence and the motivating question.

## 2. Reproduction

I copied `code/` to a scratch directory and reran every script (`run_all.sh`, with E4–E7 run in parallel copies). **Every output JSON (exp1, exp1b, exp2, exp2c, exp4, exp5, exp6, exp7) matches the stored `results/*.json` bit for bit.** I checked each number in Sections 5–7 against the JSONs, and all match within rounding:

- identity error 1.6e-14
- κ fits 0.93333 / 0.87499 / 0.82352 / 0.77775
- A* = 0.853 / 0.975 / 0.996 and 0.392 / 0.596 / 0.773
- Part A table (all 12 cells)
- Part B peaks of 0.93, 0.93 and 0.61
- E2c: r = 0.78, mean |log10 error| = 0.36, median ratio 0.87
- E4 collapse counts of 3/3, 1/3 and 0/3, with peaks 0.56–0.85
- E5 t90 values (775 / 390 / 270 / 190 / 140 / 100) and balanced accuracies 0.721 / 0.714 / 0.789 / 0.599
- E6 values 0.455 / 0.332, 0.784 / 0.944, and ≥0.964
- E7 power laws 12.62·h0^0.800 and 8.89·h0^0.762, and AUCs 0.984 / 0.767 / 90%

Figures are produced by the same scripts. **No mismatches.**

## 3. Reference spot-check (web search)

I verified the following. Each exists and says what is claimed.

- Rad et al. 2026, arXiv:2601.04411, ICML 2026: replicator flow; Youden's J decides the outcome; "rate, not fate".
- Cai et al. 2025, arXiv:2510.00915: asymmetric noise channel; backward and forward corrections.
- Lv et al. 2025, arXiv:2505.22653: 40% flips give 72.02% vs 75.85% on MATH-500, Qwen-2.5-7B.
- Zhao et al. 2025, arXiv:2507.08794, NeurIPS 2025: "master keys"; false positive rates up to about 80%.
- Huang et al. 2025, arXiv:2505.22203: v2 title "From Accuracy to Robustness"; model-based verifiers are hacked during RL.
- Shao et al. 2025, arXiv:2506.10947, ICML 2026: spurious rewards; clipping-bias explanation. A later paper, arXiv:2512.16912, disputes the clipping account, which the paper could note.
- Gunjal et al. 2025: Rubrics as Rewards; gains over Likert-based judges.
- Su et al. 2025, arXiv:2503.23829: RLVR in medicine, chemistry, psychology, economics and education.

I found no fabricated or misrepresented citations.

## 4. Strengths

- **S1. Clean, correct elementary theory.** I checked every derivation line by line.
  - Theorem 1 and Corollary 1, including dJ_t/dt = Var_c − Var_w, are correct.
  - In Corollary 3, setting d log A/dt = α − f̄_t = 0 gives w0(α−β)e^{βt} = h0(η−α)e^{ηt}. With R = e^{(η−β)t*}, A* = 1/(1 + (w0/c0)R^{−κ}(η−β)/(η−α)). This matches the paper, and the rise condition w0(α−β) > h0(η−α) is right.
  - κ = J/(η−β) and the 2^{1/κ} factors (2.3 and 6.8) check out.
- **S2. A useful conceptual split.** Separating class-conditional from response-dependent error explains the apparent conflict between "noise is benign" (Lv et al.) and "verifiers get hacked" (Huang et al., Zhao et al.).
- **S3. The idealised theory and practical PG are compared honestly.** It is informative that KL-GRPO reaches the tilt only for large λ, with the effective-temperature correction t_eff = 1/(λσ*) verified (Part A). So is the finding that softmax PG sits in a metastable exploit-free state at small λ (E6).
- **S4. Exemplary reproducibility.** The code is small, seeded and fast, and every number traces to a JSON.
- **S5.** The pre-registered protocol is concrete and falsifiable.

## 5. Weaknesses

### W1 (major): Missing closely related prior and concurrent work, which weakens the novelty claims

The paper frames response-dependent, exploitable verifier error as "the case that theory leaves open". Several 2024–2026 papers address exactly this, and none is cited.

- **Egashira et al. 2026**, "Delay, Plateau, or Collapse: Evaluating the Impact of Systematic Verification Error on RLVR" (arXiv:2605.02909, COLM 2026). It studies systematic, response-dependent false positives in RLVR with controlled experiments on a real LM. It finds that outcomes range from delay to plateau to collapse, and depend on the error *pattern* rather than the overall error rate. This is the empirical core of Prop. 2 and E5, already shown with LLMs.
- **Moya, Thornley & Lin**, "Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control" (arXiv:2609.35677). It uses gradient flow with a fixed verifier to characterise when reward rises while correctness falls, and proposes audit-based correction. This overlaps directly with the rate and fate analysis and the "audit" recommendation.
- **"Sparse Reward Auditing for RLVR under Policy-Dependent Verification Error"** (NeurIPS 2026). This is the on-policy audit idea, studied directly.
- **Kwa, Thomas & Garriga-Alonso 2024**, "Catastrophic Goodhart" (NeurIPS 2024). It shows that the KL-regularised optimum's outcome under misspecified reward is governed by the *tails* of the error distribution. Corollary 2 ("fate is set by upper tails, not means") is a bounded-reward, binary-label instance of this known phenomenon.
- **"Checking the Checkers"** (arXiv:2610.17824), a 2026 survey of verifier noise in RLVR.

Corollary 2 itself (tilting by t → ∞ concentrates on the argmax of f) is a textbook property of Gibbs measures. Theorem 1 is acknowledged as elementary. What remains novel is mainly the three-mode closed form, the error-floor exponent κ, and the toy-scale race measurements. The paper must position itself against the work above and narrow its novelty claims.

### W2 (major): "On-policy audit beats offline audit" is an artefact of a weak offline baseline

The headline practical claim (abstract item 3, Section 7 E7b, and "Audit verifiers on-policy, not offline") compares the 100-step on-policy slope (AUC 0.984) with a *single noisy scalar*: log(false/true accepts) from 256 samples at step 0 (AUC 0.767). The comparison is not fair. The on-policy statistic uses 11 audits (about 2,800 samples) along the trajectory. The offline one is an aggregate rate, which the paper's own theory (Prop. 2, Cor. 2) says *cannot* capture the tail. An offline audit consistent with the theory would look at disaggregated acceptance, that is, at the tail.

I tested this on the same 60 configurations (same seeds, script reused from `exp7`). Purely offline features computed from the **base policy only, with no training**, predict collapse as well as or better than the on-policy audit:

| predictor (base policy only unless noted) | AUC |
|---|---|
| on-policy slope, first 100 steps (paper) | 0.984 |
| step-0 log(FA/TA), the paper's "offline audit" | 0.767 |
| offline mean-acceptance gap E_π0[f \| x0=0] − E_π0[f \| x0=1] (initial gradient on the shared feature) | **0.988** |
| offline h0/J, or the margin J*(h0) − J | **1.000** (rule accuracy 98.3%) |

The informative distinction is therefore **aggregate vs disaggregated** (tail-aware) auditing, not on-policy vs offline. To be fair, in a real LLM the exploit feature is unknown, which is the argument for on-policy auditing. But the experiment as designed does not show that. Also, at J ≈ 0.8 the run reaches 90% accuracy in about 100 steps, so the "early" window often overlaps the decisive phase. The early-warning result is then closer to observing the outcome than to forecasting it. The claim must be reframed, and the baseline must include tail-aware offline audits, matched sample budgets, and confidence intervals on the AUCs (n = 60).

### W3 (major): The evidence is far from the research question

The question is whether RL can progress as fast outside math and code. The evidence is (i) elementary theory for idealised dynamics that the paper itself shows practical GRPO does not follow, and (ii) toy simulations with at most 1,024 responses, a single prompt, a hand-built exploit feature (x0 = 0 ∧ x8 = x9 = 1) with fixed η = 0.95, plain SGD and no clipping. The motivating domains (medicine, law, writing) have no counterpart: there is no estimate of J, h0 or the tail of any real judge. The conclusion "RL can move a non-math domain only at a rate set by on-policy discrimination and only below an exploit boundary" is a restatement of the model's assumptions, not an empirical finding about those domains. At minimum, the paper should measure J and a tail proxy for at least one real judge, such as the master-key FPRs from Zhao et al. 2025 on a non-math benchmark, so that the toy boundary can be placed relative to real verifiers.

### W4 (moderate): The "takeover boundary" power law is a fitted summary of an engineered low-dimensional system and is not a power law

- **Local slopes drift.** The local log–log slopes of J*(h0) fall steadily: 1.07, 0.86, 0.85, 0.67, 0.46 and 0.41 for the factorised policy. The fitted exponent 0.80 ± 0.05 (OLS standard error, 7 points) averages over strong curvature. J* is bounded above by η − β = 0.8, where the exploit loses its acceptance advantage, so saturation is built in. Prediction P2 ("slope > 1 in log h0* vs log J") inherits this fragility, and is in fact sensitive to which range of h0 is used.
- **The boundary could be derived.** In the factorised model, collapse is a race in essentially one logit (x0). At initialisation the x0 gradient sign changes at J ≈ 0.8·p_s²/0.1296 ≈ 15.4·h0. The full boundary could be obtained from the mean-field ODE over 10 logits instead of from bisection with 8 seeds and 6 halvings (resolution about 0.015 in J). A derived boundary would be far more valuable than the fitted constants 12.6 / 0.80.
- **Degenerate design point.** E5's J = 0.8 point sets α = 0.95 = η, so the exploit is no longer an exploit there. "0/8 collapse at J = 0.8" is uninformative and should be stated as such.

### W5 (moderate): Statistics

- Most cells use 3 seeds (E2, E4, E6) or 8 (E5, E7). No confidence intervals are given for the collapse fractions, AUCs, t90, power-law parameters or the E2c correlation.
- "t90·J within 77.5–85.5, matching 1/J to within about 10%" uses medians over sometimes only 2 non-collapsing seeds (J = 0.1). It also ignores that std-normalised advantages predict a rate ∝ J/σ, not J. The agreement with 1/J needs explaining rather than being cited as confirming Corollary 1, which is derived for the tilt, not for softmax PG.
- **Small bias in Part A.** With centred advantages the group mean includes the sample itself, so the expected update is scaled by (G−1)/G. The fixed point is then t = (G−1)/(Gλ), not 1/λ. The resulting error is small near the peak, but it should be stated, especially since the claimed agreement is "within 0.003".

### W6 (minor–moderate): Proposition 2 is mis-stated

"For any offline TPR > FPR" is false at the boundaries.

- The construction needs FPR > 0, because lowering f on the remaining incorrect mass requires δ′ ≤ FPR(1−A0).
- It also needs TPR < 1. If TPR = 1 then f ≡ 1 on correct responses, s_c = s_w = 1, and A_t → A0/(A0+δ′), not 0.

The parameter δ is introduced and then plays no role beyond δ′ ≤ δ. Please restate as: for any 0 < FPR < TPR < 1 and any δ′ ∈ (0, FPR(1−A0)].

### W7 (minor): Other issues

- The rate law (Cor. 1) and fate law (Cor. 2) hold for the idealised objective. The abstract says "Learning speed is A(1−A) times an on-policy Youden index", and the Discussion's "Rate" item cites E5 as confirming it. The paper should say clearly that the E5 match is empirical and for a different dynamics.
- The E4 "parameter sharing makes exploits discoverable" result depends heavily on the chosen exploit encoding. x8 and x9 are irrelevant to correctness, so they drift up for free. Show at least one alternative encoding, for example an exploit feature that is partly anti-correlated with correctness, or one with η < 1 drawn from a distribution.
- The Discussion recommends entropy bonuses and natural-gradient preconditioning as *risk* factors. This is a novel and testable prediction (P5) but is untested even in the toy. One run with an entropy bonus or a natural-gradient update in E2/E4 would be cheap and would strengthen the paper considerably.
- Mixing "≈ four times as many steps" (J = 0.2 vs 0.8) with Lv et al. is misleading. Lv et al. report similar *convergence speed* at 40% flips. The paper should note this apparent tension with its own 1/J prediction rather than cite Lv et al. as consistent.
- Figure 6's tilt prediction at λ = 0.3 (0.595) lies *below* GRPO (0.784 / 0.944), while the text implies the tilt is an upper "cap". The "metastable" interpretation should be quantified, for example as exploit mass versus time.

## 6. Questions for the authors

1. How does your work differ from Egashira et al. (2026), Moya et al. (2026), Sparse Reward Auditing (NeurIPS 2026) and Kwa et al. (2024)? What is new relative to each?
2. Can you compare the early-warning audit with tail-aware offline audits, such as per-cluster acceptance and max-acceptance on wrong outputs, at matched sample budgets, with bootstrap CIs on the AUC?
3. Can the factorised-policy boundary J*(h0) be derived from the 10-dimensional mean-field ODE? If so, does it explain the curvature (local slope falling from 1.07 to 0.41)?
4. What are J and a tail proxy for a real LLM judge on a non-math dataset (for example, Su et al.'s data)? Where would that judge sit relative to your boundary?
5. Does adding an entropy bonus or a natural-gradient update to E2 or E4 move outcomes toward the tilt's pessimistic fate, as your Discussion predicts?
6. How sensitive are the E4/E5 conclusions to the learning rate (0.5), group size (16) and exploit encoding?

## 7. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision (weak reject as submitted).**

Correctness is good: the reported numbers are real and the derivations hold up, apart from the Prop. 2 boundary cases. The submission falls short of a top venue for three reasons. The theory is elementary and overlaps with recent work that goes uncited. The main practical claim (on-policy beats offline auditing) does not survive a fair offline baseline. The toy evidence answers a narrower question than the one posed.

**Single most valuable change:** add a real-verifier anchor (measure J and an exploit-tail proxy for an actual LLM judge on a non-math task) and replace the strawman offline audit with tail-aware offline baselines. Then position the contribution explicitly against Egashira et al. 2026, Moya et al. 2026 and Kwa et al. 2024.
