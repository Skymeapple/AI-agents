# Review (Round 1): "Proxy-Only Detectors Cannot See Shared Exploits"

Reviewer role: area-chair-level, independent. Code rerun: yes (all three scripts, on a copy, in a scratch directory). References checked: 10.

## Summary

The paper builds an "exploit-mixture" toy model of reward hacking. A base policy rarely emits exploits that the optimised proxy overrates by mu, and each held-out proxy shares an exploit with probability q. The model is solved in closed form under exponential tilting. The paper then claims four results:

- (T1) d/dlam E[g] = Cov_{pi_lam}(g, r1).
- (T2) A peak-prevalence identity: the exploit share at the gold peak is strictly positive.
- (T3) Non-identifiability: a "twin world" W' in which shared exploits are genuinely good has the same proxy law, which gives a minimax regret lower bound for proxy-only stopping rules.
- (P4) A Hansen–Hurwitz estimator of the on-policy covariance, using leverage-weighted gold audits, is unbiased.

Simulations compare a covariance-sign stopping rule, first-hack alarms, and a held-out-gap alarm tuned at q = 0. A small learned-RM study (sklearn MLPs and ridge on 12-D synthetic features) measures q-hat between 0.42 and 1.03.

## Reproducibility check

I reran `exp_lowerbound.py`, `exp_theory.py` and `exp_learned_rm.py` on a copy of `code/`. All three result JSONs came out **byte-identical** to the shipped `results/`. I checked every number in Tables 1–3, Sections 5.1–5.3, 5.5 and 5.6, and the Discussion against the outputs, and all of them match to the stated rounding:

- Exp1a error: 0.0238.
- Peak identity relative error: 3.98e-4.
- Best-of-n peak: n = 40, G = 1.265.
- Covariance identity values.
- RMSE table.
- Table 1 entries (gap-alarm overshoot at q = 0 is 0.565, reported as 0.57).
- Table 2 entries.
- Offline bounds: 0.18, 0.86, ~1.
- Learned-RM regrets and q-hat values.
- pi* values of 0.094, 0.062 and 0.036 for delta = 1, 2, 4.

Figures 1 and 4 match the descriptions. This part of the submission is exemplary.

One sentence has no code behind it: "in W' the same audit rule observes positive covariance and continues, as it should" (Section 5.5). The audit rule is never simulated in W'. The claim is plausible, but as written it is an unbacked empirical statement.

## Reference spot-check

I verified these by web search: Deshpande et al. 2026 (TRACE, arXiv:2601.20103, ICML 2026, 63% best detection in the contrastive setting); Gabor et al. 2025 (EvilGenie, 2511.21654); MacDiarmid et al. 2025 (2511.18397); Fluri et al. 2025 (ICML, PMLR 267); Laidlaw et al. 2025 (ICLR, chi-squared occupancy regularisation); Moskovitz et al. 2024 (ICLR, periodic ground-truth queries); Kwa et al. 2024 (NeurIPS, light-tailed vs heavy-tailed); Miao et al. 2024 (InfoRM, NeurIPS, latent outlier detection); Beirami et al. 2025 (ICML, log n − (n−1)/n is an upper bound). All exist and say what the paper attributes to them. I found no fabricated or misattributed citation.

## Strengths

1. The model is clean and the closed forms are correct. I checked Prop. 0, the derivative in T2 (G' = (1−pi)[sigma_g^2 − mu pi (sigma_g^2 lam + delta)]), T3(b) (at q = 1, G_W' = sigma_g^2 lam + pi mu) and P4 line by line, and found no errors.
2. Reproducibility is fully deterministic: seeds are fixed, runtime is a few minutes, and the outputs are bit-for-bit identical.
3. The paper is honest about scope. It runs no LLM experiments, gives a pre-registered falsification protocol, and reports negative results: the control variate hurts, and H3 breaks the q-hat ordering.
4. The practical framing is useful. The stop/continue decision depends on a gold-gradient quantity, not on hack counts, and zero-tolerance tripwires couple the stopping point to the audit budget.

## Weaknesses

### Major

**M1. Novelty is thin. Each theorem is either textbook or nearly tautological.**

- T1 is the standard exponential-family score/covariance identity, which the paper itself calls "elementary".
- P4 is Hansen–Hurwitz (1943) applied verbatim.
- T2 is a one-line first-order condition in a two-component model.
- T3 formalises an observation that holds by definition: if gold never enters the proxy-generating process, proxies cannot identify gold. Any gold reassignment on a proxy-agreed region gives a twin.

The value of the paper therefore lies in assembling these pieces into a stopping recipe. Moskovitz et al. (2024) already periodically query ground-truth reward during a single RL run to locate proxy points. That is an on-policy gold-audit approach to the same stop/continue problem, and the paper should position itself against it much more carefully.

**M2. The headline claim that the rule is "invariant to shared vulnerability" holds by construction.**

- In model W, the joint law of (g, r1) does not depend on q. Only B_2, and hence r2, does.
- The non-CV covariance estimators (uniform and leverage designs) use only g and r1.
- Their regret is therefore exactly q-independent in distribution. The 0.016–0.020 spread across q in Table 1 is pure Monte Carlo noise.

Presenting this as an empirical finding ("the leverage-audited covariance test is invariant to q") overstates the evidence. It is a property of the estimator's inputs, and the paper should say so plainly.

**M3. The abstract misstates the lower bound.**

- The abstract says "every proxy-only stopping rule incurs minimax gold regret bounded below (numerically 1.09 at q = 0.5 and 1.81 at q = 1)".
- Those numbers are the deterministic-lam min-max column of Table 2.
- Any realistic proxy-only rule stops at a random lam, because the proxies are sampled. Theorem 3(c) itself only gives 1/2 inf (R_W + R_W') for such rules, which is 1.016 and 1.205.
- Moreover, the size of the bound comes entirely from the authors' choice of twin gold (mu + g0). With a different reassignment the bound can be made arbitrarily large, or arbitrarily small. The specific numbers carry no information beyond "positive". They should be reported as illustrative, or stated as a supremum over admissible twins.

**M4. Baselines are weak or straw-man, and the key gold-using baseline is missing.**

- *Top-proxy first-hack alarm.* Exploits sit at proxy +6, so this alarm fires at lam = 0 by construction (regret 0.426, which is the entire peak gain).
- *Zero-tolerance hack classifiers.* These are not how hack classifiers are usually deployed. A rate-threshold variant tuned like the gap alarm should be included.
- *Proxy-only detectors.* Only one is tested: a linear-trend gap alarm tuned in-sample on the very 200 q = 0 trajectories it is then scored on, which biases its 0.007 optimistically. Ensemble-variance and uncertainty detectors (Coste; Eisenstein; NeurIPS-2024 lightweight uncertainty; InfoRM-style outlier scores) are discussed but never run.
- *Missing gold-using baseline.* With the same label budget, compare to directly estimating G(lam) from m labelled on-policy samples per checkpoint and stopping when it decreases. Add a Moskovitz-style proxy-point search as well. Without these, the paper shows that "using gold beats not using gold", which is unsurprising. It does not show that the *covariance* is the right quantity to audit.

**M5. The label budget is under-reported.** "20 labels per checkpoint" sounds small, but the rule audits at every checkpoint until it stops: about 12 checkpoints, so about 240 labels, before reaching lam* ≈ 0.59. The comparison with offline labels in Section 5.5 ("offline labels ... whereas the on-policy audit used 20") is therefore misleading. Report cumulative labels, and compare at equal total budgets.

### Moderate

**M6. Statistical reporting.**

- No confidence intervals or standard errors are given for any regret or RMSE.
- The stopping rule has no multiple-testing or anytime-valid control (acknowledged).
- Overshoot of 0.43–0.50 at z = 0 means about half the runs stop late. With a steep post-peak decline, the regret looks good only because the step size is 0.05.
- There is no sensitivity analysis over mu, delta, p, rho, N or the schedule step (acknowledged).

**M7. The learned-RM study is too small to support the claim that "q-hat is predicted by data overlap more than by model class".**

- It uses 4 held-out RMs and one training seed per RM.
- The exploit region is hand-placed off-support.
- q-hat exceeds 1 (1.03).
- The gap alarm's maximum regret is only 0.14 out of 1.53, so the dramatic Theorem 3 failure does not show up with learned models. H3 is a counterexample to the q-ordering.

**M8. The link from exact tilting to PPO is asserted, not demonstrated.** The claim that T1 "also holds for the batch-level empirical analogue when lam is replaced by a natural-gradient step on a finite pool" is stated without proof. The 6.6e-4 check in the learned-RM study is again exact tilting on a finite pool, which is the same identity.

### Minor

- `exp_learned_rm.py` uses the disagreement design *without* the control variate, whereas Exp. 3 uses "disagreement+CV". Table 3's label is correct but the inconsistency should be made explicit.
- Contribution 5 says an H4 ensemble "eliminates the decline". It is monotone only up to lam = 12 in one toy setting.
- The Section 5.5 sentence about the audit behaving correctly in W' needs a simulation.
- Exp1a reports its error only at checkpoints with ESS > 1e4. State which lam values those are.
- README time estimates (10–15 minutes) do not match the actual runtime (about 1.5 minutes).

## Questions for the authors

1. How does the covariance test compare to a gold-*mean* change test, and to Moskovitz-style proxy-point search, at equal cumulative label budget?
2. What is the minimax value over *randomised* stopping distributions (the actual game value), rather than the weaker 1/2 inf-sum bound? How does it depend on the choice of twin gold?
3. Can you simulate the audit rule in W' and report its regret there?
4. How does regret degrade under noisy or biased gold labels, and under non-tilting (PPO-like) updates?

## Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** (weak reject) |

## Decision

**Major revision / weak reject for a top venue.** The paper is correct, fully reproducible and honestly scoped. However:

- its theorems are elementary or hold by definition;
- its central empirical "invariance" holds by construction;
- the headline lower-bound numbers are overstated and depend on an arbitrary twin;
- it lacks the gold-using baselines that would show the covariance is the right audit target.

In its current form it is a reasonable workshop paper.
