# Response to Review Round 1 (p034)

Thank you for a careful review that also reran our code. We agree with most of the criticisms. Several of them exposed claims that were wrong or overstated, and we have corrected those rather than defended them.

Every new number below comes from code we wrote and ran in this environment:

- New scripts: `exp_revision.py`, `exp_lowerbound_v2.py`, `exp_ppo.py`, `exp_learned_rm_seeds.py`, `make_fig_revision.py`, plus the helper `rev_common.py`.
- Outputs: `results/revision_results.json`, `lowerbound_v2_results.json`, `ppo_results.json`, `learned_rm_seeds_results.json`, `derived_label_efficiency.json`, and `fig5`/`fig6`.

We reran the first-version scripts after adding standard-error fields, and they reproduced every first-version number exactly. No citation was added. The one new factual statement about prior work (Moskovitz et al.'s single-run variant queries ground truth periodically and runs a derivative-free search for proxy points) was checked against the ICLR 2024 paper's public abstract by web search.

## Summary of the main changes

1. **Gold-using baselines at matched cumulative label budgets (the highest-leverage request).** We compared these rules at *m* ∈ {5, 10, 20, 40, 80} labels per checkpoint, reporting cumulative labels and regret ± SE:
   - a direct gold-mean trend test, in plain, regression-estimator, rollback and oracle-tuned-window versions;
   - a scalar Moskovitz-style proxy-point search, untuned and oracle-tuned;
   - covariance tests (plain, windowed+rollback, anytime-valid, uniform).

   The results are in Table 2 and Fig. 5 (§5.5). The PPO-like optimiser repeats the comparison (Table 6a).
2. **Robustness tests the review asked for.** We now test:
   - a non-tilting, PPO-like optimiser (§5.8);
   - noisy and biased gold labels (§5.9);
   - one-at-a-time sensitivity to μ, δ, *p*, ρ, *N* and the schedule step, including transfer of tuned thresholds (§5.9);
   - the twin world W′ (§5.7).
3. **Corrected claims.**
   - "*q*-invariance" is now stated as holding by construction.
   - The lower-bound numbers are replaced by the exact randomised game value, reported as twin-dependent and illustrative.
   - The unproved PPO/natural-gradient claim is withdrawn and replaced by a proved exact identity.
   - The W′ claim was wrong, and is corrected by simulation.
   - The 20-labels framing is replaced by cumulative budgets.
   - "Eliminates the decline" is weakened.
4. **A larger learned-RM study.** It now covers 5 seeds × 8 held-out RM types, including a data-mixture dose series (§5.10).

## Major issues

### M1. Novelty is thin; positioning against Moskovitz et al. (2024)

**We agree that the individual mathematical facts are elementary, and the paper now says so plainly.**

- The Abstract and §1 ("What is and is not new") and §7 say this directly.
- T3(a) is now described as "holds by construction … a formalisation rather than a deep result", and is restated in general form: any gold reassignment on a proxy-agreed region gives a twin. This matches the reviewer's observation.

**We have sharpened the contribution to what is not textbook.**

- The decision problem (*when to stop*) and the question of *which gold statistic to buy*. This is now tested directly against gold-level and proxy-point alternatives at equal budgets (Table 2 and Fig. 5).
  - The derivative audit beats untuned gold-mean trend tests at every budget: at about 250 labels, 0.015 versus 0.118–0.244.
  - It also beats the in-sample oracle-tuned trend test at every budget above about 60 labels.
  - At the smallest budget, a windowed covariance rule is best (0.027 at 67 labels, versus 0.060 at 57 labels for the oracle-tuned trend test).
- Remark 5, which is new, gives the variance argument for auditing the derivative rather than the level. At our checkpoint spacing the level test needs 9.5–57× more labels, depending on window length (`derived_label_efficiency.json`).
- The exact value of the proxy-only stopping game, and how it depends on the unidentifiable twin (Proposition 3′, Tables 3–4).
- A new exact identity for **any** update, Proposition 1′: ΔG = Cov_{π_t}(*g*, π_{t+1}/π_t). It is one line, and we say so. It replaces an unproved claim and is what makes the audit applicable beyond tilting.

**Positioning against Moskovitz et al.** §2 now has a dedicated paragraph.

- Their method audits gold *level* as a function of proxy values, to find reusable proxy-point thresholds enforced by constrained RL.
- Ours audits the gold *derivative* for a per-checkpoint stop decision.
- We implemented a scalar proxy-point search as a baseline.
  - Untuned, its regret is 0.23.
  - With its warm-up length tuned in-sample, it reaches 0.020 at 327 labels, versus 0.015 at 247 for the untuned covariance test.
  - The tuned warm-up (*k*_min = 16–20 checkpoints) starts fitting only after the true peak, so tuning it encodes where the peak is.
- Reusing an estimated proxy point in a shifted environment gave regret 0.154 at δ = 4 and 0.102 at *p* = 10⁻², versus 0.024 and 0.016 for a fresh covariance audit (§5.9).
- We say explicitly that our scalar version does not test their full multi-component constrained-RL method (§7).

### M2. "Invariant to shared vulnerability *q*" holds by construction

**Agreed. The criticism is correct.**

- §3.2 now notes that the joint law of (*g*, *r*₁) does not depend on *q*.
- §5.4 states that the covariance tests' regret is "exactly *q*-independent in distribution", and that the 0.016–0.020 spread is Monte Carlo noise. We now report SEs (±0.002), and the spread lies within ±2 SE.
- The abstract and contributions no longer call this an empirical finding.
- What remains empirical, and is labelled as such, is that the proxy-only rules and the disagreement design are *not* *q*-invariant: they read *r*₂, …, *r*_K.

### M3. The abstract misstates the lower bound, and the numbers depend on the twin

**Agreed on both counts.**

- `exp_lowerbound_v2.py` computes the exact value over randomised proxy-only rules by linear programming over stopping distributions, V = min_F max(E_F R_W, E_F R_W′). It checks V against the dual max_α min_λ.
- For the proxy-consistent twin, V = 1.075 at *q* = 0.5 and 1.743 at *q* = 1. This is correctly bracketed by the first version's ½inf-sum (1.016, 1.205) and the deterministic min–max (1.089, 1.814) (Table 3). We also report the optimal mixtures.
- We swept the twin constant *c* (Table 4). V ranges from 0.003 (*c* ≤ −2) to 1.98 (*c* = 12) at *q* = 1, confirming that the magnitude is set by the twin.
- The abstract now gives V, says it is illustrative and twin-dependent, and quotes the range.
- We also justify the default twin: it is the one in which the agreeing proxies are *correct* (*c* = μ). We do not claim it is canonical.

### M4. Weak or straw-man baselines; missing gold-using baseline

**Agreed. All four sub-points are addressed (Exp. R1, R2; §5.5).**

- **Top-proxy first-hack alarm.** It is now explicitly labelled as firing at λ = 0 by construction and kept only as a reference (§5.4).
- **Hack classifiers as rate thresholds.** We added a hack-rate alarm with a *perfect* classifier on 20 calls per checkpoint, with its threshold tuned out of sample.
  - It is strong in W: 0.023–0.027, about 1.5–2× the covariance test's regret, and *q*-invariant by construction. We report this as a strength.
  - Its weaknesses show elsewhere: under transfer (0.111 at δ = 4 and 0.095 at step 0.1, versus 0.024 for the covariance test), and in the twin world (2.2–6.8).
- **More proxy-only detectors, tuned out of sample.**
  - The gap alarm's τ is now tuned on 200 independent *q* = 0 trajectories. Its regret goes from 0.011 at *q* = 0 to 2.17 at *q* = 1.
  - We added an ensemble-std alarm with *K* = 4. Its regret goes from 0.004 at *q* = 0 to 2.15 at *q* = 1. It is the *best* rule we tested at *q* = 0, and we say so in §6.
  - We did not run InfoRM-style or other learned-uncertainty detectors. In this model they are functions of proxy features, so Theorem 3(a) covers them. We acknowledge that an empirical test with learned latent features would be stronger.
- **Gold-using baselines at equal cumulative budget.** These are the main addition, described under M1.
  - The gold-mean trend test was run plain, with a regression estimator, with rollback, and with its window oracle-tuned. Its regret never fell below 0.055, even at about 950 labels.
  - The proxy-point search is described under M1.

### M5. The label budget is under-reported

**Agreed.**

- Every stopping table now reports mean cumulative labels until stopping. The headline rule uses about 245 labels.
- The abstract no longer says "20 labels" without the cumulative figure.
- §5.6 now compares offline labels with the ~245 cumulative on-policy labels. The offline argument still holds: offline labels of size 100 or 1000 contain a shared exploit with probability at most 0.18 or 0.86, whereas the on-policy audit concentrates labels where π(λ) ≫ *p*.

### M6 (in the major list). The Section 5.5 "continues, as it should" claim was never simulated

**We simulated it, and the claim was wrong for the plain sign test (Exp. R3, §5.7, Table 5, Fig. 6).**

- In W′, gold increases monotonically, but the z = 0 sign test stops early. Its regret is 1.24 at *q* = 0.5 and 0.45 at *q* = 1, because shared exploits carry gold ≈ 6.6 at high leverage. This makes the audit terms heavy-tailed, and a negative estimate becomes likely over many looks.
- The paper now reports the trade-off across worlds:

  | Rule | Regret in W | Regret in W′ |
  |---|---|---|
  | Anytime-valid | 0.23–0.24 | 0.00–0.05 |
  | Windowed with rollback | 0.011–0.012 | 0.13–0.31 |
  | Proxy-only and classifier rules | up to 2.2 | up to 6.8 |

- We state that no rule we tested is best in both worlds.

### M7. The learned-RM study is too small

**Addressed with Exp. 4b (5 seeds × 8 held-out types, §5.10, Table 7).** The claim has been revised to match the new evidence.

- **What drives *q̂*.** *q̂* is ≈ 0.83–0.97 for *every* RM trained only on distribution A, whether MLP, small MLP or quadratic ridge. It is ≈ 0.41–0.46 for every RM with any decorrelated data. The effect is threshold-like: 25% B is enough. Pooled Spearman correlation of B-fraction with *q̂* is −0.78.
- ***q̂* predicts gap-alarm failure.** The pooled Spearman correlation of *q̂* with gap-alarm regret is 0.82 (*p* = 6 × 10⁻¹¹).
- **Size of the effect.** The gap alarm's regret reaches 3.45 against a peak of 1.47, so the first version's small effect (at most 0.14) was seed-specific.
- **H3.** The first version's H3 exception does not recur systematically (H3 regret ranges from 0.02 to 2.41 across seeds).
- **q̂ above 1** is now explained in §7: the estimator is a ratio of overestimation magnitudes, not a probability.
- **The exploit region** is still hand-placed. We list this as a limitation.
- **A new negative finding.** In one seed, gold never declines within the schedule. There the plain covariance test stops early on noise (regret 0.51 of a 3.59 gain), whereas the windowed+rollback rule gets 0.076. We report this rather than hide it.

## Moderate issues

**M6 (statistics).**

- *Standard errors.* All regrets and RMSEs now have SEs. `exp3_audit.py` and `exp_learned_rm.py` were modified only to add fields.
- *Anytime-valid control.* An anytime-valid variant is now evaluated (union-bound α-spending). It is very conservative in W and protective in W′.
- *Overshoot.* The reviewer noted that about half the runs stop late. Overshoot is reported, and the step-size sensitivity is now tested: at step 0.1 the covariance test has regret 0.024, and at step 0.025 it has 0.024. The concern that it "looks good only because the step is 0.05" is therefore not borne out.
- *Sensitivity.* A one-at-a-time sweep covers μ, δ, *p*, ρ, *N* and the step. Plain covariance regret is 0.008–0.041 across 13 settings; the gap alarm is 1.45–3.37 at *q* = 1.

**M8 (tilting versus PPO).**

- The unproved sentence is withdrawn.
- Proposition 1′ (exact, any update) is proved and verified to 2.2 × 10⁻¹⁴ under a PPO-like optimiser. That optimiser uses Adam with clipping, a KL penalty and a 3-feature exponential-family policy, and its parameters are not proportional to the tilting direction.
- The tilting surrogate Cov(*g*, *r*₁) agreed in sign with the realised gold change in 99.1% of iterations.
- With a window, covariance audits stayed within 0.003–0.015 of peak gold (1.22). The oracle-tuned gold-mean trend test reached 0.075–0.16.
- The exact ratio covariance turned out *noisier* as an audit target than the surrogate, which we report.
- We note that this is one toy policy class, and that sequence-level ratios in LLMs may be much noisier (§3.3, §7).

## Minor issues

| Issue | What changed |
|---|---|
| No CIs or SEs | SEs added everywhere (see M6). |
| No multiple-testing control; no sensitivity sweep | Anytime-valid variant and windowed rule added (see M6); one-at-a-time sweep in §5.9. |
| Exp. 4 uses disagreement without CV, Exp. 3 uses disagreement+CV | Now flagged explicitly in §5.10, with the reason: the CV hurts at high *q*. |
| T1 extension to natural-gradient steps not proved | Withdrawn; replaced by Proposition 1′. |
| H4 ensemble "eliminates the decline" | Now: "no decline was observed up to λ = 12 in this single setting". |
| README runtimes; Exp. 1a ESS checkpoints | README now lists measured runtimes. Exp. 1a's ESS filter keeps λ ∈ {0, 0.25, 0.5, 0.75}, stated in §4 and the README. |

## Questions for the authors

1. **Covariance test versus gold-mean change test and Moskovitz-style search at equal cumulative budget.** See Table 2, Fig. 5 and Table 6a, discussed under M1 and M4.
2. **Value over randomised rules, and dependence on the twin.** See Tables 3–4, discussed under M3.
3. **The audit rule in W′.** See Table 5, discussed under the major M6 item above. The plain rule fails there, and windowed and anytime variants do better.
4. **Noisy or biased gold, and PPO-like updates.**
   - *Unbiased noise.* At noise SD 2, the windowed rule's regret goes from 0.012 to 0.030, and the plain rule's from 0.015 to 0.127.
   - *Biased grader.* A grader that overrates exploits breaks every gold-using rule, including the gold-mean and proxy-point baselines. With half the exploit gap credited (*b* = 0.5), regret is about 1.5–1.8 for all of them. We present this as a limitation of auditing itself.
   - *PPO-like updates.* See M8.

## What we did not do

- We ran no LLM experiments. The protocol in §8 is updated with P4 (sign agreement of the surrogate) and with ratio-based audits.
- We did not implement Moskovitz et al.'s full multi-component constrained RL.
- We did not run learned-latent uncertainty detectors such as InfoRM.
- The parameter sweep is one-at-a-time.
