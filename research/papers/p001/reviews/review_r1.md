# Review (Round 1): "Deceleration Without a Break: Separating Mechanical Slowdown from Genuine Regime Change in Pretraining Scaling Laws, with a Self-Similar Model of the Data Wall"

Reviewer role: independent area-chair-level reviewer. Research question: 预训练的扩展定律是否正在放缓？下一个增长点在哪里？

## 1. Summary

The paper makes three claims.

1. **Mechanical deceleration.** Under a saturating law $L^*=E+GC^{-\gamma}$, the raw log-log slope falls as $\gamma\rho$. The gain per decade shrinks by $10^{-\gamma}$, and $\ln L^*$ is strictly convex in $\ln C$. Raw-loss "flattening" tests therefore always find a slowdown. The paper defines genuine deceleration as $\delta>0$ in the nested model $L=E+(AN^{-\alpha}+BD^{-\beta})\max(1,C/C_b)^\delta$.
2. **Empirical re-analysis.** The authors refit the law to Epoch AI's reconstruction of the Chinchilla runs (240 points). They find no within-sample deceleration: the profiled $\hat\delta=-0.039$, with 95% bootstrap CI $[-0.072,0.020]$. A compute-split backtest finds that high-compute runs did slightly better than forecast.
3. **Simulation and a data-wall model.**
   - The naive curvature test has a false-alarm rate of 81–100%.
   - The break test keeps its size.
   - Power falls with the reducible fraction $\rho$ ("signal shrinkage").
   - An unmodelled break is absorbed into $\hat E$.
   - Under Muennighoff et al.'s data-constrained law with $\alpha=\beta$, all data-wall milestones scale as $C/U^2$. The paper reports universal constants (wall at $0.3U^2$, 50% efficiency at $75\,C_{\rm wall}$, infinite shadow price of data at $161\,C_{\rm wall}$) and a "$k$-fold data equals $k^2$-fold compute" rule. From this it concludes that the next growth lever is data efficiency.

The paper also gives five falsifiable predictions and a small-scale protocol.

## 2. Reproduction and verification

**Code rerun.** I copied `code/` and `data/` into a scratch directory and ran all seven scripts in the documented order.

- Exp. 1 reproduced byte-for-byte: `exp1_fit.json` is identical to the shipped file.
  - Fit: $E=1.8172$, $A=477.8$, $B=2143.4$, $\alpha=0.3473$, $\beta=0.3672$.
  - $\gamma$: 0.1785, 80% CI $[0.1728,0.1859]$.
  - Residual SD: 0.00752.
- Exps. 2–7: all completed with exit code 0. Every result JSON (`exp2_break_real`, `exp3_simulation`, `exp4_data_wall`, `exp5_wall_sensitivity`, `exp6_confound_wall`, `exp7_signal_shrinkage`) is **byte-identical** to the shipped file, so the seeds are genuinely fixed.

**Additional reviewer experiment (W3).** I reused the paper's Exp. 7 `run_window` design (240 runs, 3 decades, fixed $C_b$), lowered the per-run noise, and ran 40 replicates per cell. The break test's rejection rates were:

| $\sigma$ | window ends at | $\delta=0$ | $\delta=0.02$ | $\delta=0.05$ |
|---|---|---|---|---|
| $10^{-4}$ | $10^{26}$ | 0.03 | 1.00 | 1.00 |
| $10^{-4}$ | $10^{28}$ | 0.08 | 1.00 | 1.00 |
| $3\times10^{-4}$ | $10^{26}$ | 0.03 | 1.00 | 1.00 |
| $3\times10^{-4}$ | $10^{28}$ | 0.00 | 1.00 | 1.00 |
| $10^{-3}$ | $10^{26}$ | 0.05 | 0.98 | 1.00 |
| $10^{-3}$ | $10^{28}$ | 0.05 | 0.63 | 0.88 |

The script is at `/tmp/claude-0/-home-user-AI-agents/6bf1b808-c21b-510f-a222-0286a315a58b/scratchpad/extra/noise.py`.

**Paper vs shipped outputs.** I checked every number in the main text against the shipped JSON/log files: Table 1, the profiled and fixed-$C_b$ $\delta$ intervals, the backtest means and bands, Table 2, the $W=3$ rows, the Exp. 6a $\hat E$ means and extrapolation errors, Exp. 6b detection rates, Table 3, the $\rho$ table, the paired-doubling pair counts, Tables 4–5, the Epoch-law variant (48/45/41/37), the floor excesses and the calendar years. All of them match to the stated rounding. I found no number that lacks code or a source behind it.

**Derivations.** I checked these line by line:

- Prop. 1(a–c) and Lemma 1: correct.
- Prop. 2: the exact identity is correct. The SNR $\propto\rho\sqrt n/\sigma$ statement is a heuristic, not a proof, although Table 3 is roughly consistent with it.
- Prop. 3: the proof is correct. It reduces to dimensional analysis once $U_N=k_N U_D$ is linear, which itself requires $\alpha=\beta$.
- Corollary 1(i): checked ($2\ln10/\ln4=3.32$; $2\ln10/\ln5=2.86$).
- Corollary 1(iii): checked ($C=6U^2/r$).
- Calendar years: I recomputed them by hand from the stated anchors, and they agree.

**References.** I checked eight by web search. All exist and support what the paper attributes to them:

- Besiroglu et al. 2024: refit 1.8172 / 482.01 / 2085.43 / 0.3478 / 0.3658, on 240 points.
- Muennighoff et al. 2023: about 400 runs, up to 9B parameters and 900B tokens; $R_D^*\approx15.4$, $R_N^*=5.3$.
- Villalobos et al. 2024: about 300T tokens, 90% CI 100T–1000T, exhaustion 2026–2032.
- Sevilla et al. 2024: 2e29 FLOP by 2030, four constraints.
- Epoch AI 2024: 4–5×/yr through May 2024.
- Epoch AI 2025: Grok-3 is the first model in their dataset above 1e26 FLOP.
- Caballero et al. 2023 (BNSL).
- DeepSeek-R1 (Nature).

Minor point: Muennighoff et al. report $\beta\approx0.353$, not exactly 0.35. Rounding it to 0.35 is what makes self-similarity exact (see W6).

## 3. Strengths

- **S1. Clear framing.** The paper separates "diminishing absolute returns", which are built into the functional form, from "a change in the parameters of the law". This is a useful framing for a debate that is usually conducted loosely, and the Chinese/English abstract states it well.
- **S2. Transparent, reproducible and honest.** Everything is labelled [computed] or [cited] and runs on a CPU. The Exp. 1 rerun is exact, and the limitations section is unusually candid: it covers digitisation, the optimistic fixed-$C_b$ test, misspecification in Exp. 6b, and extrapolation of the Muennighoff law.
- **S3. Signal shrinkage and absorption into $E$.** These are the most interesting parts of the paper. The demonstration that a modest $\delta$ appears mainly as an inflated $\hat E$ (1.817 → 1.905 at $\delta=0.02$) is practically relevant for people who compare $E$ across studies.
- **S4. Self-similarity.** The $C/U^2$ observation is elementary, but it is a neat organising principle. Exp. 5 checks it numerically, including how it drifts when $\alpha\ne\beta$.
- **S5. Falsifiable predictions.** The paper states falsifiable predictions with concrete protocols.

## 4. Weaknesses

### Major

**W1. Limited novelty of the theory.**

- Prop. 1 and Lemma 1 are textbook consequences of including an irreducible term. That consequence is why Hoffmann et al. (and Kaplan's $L_\infty$ fits) include $E$. Saying that "$\ln L$ vs $\ln C$ bends" is not a contribution at NeurIPS level.
- The "naive curvature test" is a strawman. No serious scaling-law paper tests for slowdown by fitting a quadratic to raw log-loss without $E$. The 81–100% false-alarm rate is therefore a restatement of Lemma 1, not an empirical finding about practice. The paper cites no published slowdown claim that uses this test.
- Prop. 3 is a direct scaling argument. The "$k$ in data ⇒ $k^2$ in compute" rule is the familiar $D^*\propto C^{1/2}$ Chinchilla relation applied to the onset of the wall.
- Signal shrinkage is a one-line Taylor expansion.

Taken together, the theoretical content is modest. The value lies in packaging these points and quantifying them, not in new results.

**W2. The empirical "no slowdown up to $10^{22}$ FLOP" claim overstates what the data can support.**

- In the 240-run reconstruction only 23 runs are at or above $10^{21}$ FLOP, only 4 at or above $3\times10^{21}$, and a single run lies above $5\times10^{21}$. The effective range is about $10^{18.5}$–$10^{21.5}$.
- The profiled breakpoint grid covers only the 30th–70th compute percentiles, $2.7\times10^{19}$–$2.8\times10^{20}$, so the "post-break" regime is mostly below $10^{21}$.
- The selected $\hat C_b$ lies at the **upper boundary** of the grid. Profile-likelihood bootstrap percentiles are not valid confidence bounds when the nuisance parameter sits at a grid edge. The 97.5th percentile should not be read as a "bound" of $\delta\le0.02$.
- At three of the five fixed $C_b$ values the intervals exclude zero, in *opposite* directions: positive at $2.8\times10^{19}$, negative at $\ge1.5\times10^{20}$. The authors read this as misspecification. If so, inference on $\delta$ inside a misspecified model is not a reliable bound on deceleration. The honest conclusion is "this model and this data cannot detect a deceleration of a few hundredths in the exponent", not "no slowdown, $\delta\le0.02$".
- The iid bootstrap ignores the grouped, iso-FLOP and digitised structure of the data. Points from the same model or budget are not exchangeable, so the intervals are likely too narrow.

**W3. The "loss curves are blind at frontier scale" conclusion rests on a noise assumption that does not hold for frontier labs.**

- Table 3 and P5 use $\sigma_0=0.0075$, which the authors concede is largely digitisation error, or $\sigma=0.0025$. Frontier labs evaluate on held-out sets of millions to billions of tokens, where the evaluation noise in log-loss is around $10^{-4}$. Seed-to-seed variance at scale is also well below 0.25%.
- With $\sigma=0.0025$, the break test already detects $\delta=0.1$ with power 0.97 at $10^{28}$.
- My rerun (Section 2) shows that the picture **does reverse** at realistic noise. At $\sigma\le3\times10^{-4}$, the paper's own test detects $\delta=0.02$ (11% of $\gamma$) with power 1.00 at $10^{28}$ FLOP. At $\sigma=10^{-3}$ the power is still 0.63.
- Signal shrinkage is real, but whether it makes frontier loss curves "blind" depends almost entirely on $\sigma$.
- The policy conclusions therefore depend on an unrealistic noise level: that the slowdown debate "cannot be settled" by loss curves, and that "published slowdown-detected claims are expected to be false alarms". These conclusions need a sweep down to $\sigma\in\{10^{-4},3\times10^{-4},10^{-3}\}$ and an argument for which noise level is relevant.
- Holding the design fixed at 240 runs over 3 decades is also an arbitrary choice for frontier-scale labs.

**W4. The answer to "where is the next growth point?" is largely built in by assumption.**

- The only genuine break the paper models is a cap on unique data, so the conclusion that "the next growth point for pretraining is data" follows almost by construction.
- RL, test-time compute, architecture or algorithmic-efficiency gains, multimodality and synthetic data are discussed only qualitatively. The model "makes no quantitative claim" about them.
- No alternative break mechanism is modelled and compared. Examples are an algorithmic-efficiency drift, or a shift in data quality as one moves down the quality tail of the stock.
- For a paper whose stated question includes "下一个增长点在哪里", this is a narrow answer.

**W5. The data-wall numbers rest on a heavy extrapolation of a law fitted at most at about $10^{22}$ FLOP.**

- The universal constants (64, 75, 128, 161, 942) inherit the functional form of Muennighoff et al.: exponential decay of repetition value, and the $U_N$ / excess-parameter penalty.
- That form has been challenged and revised in 2025–2026 work that the paper does not cite. Examples:
  - Lovelace et al., *Prescriptive Scaling Laws for Data Constrained Training* (arXiv:2605.01640, COLM 2026), which uses an additive overfitting penalty and finds a strong dependence on weight decay;
  - *Practical Scaling Laws: Converting Compute into Performance in a Data-Constrained World* (arXiv:2605.09189);
  - *Bridging Compute- and Data-Optimal Pretraining* (arXiv:2607.25271);
  - work on pretraining under fixed data and unbounded compute (Kim et al., 2025).
- Under these alternative forms the milestones, and possibly self-similarity itself, would change.
- Table 5 already shows a 12× spread (20–246) in the headline 50%-efficiency multiple across plausible half-lives. Presenting "75×" in the abstract and conclusion as a "universal constant" overstates precision.
- The implementation also applies the excess-parameter penalty whenever $N>0.051D$, even with fresh data. Whether that matches Muennighoff's intent in the single-epoch regime is debatable, and it affects the design used in Exp. 6b.

**W6. Self-similarity is fragile, which undermines the small-scale protocol (P1/P3).**

- Exact self-similarity needs $\alpha=\beta$. The paper rounds Muennighoff's $\beta=0.353$ to 0.35.
- With the Epoch refit ($\alpha=0.347$, $\beta=0.367$) the 50%-efficiency multiple drifts systematically: 48 → 45 → 41 → 37 over $U=3\times10^{13}$ to $10^{15}$, about a 10% decline per half-decade of $U$.
- P1 proposes running at $U\in\{10^8,10^9,10^{10}\}$ and transferring the results to $U\sim10^{14}$. That is 4–6 decades of extrapolation in $U$. At the observed drift rate the thresholds could shift by 2–4×, so the claimed "$<1.4\times$ bound" holds only within the 1.5-decade range actually tested.
- The $k_N$ constant was also kept linear for the $\alpha\neq\beta$ variant, which is internally inconsistent.
- The small-scale protocol therefore cannot validate frontier-scale constants without an explicit correction for $\alpha\neq\beta$.

**W7. The "large in compute, small in loss" asymmetry is mostly a restatement of mechanical deceleration.**

- Near the floor the raw slope $s$ is about 0.002, so any small loss gap maps onto a large compute multiplier.
- The headline $\eta=0.65$ at $10^{29}$ FLOP is a 1.5× compute penalty, which is about four months of growth at 4–5×/yr. Calling this "large" without economic framing is rhetorical.
- The claim that the slowdown debate cannot be settled "because" of this asymmetry is a hypothesis. The paper does not demonstrate it.

### Minor

- **M1. Inconsistent power figures.** The abstract says the break test has power "about 0.8 at Chinchilla scale" for $\delta=0.02$. Table 3 (window ending at $10^{22}$, $\sigma_0$) gives 0.55; the 0.74–0.87 figure comes from windows starting at $10^{18}$. Report one consistent number, or state the design.
- **M2. Simulations versus the real-data fit.** The simulations use least squares with the breakpoint fixed at the truth, while the real-data fit uses Huber with a profiled breakpoint. Size and power were never established for the procedure actually applied to the real data, which is Huber plus profiling plus bootstrap.
- **M3. Monte Carlo error.** Exp. 6 and 7 use 60 replicates (MC SE ≈ 6 points at power 0.5). Size estimates of 0.00–0.09 with 60–100 replicates are too coarse to claim "close to nominal". Report MC intervals.
- **M4. The 245-run sensitivity.** $\beta$ moves from 0.367 to 0.453 when the five excluded runs are included, which shows the fit is fragile. Robustness of $\hat\delta$ to this choice is not reported.
- **M5. The "17% of bootstrap replicates have $\hat\delta>0$" statistic.** It is not a p-value and should not be read as one.
- **M6. The paired-doubling sample sizes.** They assume $\sigma=0.001$ per run with independent noise. Within a matched pair, seed noise is shared only if seeds and data order are matched. Say so.
- **M7. Further missing related work.**
  - Choshen et al. 2024 (*A Hitchhiker's Guide to Scaling Law Estimation*) on fitting reliability.
  - Hernandez et al. 2022 and Xue et al. 2023 on repeated data.
  - Ruan et al. 2024 (observational scaling laws).
  - Recent analyses of relative excess loss and diminishing returns, e.g. arXiv:2603.28507.
  - The reference list claims it was "checked in Nov 2026" but contains nothing after early 2025 on data-constrained scaling.
- **M8. The calendar illustration.** It uses Grok-3 ($4$–$5\times10^{26}$) as a $10^{26}$ anchor. Anchoring at the actual $\sim4.6\times10^{26}$ moves the dates earlier by about a year; the script computes $C_0=5\times10^{26}$ but the paper does not report it.
- **M9. Exp. 6b design.** Exp. 6b places the window end at $m\cdot C_{\rm wall}$ for $U=10^{14}$, which implies runs with up to 200 tokens per parameter and heavy repetition. A sentence on how realistic this design is would help.

## 5. Questions for the authors

1. What is the break test's power in Table 3 at $\sigma=10^{-4}$ and $3\times10^{-4}$? Does P5 survive?
2. Can you give size and power for the exact procedure used on the real data (Huber, profiled $C_b$, bootstrap)? Can you use a cluster or block bootstrap by model size or iso-FLOP group?
3. How do the milestone constants change under the Lovelace et al. (2026) additive overfitting penalty, or under any of the 2025–26 data-constrained forms?
4. If $\alpha\neq\beta$ at frontier scale, how should P1 results at $U=10^8$–$10^{10}$ be transferred to $U=10^{14}$? Please give the predicted drift.
5. Is there any published slowdown claim that actually uses the raw-curvature test? If not, how do you justify the "naive test" as the baseline?
6. Can you model at least one competing break mechanism, for example a falling quality of the marginal token or an algorithmic-efficiency drift, so that the "next growth point" conclusion is not true by construction?

## 6. Assessment of the sub-question and conclusions

The sub-question is sharpened well, from "slowing?" to "$\delta>0$ in the reducible-loss exponent?". The conclusions follow only partly:

- "Mechanical slowdown is guaranteed": correct but trivial.
- "No within-sample deceleration": weakly supported, and only over about 3 decades ending around $3\times10^{21}$, not up to $10^{22}$ as claimed.
- "Loss curves cannot settle the debate at the frontier": depends on an unrealistic noise level.
- "The next growth point is data": a consequence of the chosen model, not an inference from evidence.

## 7. Scores (1–10)

| criterion | score |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 8. Decision

**Major revision.** For NeurIPS main track this would be a **weak reject** (5/10). The paper is careful, honest and fully reproducible, and its framing is useful. But the theoretical results are elementary, the empirical evidence covers about 3 decades with sparse high-compute support, the headline detectability conclusion depends on an unrealistic noise level, and the "next growth point" answer is fixed by the modelling choice. In its current form it fits a workshop or TMLR.

**Most valuable single change.** Re-run the detectability analysis (Table 3, P5, Exp. 6b) at realistic frontier noise levels ($10^{-4}$–$10^{-3}$) and with the exact profiled or Huber procedure used on the real data. Then either retract or strongly qualify the "loss curves cannot settle the debate" conclusion. That conclusion is the paper's main claimed significance.
