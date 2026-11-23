# Response to Round-1 Review (p001)

We thank the reviewer for a careful report, including the reproduction and the extra noise experiment. We agree with most of the criticisms. The main changes are these:

- (i) We withdrew two headline claims: "no slowdown, $\delta\le0.02$, up to $10^{22}$ FLOP" and "loss curves cannot settle the slowdown debate".
- (ii) We replaced them with calibrated, cluster-robust real-data inference (Exp. 9, 9b) and a noise-resolved detectability map that uses the exact real-data procedure (Exp. 8).
- (iii) We modelled two competing break mechanisms (Exp. 11), so the "next growth point" answer is no longer built in.
- (iv) We generalised self-similarity to $\alpha\neq\beta$ and recomputed the wall milestones under the 2026 functional forms of Lovelace et al. (Exp. 10). The "universal constants" language is gone.
- (v) We reframed the paper as a measurement paper whose theory is elementary.

Every number added in this revision comes from code in `code/` with outputs in `results/`. All of Exps. 2–11 were rerun after one numerical change: objectives are scaled by $10^6$, with BFGS gtol $10^{-7}$, so that convergence is not declared prematurely at small noise. The pre-change outputs are kept in `results/v0_unscaled_optimizer/` for comparison. The change moved the real-data estimates only in the sixth significant figure. The run log is `results/run_logs/timing.txt`.

New title: *Is Pretraining Scaling Slowing Down? Mechanical Deceleration, Noise-Limited Detectability, and How to Tell a Data Wall from Other Breaks.*

---

## Fatal issues

None were raised.

## Major issues

### W1. Limited theoretical novelty; the naive test is a strawman

**We agree.** Changes:

- Prop. 1 and Lemma 1 are now introduced in a subsection titled "Mechanical deceleration (standard facts)". It says explicitly that they are textbook consequences of including $E$ and are stated only to fix terminology (§3.2). Prop. 2 is described as an exact identity plus a heuristic, not a theorem (§3.3).
- The naive curvature test is no longer a contribution or a baseline. §3.2 says we know of no peer-reviewed paper that uses it. §5.4 reports it only to quantify informal "the curve is bending" arguments. It is gone from the contribution list and the abstract.
- The contribution list (§1) now describes a *measurement* paper:
  - a calibrated, cluster-robust re-analysis;
  - a noise-resolved size and power map for the exact procedure;
  - a comparison of break mechanisms by observable signature;
  - an audit of the data-wall milestones across functional forms.
  
  A "What is new here" paragraph (§2) says that each ingredient is known.
- The one piece of theory that is new in this revision, Prop. 3′, is modest. It is the scaling argument with the consistent excess-parameter threshold, and it extends self-similarity to $\alpha\neq\beta$ with transfer exponent $\tau=(\alpha+\beta)/\alpha$. We present it as such. It is useful because it removes the drift the reviewer found in W6.

### W2. "No slowdown up to $10^{22}$" overstates the data; grid edge; iid bootstrap

**We agree, and we withdraw the claim.** New analysis (Exp. 9, Exp. 9b, Fig. 7, Table 6, §5.2):

- **Support.** We now report the effective support instead of the nominal range. The 10th, 50th and 90th compute percentiles are $10^{18.83}$, $10^{19.92}$ and $10^{21.00}$. There are 104 runs at or above $10^{20}$, 23 at or above $10^{21}$, 4 at or above $3\times10^{21}$, 2 at or above $5\times10^{21}$, and 1 at or above $10^{22}$. These match the reviewer's counts.
- **Grid edge.** We widened the breakpoint grid to the 10th–90th percentiles (9 points). The profiled optimum then moves to the *lower* edge ($C_b=6.8\times10^{18}$, $\hat\delta=+0.024$). $\hat\delta(C_b)$ falls steadily from $+0.024$ to $-0.040$ across the grid, which confirms that the v0 optimum at the upper edge was an artefact of the grid.
- **Calibrated tests instead of percentile bounds.** We calibrate the sup statistic by parametric bootstrap under three noise models: Gaussian, wild-cluster over compute bins, and wild-cluster over size groups (300 replicates each). All three reject the constant-exponent law, one- and two-sided, at $p=0.0033$, the smallest attainable value.
- **Cluster bootstrap.** Pairs bootstraps over 36 compute-bin clusters and 36 size-group clusters give profiled-$\hat\delta$ 95% intervals of $[-0.076,0.044]$ and $[-0.084,0.043]$, so the sign of the profiled $\hat\delta$ is not identified. The fixed-$C_b$ intervals still exclude zero in opposite directions: $[0.010,0.044]$ at $6.8\times10^{18}$ and $[-0.104,-0.010]$ at $9.9\times10^{20}$.
- **Two-break model.** The local exponents are 0.187, 0.166 and 0.175. The top-segment change is $\hat\delta_2=-0.009$, with 95% cluster CI $[-0.075,0.015]$.
- **Noise decomposition.** About 73% of the residual variance is smooth misfit rather than run-level noise.
- **Power on the actual design.** With wild-cluster noise, the test detects $\delta=0.02$ with power 0.63 (break at $8.4\times10^{19}$) or 0.46 (break at $2.8\times10^{20}$).
- **New statement.** The data reject the constant-exponent law. The departure is non-monotone and located at the low-compute end. Above $2.8\times10^{20}$ (effectively up to about $3\times10^{21}$), the exponent change lies in $[-0.075,0.015]$, so a slowdown of up to about 8% of $\gamma$ is not excluded. This is the reviewer's suggested wording, "this model and data cannot detect a deceleration of a few hundredths", made quantitative. It now appears in the abstract, §5.2, §6 and §8. The v0 interval is still reported in §5.2, labelled as superseded, with the reviewer's three objections stated.

### W3. Detectability conclusions rest on digitisation-level noise

**We agree. The headline conclusion is withdrawn and replaced.** New Exp. 8 (revised Table 3, Fig. 6, §5.3):

- **Exact procedure (also addresses M2).** We use Huber loss, a breakpoint profiled over the 10th–90th percentiles, and the one-sided sup statistic calibrated by parametric bootstrap at the same design (200 null replicates per cell). This is the same function (`code/breaktest.py`) that is applied to the real data.
- **Noise sweep.** We vary $\sigma_w\in\{10^{-4},3\times10^{-4},10^{-3},2.5\times10^{-3},7.5\times10^{-3}\}$, with windows ending at $10^{22}$–$10^{28}$, and 100 alternative replicates per $\delta$ with Wilson intervals.
- **Result, which confirms the reviewer.** At $10^{28}$, $\delta=0.02$ is detected with power 1.00 at $\sigma_w=10^{-4}$ and at $3\times10^{-4}$, 0.22 at $10^{-3}$ and 0.09 at $2.5\times10^{-3}$; $\delta=0.05$ is still detected with power 0.96 at $10^{-3}$. The reviewer's fixed-$C_b$ numbers (1.00 at $\sigma\le3\times10^{-4}$, 0.63 at $10^{-3}$) agree at low noise. At $10^{-3}$ ours is lower, as expected, because profiling costs power.
- **New finding that refines the reviewer's point: correlated noise.** A per-budget shared offset of $10^{-3}$, for example from recipe re-tuning, with $\sigma_w=10^{-4}$, gives power 0.13 ($10^{26}$) and 0.10 ($10^{28}$) at $\delta=0.02$. An offset of $3\times10^{-3}$ gives 0.05–0.06. A small design (48 runs) at $\sigma_w=3\times10^{-4}$ keeps power at 1.00 ($10^{26}$) and 0.70 ($10^{28}$).
- **Which noise is realistic.** An assumption-driven evaluation-noise budget (Exp. 12) gives $3\times10^{-5}$–$1.5\times10^{-4}$ for $10^9$ evaluation tokens. We agree that evaluation noise is not binding. We do not claim to know frontier seed or recipe variance, which is not public. We cite PolyPythias for small-scale seed stability without importing numbers.
- **New claim.** Whether loss curves can settle the question depends on the correlated part of the noise. Matched internal sweeps may settle it; public, heterogeneous loss data cannot. P5 has been rewritten as a two-sided, noise-conditional prediction. The v0 sentence "published slowdown-detected claims are expected to be false alarms" has been deleted.
- **Design.** We added the 48-run design, so the result does not depend only on the 240-run, 3-decade choice. We say in §5.3 that the design is held fixed only to isolate the effect of noise.

### W4. "The next growth point is data" is true by construction

**We agree.** New Exp. 11 (Table 8, §5.6) models three mechanisms with the same onset: M1, a unique-data cap; M2, a declining quality of the marginal token ($\theta=0.5$); and M3, a non-data exponent break ($\delta=0.05$). The results:

- The frontier loss-curve test does not identify the mechanism. Its power is 0.92, 0.23 and 1.00 for M1, M2 and M3 at $100\,C_x$ with $\sigma_w=3\times10^{-4}$, and 0.18, 0.06 and 0.75 at $10^{-3}$. It detects the non-data break best because that break is exactly its model. The quality decline is mostly absorbed into $\hat E$.
- The matched data-doubling gap separates them: it grows without bound for M1, saturates near $5\times10^{-4}$ for M2, and is exactly 0 for M3.
- The compute value of doubling the data lever is ∞, about $1.5\times$, and $1\times$ respectively.

The conclusion is now conditional. Under M1 the lever is new unique data or repetition tolerance, under M2 it is data quality, and under M3 it is algorithmic. Matched data-doubling experiments decide between them. The v0 sentence that the next growth point "is data" is withdrawn (abstract, §5.6, §6, §8). RL and inference-time compute are still outside our quantitative scope; §6 and §7 say so. We did not model an algorithmic-efficiency *drift*, which would be a multiplicative compute shift that leaves $\delta$ unchanged. It would not be a break in our sense, and it is observationally equivalent to a rescaled $C$.

### W5. Data-wall constants rest on extrapolating one functional form; missing 2025–26 work

**We agree.** Changes:

- **Missing work added and verified** by web search: Lovelace et al. 2026 (arXiv:2605.01640), Bryant & Liu 2026 (arXiv:2605.09189), Qin et al. 2026 (arXiv:2607.25271) and Kim et al. 2025 (arXiv:2509.14786), discussed in §2.
- **Recomputed under Lovelace et al.'s forms (Exp. 10c, Table 7).** We use the parameters published in their repository (`dcscaling/data/fits/standard` and `wd_sweep/wd1.0`), read directly from the JSON files. We checked our implementation against their `laws.py`: the additive penalty $P\,R_D\,N/U_D$, the 4-parameter $P\,R_D^\delta(N/U_D^\gamma)^\kappa$ masked at $R_D=0$, and their Muennighoff-D refit. The 50%-efficiency multiple is:
  - 28–33 for the additive forms at $U\ge10^{10}$;
  - 1,000 and 7,822 for their effective-data refits (standard and wd 1.0);
  - 69–75 for the Muennighoff constants.
  
  The milestones therefore span more than two orders of magnitude across forms that fit the same 2026 runs.
- **Self-similarity under the new forms.** The additive forms are not self-similar at small $U$: their milestones move by more than $10\times$ between $U=10^8$ and $10^{10}$ before settling into a "hard wall" regime. We say so explicitly, and it limits P1 (see W6).
- **"Universal constant" removed** from the abstract, the conclusion and the table captions. "75×" now appears only as the value under the Muennighoff form, next to the alternatives.
- **Excess-parameter penalty in the single-epoch regime.** With the consistent rule $U_N=\min(N,cU_D^{\beta/\alpha})$, which is also how the Lovelace et al. code implements Muennighoff-DN, the penalty applies only when $N$ exceeds the compute-optimal size for $U_D$. We kept the v0 implementation for Exp. 4 and Exp. 6b so they remain comparable, and we note the issue there.

### W6. Self-similarity is fragile when $\alpha\neq\beta$; P1 transfer; linear $k_N$ is inconsistent

**We agree on all three points.** Changes:

- **Prop. 3′.** With the consistent threshold $U_N=\min(N,c\,U_D^{\beta/\alpha})$, which is Muennighoff et al.'s own definition, $L^*_U-E=U^{-\beta}\Psi(C/U^{\tau})$ with $\tau=(\alpha+\beta)/\alpha$. Self-similarity is then exact for any $\alpha,\beta$.
- **Numerical check (Exp. 10a).** On the refit base law, the linear rule drifts from 124 to 36 over $U=10^8$–$10^{15}$, a factor of 3.4 that confirms the reviewer's estimate. The consistent rule gives exactly 69 at every $U$.
- **$\beta=0.353$.** The linear rule drifts from 63 to 52 between $U=10^8$ and $10^{14}$; the consistent rule gives 71 at both. §5.5 states that the v0 rounding to 0.35 is what made v0 exact (also M9 in the review's list).
- **Transfer exponent uncertainty.** $\tau=2.06$, 80% CI $[1.98,2.18]$, which over 6 decades corresponds to a factor of 0.32–5.2. P1 now requires $\tau$ to be estimated from the same sweep. The v0 "$<1.4\times$" bound is withdrawn (§5.5).

## Minor issues

- **M1 (inconsistent power figure).** The "about 0.8" in the abstract is gone. The abstract now quotes only Exp. 8 numbers with their design: 240 runs, 3 decades, exact procedure. §5.4 explains where the v0 "0.8" came from (fixed-$C_b$ test, designs starting at $10^{18}$) and gives the profiled test's 0.23 at $\sigma_0$ for a window ending at $10^{22}$.
- **M2 (simulated vs applied procedure).** Exp. 8, Exp. 9(f) and Exp. 11 now use the applied procedure (`breaktest.py`). We added a fitted-null calibration check: 0/20 rejections in each of two cells, Wilson 95% interval $[0,0.16]$. We say that this shows the test is not anti-conservative at this resolution, but not that its size is exactly nominal.
- **M3 (Monte Carlo error).** Exp. 8, 9 and 11 report Wilson intervals. The "close to nominal size" claim for Exp. 3 now gives the pooled rate over 800 null replicates (44/800 = 0.055, Wilson 95% interval [0.041, 0.073]) with its interval (§5.4). Exp. 8 uses 200 null and 100 alternative replicates per cell.
- **M4 (245-run sensitivity).** This is now reported (Exp. 9g, §5.2 and §7). With all 245 runs the profiled $\hat\delta=-0.096$ (at $2.8\times10^{20}$), and $\hat\delta(C_b)$ ranges from $+0.040$ to $-0.104$. The sign pattern is the same, but the magnitude roughly doubles, so $\hat\delta$ is not robust to the exclusion rule.
- **M5 ("17% of replicates").** Removed. §5.2 now says explicitly that it was not a p-value.
- **M6 (paired-doubling noise).** §5.6 states that the v0 pair counts assume independent per-run noise of $10^{-3}$. With matched seeds and data order, the relevant quantity is the difference noise, so the counts are conservative.
- **M7 (missing related work).** Added and verified: Choshen et al. 2025 (ICML; arXiv:2410.11840), Hernandez et al. 2022 (arXiv:2205.10487), Xue et al. 2023 (arXiv:2305.13230), Ruan et al. 2024 (arXiv:2405.10938), Lu 2026 (arXiv:2603.28507), plus the four data-constrained papers listed under W5 and PolyPythias (arXiv:2503.09543). Note on arXiv:2603.28507: the paper exists (Chien-Ping Lu; v2 title "Continued AI Scaling Requires Repeated Efficiency Doublings"). It argues that diminishing returns under a constant exponent require repeated efficiency doublings. We could not confirm the phrase "relative excess loss" in it, so we cite it only for that argument.
- **M8 (calendar anchor).** The calendar now uses Epoch's 4.6e26 FLOP estimate for Grok-3, verified on Epoch's "models over 1e25 FLOP" page. The dates move about a year earlier: for $U=10^{14}$ the wall arrives in 2026.2–2026.7 and 50% efficiency in 2028.8–2030.6. §5.5 notes that the spread across functional forms is larger than the spread across growth rates.
- **M9 (Exp. 6b design realism; also the $\beta$ rounding).** The $\beta$ rounding is addressed under W6. Exp. 6b (up to 200 tokens per parameter with heavy repetition) is no longer used for any conclusion. Detectability of the wall is now assessed in Exp. 11 at realistic noise, on a 240-run window ending at $10$ or $100\,C_x$.

## Other changes

- **W7 (the "large in compute, small in loss" rhetoric).** We added economic framing (§5.5). The compute penalty is $1.53\times$ at $10^{29}$ FLOP, which is 3.2–3.7 months of growth at 4–5× per year. At $10^{30}$ it is $4.5\times$, or 11–13 months. The word "large" was removed for $10^{29}$. The claim that the debate cannot be settled "because of" this asymmetry is deleted.
- **Optimiser scaling, and a correction we owe the reviewer.** While rerunning the experiments we found that the v0 objectives were unscaled (of order $10^{-6}$–$10^{-3}$). With BFGS's default tolerance, fits near the irreducible loss could stop before converging. We now scale the objectives by $10^6$ and use gtol $10^{-7}$ throughout.
  - Real-data results changed only in the sixth significant figure.
  - Exp. 3 rejection rates are identical.
  - Exp. 6 changed by at most one replicate per cell.
  - The high-compute cells of the v0 Table 3 (Exp. 7) changed materially, for example 0.10 to 0.20 and 0.40 to 0.62 at $10^{28}$ with $\sigma=2.5\times10^{-3}$. Part of the v0 "blindness" at frontier scale was therefore a numerical artefact.
  - A preliminary Exp. 8 run with the old objective differed from the final one by up to 0.27 in individual cells, in both directions.
  
  The paper reports only the corrected numbers and says so in §5.3. The old outputs are archived in `results/v0_unscaled_optimizer/` for comparison.
- **Limitations (§7)** were rewritten to cover: the digitisation and small-model explanations we did not test for the rejection of the constant-exponent law; the oracle calibration in Exp. 8; the lack of public frontier noise data; the extrapolation of all data-constrained forms beyond about 1B parameters and 6B unique tokens; and the illustrative (not estimated) parameters of M2 and M3.

## Where we disagree, or only partly agree

- **W3, "at realistic frontier evaluation noise the headline conclusion reverses".** We agree for *independent* noise. Our grouped-noise cells show, however, that a per-budget shared offset of $10^{-3}$ restores the blindness, even when the per-run noise is $10^{-4}$. Whether frontier sweeps have offsets of that size is unknown publicly, so we state the conclusion conditionally rather than reversing it outright.
- **W1, "signal shrinkage is a one-line Taylor expansion".** We agree that it is, and say so. We keep it because it predicts how the noise requirement scales with compute, and Table 3 checks that prediction.
