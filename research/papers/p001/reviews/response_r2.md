# Response to Round-2 Review (p001)

We thank the reviewer again for a reproduction-level review, which included rerunning our code and adding a new experiment (power against the number of budgets). We agree with the main diagnosis: the contribution was incremental, and two new headline claims were not established. This revision does more than patch them. It re-scopes the paper around a new result that answers the reviewer's central request ("a realistic frontier-detection design … with an empirically anchored correlated-noise level").

**New title:** *How Much Slowdown Can a Loss Curve Hide? A Compute-Equivalent Audit Ceiling for Pretraining Scaling Laws.*

## 0. Diagnosis and the new contribution

**Why revision 1 was not a strong accept.** It answered the question "is scaling slowing?" with "public data cannot tell", followed by conditional maps. A reader learned that detection is hard, but not *how* hard in units that matter, nor *why*, nor what would fix it. The reviewer's suggested design (a small sweep plus a few frontier runs, tested on forecast residuals) turns out to have a clean closed-form answer. Revision 1 had not found it.

**New contribution (Theorem 1, §3.2).** We express any deviation from the law as a compute penalty $P$. A loss comparison with total log-scale uncertainty $s$ can then certify at most $C_{\rm cert}(C)=\big(G/((1+k)R(C)+kE)\big)^{1/\gamma}$ of effective compute, with $k=e^{zs}-1$. This quantity is strictly bounded by the **audit ceiling** $C^\dagger=(G/kE)^{1/\gamma}$, where $R(C^\dagger)=kE$. Four further results follow:

- The undetectable penalty grows linearly in $C$ beyond the ceiling.
- The ceiling's elasticity in $s$ is $-1/\gamma\approx-5.6$.
- For a forecast check, $s\ge\mathrm{sd}(\ln\hat E)$, however good the frontier measurement is (Corollary 1).
- At matched efficiency, all break mechanisms are *exactly* equally detectable by a residual check (Corollary 2).

**What an expert did not know, and why it matters.**

- *The public ceiling.* For the public Chinchilla law, the audit ceiling at 50% power is $1.3$–$7.6\times10^{26}$ FLOP, which is exactly today's frontier (Epoch: Grok 4 at about $5\times10^{26}$).
- *A nonparametric certification bound.* A $5\times10^{26}$ FLOP run that lands *exactly on the forecast* certifies, at 95% confidence, only $1.3$–$3.8\times10^{25}$ FLOP of effective compute (Table 3). A slowdown worth 13–40× in compute is therefore consistent with "landing on the curve".
- *Real forecasts.* On the Gadre et al. (2024) suite, real out-of-sample errors of a few per cent in loss at about one decade of extrapolation correspond to multi-× errors in compute. The irreducible loss is not identified by that sweep (Table 4).
- *A design law.* We derive $\mathrm{sd}(\ln\hat E)\propto\sigma_b^{1.00}K^{-0.38}$ and an exact reach identity. We anchor $\sigma_b$ empirically and give an $E$-free ladder test (Proposition 4). Both turn the detectability section from a conditional map into a protocol.

We do not call this a breakthrough. The theorem is elementary, which we say in the paper (§2). Its value is that it makes the measurement question quantitative. We searched for prior statements and did not find the theorem or the audit-ceiling consequence in the scaling-law literature, but we cannot exclude that it exists informally.

New experiments: 13–20 (code in `code/`, outputs in `results/`, run script `results/run_logs/run_all_r2.sh`).

---

## Weaknesses

### W1. Limited novelty and significance

**Agreed for revision 1; addressed by re-scoping.** The paper's centre is now Theorem 1 and its application to two public suites (§3, §5.1–5.5). The round-1 material is condensed (§5.6, §5.7, Appendix A). The research question now gets a direct and quantitative answer:

> Public loss data cannot tell whether frontier scaling has slowed, because against the best public law slowdowns of 13–40× in compute are invisible at $5\times10^{26}$ FLOP. Matched contrasts that cancel $E$ can tell, provided per-run noise is near $10^{-4}$.

The "next growth point" half is answered at the level our tools support (W5).

### W2. "Correlated noise is binding" rests on an assumed $\sigma_b$; power vs $K$; perfect confounding

**Agreed on all three parts; claim replaced.**

- **(a) Confounding stated.** §5.4 now says explicitly that under model (2) a per-budget offset is a function of $C$ alone, like the break, so the in-sweep test has effectively $K$ observations of the break signal. The revision-1 phrase that correlated noise "destroys" power is withdrawn (Appendix A, corrections (a)).
- **(b) $\sigma_b$ anchored (Exp. 15; §5.4).**
  - Chinchilla between-compute-bin component: $3.9\times10^{-4}$ (within-bin SD $7.5\times10^{-3}$).
  - Gadre et al.: the best versus second-best warm-up gap has median $2.4\times10^{-3}$ (IQR $1.2$–$4.3\times10^{-3}$, 126 configurations), and the SD across head counts has median $8.1\times10^{-3}$ (IQR $3.4\times10^{-3}$–$1.3\times10^{-2}$).
  - Gadre et al.: residuals of fits on the three datasets correlate at 0.87 per configuration, so the misfit is structural.
  
  We read these as $\sigma_b$ of a few $10^{-4}$ for a fixed recipe and $10^{-3}$–$10^{-2}$ when the recipe is re-tuned per budget. These are small-scale anchors, and we say so (Limitations).
- **(c) Power against $K$ (Exp. 17a; Table 6).** We reproduced the reviewer's experiment and extended it to $K\in\{6,12,24,48,120\}$ at $n=240$. Power at $\delta=0.02$ is 0.09, 0.09, 0.17, 0.43 and 0.20; at $\delta=0.05$ it is 0.35, 0.54, 0.88, 1.00 and 0.96. This matches the reviewer's 0.075 (12×20) and 0.43 (48×5). At 120×2 we get 0.20 against the reviewer's 0.38 (Wilson intervals $[0.13,0.29]$ against $[0.28,0.48]$). We report the discrepancy and do not explain it.
  
  For the forecast design, the same lesson appears in closed form: $\mathrm{sd}(\ln\hat E)\propto K^{-0.38}$, and the reach improves by about 1.6 decades from 12 to 48 budgets (Table 5).
- **Recommendation (4)** is rewritten: measure $\sigma_b$, use a fixed recipe and many budgets, and calibrate with cluster bootstraps.

### W3. The detection design is not one any lab would run

**Agreed; this is now the centre of the paper.**

- **Forecast design.** The realistic design (a small sweep plus 1–3 frontier runs, tested on the forecast residual) is analysed in closed form (Theorem 1) and validated by Monte Carlo (Exp. 14, 30 cells × 400 replicates). With independent noise, the empirical forecast SD matches the delta method within 5%, the size is 0.033–0.068, and the median absolute power error is 0.003.
- **Where the closed form fails.** With $\sigma_b$ estimated from 12 budgets, the plug-in test over-rejects (size 0.06–0.105). We report this as a limitation and recommend cluster-bootstrap calibration.
- **Dependence on $E$.** The reviewer's point that this design depends strongly on the extrapolation of $E$ is exactly Corollary 1. As $C\to\infty$ the forecast SD tends to $\mathrm{sd}(\ln\hat E)$, which sets the audit ceiling.
- **The reviewer's question 2** ("a small sweep up to $10^{23}$ plus 1–3 runs at $10^{26}$"): with a fixed recipe ($\sigma_b=0$) and $\sigma_w\le10^{-3}$, $P=2$ and $\delta=0.02$ at $10^{26}$ are detected with power 0.98–1.00 (Exp. 14). With $\sigma_b=10^{-3}$ the powers at $10^{26}$ fall to 0.43–0.48 for $P=2$ and 0.51–0.56 for $\delta=0.02$ (closed form; the Monte Carlo is higher because of the anti-conservative plug-in). Taking a third frontier run barely helps, because the sweep term dominates.
- **The P5 claim** that labs "can detect small breaks" is replaced by conditional statements tied to $\mathrm{sd}(\ln\hat E)$ and to per-run noise (§6, P1–P3).
- **Ladder design.** We add a second realistic design, a one-recipe ladder at $C_0$, $10C_0$ and $100C_0$, costing about 1.11 times the top run (Prop. 4, Exp. 18). It is $E$-free and invariant to constant penalties: the size under $P=5$ is 0.051–0.053. Its closed-form power matches Monte Carlo within 0.005 over 80 cells. At $5\times10^{26}$ it resolves $\delta=0.017$ (50% power) with per-run noise of $10^{-4}$, and nothing below $\gamma/2$ at $10^{-3}$.

### W4. The mechanism comparison is by construction, and its power comparison is confounded

**Agreed.**

- The revision-1 in-sweep power comparison across mechanisms did not match their magnitudes and is withdrawn (§5.7).
- The reviewer's question 3 (power at matched $\eta$) has an exact answer for the residual design. By Corollary 2, the signal depends only on $\eta$ and $C$, so at matched $\eta$ all mechanisms are equally detectable.
- We state that matched data-doubling experiments are already run by Muennighoff et al. (2023), Lovelace et al. (2026) and Kim et al. (2025), and we claim no novelty for them. Table 8 and the levers were restatements of illustrative assumptions and have been removed from the main text.

### W5. "Next growth point" remains mostly outside the analysis

**Partly agreed; partly addressed.** We still do not model RL, inference-time compute or synthetic data quantitatively, and the Limitations section says so. The compute-equivalent framing does, however, give a concrete, quantitative statement about the levers practitioners debate, all of which are compute multipliers (§6):

- A 2× lever moves frontier log-loss by 0.0033 at $5\times10^{26}$ and by 0.0020 at $10^{28}$ (Exp. 20).
- These levers are below the resolution of forecast checks against public laws.
- They are within the resolution of matched A/B contrasts: $P_{\min}=1.43$ at a difference noise of $10^{-3}$ (Corollary 3).

So the answer to "where is the next growth point?" that our analysis supports is about *measurement*. Frontier loss compared with a law cannot rank the levers; matched contrasts can. We now treat algorithmic-efficiency drift as the central quantity, as the reviewer suggests: it is exactly the compute penalty $P$, read in reverse.

### W6. Table 7 extrapolations and the calendar

**Agreed.**

- §5.7 now states that the additive forms' hard-wall values (28–36×) come from extrapolating a penalty that is linear in $N/U_D$ by 2–6 decades, and are not established properties at scale.
- The calendar is moved to Appendix A with stronger caveats, and the Grok 4 estimate (about $5\times10^{26}$, verified on Epoch's model page) is noted.

---

## Status of the remaining round-1 items (reviewer §3)

- **Post hoc two-break breakpoints.** These are now stated as chosen after inspecting the one-break profile, and the CI is stated not to account for the selection (§5.6).
- **"Rejects $\delta=0$".** Replaced throughout by "rejects the five-parameter constant-exponent form". §5.6 notes that 73% of the residual variance is smooth misspecification.
- **"Size-group clusters give similar intervals".** Corrected. At $9.9\times10^{20}$ the size-group interval $[-0.077,0.015]$ includes zero, while the compute-bin interval excludes it (§5.6).
- **Xu et al. (2026) and arXiv:2610.37842.** Both were verified by web search and are now cited and discussed (§2, §5.7). We did not implement SoftQ, because that would require its fitted parameters and a check of our implementation against the authors'. The paper says this explicitly, and says that SoftQ widens rather than narrows the functional-form uncertainty. This is a partial answer to the reviewer's question 4.

## Minor issues

- **71× against 74–75×.** Explained in §5.7. Exp. 4 and Exp. 10 use the rounded linear threshold $k_N=0.051$, whereas the consistent constant is $c=(\alpha A/\beta B)^{1/\alpha}=0.050$.
- **Huber threshold (Exp. 17c).** At $\sigma_w=2.5\times10^{-3}$, $10^{26}$, the power at $\delta=0.02$ / $0.05$ is 0.12 / 0.81 with $\delta_H=10^{-3}$, 0.23 / 0.86 with $2.5\times10^{-3}$, and 0.22 / 0.93 with $5\times10^{-3}$. Scaling $\delta_H$ with the noise helps modestly, as the reviewer anticipated. Exp. 8 gave 0.17 / 0.87 for the first setting with other seeds.
- **Fitted-null power (Exp. 17b).** At $\sigma_w=10^{-3}$, $10^{26}$, the fitted-null procedure rejects 2/40 null datasets (0.05) and 39/40 datasets at $\delta=0.02$ (0.975), against an oracle power of 0.82. We find no evidence that the oracle powers overstate the power of the real-data procedure.
- **Abstract wording.** Changed as requested.
- **`code/README.md`.** It now carries the new title and documents Exps. 13–20. Exp. 12 has been added to `run_all_r1.sh`, and `run_all_r2.sh` runs the revision-2 experiments.
- **Post hoc two-break breakpoints.** Stated.

## Answers to the reviewer's questions

1. **Basis for $\sigma_b$; power against $K$.** See W2(b) and (c). The anchors range from $3.9\times10^{-4}$ to about $8\times10^{-3}$, and power peaks near 48 budgets at fixed $n$.
2. **Power of the forecast-residual test.** See W3 and Theorem 1. The resolution is closed-form, and the ceiling is set by $\mathrm{sd}(\ln\hat E)$.
3. **Power at matched $\eta$.** For the residual design it is exactly equal across mechanisms (Corollary 2). The in-sweep comparison is withdrawn.
4. **Table 7 under SoftQ.** Not computed. Xu et al. is discussed as additional form uncertainty.

## Integrity notes

- Every new number comes from Exps. 13–20, whose outputs are in `results/`.
- The Gadre et al. analysis uses the Chinchilla form, not the authors' law, and the paper states that it does not assess their forecasts.
- References were verified by web search in November 2026: Gadre et al. 2024/ICLR 2025; Xu et al. 2026; arXiv:2610.37842; Epoch's Grok 4 estimate. We cite arXiv:2610.37842 by title only, because we could not confirm its author list.
- Removed from the reference list because they are no longer cited: Epoch 4–5×/yr (it now appears only in the appendix calendar via the r1 log), Hestness et al. 2017, Sardana & Frankle 2024, Shumailov et al. 2024, Sorscher et al. 2022.
