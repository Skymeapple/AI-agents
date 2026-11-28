# Response to Review, Round 1 — p016

Paper: "The Stability Gap Is a Property of the Path, Not the Objective"
Reviewer recommendation: major revision (5/10).

We thank the reviewer for a careful, reproducible review. We accept almost every point. This note lists each issue, what changed, and where the supporting output is. Section and theorem numbers refer to the revised `paper.md`. Three requests were **not** met; they are listed at the end and in §8 of the paper.

How this revision was produced: the first revision session was interrupted mid-edit. It left new scripts and partial outputs but no paper changes. We finished and re-ran the parts that the revised paper uses, and moved the rest, unverified, to `code/exploratory_unfinished/` and `results/exploratory_unfinished/`. Nothing in the paper depends on that folder.

## A. Numbers in the prose that did not match the outputs (review §2, Weakness 1)

All six were wrong as stated and are corrected. `linear_revision.py` part E recomputes each from the stored outputs (`results/linear_revision.json`, key `E_audit`); `audit_mlp_r1.py` does the same for the network (`results/mlp_audit_r1.json`).

| # | Old claim | Output | New text |
|---|---|---|---|
| 1 | closed form agrees "within $10^{-6}$" (Table 1, abstract) | grid error up to $1.66\times10^{-6}$ ($\rho=0.3$), $1.05\times10^{-6}$ ($\rho=0.5$) | Table 1 now uses a real ODE integrator (see B); relative error $\le2.4\times10^{-13}$. The old grid error is reported as "up to $1.7\times10^{-6}$". Abstract says "below $3\times10^{-13}$". |
| 2 | terminal loss "$<2\times10^{-7}$" | max $4.8\times10^{-7}$ | "below $5\times10^{-7}$ (maximum $4.8\times10^{-7}$)" |
| 3 | commuting control "$<10^{-17}$" | max $5.3\times10^{-17}$ | "$<10^{-16}$ (maximum $5.3\times10^{-17}$)" |
| 4 | recovery slope "$-0.79$" | $-0.82$ on mean times | "$-0.82$", and the paragraph now explains why this metric and range are not asymptotic (E5) |
| 5 | "GN 4–7 steps, GD 15–22" | GD per-$\rho$ means 18.6 / 20.2 / 24.4, runs 12–46; damped GN $\delta=0.1$ up to 12 | E3 now gives ranges and per-$\rho$ means for every method |
| 6 | half-recovery "by step 600–1,600" in split | that was the range of seed means | per-run ranges: 484–2,445 ($\rho=0.1$), 244–1,189 (0.3), 804–1,770 (0.5); perm censoring counts added |

Also corrected while checking: three third-decimal entries of the optimizer table (now Table 6: split $\rho=0.01$ GGN-flow 0.671; perm $\rho=0.01$ $\delta=10^{-2}$ final $L_2$ 0.034; perm $\rho=0.1$ $\delta=10^{-2}$ 0.173; split $\rho=0.3$ $\delta=10^{-2}$ sd 0.013). The table is now generated from `mlp_summary.json` with a fixed format, and its caption says the sd is the population sd.

## B. E1 "numerical integration" was the same eigendecomposition (Weakness 7)

Correct. E1 is redone with scipy `DOP853` (rtol $10^{-13}$), which does not use the eigendecomposition, with the peak found by bounded maximization on the dense output (`linear_revision.py` part A). Table 1 reports both and the paper states plainly what the old comparison did and did not check. E2 text now says that its trajectories are evaluated from the exact solution.

## C. Prop. 5 does not prove "recovery takes time of order $1/\rho$" (Weakness 2, Question 1) — major

Correct; this was a gap between proof and claim.

- Prop. 5 is retitled "a slow mode exists; an upper envelope", and the text after it says explicitly that it does not show slow recovery and that the first submission's inference was unjustified.
- **New Theorem 7** proves two-sided bounds on $T_\varepsilon(\rho)=\sup\{t:L_1(t)\ge\varepsilon F_\infty\}$. The lower bound is the one the reviewer suggested, from the Duhamel estimate: $T_\varepsilon\ge(1-\sqrt\varepsilon)\sqrt{2F_\infty}/(2\rho\|\Sigma_1\|^{3/2}\|e_0\|)$ under an explicit smallness condition on $\rho$. The upper bound is $\ln(1/(\varepsilon\beta^2))/(2\min(\rho,1-\rho)\nu)$, with $\nu=\lambda^+_{\min}(\Sigma_1+\Sigma_2)$. Inequality (4.1) is now stated for every $t$ in Theorem 4 so that Theorem 7 can cite it.
- **New Proposition 8** identifies the constant: $\rho T_\varepsilon\to T^\star_\varepsilon$, the exit time of the gradient flow of $L_1$ restricted to $\ker\Sigma_2$. The proof is given in outline (standard large-coupling limit) and this is listed as a limitation.
- **New experiment E5 / Table 3** (`linear_revision.py` part B): $\rho\in[10^{-5},10^{-1}]$, 10 instances. Slope $-1.001$ for $\rho\le10^{-3}$; $\rho T_{0.1}$ within $8\times10^{-5}$ of $T^\star_{0.1}$ at $\rho=10^{-5}$ and within 0.8% at $10^{-3}$; both bounds hold on all 90 pairs.
- What we do not hide: the bounds are loose by factors of about 350 (lower) and 200 (upper), and the lower bound's sufficient condition holds for all seeds only at $\rho\le3\times10^{-5}$. The paper says the bounds establish the exponent and nothing more.
- The reviewer's observation that the measured slope was $-0.82$/$-0.89$, not $-1$, is explained in E5: the old metric (10% of the peak) at $\rho\ge0.003$ is not in the asymptotic range.
- The network version of this claim is **not** established. Prediction 3 in §6 and P3 in §7 now say "untested in any network".

## D. Thm 6(a) is the straight-path observation; "artefact of Euclidean geometry" overstated (Weakness 3, Question 2) — major

Accepted in full.

- Theorem 6 is restated as "paths that stay in $\ker\Sigma_1$ have no gap". Exact GN is instance (a). **New instance (a′)**: projected gradient (the OGD analogue) has no gap for any $\rho\in[0,1)$ including $\rho=0$, and still drives $L_2\to0$; proof included.
- A paragraph "What this theorem does and does not say" states that (a) is the straight-segment fact, credits Kamath et al. (2024) for the empirical counterpart, says the joint-curvature metric is not privileged, and describes the ill-conditioning the reviewer pointed out ($\rho\to0^+$ endpoint vs. $\rho=0$ endpoint; errors of order $\rho\|\Sigma_1\|$ destroy the result).
- Answer to Question 2, with data (`linear_projection.py`, Table 2): OGD-style projection gives $G\le4.6\times10^{-17}$ at all three $\rho$; a GEM-style single aggregate constraint does not (0.0156 / 0.0154 / 0.0143 against GD's 0.0664 / 0.0341 / 0.0182). What is specific to GN is only that it needs no separate projector; that is now all the paper claims.
- "The sharpest statement in the paper" and "artefact of Euclidean gradient geometry" are removed from the abstract, contributions, §4, Discussion and Conclusion. The Discussion has a paragraph that withdraws them by name.

## E. "Diagonal does no better than GD"; "Adam-like" label (Weakness 8)

Accepted. The row is relabelled "Jacobi preconditioner, $\mathrm{diag}(H_\rho)+10^{-3}\lambda_{\max}$". The claim is removed from the contributions and the abstract; E3 says it is a single instance and not a theorem, and that Theorem 6(c) only says a general $P$ leaves a nonzero first-step coefficient.

## F. Optimizer comparison not matched; 4 seeds, no CIs (Weakness 5, Question 3) — major

- **Paired CIs for the existing 4-seed sweep** (`audit_mlp_r1.py`): every relative change is now a paired mean with a 95% $t$-interval. The headline "36–60%" (a ratio of means) is replaced by paired values, e.g. perm $\rho=0.3$: −40% (CI −56 to −24), the case the reviewer flagged as within 1 sd unpaired. The paper says the unpaired means overlap there and that 4-seed intervals are fragile.
- **New progress-matched experiment** (`mlp_matched.py`, Table 7, Figure 6), as requested in Question 3: GD at $c/\lambda_{\max}$ for $c\in\{1,0.3,0.1\}$, Adam at two learning rates, GGN-flow; gap measured up to the first step with $L_2\le qL_2(0)$, $q\in\{0.5,0.25,0.1\}$; $L_1$-vs-$L_2$ curves plotted.
  - GD with a 10× smaller step: gap changes by about 1% (9–13% at $q=0.5$). Step size is not the confound.
  - GGN-flow vs GD at matched progress: −34% to −75% over the 12 cells, lower in all 6 seeds in every cell.
  - Adam: −14% to −26% at $q\le0.25$, not removed by a lower learning rate.
- **Seeds:** 8 were launched; a machine restart killed seeds 6–7 and they were not re-run. All matched statistics use 6 seeds and the paper says so in §6, the abstract and §8.
- Not fixed: the damping test still has 2 seeds (the paper now says no interval is meaningful there), intervals are uncorrected for multiplicity, and the comparison is not compute-matched.

## G. One-step "within 2%" is near-tautological (Weakness 6)

Accepted. Prediction 1 now says so in its heading and text, and the abstract no longer presents it as a confirmation. The suggested multi-step test was run with a common step for all $\rho$ (`mlp_matched.py`, `short`): the normalized curves collapse at $k=1$ (within 0.6%), stay within 5% to $k=30$ at $\rho=0.01$, and separate quickly at larger $\rho$ (0.86–0.89 at $k=10$ for $\rho=0.1$; 0.55–0.60 at $k=30$ for $\rho=0.3$). The law is renamed "first-step law" throughout. P1 in §7 is marked as a weak test.

## H. Novelty is incremental; conceptual messages restate prior work (Weakness 4, review §4)

We agree with the assessment and now say it ourselves: §1 has a paragraph "What is and is not new" that attributes each conceptual message to its source (De Lange et al.; Kamath et al.; Hess et al.) and calls the mathematics elementary and classical. §8 has a "Novelty is incremental" item. Of the four extensions the reviewer listed, the revision supplies two and a half:

- matching upper and lower bounds on recovery time: Theorem 7 (rates match; constants do not) and Proposition 8 (constant);
- SGD / finite buffer (also Question 4): **Theorem 9** (exact second-moment recursion; expected gap never below GD's at the same step size; first step $=\tfrac12\eta^2(1-\rho)[(1-\rho)a+(b-a)/B]$, so the noise part scales as $(1-\rho)/B$, which answers the question of how replay noise changes the $(1-\rho)^2$ law) and **Proposition 10** (finite buffer: $\rho$-independent terminal floor). Checked in E6 (Table 4; Monte-Carlo agreement, max $|z|=2.0$) and E7 (Table 5). E6 uses one small step size, stated as a limitation;
- linearized network: an **experiment only** (§6, "How far does the linearization…"): $G_{\rm lin}/G$ between 0.97 and 1.09 across scenarios and $\rho$; trajectories diverge after the peak. No proof; stated in §8;
- more than two tasks: not done.

We do not claim this lifts the paper above "incremental". That judgement is the reviewer's.

## I. Over-reached conclusions (review §6)

- "Artefact of Euclidean gradient geometry": removed (see D).
- Łapacz et al. head-localization explanation: withdrawn explicitly in §7, with the reviewer's three reasons (cross-entropy, non-realizable head, moving features).
- Pre-registered predictions: retitled "Predictions for larger models, with their status". Each now carries its evidence. P2 is narrowed to $\rho\le0.01$ because perm at $\rho=0.03$ is already below the 50% line (0.37). P4 drops the damping sub-claim, which failed. P3 and P5 are marked untested in any network.

## J. Smaller points

- Thm 4 / Contribution 4: the bound is now stated on $\sqrt G$, with the squared form $F_\infty-O(\sqrt{F_\infty}\rho\log(1/\rho))$ given once.
- Thm 1: the $\rho\to0^+$ limit vs. the monotone $\rho=0$ case is stated.
- Train vs. test: §6 states once, in the design paragraph, that $G$ is always training loss and accuracy always test.
- Reference 20 (Mahdaviyeh et al.) updated to CoLLAs 2025, PMLR 330:15–42, after re-checking by web search. No reference was added or removed.
- Scale / practical relevance (Weakness 9): unchanged in substance. §8 states the distance from practical regimes. E7 adds one relevant observation: with a small buffer forgetting is mostly terminal and the transient excess is small, so the gap as a distinct phenomenon belongs to the good-replay regime.
- A plotting bug in the inherited Figure 5b (GD/SGD curves drawn for $\rho=0.5$ next to a $\rho=0.1$ Monte-Carlo curve) was fixed; no number changed.

## K. Not resolved

1. **Question 5 (commutator criterion in the MLP)** was not run. P5 is marked untested.
2. **Recovery-time scaling in a network** has no verified measurement. An over-parameterized-network experiment existed only as partial output from the interrupted session; we did not re-run or check it, so the paper reports nothing from it.
3. **More than two tasks, an NTK-level proof, large-step SGD, K-FAC-type approximations**: not done.
4. **Significance for a main-track venue** (scores for novelty and significance): the added theorems are again elementary. We have tried to make the paper accurate about its own scope and do not argue the point.
5. The MLP main sweep and damping JSONs were not regenerated in this round (the reviewer had also not confirmed bit-for-bit regeneration). Phase 1 was re-run and reproduces the stored base losses and one GD trajectory maximum to all printed digits.
