# Review (Round 1) — p039
**"Disagreeing About p(doom), Agreeing About the Budget? Belief Pooling, Tractability, and the Value of Information in Loss-of-Control Mitigation Spending"**
Target: AIES. Reviewer role: area chair / senior reviewer.

## 1. Summary
The paper builds a static expected-cost model. A planner picks PV mitigation spending S against a catastrophe probability p, a loss L and an isoelastic residual-risk curve g(S) = (1-m) + m(1+S/k)^(-beta). From it the authors derive (i) closed-form S*, a zero-spend threshold and the budget elasticity 1/(1+beta) with respect to belief; (ii) an ordering of budgets under log-odds pooling, mean of individual optima and linear pooling; (iii) closed-form lognormal EVPI/EVSI shares; (iv) a functional-form-free marginal test (spend more iff -pLg'(S_c) > 1); (v) numerical minimax regret, Sobol analysis and a (k, beta) "frontier". The model is calibrated with US VSL x world population, XPT medians (0.38% / 3%) and a logit-normal fitted to Grace et al. (median 5%, mean 16.2%). The authors conclude that pooling is a first-order budget choice, that EVPI is a large fraction of the budget, and that the crux is marginal effectiveness at current spending. A pre-registered (not run) elicitation protocol closes the paper.

## 2. Reproducibility check (done)
- I copied `code/` to a scratch directory and reran `python3 code/run_all.py` (about 11 s). I diffed the regenerated `results.json` against the shipped one field by field. Every numeric field is identical to within 1e-9. The only difference is `runtime_sec`.
- I checked every number in the abstract, Tables 1–4, §5.1–5.6, §6.2–6.3 and §7 against `results.json`. All match after rounding: compression ratios 6.54/4.90/2.81, elasticities, p_min, pooled beliefs and budget ratios, mean-of-optima ratios, minimax-regret budgets and regrets, EVPI shares, E6 quantiles ($19B/$86B/$361B per yr), breakeven table, frontier shares 54–84%, and k_max = 5.1e13 / 5.1e12.
- The "11% of mass above 1/2" figure is not stored, but I recomputed it from (mu, sigma): P(Z > 1.22) = 0.111. Correct.
- **Discrepancy (methods misreported, not a number mismatch).** §5.6 says the Sobol table uses "Jansen estimators with 2^17 base samples and 100 bootstrap replicates ... CIs no wider than ±0.006". All four rows of that table, row 1 included (scale k = 0.0656 → "0.07"), come from E8 (`sobol_ST`, n = 2^15, **no bootstrap**). Only E5 (shown only in Fig. 4) uses 2^17 with bootstrap CIs. The CI claim is therefore unsupported for three of the four rows. The rows will not change materially, but the text must be corrected.
- Table 1 omits the prior on L used in E5, E6 and E8 (log-U[L_low, L_high] in the code). §5.5 says L is drawn "from the baseline priors in Table 1".

## 3. Derivations (checked line by line)
- Prop 1: FOC, threshold and elasticities are correct. d ln S*/d ln k → beta/(1+beta) is correct. The exponential limit (beta → ∞, k/beta fixed) is correct.
- Lemma 2: correct (pLm·A^(-a·beta) = (k/beta)·A^a because 1 - a·beta = a). Concavity via the pointwise-minimum argument is correct.
- Prop 3: (i) logit concavity on (0, 1/2] plus Jensen is correct. (ii) Correct given p_i > p_min. Note that the AI-author grid violates this condition: about 0.07% of the mass is below p_min, since probabilities are clipped to 1e-12. S* = max(0, concave) is not concave, so the ordering there rests on numerics, as for (i). (iii) Correct asymptotically.
- Prop 4: (ii) and (iii) are both correct. I re-derived E[p_hat^a] / p_bar^a = exp(-a(1-a)(sigma0^2 - sigma1^2)/2). However, the EVPI/C*(p_bar) formula implicitly assumes m = 1 and drops the -k and pL(1-m) terms from the denominator. That should be stated. Lognormal p with sigma ≈ 2.4 puts substantial mass above 1, which the authors partly acknowledge (0.40 closed form vs 0.15 numerics).
- Prop 5: correct and elementary. One inconsistency: the "relative" breakeven 1/(pL) is relative to the baseline p, not to the residual risk at current spending, p·g(S_c). Under the paper's own calibration (k = $1B, S_c = $3.29B), g(S_c) ≈ 0.65, so the distinction matters for H2 elicitation.

No mathematical errors, but the results are mostly textbook-level once the functional form is fixed.

## 4. Strengths
1. Exceptionally transparent and reproducible. Closed forms are verified numerically (E0), the full pipeline runs in seconds and outputs reproduce exactly.
2. The derivations are correct, and the authors are careful about where results are conditional (Prop 3(i) remark, Limitations 7).
3. Honest framing. The paper explicitly declines to give a new p(doom) or a budget recommendation, and §6.3 deliberately breaks its own "near-unanimity" result.
4. Two policy-useful framings: (a) a pooling rule is an implicit budget decision and should be reported; (b) the status-quo dispute reduces to marginal effectiveness at current spending, independent of functional form.
5. A pre-registered falsification protocol, which is rare in this literature.

## 5. Weaknesses
**W1. Limited novelty; the main analytical results follow from the assumed form.**
- Compression with elasticity 1/(1+beta) is just the power-law FOC.
- Prop 5 is the standard marginal-benefit = marginal-cost condition.
- "Basis points of x-risk per $1B" breakeven framings are common in the EA/x-risk cost-effectiveness literature (e.g. Millett & Snyder-Beattie 2017 use the same logic for biorisk).
- EVPI on an uncertain damage/probability parameter is standard in climate IAM work (e.g. Peck & Teisberg; Nordhaus & Popp 1997, not cited).
- Combining expert distributions for risk *decisions* (rather than accuracy) is discussed by Clemen & Winkler (1999, Risk Analysis), also not cited.

The "gap" claim in §2 (no paper links dispersion to budget, compares pooling rules by budget, or prices disagreement resolution) may be technically true for AI specifically. But the contribution is an application and synthesis of standard tools, and it is incremental relative to Jones (2025).

**W2. The quantitative conclusions are artefacts of priors and box bounds.**
- The headline "100% of 20,000 draws" is fixed by the narrow baseline priors (k in [1e8, 1e11], beta in [0.05, 1]). The authors admit this, but it is still in the abstract.
- The "agnostic" 54–66% figure depends entirely on the chosen box (k up to 1e16, i.e. about 100x world GDP; beta down to 1e-3) and on fixing m = 1, which is inconsistent with the m ~ U[0.1, 1] used elsewhere.
- A share of a log-uniform box has no probabilistic meaning. Either report the frontier only, or justify the measure.
- None of k, beta or m has any empirical anchor (Limitations 1). Every numerical claim, including the EVPI-to-budget ratios of 19–114% used in policy implication (b), inherits this.

**W3. A conceptual mismatch between the forecasts and the model's p.**
- In the model, p is the catastrophe probability "absent additional mitigation".
- Published XPT and Grace forecasts are unconditional. They already price in each forecaster's expectations about future safety work, regulation and lab spending.
- Plugging them in as baseline p double-counts or under-counts mitigation, and makes p endogenous to anticipated S. This matters most for Prop 5 and Table 4, where the paper's most "robust" claims sit.
- Relatedly, the paper compares PV optima over 75 years with current annual philanthropic spending ($189M/yr, annuitised over 25 years) without modelling how future spending is expected to grow.

**W4. EVPI interpretation is fragile.**
- Treating the cross-forecaster distribution as the planner's prior over an objective chance p, and then "learning p", is ill-defined for a one-shot event.
- Limitation 6 notes this but does not temper the policy claim in §5.4 / §7(b): "a portfolio that spends almost nothing on resolving the disagreement is unlikely to be optimal".
- There is no model of how much variance real evaluations or forecasting can remove, or at what cost. EVPI/S* > 1 for the XPT mixture is driven by a within-group dispersion borrowed from the AI-author fit, which is an assumption.

**W5. Belief inputs are thin reconstructions.**
- The fit is a two-moment logit-normal with a single "validation" statistic. The fit gives 37.8% at or above 10%, against a reported 37.8–51.4% range spanning different question framings, so the agreement is partly coincidental.
- The XPT dispersion is borrowed rather than estimated.
- The "domain experts 3%" group is small and self-selected.
- No uncertainty is propagated from the belief reconstruction itself.

**W6. Overclaims and imprecision in the abstract and text.**
- Abstract item 3 says linear pooling "always" implies at least as much spending as log-odds. That is false in general, and the paper's own Prop 3(i) is conditional.
- Item 6 / §5.3 conflates "the linear-pool optimum" with "the optimum at the interval midpoint". For the AI-author 10–90% interval, minimax regret ($81.7T) is 26% below the midpoint optimum ($110.9T), which is not "close".
- §3.5 says the minimax-regret grid uses "60 values of p". The reviewer could not confirm this count in the code shown and asks the authors to state it explicitly.

**W7. Missing structure that would change conclusions.**
The analysis has:
- no absorptive capacity or talent constraint, hence the $4–20T optima;
- no counterproductive spending (safety work that speeds up capabilities);
- no free-riding or multi-actor public-good structure, since a single global planner is assumed;
- no dynamic learning, even though the VOI analysis begs for a sequential model.

Of these, only Limitations 1–2 are acknowledged.

**W8. Sharpness of the sub-question.**
The motivating question asks for a probability and a resource level. The paper answers neither and offers a decomposition instead. That is a legitimate choice, but the title and framing should make the reframing explicit, and the empirical payoff rests on a survey that has not been run.

## 6. Questions for the authors
1. Can you redo Prop 5 / Table 4 with p defined as risk at *current* spending (relative to p·g(S_c)), and discuss how forecasts conditional on expected future spending should enter?
2. Can you justify the frontier box bounds, or replace the "share of box" with a prior-free statement such as the frontier curve plus a credible range from any external anchor (e.g. cost per unit of safety research progress)?
3. Can you report Sobol CIs for all four rows of the §5.6 table (or relabel the table as coming from E8 with n = 2^15), and add the L prior to Table 1?
4. How sensitive are the EVPI shares to the assumed XPT within-group sigma? Please give a sweep.
5. How do the conclusions change with an upper bound on absorbable spending, or with a probability that spending is counterproductive?
6. Can you add a sequential two-period version in which part of S buys information, so that §5.4's policy claim is derived rather than suggested?

## 7. Reference spot-check
**Live verification was blocked in this session.** The Tavily quota was exceeded, the WebSearch per-turn budget was exhausted, and direct fetches to arxiv.org, nber.org, transportation.gov and lesswrong.com fail DNS resolution in the sandbox. I therefore checked references against my own knowledge (through mid-2026).

**Consistent with my knowledge, both existence and the content claimed:**
- Grace et al. 2024 (arXiv:2401.02843): 2,778 respondents; the 37.8–51.4% at-or-above-10% range is in the abstract.
- Karger et al., IJF 2025: XPT superforecaster 0.38% and domain-expert 3% AI-extinction medians by 2100.
- Jones 2024 (AER: Insights) and Jones 2025 (NBER WP 33602, ">= 1% of GDP").
- Carlsmith 2022 (~5% by 2070).
- Bengio et al. 2024 (Science).
- Field 2025 (arXiv:2502.14870; 111 experts; two clusters).
- Martin & Pindyck 2015; Weitzman 2009; Millett & Snyder-Beattie 2017; Satopää et al. 2014; Genest & Zidek 1986; Gilboa & Schmeidler 1989; Klibanoff, Marinacci & Mukerji 2005.

**Could not verify:**
- Truong, Puranam & Koçak 2025, arXiv:2512.06350. I do not recognise this paper and could not check it.
- The specific "$189M in 2024" figure attributed to the McAleese January 2025 update. The 2023 post exists; the updated figure is unconfirmed.
- US DOT VSL = $13.7M for base year 2024. This is plausible given $13.2M for 2023, but unconfirmed.

These are flagged as unverifiable because of the tooling outage, **not** because there is evidence that they are fabricated. Before acceptance, the authors should provide stable links or DOIs and quote the exact passages.

## 8. Scores (1–10)
| Criterion | Score |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 9. Decision
**Weak reject / major revision.** The paper is clean, correct within its model and fully reproducible. But its analytical contributions are standard consequences of an assumed isoelastic form. Its numerical findings are driven by unanchored priors and arbitrary box bounds. A conceptual mismatch (unconditional forecasts used as no-mitigation baselines) undercuts the claim that it is most robust (Prop 5 / Table 4). At AIES it would be borderline at best in its current form.

**Single most valuable change.** Replace the unanchored tractability priors with even a crude empirical or elicited anchor for marginal risk reduction per dollar, i.e. run a pilot of the §7.1 elicitation. At the same time, fix the baseline-vs-conditional definition of p in Prop 5. This would turn a decomposition exercise into an evidence-bearing answer to "how much should we spend".
