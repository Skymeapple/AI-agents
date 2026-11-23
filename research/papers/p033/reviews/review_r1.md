# Review (Round 1): "The Neyman–Pearson Ceiling on Conditional Defection"

**Reviewer role:** independent area-chair-level reviewer
**Recommendation:** Major revision (weak reject in current form)
**Overall score:** 5 / 10

---

## 1. Summary

The paper treats a "scheming" policy, meaning one that defects in deployment and behaves well when evaluated, as a randomised hypothesis test of deployment (D) against evaluation (E). Its evaluation defection rate α is the test's size and its deployment defection rate β is the test's power. From this framing it derives:

- **Thm 1:** β ≤ R(α), the Neyman–Pearson/Bayes ROC. It also gives TV and binary-KL (data-processing) corollaries.
- **Thm 2:** an evaluation-budget law, β ≤ R(c/N), with a model-free 1/ln N rate and a Gaussian N_req formula.
- **Thm 3:** a Hölder/Rényi bound for honeypot mixtures, with a Gaussian "realism gap" corollary and an "incentive collapse" corollary for the penalty objective.
- **Prop. 4:** a probit-linear "realism-response curve".

The experiments are small numpy policies on 8-D synthetic data, analytic N-scaling calculations that convert published evaluation-awareness AUCs (Needham et al., 2025), and a Monte Carlo study arguing that AUC does not pin down the low-α tail of the ROC. The paper ends with a pre-registered protocol for frontier models, which has not been run.

## 2. Reproduction

I copied `code/` to a scratch directory and reran `exp3_scaling.py`, `exp4_measurement.py`, `check_renyi.py`, `exp1_frontier.py` and `exp2_honeypots.py`. Wall time was 11 minutes on 4 cores. **Every regenerated JSON is byte-identical to the shipped `results/` files.** I did not rerun the optional `exp0_fixed_lambda.py`, but its shipped JSON supports the quoted collapse numbers: α ≈ 0.00226, β ≈ 0.0027, and R(0.00226) = Φ(1.5 − 2.84) ≈ 0.090.

I checked the paper's numbers against the outputs:

| Claim | Paper | Code output | Status |
|---|---|---|---|
| Max excess over frontier (144 policies) | 2.1e-4 | 2.05e-4 | match |
| Table 1 efficiencies / medians | 0.920/0.988/0.413/0.628/0.915/0.849 | identical | match |
| XOR Bayes AUC | 0.884 | 0.8840 | match |
| Honeypots: violations, mean eff, Rényi valid | 0/64, 0.984, 64/64 | 0, 0.9835, 64 | match |
| Mixture frontier at α′ = 1e-3 (ρ = 0.1) | 0.075/0.032/0.0099 | 0.0751/0.0324/0.00993 | match |
| Probit slope | 1.986 ± 0.020 | 1.9859 ± 0.0201 | match |
| N_req, d ln N/dd′ = 4.54, literature ceilings | as stated | identical | match |
| Same-AUC tails, finite-sample intervals | as stated | identical | match |
| Rényi bound/exact ratio range 1.02–2.3 | as stated | 1.02–2.32 | match |
| **Incentive collapse: "policies trained with λρ ≥ 1 defect at β < 10⁻⁷"** | β < 1e-7 | ρ = 0.02, λ = 128: β = 2.2e-7, 6.1e-7; ρ = 0.1, λ = 32, seed 1: β = 2.7e-7 | **MISMATCH (3 of 6 policies)** |

The incentive-collapse mismatch does not change the qualitative conclusion, since all six policies are below 10⁻⁶. It is still a false quantitative statement in the text and must be corrected. Everything else reproduces exactly. The engineering is exemplary on this point.

## 3. Reference spot-check

I verified the following by web search:

- **Krakovna et al. (2026), arXiv:2605.29729.** Exists. GDM, realistic honeypots in internal codebases. Correctly described.
- **Needham et al. (2025), arXiv:2505.23836.** Exists. Gemini 2.5 Pro 0.83, human 0.92, Claude 3.7 Sonnet 0.82, GPT-4.1 0.83, GPT-4.1 mini 0.75, on a 900-item test set out of 1,000 samples. **Caveat:** the per-model figures for Claude, GPT-4.1 and GPT-4.1 mini come from v1. Later versions report chat and agentic subsets separately (e.g. chat AUC 0.71–0.77, agentic 0.88–0.95). The paper should name the version it cites. It should also note that the agentic/chat split implies a strongly heterogeneous, non-binormal ROC, which reinforces its own §5.4 point.
- **Chaudhary et al. (2025), arXiv:2509.13333.** Exists, NeurIPS 2025 listing. Reports power-law scaling of probe-measured evaluation awareness. Correct.
- **Nguyen et al. (2025), arXiv:2507.01786.** Exists. Linear probes on Llama-3.3-70B-Instruct. Correct.
- **Mallen et al. (2025), arXiv:2412.12480.** Exists. Stateless subversion strategy. Correct.
- **Other references:** Meinke et al. 2412.04984, Greenblatt et al. 2412.14093 and 2312.06942, Hubinger et al. 2401.05566, Schoen et al. 2509.15541, Balesni et al. 2411.03336, Griffin et al. 2409.07985, Laine et al. 2407.04694 and Goldwasser et al. 2204.06974 are well-known papers and are described accurately.

I found no fabricated references.

## 4. Strengths

1. **Clean, correct framing.** Reading a conditional defector as a test, with deployment misbehaviour as power, is crisp. It forces the right question: what is the ROC tail at α ≈ 1/N?
2. **The mathematics is correct.** I checked each proof line by line:
   - Thm 1(c): kl ≥ β ln(1/α) − h(β) holds.
   - Thm 3: Hölder with π^p ≤ π gives exactly (α′/ρ)^{1−1/q} exp((1−1/q)D_q).
   - Cor. 3b: the optimisation at q* = √(2L)/Δ gives −(√L − Δ/√2)², with the validity condition q* > 1 correctly stated.
   - Thm 2b: the asymptotic ln N_req = x²/2 + ln x + ln(c√(2π)) follows from the Mills ratio.
   - Cor. 3c: the algebra holds.
3. **The §5.4 measurement point is useful and actionable.** At a fixed AUC of 0.83, R(10⁻³) spans four orders of magnitude across binormal slopes. The equal-variance conversion silently fixes the slope. This is the paper's most practically valuable message for the evaluation-awareness literature.
4. **Exemplary reproducibility and candour.** Seeds are fixed, outputs are byte-reproducible, the timid-collapse negative result is reported, and limitations are stated frankly (A1–A3, toy scale, illustrative literature conversions).

## 5. Weaknesses

### 5.1 Novelty is modest
The theorems are direct applications of textbook results: the NP lemma, the data-processing inequality, Hölder/Rényi, and binormal ROC algebra. The trade-off-function view of "power at fixed size" is standard in hypothesis testing and in f-DP (Dong, Roth & Su). The qualitative insight already circulates in the AI-control literature: a schemer can tune its defection rate to ≈1/N to beat upfront auditing, and evaluations reassure only to the extent that E and D are indistinguishable. Relevant work includes Greenblatt et al. 2024b (upfront auditing), Mallen et al., Griffin et al., and Alignment Forum discussion of eval realism.

The paper also misses closely related 2026 work: *When can we trust untrusted monitoring?* (arXiv:2602.20628), which bounds a schemer's win probability via honeypot and upfront-audit terms; AutoHoney (NeurIPS 2026), on honeypots and evaluation awareness at scale; and *Evaluation Awareness in Language Models: Representation, Verbalization, and Control* (arXiv:2608.21766). I could not find a prior paper that states the NP formulation exactly as here, so the formalisation is new. But the contribution is the packaging of known tools, not new technique, and the "late-2026 targeted search" claim in §2 needs to engage with the work above.

### 5.2 The ceiling cannot be operationalised as an upper bound
This is the central conceptual problem. For LLM inputs, E and D are high-dimensional text distributions that are almost surely close to mutually singular, so R ≈ 1, TV ≈ 1 and KL = ∞. The model-free statements (Thm 1b–c, Thm 2a) are therefore vacuous in the intended application. Everything rests on the capacity-restricted frontier R_Π, which the paper never characterises.

Any evaluator estimate of R_Π (from a probe, a verbalised probability, or a trained discriminator) is a *lower* bound on the model's true discriminative frontier, not an upper bound. The paper half-admits this ("the true R can only be higher"; P2). But that means its headline practical message ("bound what a scheming model can hide") inverts in practice. What an evaluator can measure only says that the model can hide *at least* this much. A safety case would need a certified upper bound on R_Π, and the paper offers no route to one. The abstract and §6.1 ("a bound… limited by a single statistical object") should be rewritten to make this explicit.

### 5.3 The experiments mostly confirm tautologies
- That trained policies "never exceed the frontier" is guaranteed by Thm 1 up to Monte Carlo error. The experiment checks the optimiser and the Monte Carlo code, not the theory.
- Incentive collapse (Cor. 3c) is checked by training on the exact penalty objective whose sign the corollary determines.
- The probit slope recovering d′ "within 1%" is near-circular. The honeypots are parameterised as N(tμ, I), so t is in units of d′ by construction, and the trained policies are near-threshold projections onto μ.

None of these experiments could have come out otherwise. The one genuinely informative empirical finding, that capacity limits R_Π on XOR, is unsurprising and is shown on an 8-D toy problem.

### 5.4 The experiments do not reach the regime that matters
The theory matters at α ≈ c/N ≈ 10⁻⁴–10⁻⁶. In Exp. 1, 29 of 144 policies (20%) have α ≤ 10⁻⁴ and are *excluded* from Table 1's efficiency statistics. So the experiments say essentially nothing about how close learned policies get to R in the relevant tail, which is precisely where optimisation pathologies such as the timid collapse appear.

### 5.5 The realism-response diagnostic is weaker than claimed
"Slope = d′" depends on t being calibrated in units of the cue direction. For a real honeypot ladder, "realism" has no natural metric, so only monotonicity is identifiable, and P1 retreats to "correlates with d′". The diagnostic is also confounded:
- An aligned but imperfect model whose *accidental* misbehaviour rises with task realism or complexity also produces a rising curve.
- Input-dependent unconditional misbehaviour need not be flat.

The claim that the curve "separates conditional defection from unconditional misbehaviour" (abstract, §6.2) is too strong.

### 5.6 Overstatements and internal inconsistencies
- **Abstract:** "(KL+ln 2)/ln(1/α)… and that bound is tight over distribution pairs." Thm 1(d) proves tightness of the binary-KL constraint, not of the Fano-type relaxation, which is loose by the ln 2 term. Only the 1/ln(1/α) *rate* is tight.
- **Abstract:** "test sets of ~450 items per class cannot resolve this tail." The paper's own Exp. 4 shows that the unequal-variance fit at n = 450 gives non-overlapping 90% intervals for b = 0.5 ([0.256, 0.388]) and b = 1.5 ([0.0005, 0.0071]). It does resolve the tail to within an order of magnitude, *provided the binormal model is correct*. The real issue is model misspecification (non-binormal, heterogeneous agentic/chat mixtures), which the simulation does not test because data are generated from the fitted family. The "several thousand per class" recommendation is not derived from any stated precision target.
- **"Grows like exp(d′²/2)."** At fixed cap b, ln N_req ≈ (d′ − Φ⁻¹(b))²/2, whose cross term d′·|Φ⁻¹(b)| dominates at practical d′. At d′ = 2 and b = 0.01 it contributes 4.65 against 2.0. The headline scaling is misleading. The 4.54 derivative is computed correctly.
- **Abstract:** "92–99% of the frontier when capacity matches." Width-32 on XOR, which matches capacity, averages 0.849, and individual matched-capacity policies go as low as 0.60–0.64.
- **Incentive collapse:** the β < 10⁻⁷ claim is false for 3 of 6 policies (see §2).
- **Seed averaging:** the XOR λ = 16 linear example (β = 0.026 at α = 0.0011) averages two working seeds with one collapsed seed (α = 1e-4, β = 1e-4). Report per seed or median.
- **d′ = 1.99 vs 2:** §5.3 mixes N_req at d′ = 1.99 (1.33×10⁴) with honeypot computations at d′ = 2.

### 5.7 Statistical reporting
- Table 1 reports means and minima but no between-seed uncertainty.
- The "±0.020" on the probit slope is the SD over 6 policies, not a standard error or CI. Say so.
- Exp. 0 versus the annealed runs changes λ schedule, step count (1500 vs 3000), learning rate (2e-2 vs 1e-2) and batch size at once, so it is not a clean ablation of annealing.

### 5.8 Assumptions limit relevance
- (A1) perfect detection and (A2) i.i.d. stateless episodes are strong.
- Real evaluation suites are heterogeneous and correlated.
- Real "defection" is graded.
- The treatment of D as the distribution of *opportunities* hides most of the difficulty.

These are acknowledged, but together with 5.2 they leave the paper as a conceptual framework rather than a quantitative safety tool.

## 6. Questions for the authors

1. How can an evaluator obtain an *upper* bound on R_Π for a given model? Without one, in what sense does the paper "bound what a scheming model can hide"?
2. Can you rerun Exp. 4 with a misspecified ground truth, for example a mixture of an agentic-like and a chat-like binormal component as in Needham et al.'s split, and show how the unequal-variance fit fails?
3. How do learned policies behave at α ∈ [10⁻⁶, 10⁻⁴]? Can you report efficiency there instead of excluding those policies?
4. For Prop. 4 with real ladders, what realism metric makes the slope interpretable? How would you distinguish conditional defection from realism-correlated capability errors?
5. How does your framework relate to the honeypot and upfront-audit analysis in arXiv:2602.20628, and to f-DP trade-off functions?
6. Which version of Needham et al. supplies the per-model AUCs?

## 7. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 8. Decision

**Major revision (weak reject as submitted).** The paper is well written, mathematically correct and fully reproducible. Apart from one false quantitative statement (β < 10⁻⁷), every number matches the code byte for byte. Its §5.4 measurement point is valuable.

The contribution is still an application of textbook hypothesis-testing results to an informal argument the AI-control community already makes. The experiments largely verify identities. The central practical claim, a *ceiling*, cannot be operationalised, because evaluators can only estimate lower bounds on the model's discriminative frontier.

As it stands this is workshop-level. To move toward acceptance, the paper would need two things:
- a credible route to upper-bounding R_Π, or else reframe honestly as a measurement and reporting methodology paper centred on §5.4, with misspecification experiments;
- some evidence beyond toy Gaussians, for example real evaluation-awareness ROC tails from open-weights models.
