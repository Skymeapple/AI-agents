# Review (Round 1): "When Does Bootstrapped Oversight Keep Stronger Models Aligned? Persistence, Correlated Blind Spots, and the Resampling Trap"

**Reviewer role:** independent area-chair-level reviewer
**Recommendation:** Major revision (weak reject in current form)
**Overall score:** 5 / 10

---

## 1. Summary

The paper proposes the Bootstrapped Oversight Schedule (BOS) model, a stylized probabilistic model of nested/iterated oversight. Each new model has a misalignment hazard μ(Δ)=1−(1−μ₀)e^{−λΔ}. Overseers detect misalignment with an Elo-logistic probability d(δ). A misaligned parent corrupts its child with probability γ. Misaligned overseers collude with probability κ. Overseer errors follow a one-parameter mixture ρ of the comonotone and independence copulas. Flagged models are retrained up to R times before the run aborts.

The paper states five results:
- **Thm 1:** under γ=1, a floor P(S) ≤ exp(−λ(1−d̄)G − n c₀).
- **Thm 2:** an exact exponential law for chain oversight, P(S)=exp(−G h(G/n)), with a G-independent optimal step size and a first-order closed-form rate.
- **Prop 3:** the Fréchet–Hoeffding bounds for an ensemble of overseers.
- **Prop 4:** a "resampling trap" in anchored chains with ρ=1, where P(S) and P(F) both rise with R and anchoring gives no protection as R→∞.
- **Prop 5:** a gap-independent limit a/(a+ε₀) when γ<1.

Exact dynamic programs are checked against Monte Carlo. A toy MLP experiment estimates the dependence weight ρ̂≈0.42–0.47 and shows that best-of-20 adversarial selection roughly triples the all-miss rate.

## 2. Reproduction and verification

- **Code rerun.** I reran `run_theory.py`, `gamma_limit.py` and `toy_rho.py` from a clean copy in a scratch directory (about 50 s plus about 2 min). All three JSON outputs match the committed `results/` files exactly (tolerance 1e-9, every field).
- **Numbers checked against JSON.** Every number I checked in Tables 1–4, §5.1–5.5 and the abstract matches the JSON:
  - r* = 1.174e-4, Δ* = 45.4, r_lin 4.3% below r*, half-lives 1000 / 5902 / 10000;
  - n* = 9 / 22 / 44 / 88;
  - DP–MC maximum discrepancy 0.0019;
  - property test: 0 violations, minimum slack 2.6e-5;
  - every cell of Table 2;
  - ρ sweep 0.0065 … 0.112;
  - Table 3;
  - toy rates 0.115 / 0.014, 8× and 17×, ρ̂ per seed, seed-0 CI [0.33, 0.38];
  - κ=0.5 sensitivity 0.985 → 0.991.
- **Small labelling problems:**
  - The Monte Carlo caps retries at `max_attempts=50`. The Monte Carlo columns labelled "R=∞" (ρ=0.44, ρ=0 in Table 2, and the ρ sweep) are really R=50. That is why they show nonzero aborts (0.002, 0.001), which is impossible at true R=∞.
  - "Best n" in the κ/γ sensitivity (n=150) sits on the search cap (nmax=150), so it is not an interior optimum.
  - The toy measurement is cross-referenced as §5.4 (in §3, the Figure 4 caption and the Reproducibility note), but it is §5.5.
- **References.** Spot-checked by web search: Engels et al. 2025 (all four NSO rates confirmed, including Wargames 9.4%), Kim et al. 2025 (60% figure on one leaderboard confirmed), Bhatt et al. 2025 Ctrl-Z (58%→7% at about 5% usefulness cost confirmed), Goel et al. 2025 (CAPA, judge bias, weak-to-strong similarity, rising error similarity confirmed), Cloud et al. 2025 (subliminal learning requires a shared base model, confirmed), Littlewood & Miller 1989 (TSE 15(12), DOI 10.1109/32.58771, diversity reduces coincident failure, confirmed), Charikar et al. 2024 (NeurIPS 2024, misfit-based gain, confirmed), Griffin et al. 2024 (POSG formalisation, confirmed). I found no fabricated or misrepresented reference. One nit: Ctrl-Z is cited as "Bhatt, A., Shlegeris, B., et al." although Shlegeris is the last author.

## 3. Strengths

1. **A clean, well-posed abstraction.** The model isolates three mechanisms (persistence, correlated blind spots, resampling) that the NSO analysis of Engels et al. leaves out, and states which quantities matter: γ, d₀, ρ.
2. **Correct and mostly clean derivations.** I checked them line by line:
   - Thm 1's per-step bound x/(x+y) ≤ (1−μ)/(1−d̄μ) and the convexity-style step dg/dΔ ≥ λ(1−d̄) are correct.
   - Thm 2 is immediate and correct.
   - The closed-form algebra for n_lin and r_lin = (√(λ(1−d₀))+√(μ₀d₁))² is correct.
   - The Prop 4 conditioning formula and monotonicity argument are correct.
   - The Prop 5 two-state Markov-chain argument is correct.
3. **Excellent reproducibility.** The code is short and deterministic, regenerates everything, and is honestly scoped. The limitations section is candid.
4. **A useful practical message.** Retrying until the overseer passes is best-of-R optimisation against the overseer, and abort or investigate beats resample. The connection to Ctrl-Z's "resample for incrimination" is apt.
5. **Concrete falsifiable predictions (P1–P4) and a pre-registrable protocol.**

## 4. Weaknesses

### W1 (major): The headline "floor" is misinterpreted for ensembles, and the paper's own Table 2 contradicts that reading
Thm 1 needs an *upper* bound d̄ on per-attempt detection. Prop 3 (Fréchet–Hoeffding) gives a *lower* guarantee: detection is at least max_j d_j, and it can be as high as min(1, Σd_j).

Despite this, §4.1 says the single-overseer value d̄=d₀ "is also the guarantee available when error dependence is unknown". §4.3 says "Theorem 1 therefore applies to anchored oversight with d̄ = d₀ whenever dependence is unknown". Both have the direction wrong. With unknown dependence the only valid upper bound is d̄ = min(1, Σd_j), which is 1 for anchored chains, and then the floor is vacuous.

The paper's own numbers show this. In Table 2, anchored oversight with ρ=0 gives P(S)=0.993 at R=∞, G=1000, n=22. The Thm 1 bound with d̄=d₀=0.9 gives exp(−0.0693−22·0.00091)≈0.915. So the anchored ensemble beats the claimed floor by a wide margin.

As a result, these claims hold only for chain oversight or comonotone ensembles:
- C2 ("oversight can slow the decay by at most 1−d̄ … cannot stop it");
- the Interpretation paragraph ("peer-level detection d₀ is the single quantity that bounds what bootstrapping can achieve");
- Discussion lever 2 ("better weak-to-strong transfer across large gaps cannot beat the floor").

They are not general statements about bootstrapping. Once d̄ is defined properly as the ensemble's detection, Thm 1 is nearly a tautology: if each step leaks at least μ(1−d̄), losses compound.

**Required:** restate C2 and the abstract accordingly. Make d̄ explicitly the ensemble's (dependence-dependent) detection. Remove the "applies with d̄=d₀ whenever dependence is unknown" claim.

### W2 (major): Novelty and depth are limited
Each formal result is close to textbook:
- Thm 2 is the product structure of NSO with G factored out of the exponent.
- Prop 3 is the Fréchet–Hoeffding bound, cited from Nelsen.
- Prop 4 is 1−d^R → 1 under independent retries.
- Prop 5 is the stationary distribution of an ergodic two-state chain versus an absorbing one.

The genuinely new content is the framing and the G-independence of Δ*. That is useful, but it is closer to a careful workshop or position paper than a main-track NeurIPS theory contribution.

Related work also misses directly relevant literature on correlated failures across safety layers, e.g. "AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures?" (arXiv:2510.11235), and recent bootstrapped-monitoring work (e.g. arXiv:2606.11998).

### W3 (major): The persistence result depends on how parameters scale with step size
Prop 5's limit a/(a+ε₀) does not involve λ at all. Recovery a=(1−μ₀)(1−γ) and leak μ₀ are both charged *per step*, independent of Δ, so infinitely many tiny steps give infinitely many recovery chances.

Gap-independence itself is generic for any recovering (ergodic) chain at fixed Δ. But the specific limit, and the policy claim that "persistence γ, not capability gap, decides", depend on γ being Δ-independent. It is at least as plausible that a parent nearly identical to its child corrupts it almost surely, i.e. 1−γ(Δ)→0 as Δ→0. Under that parameterisation the n→∞ limit goes to 0, and the optimal schedule is finite and gap-dependent.

Table 3 already shows the tension: at γ=0.999 the limit (0.495) is far below the best finite-n value (0.954 at G=400). Please analyse γ(Δ) parameterisations, and state the result as "absorbing vs ergodic" rather than as a scaling law.

### W4 (major): The toy ρ̂ does not measure the BOS ρ
In BOS, ρ governs dependence of overseer misses *over draws of misaligned models* (one draw per training attempt). In the toy, ρ̂ is dependence *over input points* of a fixed classifier ensemble, driven by points near a noisy decision boundary, i.e. item-difficulty heterogeneity. That dependence of this kind appears is unsurprising and is not specific to oversight.

The step "plugging ρ=0.44 into the BOS model … P(F)=0.059" therefore links two different quantities. Two further problems:
- The code computes pairwise ρ̂, which comes out at 0.36–0.56 and differs from the 4-way ρ̂. That suggests the one-parameter mixture is misspecified, yet the paper does not report it.
- Three seeds, one task and one training-set size are thin even for a toy.

The paper's limitations section concedes most of this, but the abstract and §5.5 still present ρ̂ as "a toy measurement of ρ" that plugs into the model.

### W5 (moderate): The Monte Carlo is validated only where an exact solution exists
The ρ<1 Monte Carlo branches (copula mixture for misses and false positives, honesty fixed within a step) are never checked against any analytic special case, e.g. ρ=0 with a single overseer, or a small-n brute-force enumeration. Table 2's ρ=0.44 and ρ=0 columns and the whole ρ sweep rest on this unvalidated code path. The R=50 cap labelled as R=∞ (see §2) also needs fixing.

### W6 (moderate): The illustrative parameters drive the qualitative message more than acknowledged
Claims such as "a single step is almost worthless", "most of the benefit is lost at moderate ρ" and "R=5 is a reasonable operating point" are tied to d₀=0.9, k=1, μ₀=0.01 and f=0.02. The sensitivity analysis covers r*/λ only. It does not cover the anchored, retry or ρ conclusions (Table 2, Figure 4).

### W7 (minor)
- Theorem 2(a): "equality whenever G/Δ* is an integer" should say the supremum over integer n equals exp(−G r*) only then. As stated it reads oddly.
- The closed form is called "first order in λΔ and d₁Δ", yet the decisive term λd₁Δ² is second order. Please clarify that the expansion keeps the leading cross term.
- The Thm 1 property test covers only chain oversight with κ=1, the case where d̄=d₀ holds. It provides no evidence for the anchored or ensemble claims.
- Section cross-references §5.4 vs §5.5 (see §2).
- Prop 4's limit claim "P(F) → chain silent-failure probability" depends on f: s, q in the chain formula should use the same f′. Please state this.
- The abstract's "ρ̂ ≈ 0.42–0.47" is a range of per-condition means. Individual seeds span 0.36–0.48.

## 5. Questions for the authors

1. How do you reconcile Table 2 (anchored, ρ=0: P(S)=0.993) with the claim that Thm 1 applies to anchored oversight with d̄=d₀ under unknown dependence?
2. What happens to Prop 5 and the "persistence decides" message if 1−γ scales with Δ, i.e. recovery is per unit of capability rather than per step?
3. Can you validate the ρ<1 simulator against an exact computation (e.g. enumerating overseer honesty and miss patterns for small n)?
4. Why should per-input error dependence of fixed classifiers predict per-model-draw miss dependence in BOS? Can you design a toy in which overseers judge *trained models* (e.g. backdoored small networks) rather than inputs?
5. What do the pairwise ρ̂ values (0.36–0.56) imply about the adequacy of the one-parameter copula?

## 6. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 (theorems correct; interpretation of Thm 1 for ensembles wrong) |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 7. Decision

**Major revision / weak reject.** The paper is honest, reproducible and clearly written. The derivations are correct, and every reported number matches rerun code. The references I checked exist and say what is claimed. However:
- the central "alignment-loss floor" message is wrongly extended to ensembles under unknown dependence, and the paper's own Table 2 contradicts that extension;
- the formal results are thin relative to the target venue;
- the persistence result depends on a per-step parameterisation of γ;
- the toy measurement does not measure the model's ρ.

With W1 fixed, W3 analysed and a better-matched empirical estimate of ρ, this could make a solid workshop paper, or a main-track paper if the analysis were deepened, e.g. optimal R and abort policies, or γ(Δ) scaling.
