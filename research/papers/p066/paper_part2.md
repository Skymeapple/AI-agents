## 3 Model

### 3.1 Primitives

Time is discrete. A period is one practice session. A learner has two states:

- **Skill** s in [0,1]: the probability of solving a representative problem correctly when the learner attempts it seriously.
- **Persistence** g in [0,1]: the probability of attempting seriously rather than giving up. It summarizes the academic self-efficacy and persistence documented by Liu et al. (2026) and is grounded in Bandura's (1977) account of mastery experiences.

Unassisted performance, which is what an exam without AI measures, is E = g·s.

An **assistant** is characterized by two numbers:

- **In-the-moment quality** q in [0,1]: the expected score on work it performs.
- **Pedagogical retention** sigma in [0,1]: the share of the learning from one's own work that is preserved when the assistant does the work. An answer-giving assistant has sigma = 0. A hint-giving tutor that leaves the key steps to the learner has sigma > 0. sigma = 1 would be an assistant that teaches as effectively as solo practice.

Each period the learner chooses an **offloading share** a in {0, 0.1, ..., 1}.

**Assisted performance and payoff.** Assisted (practice) performance is P(a) = (1−a)·g·s + a·q. The flow payoff is

u(a; s, g) = w·P(a) − c·(1−a)·(1 − θ·s),                                    (1)

where c is the effort cost of solo work. The cost falls with skill (θ = 0.5), and w = 1 normalizes the grade value of practice.

**Skill dynamics.**

s' = s + η·φ_t·[(1−a)·g + σ·a]·(ε + s)·(1 − s) − λ·s,                    (2)

Learning requires engaged solo work, since (1−a)·g is the share of work done and not abandoned, or retained learning from assisted work, σ·a. Learning is faster when the learner already has some skill: the self-productivity factor (ε + s), with ε = 0.05, follows Cunha and Heckman (2007). Learning slows near mastery, through (1 − s). Skill decays at rate λ. Plasticity is η·φ_t, where φ_t is an age profile that we vary.

**Persistence dynamics.**

g' = g + κ·[(1−a)·s·(1−g) − ρ·a·(1−σ)·g] + μ·(ḡ − g),                       (3)

Solo work builds persistence in proportion to success (s). Receiving complete answers erodes it at relative rate ρ, a reduced-form version of the mechanism Liu et al. (2026) propose, namely that AI "conditions people to expect immediate answers". Retained-learning help erodes it less, through the factor (1−σ). The last term is mean reversion to the individual's baseline ḡ.

**Choice.** The learner is present-biased with quasi-hyperbolic weight β ≤ 1 (Laibson, 1997). In the reduced model (Section 4) the learner values next-period skill at a linear shadow value v, discounted by βδ. In the full DP (Section 4.4) the learner is naive (O'Donoghue and Rabin, 1999): it uses the exponential continuation value but weights it by β. In the calibrated two-state model (Sections 5 and 6) the learner uses a one-step-lookahead rule,

a* = argmax_a  u(a; s, g) + β·Ω·g'(a)·s'(a),                                    (4)

where Ω is the stake attached to the next unassisted test. We use (4) for calibration and projection because Bastani et al.'s sessions end with an unassisted exam on the same material, so (4) is the natural description of within-session incentives. It also requires no assumption about the learner's horizon. Using the same rule in projections avoids mixing a behavioral rule fitted to data with a different rule for counterfactuals.

### 3.2 What the model leaves out

The model has one generic skill and no curriculum structure. Grading regimes, peer effects, teacher responses and socio-emotional uses of AI (companionship) are absent. The assistant does not adapt to the learner, and its quality does not change over time except in an explicit scenario. Section 9 discusses these omissions.

## 4 Theory: Thresholds, Traps and Retention

For the analysis we fix g = 1 and use a linear shadow value of skill: the learner perceives continuation βδ·v·s'. Because u and s' are linear in a, the choice is bang-bang (a ∈ {0,1}). The perceived net benefit of attempting alone rather than offloading is

D(s) = w·(s − q) − c·(1 − θ·s) + B·G(s),   where G(s) = (ε + s)(1 − s),  B = β·δ·v·η·(1 − σ).   (5)

D is a concave quadratic in s when B > 0. Let s* denote the positive steady state of solo practice, which solves η·G(s) = λ·s, and let s_σ denote the steady state under perpetual assisted work with retention σ, which solves σ·η·G(s) = λ·s.

**Proposition 1 (threshold rule).** Suppose B > 0 and D(0) = B·ε − w·q − c < 0, and suppose D has a real root in (0,1). Then the learner offloads (a = 1) exactly on [0, s_L) ∪ (s_H, 1], where

s_L = [b₁ − √(b₁² + 4·B·b₀)] / (2B),  b₁ = w + c·θ + B·(1 − ε),  b₀ = B·ε − w·q − c,

and s_H ≥ s_L is the other root (if s_H ≥ 1, the upper offloading region is empty).

*Proof.* D(s) = −B·s² + b₁·s + b₀ is a downward parabola with D(0) = b₀ < 0. Its lower root is s_L, and D < 0 exactly outside [s_L, s_H]. ∎

**Proposition 2 (offloading trap; σ = 0).** Let σ = 0, s_L ∈ (0, s*), and s_H ≥ s*. Assume the solo-practice map f(s) = s + η·G(s) − λ·s is increasing on [0,1], which holds whenever η·(1+ε) + λ < 1. Then (i) for every s₀ < s_L, s_t = (1−λ)^t·s₀ → 0, so 0 is absorbing; (ii) for every s₀ ∈ [s_L, 1], s_t → s*.

*Proof.* (i) On [0, s_L) the learner offloads, so s' = (1−λ)·s < s, and the trajectory stays in [0, s_L) and converges to 0. (ii) On [s_L, s*], f(s) ≥ s because η·G(s) ≥ λ·s below s*. Since f is increasing, f(s) ≤ f(s*) = s*. The sequence is therefore nondecreasing, stays in [s_L, s*] ⊂ [s_L, s_H] (so the learner keeps attempting), and converges to the unique fixed point s*. For s₀ ∈ (s*, s_H] the symmetric argument gives a nonincreasing sequence bounded below by s*. For s₀ > s_H the learner offloads, s decays geometrically, and the trajectory enters [s*, s_H] if s_H ≥ s*, after which the previous case applies. ∎

**Proposition 3 (comparative statics).** At an interior threshold, D′(s_L) > 0, and by the implicit function theorem
∂s_L/∂q > 0, ∂s_L/∂c > 0, ∂s_L/∂β < 0, ∂s_L/∂η < 0, ∂s_L/∂v < 0, ∂s_L/∂σ > 0.

*Proof.* ∂s_L/∂x = −(∂D/∂x)/D′(s_L). The partial derivatives are ∂D/∂q = −w < 0; ∂D/∂c = −(1 − θ·s) < 0; ∂D/∂β = δ·v·η·(1−σ)·G > 0, and similarly for η and v; ∂D/∂σ = −β·δ·v·η·G < 0. ∎

The sign for σ is worth stating plainly: **a pedagogically better assistant makes offloading more attractive.** It does so because it reduces the learning advantage of solo work. This does not harm the learner (Proposition 5); it changes what "using the tool" means.

**Corollary 4 (β·η invariance).** In the reduced model, s_L depends on β, η, v, δ and σ only through B = β·δ·v·η·(1−σ). In particular, a fall in plasticity by factor k and a fall in β by factor k shift the threshold identically.

The developmental implication follows directly. Early adolescents tend to have lower β (steeper discounting; Green et al., 1994; Steinberg et al., 2009) and possibly higher η (Fuhrmann et al., 2015). Vulnerability to the trap therefore rises with age if the product β_t·η_t falls with age, and falls with age if the product rises. Neither growth rate is currently measured well enough to sign the product, so this is a prediction to test (P3 in Section 7), not a conclusion.

**Proposition 5 (retention removes the absorbing state but bounds skill below s*).** Let σ > 0. Then (i) 0 is not a fixed point, since at s = 0 with a = 1, s' = σ·η·ε > 0. (ii) Under perpetual assisted work (a ≡ 1), s_t → s_σ, which is strictly increasing in σ, with s_σ < s* for σ < 1 and s_1 = s*. (iii) Under the optimal threshold rule, long-run skill is s* if the trajectory ever reaches the attempt region [s_L(σ), s_H(σ)] and s_σ otherwise. In either case it is at least min(s₀, s_σ) for any start s₀.

*Proof.* (i) is immediate. (ii) s_σ is the positive root of σ·η·G(s) = λ·s; the left side increases in σ for s > 0, so the root increases in σ. (iii) Above s_L the argument of Proposition 2 applies. In the offload region the map is f_σ(s) = s + σ·η·G(s) − λ·s, which is monotone and converges to s_σ. ∎

Proposition 5 says that a hint-only tutor does not need to change *whether* students offload to prevent the trap. It works by changing what offloading costs. The relevant design quantity is σ, and the long-run loss under perpetual tutor use is s* − s_σ.

### 4.1 Numerical verification (code: `theory.py`; output: `results/theory_results.json`)

Baseline reduced-model parameters: w = 1, c = 0.3, θ = 0.5, q = 0.9, σ = 0, η = 0.15, λ = 0.03, ε = 0.05, β = 0.7, δ = 0.98, v = 25. These are illustrative values, not estimates.

- **Threshold and basin boundary.** The analytic threshold is s_L = 0.431. Simulating 199 initial conditions on a 0.005 grid for 600 periods gives a basin boundary of 0.435, within one grid step. Trajectories above the boundary converge to 0.812, which equals the analytic s* = 0.812. Figure 1B shows trajectories.
- **Comparative statics** (central finite differences of the analytic s_L): ∂s_L/∂q = +0.73, ∂s_L/∂c = +0.57, ∂s_L/∂β = −0.73, ∂s_L/∂η = −3.41, ∂s_L/∂v = −0.020, ∂s_L/∂σ = +0.51. All signs match Proposition 3.
- **β·η invariance (reduced model).** Holding β·η = 0.105 and varying β from 0.4 to 1.0 leaves s_L = 0.431 to machine precision. Figure 1C shows hyperbolic iso-threshold curves.
- **Retention.** Under perpetual tutor use, long-run skill starting from s₀ = 0 rises from 0 (σ = 0) to 0.046 (σ = 0.1), 0.200 (0.2), 0.406 (0.3), 0.629 (0.5), 0.791 (0.9) and 0.812 (σ = 1 = s*). With q = 0.9 the threshold s_L(σ) reaches 1 for σ ≥ 0.4, so the learner always offloads, and long-run skill equals s_σ exactly as Proposition 5(ii) states (Figure 2).

### 4.2 Where the reduced model breaks: stationary dynamic program

To check whether the analytic conclusions survive when the shadow value of skill is endogenous, we solve an infinite-horizon dynamic program on a 401-point grid. The naive learner chooses using β·δ·V(s′), where V solves the exponential Bellman equation with flow payoff (1) plus ω·s (ω = 0.5, the value of unassisted competence). We set q = 0.7 here, because with q = 0.9 > s* an even fully patient learner rationally offloads forever: if the assistant is always available and assisted performance is what is rewarded, offloading is optimal. This is a substantive point in its own right. The harm in the model comes from the gap between what is rewarded now (assisted performance) and what is valued later (unassisted competence). It does not come from offloading as such.

Results (`dp_threshold_checks`, `dp_vs_q`, `dp_vs_beta`):

- In every case checked, the DP policy has the single-crossing threshold form of Proposition 1.
- The DP threshold rises with q (0.08, 0.35, 1.0, 1.0 for q = 0.5, 0.7, 0.9, 1.0) and falls with β (0.745, 0.54, 0.173, 0.095 for β = 0.4, 0.6, 0.8, 1.0). Both signs agree with Proposition 3.
- **The β·η invariance does not survive.** At β·η = 0.07, DP thresholds are 0.673, 0.515 and 0.750 for β = 0.5, 0.7 and 1.0. At β·η = 0.105 they are 0.638, 0.348 and 0.425. With an endogenous value function, η also changes how valuable skill is (through s* and the speed of accumulation), so the product is no longer sufficient, and the effect is non-monotone. Corollary 4 should therefore be read as a first-order prediction for local, short-horizon decisions, not as a structural invariance. Section 7 proposes a test of exactly this distinction.

