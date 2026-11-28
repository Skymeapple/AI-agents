# Review (Round 1): "Decodable Is Not Modelled: Suffix Ceilings and Counterfactual Rollouts for Testing World-Model Claims in Sequence Models"

## Summary

The paper asks when probe decodability of a latent world state counts as evidence of an internal world model. It proves three things:
- **Theorem 1 (suffix ceiling).** For a λ-contracting recurrent encoder, any linear probe's accuracy is at most Acc*_k + P(margin ≤ L·δ_t(k)), with δ_t(k) ≤ λ^k D.
- **Corollary 2.** For finite-context models, Acc ≤ Acc*_w.
- **Proposition 3 (free decodability).** Local states are linearly decodable from any encoder whose k-suffix features are in affine general position.

From these it derives two diagnostics: the effective suffix length k_eff, and counterfactual rollout agreement (CRA), which is scored only on discriminating steps and compared against random-subspace and full-swap controls. The testbed is a 5×5 torus gridworld with three colourings (unique, two-colour, 3 landmarks), in which Acc*_k is computed exactly by an HMM filter. The encoders are trained and untrained GRUs, two ESNs and small transformers, with three seeds each (transformers n = 1). A 40-epoch training-dynamics study reports an ordering: behaviour comes first, then a decodable state, then a causally used state.

## Verification performed

- **Code read in full** (`world.py`, `models.py`, `run_experiment.py`, `training_dynamics.py`, `aggregate.py`). The implementation matches the stated method:
  - The restart uses the 2k tokens obs_{t−k}..m_t.
  - The filter conditions on the same window.
  - The a priori bound uses λ^{2k}√d.
  - CRA uses closed-loop feedback and is scored on discriminating steps only.
- **Archived outputs against the paper.** I checked every quantitative claim in Sections 4–7 against `results/raw_*.json`, `dyn_*.json`, `summary.md`, `bound_tightness.json` and `cra_vs_probe.json`. These all match:
  - every Table 1 entry;
  - the suffix curves;
  - the δ values;
  - the bound infima (0.52/0.70/0.32/0.34);
  - the maximum truncation excess +0.010 over 324 cells;
  - the 6/8 checkpoints where CRA(probe) < CRA(random);
  - the CRA ranges 0.64–0.89 and 0.25–0.42 at k_eff = ∞ checkpoints;
  - Pearson r = 0.92 with n = 34, the 10 evaluations with gain ≤ 0, and the 21 with gain ≥ 0.17.
- **Independent rerun (partial).** I reran all code in a scratch copy. The machine was heavily shared (load average about 18), so the rerun had not finished when the review was due.
  - Completed and bit-identical to the archived outputs: the seed-0 suffix curves for all three worlds, and training-dynamics checkpoints at epochs 1–2 for all four runs.
  - Not completed: the full main-table rerun.
  - The code is deterministic and seeded, and what did finish matches exactly. I therefore have reasonable but not complete confidence in reproducibility.
- **One mismatch found (see Fatal 1).**

## Strengths

1. **A sharp, operational criterion.** The question "does the model have a world model?" is reduced to "does probe accuracy exceed the suffix ceiling of the encoder's effective memory?" This is a genuinely useful reframing and can be subtracted from probe results.
2. **The testbed makes the ceiling exact.** Varying how local the state is (unique → two-colour → landmarks) is a clean manipulation.
3. **Good intervention design.** CRA on discriminating steps only, with random-subspace and full-swap controls, is a real methodological improvement over single-step interventions. The observation that probe-subspace edits *underestimate* causal use (0.70–0.89 vs 1.00 for a full swap) is useful.
4. **Honest limitations section.** It covers vacuous a priori bounds, loose data-dependent bounds, n = 1 transformers and the schedule mismatch. Per-seed values are reported rather than hidden behind means.
5. **The proofs are correct.**
   - Theorem 1's perturbation argument holds: P(f(h_t)=s_t) ≤ P(f(h^{(k)})=s_t, E) + P(E^c).
   - Lemma 1 is correct.
   - Proposition 3: the augmented M×(d+1) matrix has full row rank iff the points are affinely independent.
   - The analytic-genericity remark is correct.
6. **All six-plus spot-checked references exist and say what is claimed.**

## Weaknesses

### Fatal (as defined by the review protocol)

**F1. A reported number does not match the outputs.**
- **The claim.** Section 5.3 states: "The trained GRU of seed 0 reaches 0.937 and predicts landmark hits with recall 0.85 (Table 1, per-seed values in results/summary.md), yet its state probe is only 0.250."
- **The output.** `raw_landmark_s0.json` gives `landmark_recall = 0.607` for this model. The value 0.85 (0.847) is the 3-seed *mean*, which is dominated by seeds 1 and 2. Those seeds have probe accuracy 0.99 and 0.75, not 0.25.
- **The effect.** The sentence overstates the "good behaviour, no state" dissociation for the very model used to illustrate it. The qualitative point survives through the dynamics runs (recall 0.86 with probe 0.34 at epoch 5), but the sentence must be corrected.

### Major

**M1. The "untrained encoders have a 2–3 step window" result is an artefact of hyperparameters, yet it is stated as general.**
- **Evidence.** The ESN with λ = 0.99 has an effective window of about 3 steps only because input scaling is 1.0 (U ~ N(0,1)) and the tanh units saturate. The paper itself shows this.
- **Missing sweep.** There is no sweep over input scale, spectral radius or leak rate, and no orthogonal or near-identity random RNN.
- **Over-general wording.** The abstract says "all untrained and fading-memory encoders have k_eff of 2–3". This is a statement about these particular initialisations.
- **Fix.** A memory-capacity-tuned reservoir (small input gain, λ → 1, d = 512) could plausibly hold 8–16 steps and push k_eff much higher without training. That is exactly the regime where the paper's criterion would be stress-tested. This sweep is needed before the claim can be made.

**M2. The abstract over-generalises the post-transition CRA range.**
- "After it, they reach CRA 0.64–0.89" holds for the training-dynamics checkpoints only.
- In the main runs, landmark seed 2 has k_eff = ∞ with CRA(probe) = 0.533, and seed 1 has 0.659.
- The abstract and P4 ("as we found 0.70–0.89") should report the full range, or state the restriction.

**M3. Proposition 3 is invoked where its hypotheses fail.**
- In the unique world, M = 25 × 4 = 100 reachable 1-suffixes. Affine independence therefore needs d ≥ 99.
- The d = 64 GRUs and ESN nonetheless reach 1.000. "This is Proposition 3 at work" (Section 5.1) is therefore not literally true: the proposition gives a sufficient condition that is unmet.
- **Fix.** Either state a separability-based version (Cover-style capacity for 25 classes) or rephrase the claim.

**M4. Mismatch between the theory's ceiling and the measured ceiling.**
- Definition 1's Acc*_k is defined under the true data distribution. Under that distribution, a known start gives Acc*_t = 1.
- The measured Acc*_k uses a uniform prior at the window start.
- **Consequences.**
  - The "∞" category is defined relative to the measured curve, not the theorem's.
  - The empirical check of Theorem 1 is against a different quantity. The +0.010 excess is attributed to this mismatch, but it is not quantified.
- **Fix.** Compute the exact finite-time marginal from cell 0. This is trivial in this world and would remove the approximation entirely.

**M5. Statistics are thin.**
- Three seeds per condition. The trained GRUs are bimodal across seeds, as the authors acknowledge.
- No confidence intervals on CRA or probe accuracy. About 1,920 pairs is fine within a run, but between-seed variance dominates.
- Pearson r = 0.92 is computed over 34 points, 28 of which are repeated checkpoints of 4 training trajectories. The points are not independent, and the effective n is small.
- Transformers: n = 1, with no CRA and no truncation test. Corollary 2 is therefore never exercised empirically, because the transformer's context covers the whole sequence.

**M6. Novelty is moderate.**
- **Already known.** Probing has long been compared against random-initialisation baselines (Li et al., 2023), control tasks (Hewitt & Liang, 2019) and amnesic or causal checks. Fading-memory approximation results are classical (Boyd & Chua, 1985; Grigoryeva & Ortega, 2018). Theorem 1 is a short perturbation argument, and Corollary 2 is trivial by the authors' own admission.
- **New.** The packaging of these as a subtractable, exactly computable ceiling, plus discriminating-step CRA.
- **Missing related work.**
  - Yuan & Søgaard (2025), "Revisiting the Othello World Model Hypothesis".
  - Mechanistic evidence that Othello-GPT relies on heuristics.
  - Concurrent work on observation-history baselines and editability, e.g., Brodjian, Hobley & Perona (2026), "When Generative Models Act as Simulators".
- **Verdict.** The headline empirical findings (random reservoirs are short-memory; trained RNNs path-integrate; local states are free) are largely expected. The large-model predictions P1–P5 are not tested.

### Minor

- k_eff compares a *linear* probe against a *nonlinear* Bayes ceiling. A small k_eff is therefore only a lower bound on the encoder's memory. The truncation test is the better memory measure and should be foregrounded.
- The trained GRU's truncated-probe accuracy "keeps rising with k". On its own this does not show long memory; any longer-window encoder would show the same. The decisive evidence is probe accuracy above the ceiling.
- The longer-training runs use a different LR schedule from the main runs, which confounds the "all four longer runs reach ∞" statement.
- For untrained encoders, the next-observation readout is fitted. "Untrained encoder passes CRA" therefore involves a trained readout; say so explicitly.
- `run_experiment.py` mislabels the log line "k=1,4,8,16,40" (acknowledged in the README).
- Corollary 2: the receptive field of n layers with window w′ is n(w′−1)+1. Using w = n·w′ is a valid upper bound but should be stated as one. Absolute positional embeddings also make h_t depend on t, which is harmless only because the bound is applied per fixed t.
- The abstract is very long and number-dense.

## Questions for the authors

1. How do k_eff and the truncation curves change for ESNs with input scale in {0.01, 0.1} and λ ∈ {0.99, 0.999}, and for orthogonally initialised random RNNs?
2. Can you recompute Acc*_k with the exact finite-time marginal from the known start, and report the "∞" criterion against it?
3. Can you run CRA on the transformer with edits propagated to all subsequent positions or to the KV cache? Alternatively, run the masked-attention truncation test so that Corollary 2 is exercised.
4. Can you report r with trajectory-clustered uncertainty (e.g., a bootstrap over runs), and add more seeds for the trained GRU in the 10-epoch condition?

## References spot-checked (all verified)

- Vafa et al. 2025 (ICML, PMLR 267: 60727–60747)
- Jin & Rinard 2024 (ICML, PMLR 235: 22160–22184)
- Karvonen 2024 (COLM 2024; acceptance confirmed by the author)
- Grigoryeva & Ortega 2018 (Neural Networks 108: 495–508)
- Ravichander, Belinkov & Hovy 2021 (EACL, 3363–3377)
- Wieting & Kiela 2019 (ICLR)
- Li, Nye & Andreas 2021 (ACL, 1813–1827)

No fabricated or misattributed citation was found.

## Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 5 |
| Rigor | 5 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 8 |
| **Overall** | **5 (weak reject)** |

## Decision

**Major revision.** The idea is clean and the code is reproducible and consistent with almost all reported numbers. However:
- one reported number is misattributed (F1);
- the central "untrained encoders are pinned at 2–3 steps" claim rests on a single, saturating reservoir configuration with no sweep (M1);
- the theory–measurement gap in Acc*_k and the thin statistics weaken the empirical case.

The most valuable addition would be a reservoir memory sweep showing whether the suffix-ceiling criterion still separates trained from untrained encoders when untrained encoders have long memory.
