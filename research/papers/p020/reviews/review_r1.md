# Review (Round 1): "Same Scores, Different Crowds: Authorship Information and the Structure of Human and Machine Creative Novelty"

**Reviewer role:** area-chair-level independent reviewer
**Recommendation:** Major revision (weak reject in current form), overall 5/10

## 1. Summary

The paper asks whether LLM "innovation" is the same kind of thing as human innovation. It recasts the question as one about population structure. Population novelty H(X) splits into within-agent entropy G plus authorship information I(X;A), which is the generalised JSD. The theory has four parts:
- Prop. 1: per-response score parity says nothing about this split.
- Prop. 2: an identity, N + C = I − Σw KL(r‖p), where competence C is the expected log-probability under the agent's own competence distribution.
- Prop. 3: temperature monotonically raises novelty and lowers competence.
- Prop. 4: with one list per agent, stable agents cannot be told apart from a persona-resampling generator.

Simulation 3 shows that response-level permutation nulls fail in both directions under within-list dependence, and that list-level nulls with replicates hold their size. The empirical part re-analyses Nath et al. (2024), with 220 humans and 8 LLMs × 11 temperatures × 5 replicates. It reports four findings:
- Model identity accounts for about 0.37–0.49 bits (10–12%) of LLM population entropy.
- Temperature raises run-to-run divergence and erodes model identity.
- In a "divergence ladder", 5 humans diverge more than 5 different LLMs (on all tasks at string level, on 2 of 3 at category level).
- A "norm-reference paradox": LLM lists look more original against human norms and much less original against LLM norms.

The paper ends with a pre-registered replicate protocol.

## 2. Reproduction check

- I copied the package to a scratch directory and confirmed the SHA-256 of both data CSVs against the README.
- I ran `simulate.py` (21 s), `empirical.py` (150 s), `human_within_entropy.py` and `make_figures.py`.
- **Every regenerated CSV and JSON in `results/` is byte-identical to the shipped files.**
- I checked the numbers in the paper against the outputs: Table 1 (sim3), Table 2 (B), the 5.4 Spearman table (C, log.json), Table 3 and the disjoint tests (D, D2), the 5.6 table (E), the Welch tests (log.json), the 5.7 exchange ratios (F), human G = 2.76/2.85 bits, and the derived percentages (89–98%, 85–91%, 59–78%). All match to the stated rounding.
- One wording slip: in Sec. 5.5 the g = 3 robustness check calls the 3-model groups "five-model groups" (0.78 is the g = 3, any-T value).
- No result lacks code behind it.

## 3. References spot-checked (8 of 23)

All of the following exist and say what the paper claims:
- Wenger & Kenett 2026, PNAS Nexus 5(3) pgag042 (preprint arXiv:2501.19361)
- Bellemare-Pepin et al. 2026, Sci. Rep., DOI 10.1038/s41598-025-25157-3
- Meincke, Nave & Terwiesch 2025, Nat. Hum. Behav.
- Nath, Dayan & Stevenson, ICCC'24, pp. 325–330
- Zhang et al., NoveltyBench, COLM 2025
- Peeperkorn et al., ICCC'24: temperature is weakly related to novelty and moderately related to incoherence
- Anderson, Shah & Kreminski, C&C'24, DOI 10.1145/3635636.3656204
- Si et al., ICLR 2025

I found no fabricated or misattributed citations.

## 4. Strengths

1. **Reproducibility is exemplary.** The data are pinned to a commit with hashes, seeds are fixed, the code is small and readable, and outputs regenerate bit-for-bit in about 3 minutes.
2. **The identification point is correct and useful to practitioners.**
   - Prop. 4 is simple, but it is stated clearly.
   - Sim 3 is a clean demonstration that response-level permutation nulls (implicit in many homogeneity analyses) are invalid in both directions under within-list dependence.
   - The T = 0 replicate "known-truth" check on real data (I* ≈ −0.3 to −0.5 bits where the truth is ≈ 0) is a nice touch.
3. **Matched comparisons.** The divergence ladder holds g and k fixed, so plug-in bias is comparable across structures. This is a sensible design for a bias-prone estimator.
4. **The temperature analysis is informative.** Temperature raises the replicate floor (ρ ≈ 0.96–0.98), lowers between-model information (ρ ≈ −0.93), and does not move within-list entropy or ratings.
5. **The paper is unusually honest about scope.** It says explicitly that the central human quantity is not identified, and that the competence side of Prop. 2 is untested.

## 5. Weaknesses

### 5.1 Novelty of the theory is overstated

- **Eq. (1) is not new.** It is the classical additive Shannon partition of gamma diversity into alpha (within) and beta (between) diversity from ecology (e.g., Lande 1996; Jost 2007), and the JSD identity of Lin (1991). The paper cites Lin but does not cite the diversity-partitioning literature, which already covers alpha/beta estimation, bias, and permutation nulls.
- **Props. 1 and 4 are one-line observations** about marginals and mixtures.
- **Prop. 3 is the standard exponential-family result:** d/dβ E[ℓ] = Var(ℓ).

None of this is wrong, but the "three results we prove" framing promises more theory than is delivered.

### 5.2 Prop. 2 is close to tautological, and its consequences depend on a contestable definition

- Competence is defined as E_{r_i}[log p_i], a cross-entropy against the agent's *own* competence distribution. With this definition, N + C = I − KL is just the chain rule rewritten.
- The claim that heterogeneous agents gain novelty "for free" follows from the definition. A narrow specialist (low H(p_i)) is automatically "more competent", whatever it produces.
- Fig. 1 shows this directly. Along the "perspective route", competence rises by about 3 bits while population novelty rises by only about 0.5 bits, because each tilted agent's distribution becomes *more peaked*. The figure therefore mostly shows that individual entropy falls, not that between-agent diversity is free.
- Competence values for different p_i are not on a common scale, so comparing C across the temperature, perspective and ensemble routes compares quantities with different units.
- The "max residual 2×10⁻¹⁵" check is vacuous for the perspective and ensemble routes. The code computes I as N − mean H and then checks N + C − I, which is 0 by construction.
- The "one-for-one" trade in Prop. 3 and Fig. 1 holds only at T = 1. In general dN/dC = −β.

### 5.3 Pseudo-replication in the human-vs-LLM tests

- The LLM arm has 8 models. The 403–440 lists are treated as independent units in:
  - the Welch tests (Sec. 5.1);
  - the SEMs in Sec. 5.6;
  - the Mann–Whitney tests in D2.
- Many of these lists are near-duplicates: T = 0 replicates have plug-in divergence of 0.04–0.21 bits. The effective sample for generalising to "LLMs" is about 8.
- The p-values (e.g., 5.9×10⁻¹¹, 4.9×10⁻¹⁶) are therefore conditional on these 8 particular models and badly overstate the evidence about LLMs in general.
- A model-level bootstrap, or a mixed model with model as a random effect, is needed. Conclusions should be phrased as being about these 8 models of the 2023–24 generation.

### 5.4 The norm-reference paradox is partly an artefact of the peer pool

- In `empirical.py`, the "LLM-ensemble peers (100)" norm is drawn from all other LLM lists, *including runs of the same model*. That is about 12 of the 100, and among them the nearly identical replicates at the same temperature.
- Part of the drop in uniqueness (0.56 → 0.18) is therefore self-similarity, not cross-model homogeneity.
- The "same-model norm" also includes same-T replicates, which at low T are almost copies.
- Two comparisons are needed to separate the effects:
  - an "other-models-only" norm;
  - a same-model norm that excludes same-(model, T) replicates.
- The human-vs-LLM uniqueness contrast also mixes translated Dutch responses with native English LLM output. The authors acknowledge this, but the abstract still headlines the string-based 0.56 → 0.18 figure.

### 5.5 The headline conclusion goes beyond the identified evidence

- The abstract says human novelty "is spread across many authors". By the paper's own Prop. 4, that is precisely what cannot be established from single-list human data. Human list divergence could equally be persona resampling inside each person.
- What the data do support is narrower: lists from different people diverge more than lists from 5 of these 8 models.
  - At the category level the gap is small: 89–98% of the human value.
  - The paperclip category gap is not significant at any k.
  - The difference is largest at string level, which is the level most confounded by translation and phrasing.
- The answer to the stated research question ("are model and human innovation homologous?") is therefore "not identifiable with current data", plus a protocol. That is honest, but it is a modest empirical contribution.

### 5.6 Missing closely related prior work

- **Jiang et al. (2025), "Artificial Hivemind: The Open-Ended Homogeneity of Language Models"** (NeurIPS 2025 D&B oral, the same track this paper targets) explicitly separates intra-model repetition from inter-model homogeneity. That is essentially the replicate-versus-model decomposition reported here, at much larger scale. The paper must position itself against it, and against the follow-up critique of its null baselines, which is directly relevant to the permutation-null argument made here.
- The ecology alpha/beta diversity literature (see 5.1) is also missing.
- Persona-conditioning diversity work in LLMs is relevant to P3.

### 5.7 Estimator magnitudes

- Under heterogeneity, the list-permutation I* in Sim 3 recovers only 0.03–0.06 bits of a true 0.20 bits. That is a 70–85% downward bias, and power is 0.30 at ψ = 4.
- This bias should be reported in Table 1's discussion. It also matters for the pre-registered protocol: an N ≈ 200, 2-replicate human design may be underpowered to detect a human π_stable of plausible size.
- The protocol needs a power analysis based on Sim 3-type simulations.

## 6. Questions for the authors

1. How does the norm-reference result change when the LLM norm group excludes the target's own model, and when the same-model norm excludes same-(model, T) replicates?
2. Can you redo the Sec. 5.1, 5.5 and 5.6 inference with a model-level cluster bootstrap (resampling the 8 models)?
3. Is there a definition of competence on a common scale (for example, external usefulness ratings, or log-probability under a fixed reference model) under which Prop. 2's "free novelty" claim still holds non-trivially?
4. How does your decomposition differ from the intra-model/inter-model split in Artificial Hivemind, and from Jost-style alpha/beta partitions?
5. What is the power of the Section 7 protocol for π_stable values in the range implied by the human-vs-LLM ladder gap?
6. Since the human responses were translated, would re-categorising both populations with a single, model-independent codebook change the category-level ladder results?

## 7. Minor issues

- In Sec. 5.5, the g = 3 check says "five-model groups"; it should say "three-model groups".
- "Trades novelty one-for-one against competence" (Fig. 1 caption, Sec. 1) is exact only at T = 1.
- The Sim 2 identity "check" is tautological for the perspective and ensemble routes (see 5.2). Remove it or reword it.
- Table 2 reports the replicate floor as plug-in without a bias estimate. Its comparison with the model I* (permutation-corrected) mixes estimator types.
- The within-list entropy trend for paperclip is ρ = −0.62, p = 0.043. Saying temperature "leaves within-list entropy unchanged" is a slight overstatement for that task.
- The bilingual abstract is long. A NeurIPS D&B submission would need it compressed considerably.

## 8. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 9. Decision

**Major revision (weak reject as submitted).** The work is careful, honest and fully reproducible, and its methodological warning about response-level nulls is valuable. However:
- The theory is largely a relabelling of known identities.
- The key quality-side claim rests on a definition of competence that makes it nearly circular.
- The human-vs-LLM inferences are pseudo-replicated over 8 models.
- The peer-norm analysis mixes same-model runs into the "ensemble" norm.
- The core human question is, by the authors' own analysis, not identified.

Positioning against Artificial Hivemind and the alpha/beta diversity literature is essential.
