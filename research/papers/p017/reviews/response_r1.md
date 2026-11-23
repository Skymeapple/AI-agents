# Response to Review Round 1

We thank the reviewer for a careful review, including an independent partial rerun. The review identified one misattributed number (F1), several major issues (M1–M6) and a number of minor ones. We fixed all of them with new experiments rather than rewording. In doing so we found a second error of our own, which we report below (M1).

We also took M6 (novelty and significance) as the binding constraint and re-scoped the paper around a stronger contribution, described first. All new numbers come from code in `code/` that we ran on a shared 4-core CPU. Raw outputs are in `results/`; aggregates are in `results/summary_r2.md` and `results/summary_othello.md`.

## The new contribution, in brief

Round 1's message was "probe accuracy should be compared with the suffix ceiling". The reviewer rightly judged its empirical findings largely expected. The revision makes three changes.

1. **Tightness theorem (new Proposition 4).** The suffix ceiling is exactly attainable by an *untrained* contracting encoder with random tanh features and a linear probe, once its width reaches the number of reachable k-suffixes. The proof shows that tanh features at distinct pre-activation shifts span ℝ^M, using the distinct complex poles of sech². So for fading-memory encoders, decodability up to Acc\*\_k is free in principle and costs width exponential in k. What training buys is memory at sub-exponential width, not decodability. This turns Theorem 1 from a loose bound into a characterisation.

2. **A free-information floor for attention, tested on Othello (new Section 6).** The suffix ceiling says nothing about full-context transformers. For them the cheap information is order-free. One attention head with uniform weights and position-parity-dependent values exposes the *placement bag*: which squares each player has played. We therefore did the following:
   - wrote an Othello engine and generated games from Li et al.'s synthetic distribution;
   - trained a small Othello GPT on CPU (4 layers, d = 128; 92.4% legal top-1 at 6,000 steps), plus ⟦LONG-RESP⟧;
   - probed it and its exact untrained initialisation;
   - compared both with learned floors:
     - t-only;
     - the last k moves;
     - placement;
     - placement + last 8 moves.

   All of these are stratified by *square age*. We calibrated the floor estimators against exact ceilings on the torus.

   **Finding.**
   - The trained model's linear probe (0.812) beats its untrained twin (0.698) by 11 points, the conventional evidence of a learned board.
   - It stays below the placement floor (0.879), as does a nonlinear probe (0.835).
   - The untrained twin understates the free information by 18 points.
   - Aggregate probe accuracy equals that of a 16-move window decoder (0.818), but the age profiles are opposite. The window decoder fails on old squares (0.475 at age ≥ 16); the probe is flat across age (about 0.63). The probe does use information older than 16 moves, but on those squares it stays below what placement gives (0.818).
   - ⟦LONG-RESP2⟧

   We think this is the result an expert would not have predicted: the conventional untrained-twin comparison and the architecture-aware floor give *opposite* verdicts on the same model. Garg (2026), which appeared during our revision, independently proposes floors of simple inputs and discusses Othello placement for the original Othello-GPT. We cite it and explain the differences: our floors are derived from the encoder class with theorems, computed exactly where possible, and stratified by age.

3. **Stress test of the gridworld criterion (M1), with run-level statistics (M5).** Both items are described point by point below.

The paper was retitled "What a Probe Gets for Free: Suffix Ceilings, Placement Floors, and When Decodable World States Are Evidence of a World Model" and rewritten as a whole, not patched.

## Point-by-point

### F1 (misattributed recall for landmark seed 0). **Fixed.**

The reviewer is right. Seed 0's landmark recall is 0.607 (`raw_landmark_s0.json`). The value 0.85 was the three-seed mean.

The dissociation claim now rests on 10 seeds rather than one example (Section 5.3, Table 3):
- After 10 epochs, landmark GRUs have next-observation accuracy 0.959 ± 0.018 but probe accuracy 0.537 ± 0.196.
- The clearest single case is seed 5: next-observation accuracy 0.962 and recall 0.815, with probe accuracy 0.394 (k\_eff = 8).
- Seed 0 is now quoted correctly: recall 0.607, probe accuracy 0.250.

### M1 (the 2–3-step window of untrained encoders is a hyperparameter artefact). **Addressed with a sweep, and an error of ours corrected.**

**The error.** While building the sweep we found that round 1's explanation was wrong. The λ = 0.99 ESN's short memory was not caused by tanh saturation. A Gaussian matrix scaled to spectral *norm* 0.99 has spectral *radius* of only about 0.51 (we computed 0.506 and 0.514 for our d = 512 seeds), so it contracts by roughly half per token. Round 1's "nominal memory of 100 tokens" was a misreading. The paper now says so explicitly (Section 5.1).

**The sweep.** We then swept nine untrained d = 512 reservoirs × 2 worlds × 3 seeds (`reservoir_sweep.py`, Table 2):
- Gaussian scaled to norm 0.99 (the provable contraction);
- Gaussian scaled to *radius* 0.99 (standard ESN practice);
- λ·orthogonal with λ ∈ {0.99, 1.0} (near-isometries with long memory; the restart distance δ(8) is about 1.8–2.1, against about 10⁻⁶ for the norm-scaled reservoir);
- each at input gains {0.1, 0.3, 1.0}.

Every configuration was tested with linear and MLP probes, the truncation test and CRA.

**Result.**
- With linear probes, no untrained reservoir exceeds k\_eff = ⟦SWEEP-LIN-MAX⟧, even with long memory.
- MLP probes raise long-memory reservoirs to k\_eff ≤ ⟦SWEEP-MLP-MAX⟧.
- Trained GRUs reach k\_eff = 15–24 after 30 epochs.

So the criterion still separates trained from untrained encoders under the stress the reviewer asked for. The nuance, now stated in the paper, is that nonlinear probes on long-memory reservoirs extract noticeably more. Proposition 4 explains why linear probes cannot: they need width exponential in k. The abstract now states the claim with its scope: "across 9 configurations of untrained d = 512 reservoirs…".

### M2 (the post-transition CRA range in the abstract is over-generalised). **Fixed.**

The range is now taken over all 40 independent GRUs with k\_eff ≥ 12: CRA(probe) is 0.534–0.927 and CRA(random) is 0.207–0.452. It includes the main-run seeds that the reviewer noted (0.533 and 0.659). Prediction P3 quotes these ranges.

### M3 (Proposition 3 is invoked where its hypotheses fail). **Fixed.**

Proposition 3 is now stated with *multiclass linear separability* of the restarted features as the hypothesis. Affine independence (d ≥ M − 1) is given as a sufficient condition.

Section 5.1 states explicitly that the d = 64 encoders do not meet the sufficient condition (M = 100) and then checks separability directly. The restarted-state probe reaches 1.000 for the d = 64 ESN at k = 1, and 0.983 for the random GRU at k = 2.

Proposition 4 gives the matching construction in which the sufficient condition provably holds.

### M4 (the measured ceiling uses a stationary prior, not the true distribution). **Fixed.**

`world.suffix_bayes_exact` computes the posterior from the exact finite-time marginal under the known start. This is Definition 1's quantity, so Acc\*\_t = 1. All round-1 runs were re-scored without retraining (`recompute_exact.py`, `results/exact_ceiling.json`):
- *Truncation test.* Over the 324 cells the maximum excess falls from +0.010 to **+0.0022**, with no cell above 0.005. The stationary prior accounted for most of the earlier excess.
- *The "∞" category.* It disappears, and every k\_eff is now finite. Trained GRUs score 20/5/8 (two-colour) and 5/22/15 (landmarks). The λ = 0.99 ESN goes from 3 to 2.

### M5 (thin statistics). **Fixed.**

1. **Trained GRUs.** We trained 40 independent GRUs: 10 seeds × 2 worlds × {10, 30} epochs, all with the *same* one-cycle schedule, which also removes the schedule confound flagged in the minor points (`gru_seeds.py`, Table 3). Seeds 0–2 at 10 epochs reproduce the round-1 networks. Their next-observation accuracies are identical. Probe accuracy and CRA agree to within 0.0005 and 0.0013 respectively; the residual difference comes from thread-dependent L-BFGS probe fits.
2. **Correlation.** The probe-gain correlation is now computed over independent runs only: Pearson r = 0.880 (95% bootstrap CI over runs [0.805, 0.928]) and Spearman ρ = 0.902. This replaces round 1's r = 0.92 over 34 dependent checkpoints.
3. **Transformers.** These now have 3 seeds per world, with the attention-window truncation test, so Corollary 2 is exercised (`tf_window.py`; Section 5.4). The corollary's bound holds in every cell (maximum excess ⟦TF-MAXVIOL⟧) but is loose. Windowing a trained transformer destroys its state computation rather than truncating it. This is one reason the paper moves to the placement floor for transformers.

We still did not run CRA on transformers, and we say why in the Limitations section.

### M6 (moderate novelty; missing related work). **Addressed by re-scoping (see above).**

**Related work.** We added:
- Yuan & Søgaard (2025): Othello world-model features across seven LM architectures;
- Brodjian, Hobley & Perona (2026): simulators and editability;
- Garg (2026): floors, ceilings and headroom in probing.

We verified all three and positioned them in Section 2 and the Discussion. We did not add a citation for "mechanistic evidence that Othello-GPT relies on heuristics", because we could not identify a peer-reviewed or archival source we could verify.

**Novelty.** On the substance, we agree that Theorem 1 alone is a short perturbation argument. The new elements are:
- the tightness result (Proposition 4), which characterises what training buys;
- the architecture-dependent placement floor, with a demonstration that it reverses the untrained-twin verdict on a trained Othello model;
- age stratification, which locates where above-floor evidence must appear;
- the calibration of learned floors against exact ceilings, which shows when they can and cannot be used.

The large-model predictions remain untested. We say so, and the predictions (Section 8) are now stated in terms of the placement floor and age strata.

### Minor points

- **A linear probe compared against a nonlinear Bayes ceiling.** Agreed. The truncation test is now presented as the primary memory measure (Section 3.4). The reservoir sweep reports both linear and MLP probes against the same ceiling.
- **"Truncated accuracy keeps rising" is not evidence by itself.** Agreed and removed. The evidence we use is k\_eff against the exact ceiling.
- **Schedule confound between main and longer runs.** Removed: the 10- and 30-epoch runs share the one-cycle schedule.
- **Trained readout for untrained encoders.** Now stated explicitly in Section 4: "untrained encoder" always means an untrained encoder with a trained linear readout.
- **Mislabelled log line.** Documented in the README. The new scripts print correct labels.
- **Receptive field in Corollary 2.** Corrected to n(w′ − 1) + 1. Absolute positional information is handled by defining Acc\*\_k at fixed t (Section 3.1).
- **Long, number-dense abstract.** Restructured around three findings.

### Answers to the reviewer's questions

1. **Input scale, spectral radius and orthogonal reservoirs.** See M1 and Table 2. We used input gains {0.1, 0.3, 1}, λ ∈ {0.99, 1.0}, and Gaussian-norm, Gaussian-radius and orthogonal recurrent matrices.
2. **Exact finite-time marginal.** Done; see M4. The "∞" category is gone, and k\_eff is reported against the exact ceiling throughout.
3. **CRA or masked-attention truncation on transformers.** We ran the masked-attention truncation test on 3 seeds × 2 worlds. CRA on transformers is still not done (see Limitations).
4. **Trajectory-clustered uncertainty and more seeds.** We used 40 independent runs and a bootstrap over runs; see M5.

## What did not work, and what remains open

- **The suffix ceiling is loose for transformers.** Windowing breaks a trained transformer instead of truncating it. We therefore do not use the suffix ceiling as the transformer test.
- **Learned window decoders are one-sided.** They are tight for k ≤ 4 but can be loose by up to ⟦CAL-MAX⟧ at k ≥ 6 (Table 4). We therefore use Othello floors only to *refute* above-floor claims.
- **The Othello model is small and partially trained.** We make no claim about the original Othello-GPT.
- **A single continuation phase.** Our Othello continuation phase gives only two checkpoints. Because of CPU limits, its schedule was shortened once, at step 800 of the continuation (documented in the code and the paper).
