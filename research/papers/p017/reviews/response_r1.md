# Response to Review Round 1

We thank the reviewer for a careful review, including an independent partial rerun. The review identified one misattributed number (F1), several major issues (M1–M6) and a number of minor ones. We addressed them with new experiments rather than rewording. In doing so we found a second error of our own, which we report below (M1). Two of the new runs were interrupted before they finished, and we say exactly what exists (see "Status of incomplete runs" directly below).

We also took M6 (novelty and significance) as the binding constraint and re-scoped the paper around a stronger contribution, described first. All new numbers come from code in `code/` that we ran on a shared 4-core CPU. Raw outputs are in `results/`; aggregates are in `results/summary_r2.md` and `results/summary_othello.md`.

## Status of incomplete runs

- **Reservoir sweep (M1).** Seeds 0 and 1 completed all nine configurations in both non-local worlds. Seed 2 was terminated after the first configuration (norm-scaled, λ = 0.99) in both worlds. The sweep therefore covers 38 of the 54 planned reservoirs, and eight configurations rest on two seeds. The archived aggregate in `results/summary_r2.md` (Section 3) lists only the one configuration shared by all three seeds; the paper's Table 1 is built from the per-seed files in `results/sweep/`.
- **Othello continuation.** A continuation of training from the 6,000-step checkpoint was started (4,000 further steps planned). Its log ends at step 2,200 and the resulting checkpoint was never analysed. The paper reports nothing from it, and all Othello results are for the single 6,000-step checkpoint.
- **Unarchived numbers.** No spectral radii of our reservoir matrices are saved in `results/`, so the paper does not quote any.
- **No reruns at finalisation.** The environment in which the manuscript was finalised could not execute the code, so none of the above could be completed there.

## The new contribution, in brief

Round 1's message was "probe accuracy should be compared with the suffix ceiling". The reviewer rightly judged its empirical findings largely expected. The revision makes three changes.

1. **Tightness theorem (new Proposition 4).** The suffix ceiling is exactly attainable by an *untrained* contracting encoder with random tanh features and a linear probe, once it has one random feature unit per reachable k-suffix, plus a (k − 1)-unit delay line. The encoder is a structured reservoir (a delay line feeding random tanh features), not a dense random recurrent matrix. The proof shows that tanh features at distinct pre-activation shifts span ℝ^M, using the distinct complex poles of sech². So for fading-memory encoders, decodability up to Acc\*\_k is free in principle at a width that can be exponential in k. The proposition gives a sufficient width only. It does not prove that narrower random encoders must fail, so "training buys memory at sub-exponential width" is our interpretation, and the paper says so. The construction was not instantiated numerically.

2. **A free-information floor for attention, tested on Othello (new Section 6).** The suffix ceiling says nothing about full-context transformers. For them the cheap information is order-free. Two attention heads whose patterns depend only on position parity expose the *placement bag*: which squares each player has played. (This is a motivation for the floor, not a theorem about any particular network.) We therefore did the following:
   - wrote an Othello engine and generated games from Li et al.'s synthetic distribution;
   - trained a small Othello GPT on CPU (4 layers, d = 128; 92.4% legal top-1 at 6,000 steps);
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
   - We cannot say how this changes with more training: the continuation run was interrupted and not analysed.

   We think this is the result an expert would not have predicted: the conventional untrained-twin comparison and the architecture-aware floor give *opposite* verdicts on the same model. Garg (2026), which appeared during our revision, independently proposes floors of simple inputs and builds a no-flip floor for the released Othello-GPT checkpoint, where it reports that a linear mine/theirs probe clears the floor comfortably. That is the opposite outcome to ours, on a much better-trained model, and the paper discusses the two together and makes no claim about the original Othello-GPT. We cite it and explain the differences: our floors are derived from the encoder class with theorems, computed exactly where possible, and stratified by age.

3. **Stress test of the gridworld criterion (M1), with run-level statistics (M5).** Both items are described point by point below.

The paper was retitled "What a Probe Gets for Free: Suffix Ceilings, Placement Floors, and When Decodable World States Are Evidence of a World Model" and rewritten as a whole, not patched.

## Point-by-point

### F1 (misattributed recall for landmark seed 0). **Fixed.**

The reviewer is right. Seed 0's landmark recall is 0.607 (`raw_landmark_s0.json`). The value 0.85 was the three-seed mean.

The dissociation claim now rests on 10 seeds rather than one example (Section 5.3, Table 2):
- After 10 epochs, landmark GRUs have next-observation accuracy 0.959 ± 0.018 but probe accuracy 0.537 ± 0.196.
- The clearest single case is seed 5: next-observation accuracy 0.962 and recall 0.815, with probe accuracy 0.394 (k\_eff = 8).
- Seed 0 is now quoted correctly: recall 0.607, probe accuracy 0.250.

### M1 (the 2–3-step window of untrained encoders is a hyperparameter artefact). **Addressed with a sweep, and an error of ours corrected.**

**The error.** While building the sweep we found that round 1's explanation was wrong. The λ = 0.99 ESN's short memory was not caused by tanh saturation. A Gaussian matrix scaled to spectral *norm* 0.99 has spectral *radius* of only about half its norm, so it contracts by roughly half per token. An earlier draft of this response quoted specific radii (0.506 and 0.514) for our d = 512 matrices. Those values are not in `results/` and could not be re-derived when the manuscript was finalised, so they are withdrawn: the paper states the general property and the measured restart distances (δ(8) ≈ 10⁻⁶) and gives no exact radius. Round 1's "nominal memory of 100 tokens" was a misreading. The paper now says so explicitly (Section 5.1).

**The sweep.** We then swept nine untrained d = 512 reservoirs × 2 worlds (`reservoir_sweep.py`, Table 1). Three seeds were planned. Seeds 0 and 1 are complete; seed 2 finished only the first configuration (38 of 54 reservoirs in total):
- Gaussian scaled to norm 0.99 (the provable contraction);
- Gaussian scaled to *radius* 0.99 (standard ESN practice);
- λ·orthogonal with λ ∈ {0.99, 1.0} (near-isometries; the restart distance δ(8) is 1.8–2.1 at input gain 0.1, about 0.5 at gain 0.3 and about 0.007 at gain 1, against about 10⁻⁶ for the norm-scaled reservoir. So saturation at unit gain does shorten memory in the orthogonal reservoirs, even though it was not the cause in the round-1 reservoir);
- each at input gains {0.1, 0.3, 1.0}.

Every completed configuration was tested with linear and MLP probes, the truncation test and CRA.

**Result.**
- With linear probes, no completed untrained reservoir exceeds k\_eff = 4, even with long memory.
- MLP probes raise long-memory reservoirs to k\_eff of 5–7 (maximum 7).
- Trained GRUs reach k\_eff = 15–24 after 30 epochs.

So the criterion separates 30-epoch trained GRUs from untrained encoders under the stress the reviewer asked for. Two nuances are stated in the paper. First, nonlinear probes on long-memory reservoirs extract noticeably more; Proposition 4 makes this unsurprising (512 units are far fewer than the reachable suffixes at k ≥ 5) but does not prove that linear probes must fail. Second, the separation from *10-epoch* GRUs is narrow: their linear-probe k\_eff is 5–22, above every linear reservoir result (≤ 4) but overlapping the best MLP reservoir results (5–7). These maxima are over two seeds for eight of the nine configurations. The abstract now states the claim with its scope: "across 9 configurations of untrained d = 512 reservoirs…".

### M2 (the post-transition CRA range in the abstract is over-generalised). **Fixed.**

The range is now taken over all 40 independent GRUs with k\_eff ≥ 12: CRA(probe) is 0.534–0.927 and CRA(random) is 0.207–0.452. It includes the main-run seeds that the reviewer noted (0.533 and 0.659). Prediction P3 quotes these ranges.

### M3 (Proposition 3 is invoked where its hypotheses fail). **Fixed.**

Proposition 3 is now stated with *multiclass linear separability* of the restarted features as the hypothesis. Affine independence (d ≥ M − 1) is given as a sufficient condition.

Section 5.1 states explicitly that the d = 64 encoders do not meet the sufficient condition (M = 100) and then checks separability directly. The restarted-state probe reaches 1.000 for the d = 64 ESN at k = 1 in all three seeds, and 0.979–0.990 for the random GRU at k = 2.

Proposition 4 gives the matching construction in which the sufficient condition provably holds.

### M4 (the measured ceiling uses a stationary prior, not the true distribution). **Fixed.**

`world.suffix_bayes_exact` computes the posterior from the exact finite-time marginal under the known start. This is Definition 1's quantity, so Acc\*\_t = 1. All round-1 runs were re-scored without retraining (`recompute_exact.py`, `results/exact_ceiling.json`):
- *Truncation test.* Over the 324 cells the maximum excess falls from +0.010 to **+0.0022**, with no cell above 0.005. The stationary prior accounted for most of the earlier excess.
- *The "∞" category.* It disappears, and every k\_eff is now finite. Trained GRUs score 20/5/8 (two-colour) and 5/22/15 (landmarks). The λ = 0.99 ESN goes from 3 to 2.

### M5 (thin statistics). **Fixed.**

1. **Trained GRUs.** We trained 40 independent GRUs: 10 seeds × 2 worlds × {10, 30} epochs, all with the *same* one-cycle schedule, which also removes the schedule confound flagged in the minor points (`gru_seeds.py`, Table 2). Seeds 0–2 at 10 epochs reproduce the round-1 networks. Their next-observation accuracies are identical. Probe accuracy and CRA agree to within 0.0005 and 0.0013 respectively; the residual difference comes from thread-dependent L-BFGS probe fits.
2. **Correlation.** The probe-gain correlation is now computed over independent runs only: Pearson r = 0.880 (95% bootstrap CI over runs [0.805, 0.928]) and Spearman ρ = 0.902. This replaces round 1's r = 0.92 over 34 dependent checkpoints.
3. **Transformers.** These now have 3 seeds per world, with the attention-window truncation test, so Corollary 2 is exercised (`tf_window.py`; Section 5.4). The corollary's bound holds in all 96 cells (the closest approach of windowed accuracy to Acc\*\_k is −0.041) but is loose. Windowing a trained transformer destroys its state computation rather than truncating it. This is one reason the paper moves to the placement floor for transformers.

We still did not run CRA on transformers, and we say why in the Limitations section.

### M6 (moderate novelty; missing related work). **Addressed by re-scoping (see above).**

**Related work.** We added:
- Yuan & Søgaard (2025): Othello world-model features across seven LM architectures;
- Brodjian, Hobley & Perona (2026): simulators and editability;
- Garg (2026): floors, ceilings and headroom in probing.

We verified all three and positioned them in Section 2 and the Discussion. We did not add a citation for "mechanistic evidence that Othello-GPT relies on heuristics", because we could not identify a peer-reviewed or archival source we could verify.

**Novelty.** On the substance, we agree that Theorem 1 alone is a short perturbation argument. The new elements are:
- the tightness result (Proposition 4), which shows the ceiling is attainable without training at sufficient width;
- the architecture-dependent placement floor, with a demonstration that it reverses the untrained-twin verdict on a trained Othello model;
- age stratification, which locates where above-floor evidence must appear;
- the calibration of learned floors against exact ceilings, which shows when they can and cannot be used.

The large-model predictions remain untested. We say so, and the predictions (Section 8) are now stated in terms of the placement floor and age strata. Only P3 is a restatement of something we measured; the others are untested.

### Minor points

- **A linear probe compared against a nonlinear Bayes ceiling.** Agreed. The truncation test is now presented as the primary memory measure (Section 3.4). The reservoir sweep reports both linear and MLP probes against the same ceiling.
- **"Truncated accuracy keeps rising" is not evidence by itself.** Agreed and removed. The evidence we use is k\_eff against the exact ceiling.
- **Schedule confound between main and longer runs.** Removed: the 10- and 30-epoch runs share the one-cycle schedule.
- **Trained readout for untrained encoders.** Now stated explicitly in Section 4: "untrained encoder" always means an untrained encoder with a trained linear readout.
- **Mislabelled log line.** Documented in the README. The new scripts print correct labels.
- **Receptive field in Corollary 2.** Corrected to n(w′ − 1) + 1. Absolute positional information is handled by defining Acc\*\_k at fixed t (Section 3.1).
- **Long, number-dense abstract.** Restructured around three findings.

### Answers to the reviewer's questions

1. **Input scale, spectral radius and orthogonal reservoirs.** See M1 and Table 1 (two complete seeds). We used input gains {0.1, 0.3, 1}, λ ∈ {0.99, 1.0}, and Gaussian-norm, Gaussian-radius and orthogonal recurrent matrices.
2. **Exact finite-time marginal.** Done; see M4. The "∞" category is gone, and k\_eff is reported against the exact ceiling throughout.
3. **CRA or masked-attention truncation on transformers.** We ran the masked-attention truncation test on 3 seeds × 2 worlds. CRA on transformers is still not done (see Limitations).
4. **Trajectory-clustered uncertainty and more seeds.** We used 40 independent runs and a bootstrap over runs; see M5.

## What did not work, and what remains open

- **The suffix ceiling is loose for transformers.** Windowing breaks a trained transformer instead of truncating it. We therefore do not use the suffix ceiling as the transformer test.
- **Learned window decoders are one-sided.** They are tight for k ≤ 4 (within 0.029) but can be loose by up to 0.53 at k ≥ 6 (Table 3). We therefore use Othello floors only to *refute* above-floor claims.
- **The Othello model is small and partially trained.** We make no claim about the original Othello-GPT.
- **The Othello continuation was not completed.** It was meant to give a second checkpoint. Because of CPU limits its schedule was shortened once (documented in the code), its log ends at step 2,200 of 4,000, and it was never analysed. The paper has one Othello checkpoint and says so in the abstract and the Limitations section.
- **The reservoir sweep is missing most of seed 2** (see the status section above).
