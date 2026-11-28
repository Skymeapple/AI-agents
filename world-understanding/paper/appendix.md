## Appendix A. Reproducibility

All code is in `world-understanding/wu/`. Each experiment is a module that writes a JSON result file to `results/`: `python -m wu.e1_world_model [seed]`, `python -m wu.e2_manifold`, …, `python -m wu.e7_shortcuts`. Every module also accepts `--quick` for a smoke test. `python -m wu.plots` regenerates all figures, `python -m wu.summary` prints every number quoted in Sections 4–5, and `python paper/build.py` rebuilds this document. Tests (`pytest tests`) check the exact Bayes filter, the n-gram baseline, TwoNN, the power-law fit, and the ridge and dMMSE reference estimators.

**Compute.** All experiments ran on a 4-core x86 CPU with JAX 0.12. Approximate wall-clock: E1 ≈ 17 min per seed; E2 ≈ 21 min; E3 ≈ 24 min; E4 ≈ 9 min; E5 ≈ {{E5_MIN}} min; E6 ≈ 12 min per condition; E7 ≈ 3 min.

**Key hyperparameters.** E1: transformer 4 layers, d = 64, 4 heads, MLP 256, AdamW (lr 10⁻³, wd 0.01, warm-up then cosine decay), batch 64, 5,000 steps, sequences of 20 action–observation pairs (length 42). Probes: multinomial logistic regression, 400 full-batch Adam steps, L2 10⁻⁴, standardised inputs, 20,000 training positions. E2: MLP 64-256-256-1, Adam 10⁻³, batch 256, 4,000 steps, early stopping on 1,000 validation points every 100 steps. E3: MLP 32-w-32, Adam 3×10⁻³ with cosine decay, full batch over 2,048 keys; 6,000 steps (capacity) or 3,000 (data). E4: Adam 10⁻³, batch 1,024, 8,000 steps. E5: AdamW (lr 10⁻³, β = (0.9, 0.98)), full batch, 20,000 steps. E6: transformer 4 layers, d = 64, read-in from ℝ⁶, Adam 10⁻³, batch 64, 5,000 steps. E7: MLP 64-128-2, AdamW (10⁻³, wd 10⁻⁴), full batch, 3,000 steps.

**Deviations and decisions made after seeing data.** We record these for transparency. (1) E1's probes were changed from ridge-to-one-hot to logistic regression after a pilot showed that the ridge readout under-reported decodability of a 120-way variable from 64 dimensions; all reported numbers use logistic probes. (2) E2's target frequencies were rescaled by 1/√d after a pilot, so that function wiggliness along any direction is matched across d. (3) E2's exponent estimator (power law plus floor) was adopted after seeing that the naive slope was dominated by an optimisation floor; both are reported. (4) E1 was shrunk from d = 128, 30-step trajectories to d = 64, 20 steps for compute reasons before any full run. No experiment or seed was dropped.

## Appendix B. Layer-wise E1 statistics

| Layer | State (120) | State, identified | Cell (30) | Heading (4) | Untrained state | Intrinsic dim. | Edit success | Random edit |
|---|---|---|---|---|---|---|---|---|
| Embedding | 1.5% | 1.6% | 4.3% | 25.1% | 1.6% | — | — | — |
| L1 | 21.1% | 25.2% | 25.5% | 30.3% | 5.1% | 3.5 | 17.9% | 2.6% |
| L2 | 47.1% | 59.9% | 43.1% | 33.2% | 4.9% | 4.6 | 55.0% | 3.7% |
| L3 | 52.8% | 67.8% | 45.4% | 34.1% | 4.8% | 5.1 | 89.0% | 3.0% |
| L4 | 49.1% | 63.0% | 40.9% | 34.0% | 5.0% | 5.9 | — | — |

Table: **Table B1.** E1 probe accuracy, intrinsic dimension of the residual stream (TwoNN, identified positions; ambient 64) and intervention success by layer; means of 3 seeds. Chance: state 0.8%, cell 3.3%, heading 25%. Bayes-MAP ceiling (all positions) 75.7%.

## References

::: {#refs}
:::
