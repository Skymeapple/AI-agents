# Response to Round-1 Review

We thank the reviewer for a careful report. The reviewer reran our code bit-exactly and computed independent baselines, and both turned out to be decisive. We accept most criticisms. Where the evidence required it, we retracted or narrowed claims rather than defending them. Every new number below comes from code we wrote and ran in this environment. Outputs are in `results/` and listed per item. The previous version is preserved as `reviews/paper_r0.md`, and the round-0 result JSONs are in `results/backup_r0/`.

## Summary of changes

| Issue | Verdict | What changed |
|---|---|---|
| M1: Gaussian twin 50% is a theorem | Accepted | Demoted to a "unit test" in Section 5.1 and the last column of Table 1. Removed from the abstract and from the claimed findings. The CI around it is no longer reported as evidence. |
| M2: mis-calibrated "observational limit" | Accepted; claim retracted and replaced | New `code/bayes_oracle.py` computes exact Bayes baselines under our prior: MEC-only 86.3% (reviewer: 86.4%), correlation-matrix 90.2–91.4%, covariance 98.8%. The overall shortcut claim is retracted for p = 0.9 and holds only by 1–2 points for p = 1. It holds clearly on MEC-unidentified queries (about +9 points at p = 1). |
| M3: out-of-prior twin confound | Accepted; ablation added | New direction-balanced, in-prior and any-MEC (including non-adjacent) twin probes, plus prior-agnostic classical reasoners. Surplus survives (Section 5.3). |
| M4: novelty / missing prior work | Accepted | Props 1–3 are now explicitly labelled restatements of known results. Montagna et al. (2024), Reisach et al. (2023) and Hyvärinen & Smith (2013) are cited, and the relation to Corr2Cause's MEC labels is stated. Contributions are repositioned around calibrated baselines and controls. |
| M5: RQ not addressed / "pre-registered" | Partially addressed; claims narrowed | Not fixable here: model-hub downloads (LLM weights, AVICI checkpoints) are blocked by the sandbox's network policy. We verified this; see below. The scope is now stated plainly in the abstract, introduction, Section 8 and limitations. "Pre-registered" is changed to "pre-specified (not registered)". |
| Minor 1: adjacent-only twins | Fixed | General MEC refit (Prop. 3 generalized and verified to 1.4e-13). The `rev_mec_*` probes include non-adjacent queries, reported separately. |
| Minor 2: few seeds, no κ CIs | Fixed (partly) | 7 new training runs, so most regimes now have 2–3 seeds. Paired-bootstrap κ CIs per run, plus the across-seed s.d. |
| Minor 3: no ablations; obs_p0 weak | Partly addressed | Weakness stated explicitly against the new Bayes bound (74.6% vs ≤ 91%). We ran no architecture ablations (CPU budget); listed as a limitation. |
| Minor 4: standardization and varsortability | Fixed and quantified | Text corrected. R²-sortability is cited, and the Bayes-R vs Bayes-MEC gap quantifies the residual scale-invariant signal. |
| Minor 5: "oracle"/"ceiling" wording | Fixed | Renamed to "MEC + label-order reference"; "81% ceiling" removed. |
| Minor 6: smoking/cancer example | Withdrawn | Section 7 explains why. |
| Minor 7: summarize.py glob, int8 | Fixed | The glob is now `*_s[0-9].json`; `int8_p0` is removed. |

## Point-by-point

§POINTS§
