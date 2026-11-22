# p004 code: How far can synthetic data stretch the human data stock?

All experiments are CPU-only (numpy/scipy/matplotlib; E5 also uses PyTorch on CPU). From this directory:

    python3 e1_knowledge_amplification.py   # ~5 s    Prop. 2 under its own model (closed form vs MC), fidelity -> results/e1_knowledge.json, fig1
    python3 e2_verified_selftraining.py     # 18-45 min on 4 cores, machine-dependent (144 runs)      -> results/e2_selftrain.json
    python3 e2_analyze.py                   # tables, N_eff, fig2                                      -> results/e2_table.{txt,json}, fig2
    python3 e2_spearman.py                  # info-vs-gain rank correlations (original analysis)       -> results/e2_spearman.txt
    python3 e2_predictors.py                # re-analysis: info vs -b/a, a-b, precision                -> results/e2_predictors.txt
    python3 e3_urn_scaling.py               # ~2 s    urn error exponent 1-b/a (blocked simulation)    -> results/e3_urn.json, fig3
    python3 e5_tinylm_retention.py          # ~17 min on 4 shared cores (78 tiny-transformer runs)     -> results/e5_tinylm.json
    python3 e5_analyze.py                   # storage-law fits, multipliers, coverage, fig5            -> results/e5_analysis.json, fig5
    python3 e4_horizon.py                   # horizon arithmetic with labelled anchors (needs e5)      -> results/e4_horizon.json, fig4
    python3 e6_verifier_design.py           # ~28 min on 4 shared cores (170 runs)                     -> results/e6_verifier_design.json
    python3 e6_analyze.py                   # decorrelated-grid predictors, verifier cost, fig6        -> results/e6_analysis.json, e6_summary.txt
    python3 e7_threshold_storage.py         # ~2 s    threshold storage: multiplier depends on alpha, N -> results/e7_threshold.json

Seeds are fixed in each script. Wall-clock times were measured on a 4-core machine shared with other jobs and
will differ elsewhere. Sourced constants used in e4 are documented in its docstring and in the paper.
