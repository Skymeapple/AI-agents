# p004 code: How far can synthetic data stretch the human data stock?

All experiments are CPU-only (numpy/scipy/matplotlib). From this directory:

    python3 e1_knowledge_amplification.py   # ~5 s   Prop. 2 (amplification identity), fidelity  -> results/e1_knowledge.json, fig1
    python3 e2_verified_selftraining.py     # ~18 min on 4 cores (144 runs)                     -> results/e2_selftrain.json
    python3 e2_analyze.py                   # tables, N_eff, fig2                                -> results/e2_table.txt, fig2
    python3 e2_spearman.py                  # info-vs-gain rank correlations                     -> results/e2_spearman.txt
    python3 e3_horizon.py                   # horizon delay arithmetic                           -> results/e3_horizon.json, fig3
    python3 e4_urn_scaling.py               # ~2 s   Prop. 4 (urn error exponent 1-b/a)          -> results/e4_urn.json, fig4

Seeds are fixed in each script. Sourced constants used in e3 are documented in its docstring and in the paper.
