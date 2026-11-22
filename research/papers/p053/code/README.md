# p053 — Grounded vs. ungrounded emotion expression (CPU reproduction)

Requirements: Python 3, numpy, matplotlib, torch (CPU). Total runtime ~25 min on 4 cores.

    cd code
    python3 theory_check.py      # Monte-Carlo check of Props 1-3 -> results/theory_check.json (~10 s)
    python3 run_experiments.py   # 6 worlds x 5 seeds GRU(64) training + analyses -> results/raw_results.json (~12 min)
    python3 robustness.py        # GRU(32) and LSTM(64), 4 worlds x 3 seeds -> results/robustness_raw.json (~10 min)
    python3 aggregate.py         # tables (results/table_main.md, summary.json, gi_validation.json,
                                 # slopes_da_z.json, table_robustness.md) and figures fig1-fig3

Files: `emo.py` (task generator, labels, model, training), `run_experiments.py` (probes, steering,
grounding index), `robustness.py`, `theory_check.py`, `aggregate.py`.
All seeds are fixed; numbers in paper.md are copied from the files in ../results.
