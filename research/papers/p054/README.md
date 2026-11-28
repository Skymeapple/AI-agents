# p054 — Identity criteria, aggregation, and self-regarding incentives in AI systems

Reproduce (CPU only, Python 3 + numpy/scipy/matplotlib/torch):

    cd code
    python3 lineage_experiment.py   # Exp. 1, ~11 min on 2 CPU threads -> results/lineage_metrics.json, results/lineage_log.txt
    python3 incentives.py           # Exp. 2, <1 min -> results/incentives.json, fig1_continuity_matrix.png, fig2_policy_sweep.png

All numbers in paper.md Sections 5-6 are read from results/lineage_metrics.json and results/incentives.json.
Seeds are fixed (model seeds 0,1,2; Monte-Carlo seed 12345). Torch is limited to 2 threads
(4 threads caused severe contention on the test machine).
