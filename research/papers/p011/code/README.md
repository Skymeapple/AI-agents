# Code for "Necessity Is Not Reliance" (p011)

CPU-only. Requires Python 3, numpy, matplotlib, torch (CPU is sufficient; tested with torch 2.15.0).

Files
- `cotlab.py`   : model (2-layer transformer, d=64), data (a,b,c -> s=(a+b) mod 23 -> y=(s+c) mod 23),
                  attention-mask ground-truth regimes, training loop, behavioural tests (truncation,
                  corruption with 3-way outcome), two-route additive-logit fit.
- `run_one.py`  : trains one configuration, logs tests every 250 steps, saves `results/runs/<tag>.json/.pt`.
- `jobs.txt`    : 24 base configurations (20 free-mechanism models + 4 attention-mask controls).
- `jobs_long.txt`, `jobs_long2.txt` : 6 longer free runs (4 x eta=0,delta=0.3 for 8000 steps; 2 x eta=0.05,delta=0 for 12000 steps).
- `jobs_strict.txt` : 2 'reliant_strict' controls (closes the a,b -> c-position leak of the 'reliant' mask).
- `tables.py`   : supplementary statistics (quadrants, Spearman correlations, regime splits of disagreement and of the cross-test identity).
- `analyze.py`  : path-cut (mechanistic) evaluations, linear probes, aggregate statistics and figures.

Reproduce (about 40-60 min wall-clock on 4 cores; each run uses 1 thread):

    cd code
    cat jobs.txt jobs_long.txt jobs_long2.txt jobs_strict.txt | xargs -P 4 -L 1 sh -c 'python3 run_one.py $0 "$@" >> ../results/sweep_log.txt 2>&1'
    COT_THREADS=4 python3 analyze.py > ../results/analyze_log.txt
    python3 tables.py > ../results/tables_log.txt

Outputs (in `results/`): `final_table.md|json`, `summary.json`, `fig1_final_vs_eta.png`, `fig2_dynamics.png`,
`fig3_additive_fit.png`, `fig4_verdict_plane.png`, `supp_stats.json`, `supp_table.md`,
`ckpt_disagree_by_regime.json`, `crosstest_by_regime.json`, per-run histories in `runs/`.
(`pilot.py` is an early exploratory script, not used for any reported number.)
All seeds are fixed (model init = seed, data split = seed, batch sampling = seed+1000, evaluation seeds fixed).
