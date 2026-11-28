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

## Round 2 (revision in response to review r1)

- `sweep_r2_spec.json` : analysis plan written before the round-2 sweep was launched (with one compute amendment,
                         recorded in the file); `results/sweep_r2_spec.sha256` holds the hash/time of the original.
- `run_r2.py`          : trains one free model, logs tests every 250 steps, saves weights at given steps (`results/runs_r2/`).
- `jobs_r2a.txt`       : regime (eta=0, delta=0.3), seeds 100-103, 16000 steps (weights at 4k/8k/16k).
- `jobs_r2b.txt`       : regime (0, 0.3) seeds 104-109 and regime (0.2, 0) seeds 100-109, 8000 steps (weights at 4k/8k).
  (`jobs_r2.txt` is the original 40-run plan; only its first four lines were run, = jobs_r2a.txt.)
- `r2_analyze.py`      : per-model behavioural tests, path cuts, removal formats, presence-gating tests (embedding
                         interpolation, activation patching), interchange interventions for Definition 3 (D).
- `r2_trainsupport.py` : behavioural tests on training vs held-out triples.
- `r2_bound.py`        : exact training-distribution answer cross-entropy and the Prop. 3(a,b) bounds.
- `r2_stats.py`        : run-level statistics with Clopper-Pearson / cluster-bootstrap intervals, Eq. 2 vs trivial
                         baseline, symmetric test-vs-reference comparison, figures fig5/fig6. Output `results/r2_stats.json`.

Reproduce round 2 (~1 h wall-clock on this machine, about 2 effective cores):

    cd code
    cat jobs_r2a.txt jobs_r2b.txt | xargs -P 4 -L 1 sh -c 'python3 run_r2.py $0 "$@" >> ../results/sweep_r2_log.txt 2>&1'
    python3 r2_analyze.py old; python3 r2_analyze.py new
    python3 r2_trainsupport.py > ../results/r2_trainsupport_log.txt
    python3 r2_bound.py > ../results/r2_bound_log.txt
    python3 r2_stats.py > ../results/r2_stats_log.txt
