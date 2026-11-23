# p009 — How much state does in-context recall need?

Code and outputs for the paper `paper.md`. CPU only.

```
cd code
python3 capacity_sim.py        # Exp A: ideal memories vs closed-form theory  (~3 min)  -> results/sim_capacity.json, fig_sim_capacity.png
./run_train_all.sh             # Exp B: trained single-layer mixers on capacity-isolating MQAR (see note)
python3 recall_demand.py       # Exp C: recall demand of real text/code         (~15 s)  -> results/demand.json, fig_demand.png
python3 analyze.py             # Exp D: capacities, efficiency eta, T*, S_req   -> results/capacity_table.json, tradeoff.json, figs
```

Data for Exp C: NLTK Gutenberg selection
(`https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip`, unzip into `data/`)
and the Go 1.25.1 standard library sources at `/usr/local/go1.25.1/src` (any Go source tree works; numbers will differ slightly).

Note on Exp B: the machine was shared with other jobs during the run, so the sweep was cut after the first
seven runs; the remainder actually executed is `run_train_rest.sh` (d_k = 64 dropped for gla/delta/gdelta,
value-dimension sweep reduced to d_v in {32, 128}). Each run: `python3 train_mqar.py --arch A --dk K --dv V --steps 1200`.
Seeds are fixed (seed 0 for training, 1234 for evaluation).
The sweep was completed in a later session by `run_train_final.sh` (Gated DeltaNet at d_k in {8,16,32};
linear/delta at d_k=16, d_v in {32,128}) and `run_train_wide.sh` (token width d=512 control for the
d_v=128 plateau: linear/delta d_v=128 and linear d_v=64; outputs `*_d512.json`). After all runs,
`python3 analyze.py` also writes results/eta_vs_sim.json (efficiency vs simulated ideal memory) and
results/fig_dvsweep.png; its console output is saved in results/log_analyze.txt.
Each training run takes ~2.5-6 min on one CPU core.

## Revision round 1 (new code and outputs)

```
cd code
python3 capacity_ci.py          # Exp A rev: Monte Carlo CIs, fine N grid, closed-form decomposition, V=41,509 check (~10 min) -> results/sim_capacity_ci.json
./run_train_seeds.sh            # Exp B rev: 3 seeds x {linear,gla,delta,gdelta} x d_k in {8,16,32}, dense eval grid -> results/train_seeds/
                                #   + Exp B' overwrite task (d_k=16, seeds 0,1)                                    -> results/train_overwrite/
                                #   (seed-0 overwrite runs were launched separately with the same command line)
python3 recall_demand_v2.py     # Exp C rev: held-out criteria D0-D4, 2-fold doc split, doc-level bootstrap, BPE, no-KJV (~6 min) -> results/demand_v2.json
python3 replay_memory.py        # Exp D' : real token streams replayed through ideal fixed-state memories (~10 min) -> results/replay_memory.json
python3 analyze_v2.py           # tables for the revision -> results/analysis_v2.json, results/fig_revision.png
```
The byte-level BPE (vocab 8192) is trained on the corpus inside recall_demand_v2.py with HuggingFace `tokenizers`.
`train_mqar.py` gained `--task overwrite --rep R`, `--dense_eval 1` and `--outdir`; default behaviour is unchanged,
and seed-0 runs reproduce the first-version models exactly (only the evaluation grid differs).
