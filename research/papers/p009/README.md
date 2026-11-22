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
