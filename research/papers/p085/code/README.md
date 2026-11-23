# p085 — Batching cannot amortize context: decode cost floors

Reproduce everything (CPU only, ~8 min for the CPU experiment, <1 min for the rest):

```bash
cd research/papers/p085/code
for r in 0 1 2; do python3 cpu_experiment.py $r; done   # toy-decoder timing, 3 replicates
python3 analyze_cpu.py      # tests T1-T4 -> results/cpu_summary.json, fig_cpu_tests.png
python3 analysis.py         # analytic model on sourced configs -> results/analysis_results.json, figs
```

Files
- `cost_model.py` — the analytic floor model (Prop. 1-3 of the paper), sourced model/hardware configs.
- `cpu_experiment.py` — numpy/OpenBLAS fp32 toy decoder (8 layers, d=1024, 16 heads, n_kv in {16,4,1});
  calibrates GEMM FLOP/s, GEMV bandwidth and attention-kernel bandwidth, then times decode steps over
  batch b, context L, KV heads, FFN width and verified positions tau.
- `analyze_cpu.py` — aggregates replicates (median), fits slopes, post-hoc two-term model.
- `analysis.py` — floors/regimes (Table 1), elasticities (Table 2), lever ladder, Monte Carlo headroom,
  hardware-per-dollar comparison, trend projection, sensitivity to eta/SLO.

Python 3.13, numpy, scipy, matplotlib. Timings depend on the machine; seeds 0,1,2.
