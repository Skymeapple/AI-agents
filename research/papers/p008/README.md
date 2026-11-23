# p008 — How far can distillation compress capabilities? (code + results)

Requirements: Python 3, numpy, scipy, matplotlib, pandas, torch (CPU). Tested with torch 2.15 on 4 CPU cores.

```
cd code
python3 run_all.py     # E1-E4 + trained teacher + E2T, ~30 min on 4 cores -> results/all_runs.json, results/teacher_h256_seed0.pt
python3 run_e5.py      # E5 (long-training capacity ceiling), ~14 min    -> results/e5_runs.json
python3 analyze.py     # figures fig1-fig5 + results/summary.json (round-1 numbers)
# round 2 (revision)
python3 run_e6.py A    # E6-A: h=32/48, 100k steps, hard / oracle-soft / teacher-soft, 5 seeds -> results/e6_runs_A.json
python3 run_e6.py B    # E6-B: h=16 (5 seeds), teacher-distilled h=8/16/24, h=128 at 40k  -> results/e6_runs_B.json
python3 run_e6.py C    # E6-C: h=48 250k steps (hard, teacher-soft), h=128 hard 200k     -> results/e6_runs_C.json
python3 analyze_r2.py  # finite-M Prop. 2 test, gamma resampling, E3 accuracy criterion, P1 calibration +
                       # quantisation, PopQA popularity analysis, E6 stats -> results/summary_r2.json, fig6, fig7
```
`analyze_r2.py` reads the public PopQA file at `data_ext/popqa/popQA.tsv` (Mallen et al. 2023).
Round-2 runs were done on a shared 4-core machine; wall-clock logs are in `results/run_e6{A,B,C}_log.txt`.

Files
- `code/skills.py`  — Zipfian multi-skill world, student MLP, training (hard / soft labels, early stopping), evaluation.
- `code/run_all.py`, `code/run_e5.py` — experiment grids (configs listed in `configs()`).
- `code/analyze.py` — fits (threshold model, gamma), head/tail analysis, figures.
- `results/run_log.txt`, `results/run_e5_log.txt`, `results/analysis_stdout.txt` — logs.
All seeds are fixed; results are deterministic up to PyTorch CPU nondeterminism.
