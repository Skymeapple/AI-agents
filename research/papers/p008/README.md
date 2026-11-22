# p008 — How far can distillation compress capabilities? (code + results)

Requirements: Python 3, numpy, scipy, matplotlib, torch (CPU). Tested with torch 2.15 on 4 CPU cores.

```
cd code
python3 run_all.py     # E1-E4 + trained teacher + E2T, ~30 min on 4 cores -> results/all_runs.json, results/teacher_h256_seed0.pt
python3 run_e5.py      # E5 (long-training capacity ceiling), ~14 min    -> results/e5_runs.json
python3 analyze.py     # figures fig1-fig5 + results/summary.json (all numbers quoted in paper.md)
```

Files
- `code/skills.py`  — Zipfian multi-skill world, student MLP, training (hard / soft labels, early stopping), evaluation.
- `code/run_all.py`, `code/run_e5.py` — experiment grids (configs listed in `configs()`).
- `code/analyze.py` — fits (threshold model, gamma), head/tail analysis, figures.
- `results/run_log.txt`, `results/run_e5_log.txt`, `results/analysis_stdout.txt` — logs.
All seeds are fixed; results are deterministic up to PyTorch CPU nondeterminism.
