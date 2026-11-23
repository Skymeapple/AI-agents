# p025 — Fluency parity for computer-use agents

Paper: `paper.md`. Code: `code/`. Outputs: `results/`.

## Reproduce

```bash
# 1. (optional, data already included in data/oswh_tasks.json)
git clone --depth 1 https://github.com/WukLab/osworld-human.git /tmp/oswh   # commit deff1a7c (2026-05-17)
python3 code/extract_data.py /tmp/oswh

# 2. run all experiments (~6 s CPU; numpy, scipy, matplotlib)
python3 code/run_all.py > results/run_log.txt
```

## Files
- `code/klm.py` — Keystroke-Level-Model estimator of expert human time for each OSWorld-Human reference trajectory.
- `code/model.py` — agent timing model (step inflation, per-step latency, context growth, recoverable errors, hazard), parity horizon (Prop. 1), time-budgeted success S(tau).
- `code/run_all.py` — experiments E1–E6 and the scenario bounds; writes `results/results.json`, `results/e1_klm_by_app.csv`, `results/fig1..fig6*.png`.
- `data/oswh_tasks.json` — 369 OSWorld tasks with the OSWorld-Human single-action and grouped-action human reference trajectories (Abhyankar, Qi & Zhang). Upstream asks that the data not be used for training; it is used here only for evaluation analysis.

All random seeds are fixed (0, 1, 2). Every parameter marked ASSUMPTION in `run_all.py` is discussed in the paper's Section 4 and Limitations.
