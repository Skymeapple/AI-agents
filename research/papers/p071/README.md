# p071 — Threshold drift: indexing compute thresholds, and who must regulate

Paper: `paper.md` (target venue: ACM FAccT 2027).

## Reproduce

```bash
cd /home/user/AI-agents/research/papers/p071
python3 code/analysis.py > results/run_log.txt   # ~30 s CPU
python3 code/figures.py                           # figures + cumulative 1e26 counts
```

Requirements: Python 3, numpy, scipy, pandas, matplotlib. Random seed fixed (20261122).

## Data

`data/*.csv` are an unchanged snapshot of Epoch AI, "Data on AI models" (CC BY 4.0),
latest publication date in `large_scale_ai_models.csv` = 2026-09-09. The snapshot was obtained
from a public GitHub repository that committed it on 2026-10-22 (epoch.ai itself was not reachable
from this environment); SHA-256 hashes are in `data/SHA256SUMS` and Epoch's licence note in
`data/SNAPSHOT_README.md`.

## Outputs (results/)

| file | content |
|---|---|
| `results.json` | all scalar results (frontier growth, intensity fit, OOS validation, rho*, growth rates, ambiguity, bunching) |
| `projected_coverage.csv` | expected covered releases/yr under each threshold rule, 2024–2030 (M2 known, M2 MAR-weighted, M1) |
| `retrospective_coverage.csv` | observed covered models / developers / countries by rule and year |
| `loss_ratios.csv` | loss of static / capability-indexed / frontier-relative rules relative to optimal path |
| `jurisdiction_by_threshold*.csv` | developer-country composition vs threshold (known; upper-bound imputation) |
| `observability_gap.csv` | share of models without public compute estimates by year and access type |
| `fig1..fig6*.png` | figures used in the paper |
| `run_log.txt` | console output of analysis.py |
