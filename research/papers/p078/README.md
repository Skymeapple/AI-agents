# p078 — Measuring frontier AI power concentration under definitional ambiguity and disclosure censoring

Paper: `paper.md`. Target venue: ACM FAccT 2027.

## Reproduce
```
bash code/run_all.sh
```
Runtime < 1 min on CPU. Python 3 + numpy, scipy, matplotlib.

## Data provenance
`data/notable_raw.md` and `data/frontier_raw.md` are the Epoch AI "Notable AI Models" and "Frontier AI Models"
CSV files (https://epoch.ai/data/notable_ai_models.csv, https://epoch.ai/data/frontier_ai_models.csv; CC-BY 4.0),
retrieved on 2026-11-22 through a web-extraction service (Tavily, markdown mode) because epoch.ai was not directly
reachable from the sandbox. The extractor replaced line breaks with spaces; `code/parse_epoch.py` restores rows by
column count and date anchors. Validation (`results/parse_validation.json`): 1,092 rows recovered; for all 114 models
present in both independently extracted files, training-compute values agree exactly. (`*_raw.txt` are earlier
plain-text extractions that lost hyperlinked text; they are kept for transparency but not used.)

## Files
- `code/common.py` loading, actor attribution (lab vs. capital-bloc), HHI
- `code/parse_epoch.py` row recovery + cross-file validation
- `code/analysis.py` definition sweep, disclosure-censoring bounds, growth rates, open-weight gap/lag, country shares
- `code/model.py` calibration of Propositions 1–5, Fisher/logit tests of disclosure, identification curve
- `results/*.json` all numbers quoted in the paper; `results/fig*.png` figures
