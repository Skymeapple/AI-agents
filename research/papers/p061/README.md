# p061: Aggregation, not exposure (reproducibility package)

Paper: `paper.md`. Every number in the paper comes from the files in `results/`, which the scripts in `code/` produce, or from a cited source.

## Data (`data/`, public, fetched 2026-11-22)
| file | source |
|---|---|
| `full_labelset.tsv`, `occ_level.csv` | Eloundou et al. task-level exposure labels, https://github.com/openai/GPTs-are-GPTs (`data/`) |
| `task_ratings.txt` | O*NET 30.0 Task Ratings (mirrored at github.com/dot-org-ai/standards.org.ai, `.source/ONET/db_30_0_text/`; origin onetcenter.org) |
| `national_M2021_dl.csv` … `national_M2024_dl.csv` | BLS OEWS national estimates, May 2021–2024 (GitHub mirrors of the bls.gov files: SAP/hr-request-data-set, trgardos/data-science-job-market-analysis, carlboy2002/carlboy2002.github.io, Budget-Lab-Yale/AI-Employment-Model) |
| `natsector_M2024_dl.xlsx` | BLS OEWS May 2024 by NAICS sector (mirror: oTullio/measuring-the-industry) |

bls.gov and onetcenter.org could not be reached from this sandbox, so we used GitHub mirrors. The 2024 national file reports 154,187,380 jobs on its `00-0000` (all occupations) row. Before you rely on the results, re-download the files from the original sources and check them.

## Run (about 1 minute on CPU in total)
```
cd code
python3 prep.py          # merge tasks, weights, employment
python3 multiverse.py    # 1,680 specifications
python3 analysis.py      # ANOVA, sector-rank robustness, FOSD test, eps* table
python3 empirics.py      # OEWS 2021->2024 descriptive regressions, rule horse race
python3 power_sim.py     # identification / power Monte Carlo (seeded)
python3 extra_stats.py   # numbers quoted in the text
python3 figures.py       # fig1..fig5
```
Dependencies: Python 3, numpy, scipy, pandas, matplotlib, openpyxl.
