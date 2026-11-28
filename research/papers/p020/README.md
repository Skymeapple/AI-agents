# p020: Authorship information in human and LLM creative idea populations

Paper: `paper.md`

## Reproduce (CPU, about 3 minutes in total)

```bash
cd code
./fetch_data.sh          # clones github.com/surabhisnath/Creative_Process at commit bd88536 and copies the 2 CSVs to ../data
python3 simulate.py      # Sim 1-3: estimator validation, novelty-competence identity, list-dependence pitfall (~35 s)
python3 empirical.py     # analyses A-F on the public data (~2.5 min)
python3 make_figures.py  # fig_ladder.png, fig_temperature.png, fig_peer_norm.png
python3 human_within_entropy.py  # results/human_within_list_entropy.json
```

Requirements: Python 3, numpy, scipy, pandas, matplotlib. Seeds are fixed in the scripts.
Only the two CSV files from the data repository are read. Its pickled embedding files are never loaded.

Expected SHA-256 of the data files:
- `data_LLMs.csv`   ed4d85aa76e3acb400362a86a65939f78e6e1b4ff624e3ff6d7667a0c0eb9ed3
- `data_humans.csv` 307af728895c8ee45e619e643b72241930aec7d5a1f3eb2d2cef46507c993548

## Files
- `code/infodecomp.py`: entropy and plug-in MI, the i.i.d.-response permutation null (`excess_mi`) and the list-level permutation null (`list_perm_mi`)
- `code/simulate.py`: writes `results/sim1_*.json`, `sim2_identity.json`, `sim3_list_dependence.json`, `fig_sim1_estimator.png`, `fig_sim2_identity.png`
- `code/empirical.py`: writes
  - `A_naive_iid_permutation.csv` (the naive estimator, shown as a pitfall)
  - `B_llm_hierarchy_listperm.csv` (decomposition by model, temperature and replicate)
  - `C_temperature_sweep.csv`
  - `D_list_divergence_ladder.csv` and `D2_ladder_disjoint_tests.csv` (main comparison)
  - `E_peer_normed_originality_*.csv`
  - `F_breadth_depth_coverage.csv`
  - `log.json` (correlations, Welch tests, descriptives)
