# p093: Pluralism vs. noise in subjective evaluation

Paper: `paper.md`. All numbers in the paper come from the scripts below; outputs are in `results/`.

## Data (public, downloaded from GitHub raw)
- HANNA (Chhun et al., COLING 2022): `data/hanna_stories_annotations.csv`
  from https://raw.githubusercontent.com/dig-team/hanna-benchmark-asg/main/hanna_stories_annotations.csv
- DICES-350 (Aroyo et al., NeurIPS D&B 2023): `data/dices/dices350.csv`
  from https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv

## Run (Python 3, numpy, scipy, pandas, matplotlib; about 5 minutes total on CPU)
```
cd code
python3 sim.py               # S1-S3 simulations -> results/s1_*.csv, s2_summary.csv, s3_summary.csv, fig_simulations.png (~4 min)
python3 analysis_hanna.py    # -> results/hanna_summary.csv, hanna_pairs.csv, hanna_rankings.csv, fig_hanna_*.png
python3 robustness_hanna.py  # -> results/hanna_robustness.csv
python3 analysis_dices.py    # -> results/dices_summary.csv, dices_pairs.csv, dices_variance_decomp.csv, fig_dices_contrasts.png
```
`code/core.py` holds the estimators: Krippendorff alpha, per-rater contrasts, DerSimonian-Laird
heterogeneity, the within-item permutation test for rater persistence, the TV reweighting radius
(Theorem 2), and the sensitivity Gamma* (Proposition 3, Gaussian closed form and discrete LP).
All random seeds are fixed.
