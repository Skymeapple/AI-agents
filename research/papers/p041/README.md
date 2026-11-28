# p041: Feature absorption is optimal, and activations alone cannot certify its absence

Paper: `paper.md`. All numbers in the paper come from the JSON files in `results/`, produced by the scripts in `code/`.

## Requirements
Python 3, numpy, scipy, matplotlib. CPU only; no GPU or external data.

## Reproduce (about 20 min on 4 CPU cores)
```bash
cd code
python3 exp1_theory.py            # E1: exact optimum of the sparse-coding objective (~2.5 min) -> results/e1_theory.json
python3 exp1b_rho_fine.py         # E1b: fine sweep of hierarchy-violation rate rho (~3 min)   -> results/e1b_rho_fine.json
python3 exp2_sae_two_feature.py   # E2: trained ReLU SAEs, 2 latents (~1.5 min)               -> results/e2_sae.json
python3 exp3_many_features.py     # E3: 48-feature toy, weight-only diagnostics (~10-15 min)   -> results/e3_many.json
python3 make_figures.py           # figures -> results/fig*.png
```

## Files
- `code/theory.py`: exact non-negative lasso for two atoms (enumerates supports), data model, dictionary search, closed forms (Prop. 1).
- `code/sae.py`: minimal numpy ReLU SAE (unit-norm decoder, no decoder bias, Adam).
- `code/exp*.py`: experiments, as listed above. Seeds are fixed inside each script.
