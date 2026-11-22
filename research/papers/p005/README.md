# p005 — Interpolation is relative to a representation

Paper: `paper.md`. All numbers in the paper come from the JSON files in `results/`.

## Requirements
Python 3, numpy, scipy, matplotlib, scikit-learn (for the UCI 8x8 digits loader and AUROC).
No GPU and no deep-learning framework: the MLPs are implemented in numpy (`code/nn.py`).

## Reproduce (about 10 minutes on 4 CPU cores)
```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 exp_modadd.py    # Exp. 1: modular addition (p=53), grokking dynamics -> results/modadd_*.json (~3-4 min)
python3 exp_digits.py    # Exp. 2: 8x8 digits, 5 seeds, pixel-shift probe   -> results/digits.json (~3 min)
python3 exp_diag.py      # Exp. 3: certificate diagnostics vs data size      -> results/diag.json (~2 min)
python3 make_figures.py  # figures fig1-3 and results/summary.json
```

## Files
- `code/hull.py` — hull projection (NNLS with a weighted sum-to-one row; the weights are renormalised,
  so the returned point is always a valid hull point and the certificate stays sound), exact LP hull membership,
  and the readout interpolation certificate (Proposition 3).
- `code/nn.py` — numpy MLP + AdamW.
- `results/digits_perpixelstd_prelim*.json/txt` — an earlier digits run with per-pixel standardisation
  (superseded; kept for transparency, see paper Section 6.2 footnote).
