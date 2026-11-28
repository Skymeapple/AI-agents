# p005 — Interpolation is relative to a representation

Paper: `paper.md`. All numbers in the paper come from the JSON files in `results/`.

## Requirements
Python 3, numpy, scipy, matplotlib, scikit-learn (for the UCI 8x8 digits loader and AUROC).
No GPU. The MLPs are implemented in numpy (`code/nn.py`); the round-2 transformer and CNN use PyTorch on CPU.

## Reproduce (about 10 minutes on 4 CPU cores)
```
cd code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 exp_modadd.py    # Exp. 1: modular addition (p=53), grokking dynamics -> results/modadd_*.json (~3-4 min)
python3 exp_digits.py    # Exp. 2: 8x8 digits, 5 seeds, pixel-shift probe   -> results/digits.json (~3 min)
python3 exp_diag.py      # Exp. 3: certificate diagnostics vs data size      -> results/diag.json (~2 min)
python3 make_figures.py  # figures fig1-3 and results/summary.json
# Round-2 revision (about 30-40 min on 4 shared cores; PyTorch CPU needed for the transformer and CNN)
python3 exp_controls.py      # Exp. 4: 17 MLP runs, margin / loss / NC1 controls -> results/controls_mlp_*.json
TF_STEPS=6000 TF_FRACS=0.5,0.3 TF_PROCS=2 python3 exp_transformer.py  # Exp. 5 -> results/controls_tf_*.json
python3 analyze_controls.py  # lags, CIs, margin-threshold fits -> results/controls_summary.json
python3 exp_digits2.py       # Exp. 6: MLP + CNN, 6 perturbations, baselines, padding -> results/digits2.json
python3 analyze_digits2.py   # per-seed CIs, leave-one-seed-out tests -> results/digits2_summary.json
python3 make_figures2.py     # fig4_controls.png, fig5_digits_auroc.png
```

## Files
- `code/hull.py` — hull projection (NNLS with a weighted sum-to-one row; the weights are renormalised,
  so the returned point is always a valid hull point and the certificate stays sound), exact LP hull membership,
  and the readout interpolation certificate (Proposition 3).
- `code/nn.py` — numpy MLP + AdamW.
- `results/digits_perpixelstd_prelim*.json/txt` — an earlier digits run with per-pixel standardisation
  (superseded; kept for transparency, see paper Section 6.2 footnote).
