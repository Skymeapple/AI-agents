# p091 — Memorised mistakes? Separating audit error, plausibility and memorisation in wrong-key agreement on MMLU

Paper: `paper.md`. All numbers in the paper come from `results/*.json` produced by the scripts below.

## Data (public)
* HELM MMLU per-instance predictions, public GCS bucket `crfm-helm-public` (`mmlu/benchmark_output/runs/v1.*`).
  80 models with all 57 subjects (latest HELM release per model/subject).
* MMLU-Redux annotations: CSVs from https://github.com/aryopg/mmlu-redux (folder `mmlu_redux/`, commit in `data/mmlu_redux/COMMIT`), copied to `data/mmlu_redux/`.
* `data/author_adjudication.csv`: author's re-adjudication of the 23 flagged items on which the 25 strongest models most often choose the original key (W_top25 > 0.2).

## Run (CPU only, ~15 min total on 4 cores)
```
pip install numpy scipy matplotlib statsmodels
cd code
python3 01_list_helm_runs.py        # list run directories   -> data/helm_runs.json
python3 02_download_helm.py         # ~600 MB download        -> data/helm_raw/
python3 03_build_matrix.py          # join with MMLU-Redux    -> data/redux_resp.npz, data/full_correct.npz
python3 ok_item_params.py           # OK-item logistic ranges -> results/ok_item_params.json
NBOOT=60 python3 04_contamination.py  # wrong-key analysis, mixture model, bootstrap, profiles
REPS=40 python3 05_simulation.py    # simulation study (multiprocessing)
python3 06_saturation.py            # ceiling, discriminability, IRT information
python3 07_verify_items.py          # numerical checks of computational items
python3 08_theory_quantities.py     # A1 support, confound threshold, per-model bounds
python3 09_figures.py               # figures -> results/fig*.png
```
`data/helm_raw/` can be deleted after step 3.
