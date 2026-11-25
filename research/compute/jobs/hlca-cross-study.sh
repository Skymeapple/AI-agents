# Lung cell atlas: leave-one-study-out annotation on the stored scANVI embedding.
set -e
pip -q install numpy scipy scikit-learn h5py fsspec aiohttp requests
python -u research/hfm-projects/biology/lung-cell-annotation/cross_study_annotation.py
