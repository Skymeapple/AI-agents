# Lung cell atlas, full release: compare non-linear classifier heads on the stored embedding
# (nearest neighbours with different k, a small neural network, and averages with the linear head).
set -e
pip -q install numpy scipy scikit-learn h5py fsspec aiohttp requests
python -u research/hfm-projects/biology/lung-cell-annotation/cross_study_annotation.py full heads
