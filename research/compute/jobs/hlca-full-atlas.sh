# Lung cell atlas, full release (2.3 million cells, all studies, Cell Ontology labels):
# leave-one-study-out annotation on the stored embedding, HFM's setting.
set -e
pip -q install numpy scipy scikit-learn h5py fsspec aiohttp requests
python -u research/hfm-projects/biology/lung-cell-annotation/cross_study_annotation.py full
