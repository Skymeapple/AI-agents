# Lung histology stand-in on LC25000 (rerun with neighbour-capped grouping): random vs near-duplicate-grouped cross-validation.
set -e
pip -q install numpy scikit-learn pyarrow pillow torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
timeout 300m python3 -u research/hfm-projects/medicine/lung-histology/lc25000_floor.py "$DATA/lc25000"
