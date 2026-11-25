# Rayleigh-Benard: full-field reconstruction from 256 sensors on three flow settings
# of The Well's public dataset (valid file to fit, test file to score), then one
# setting again with twice the fitting data and more modes, to see whether the limit
# is the amount of data.
set -e
pip -q install numpy h5py
cd "$DATA"
base=https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/data
args=""
for s in Rayleigh_1e6_Prandtl_1 Rayleigh_1e8_Prandtl_1 Rayleigh_1e10_Prandtl_1; do
  for split in valid test; do
    curl -sS -L -f -m 1800 -o "${split}_rayleigh_benard_$s.hdf5" "$base/$split/rayleigh_benard_$s.hdf5"
  done
  args="$args $DATA/valid_rayleigh_benard_$s.hdf5 $DATA/test_rayleigh_benard_$s.hdf5"
done
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/physics/rayleigh-benard-reconstruction/reconstruct.py $args
echo; echo "===== more fitting data: Rayleigh 1e8, 10 runs from the train split, 300 modes ====="
cd "$DATA"; rm -f valid_*1e6* valid_*1e10* test_*1e6* test_*1e10*
curl -sS -L -f -m 3000 -o train_rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5 "$base/train/rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5"
cd "$GITHUB_WORKSPACE"
RB_FIT_RUNS=10 RB_MODES=300 python -u research/hfm-projects/physics/rayleigh-benard-reconstruction/reconstruct.py "$DATA/train_rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5" "$DATA/test_rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5"
