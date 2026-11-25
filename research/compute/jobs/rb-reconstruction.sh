# Rayleigh-Benard: full-field reconstruction from 256 sensors on three flow settings
# of The Well's public dataset (valid file to fit, test file to score).
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
ls -la
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/physics/rayleigh-benard-reconstruction/reconstruct.py $args
