# Rayleigh-Benard 64-frame forecasting: no-network floors on The Well's public data,
# Rayleigh 1e10, Prandtl 1. Fit on 8 runs of the train file, score on the test file.
set -e
pip -q install numpy h5py scipy
cd "$DATA"
base=https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/data
n=rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5
curl -sS -L -f -m 3000 -o train_$n "$base/train/$n"
curl -sS -L -f -m 1800 -o test_$n "$base/test/$n"
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/physics/rayleigh-benard-forecast/forecast_floor.py "$DATA/train_$n" "$DATA/test_$n" 8
