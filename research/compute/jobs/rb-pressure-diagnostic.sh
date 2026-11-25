# Rayleigh-Benard: diagnose the mismatch between the pressure equation and the stored pressure.
set -e
pip -q install numpy h5py
base=https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/data
for s in Rayleigh_1e6_Prandtl_1 Rayleigh_1e8_Prandtl_1; do
  curl -sS -L -f -m 1800 -o "$DATA/$s.hdf5" "$base/test/rayleigh_benard_$s.hdf5"
  echo "== $s"
  (cd research/hfm-projects/physics/rayleigh-benard-reconstruction && python -u pressure_diagnostic.py "$DATA/$s.hdf5")
done
