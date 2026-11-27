# Weather: the same blending test on 2022, HFM's held-out year, with every forecast
# set stored for that year on the challenge grid (found by listing the bucket).
set -e
pip -q install numpy scipy xarray zarr gcsfs dask
python -u research/hfm-projects/earth-science/weather-24h/blend_forecasts.py 2022 2 discover 2>&1 | grep -vE "RuntimeError: Task|Task was destroyed|^task:"
# rerun with duplicate forecast sets removed
# rerun with latitude-band weights and a swapped-halves check
