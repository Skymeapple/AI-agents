# Weather: bias correction of single models against blending model families, on
# forecasts stored in the public WeatherBench 2 bucket (24 h lead, 1.5 degrees, 2020).
set -e
pip -q install numpy scipy xarray zarr gcsfs dask
python - <<'PY'
import xarray as xr
for name, path in {"era5": "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
                   "pangu": "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr"}.items():
    ds = xr.open_zarr(path, storage_options={"token": "anon"}, chunks=None)
    print(name, dict(ds.sizes), [v for v in ds.data_vars][:8])
    v = ds["2m_temperature"]; print("  dims", v.dims, "chunks", v.encoding.get("chunks"))
import gcsfs
fs = gcsfs.GCSFileSystem(token="anon")
for m in ("graphcast/2020", "neuralgcm_deterministic", "fuxi", "keisler"):
    print(m, [p.split("/")[-1] for p in fs.ls(f"weatherbench2/datasets/{m}")])
PY
python -u research/hfm-projects/earth-science/weather-24h/blend_forecasts.py 2
