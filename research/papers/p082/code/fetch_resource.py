"""Fetch hourly solar (NSRDB v3.0.6) and wind (WIND Toolkit CONUS v1.0.0) resource
for candidate AI data-center sites from NREL's public AWS buckets (anonymous S3).
Writes data/<site>_<year>.npz with GHI, DNI, DHI, air temp, 100m wind speed, 100m temp, pressure."""
import s3fs, h5py, numpy as np, sys, os, time
SITES = {"abilene_tx": (32.45, -99.73), "loudoun_va": (39.04, -77.49), "phoenix_az": (33.45, -112.07)}
YEARS = [int(y) for y in sys.argv[1:]] or [2010, 2011, 2012, 2013]
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
fs = s3fs.S3FileSystem(anon=True)

def nearest(coords, lat, lon):
    d = (coords[:, 0] - lat) ** 2 + ((coords[:, 1] - lon) * np.cos(np.radians(lat))) ** 2
    return int(np.argmin(d))

cache = {}
for yr in YEARS:
    t0 = time.time()
    with fs.open(f"nrel-pds-nsrdb/GOES/aggregated/v3.0.6/nsrdb_{yr}.h5", block_size=2**20) as f, h5py.File(f, "r") as h:
        if "ns" not in cache:
            c = h["coordinates"][...]; cache["ns"] = {s: nearest(c, *ll) for s, ll in SITES.items()}
            cache["ns_ll"] = {s: c[i].tolist() for s, i in cache["ns"].items()}
        ns = {}
        ti = h["time_index"][...]
        step = len(ti) // 8760 if len(ti) >= 8760 else 1
        for s, i in cache["ns"].items():
            d = {}
            for k in ["ghi", "dni", "dhi", "air_temperature", "wind_speed"]:
                ds = h[k]; sf = ds.attrs.get("psm_scale_factor", ds.attrs.get("scale_factor", 1.0))
                d[k] = ds[:, i].astype(float) / sf
            ns[s] = d
    with fs.open(f"nrel-pds-wtk/conus/v1.0.0/wtk_conus_{yr}.h5", block_size=2**20) as f, h5py.File(f, "r") as h:
        if "wt" not in cache:
            c = h["coordinates"][...]; cache["wt"] = {s: nearest(c, *ll) for s, ll in SITES.items()}
            cache["wt_ll"] = {s: c[i].tolist() for s, i in cache["wt"].items()}
        wt = {}
        for s, i in cache["wt"].items():
            d = {}
            for k in ["windspeed_100m", "temperature_100m", "pressure_100m"]:
                ds = h[k]; sf = ds.attrs.get("scale_factor", 1.0)
                d[k] = ds[:, i].astype(float) / sf
            wt[s] = d
        wti = h["time_index"][...]
    for s in SITES:
        np.savez(os.path.join(OUT, f"{s}_{yr}.npz"), ns_time=ti, wt_time=wti, ns_latlon=cache["ns_ll"][s],
                 wt_latlon=cache["wt_ll"][s], **ns[s], **wt[s])
    print(yr, "done in %.0fs" % (time.time() - t0), flush=True)
