"""Convert NSRDB/WTK resource to hourly capacity factors (UTC, 8760 h; Feb-29 dropped).
Solar: single-axis tracking PV via pvlib (PVWatts DC/inverter models), DC/AC = 1.3, 14% DC losses.
Wind: generic normalized 3-4 MW class power curve (cut-in 3 m/s, rated 11 m/s, cut-out 25 m/s),
air-density corrected hub-height wind speed, 15% wake/availability/electrical losses."""
import numpy as np, pandas as pd, pvlib, glob, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")

def solar_cf(d):
    t = pd.to_datetime([x.decode()[:19] for x in d["ns_time"]], utc=True)
    df = pd.DataFrame({k: d[k] for k in ["ghi", "dni", "dhi", "air_temperature", "wind_speed"]}, index=t)
    lat, lon = d["ns_latlon"]
    # shift timestamps to interval centre for 30-min data, then average to hourly
    sp = pvlib.solarposition.get_solarposition(t + pd.Timedelta(minutes=15), lat, lon)
    sp.index = t
    trk = pvlib.tracking.singleaxis(sp["apparent_zenith"], sp["azimuth"], max_angle=60, backtrack=True, gcr=0.35)
    trk = trk.fillna(0)
    poa = pvlib.irradiance.get_total_irradiance(trk["surface_tilt"], trk["surface_azimuth"], sp["apparent_zenith"],
                                                sp["azimuth"], df["dni"], df["ghi"], df["dhi"], model="isotropic")
    poa_g = poa["poa_global"].fillna(0).clip(lower=0)
    tc = pvlib.temperature.sapm_cell(poa_g, df["air_temperature"], df["wind_speed"],
                                     **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"])
    dcac = 1.3
    pdc = pvlib.pvsystem.pvwatts_dc(poa_g, tc, pdc0=dcac, gamma_pdc=-0.0035) * (1 - 0.14)
    pac = pvlib.inverter.pvwatts(pdc, pdc0=1.0 / 0.96, eta_inv_nom=0.96)  # AC rating = 1.0
    pac = pac.clip(lower=0, upper=1.0).fillna(0)
    hourly = pac.resample("1h").mean()
    return hourly

def wind_cf(d):
    t = pd.to_datetime([x.decode()[:19] for x in d["wt_time"]], utc=True)
    v = d["windspeed_100m"]; T = d["temperature_100m"] + 273.15; p = d["pressure_100m"]
    p = np.where(p < 2000, p * 1000.0, p)  # kPa -> Pa if needed
    rho = p / (287.05 * T)
    veq = v * (rho / 1.225) ** (1 / 3)
    ci, vr, co = 3.0, 11.0, 25.0
    cf = np.where((veq >= ci) & (veq < vr), (veq ** 3 - ci ** 3) / (vr ** 3 - ci ** 3), 0.0)
    cf = np.where((veq >= vr) & (veq <= co), 1.0, cf) * (1 - 0.15)
    return pd.Series(cf, index=t)

def drop_leap(s):
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    return s.values[:8760]

if __name__ == "__main__":
    summary = {}
    for f in sorted(g for g in glob.glob(os.path.join(DATA, "*_20??.npz")) if not os.path.basename(g).startswith("cf_")):
        name = os.path.basename(f)[:-4]
        d = np.load(f)
        s = drop_leap(solar_cf(d)); w = drop_leap(wind_cf(d))
        np.savez(os.path.join(DATA, f"cf_{name}.npz"), solar=s, wind=w)
        summary[name] = {"solar_cf": float(s.mean()), "wind_cf": float(w.mean()), "p_raw_median": float(np.median(d["pressure_100m"]))}
        print(name, summary[name])
    json.dump(summary, open(os.path.join(HERE, "..", "results", "capacity_factor_summary.json"), "w"), indent=1)
