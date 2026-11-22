"""Fleet-level application of the vintage model to the four largest US hyperscalers.

Produces:
  results/calibration.json        implied rents, obsolescence-rate cross-checks, Meta check
  results/fleet_<scenario>.csv    year-by-year accounting vs economic depreciation, values,
                                  required rental revenue
  results/fleet_summary.json      headline numbers
"""
import json, os
import numpy as np
from vintage import Asset
import data as D

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2010, 2031))


def capex_series(scenario):
    tot = {y: sum(D.CAPEX[k][y] for k in D.CAPEX) for y in range(2021, 2026)}
    tot[2026] = sum(D.GUIDE_2026.values())
    for i, gr in enumerate(D.SCENARIOS[scenario]):
        tot[2027 + i] = tot[2026 + i] * (1 + gr)
    for y in range(2020, 2009, -1):  # (A) back-cast
        tot[y] = tot[y + 1] / (1 + D.BACKCAST_GROWTH)
    return tot


def year_flows(asset, age_end):
    """Depreciation in a calendar year for a vintage placed in service mid-year.
    age_end = age at end of the year (>=0.5).  Returns (econ, acc, V_end, B_end, rent)."""
    a0 = max(age_end - 1.0, 0.0)
    econ = float(asset.Vbook_econ(a0) - asset.Vbook_econ(age_end))
    acc = float(asset.B(a0) - asset.B(age_end))
    # average rent over the in-service part of the year
    grid = np.linspace(a0, age_end, 21)
    rent = float(np.mean(asset.rent(grid))) * (age_end - a0)
    return econ, acc, float(asset.Vbook_econ(age_end)), float(asset.B(age_end)), rent


def run_fleet(params, scenario, years_out=range(2021, 2031)):
    p = params
    srv = Asset(p['g'], p['c'], p['r'], p['Tmax'], p['L'])
    stc = Asset(0.0, p['c_struct'], p['r'], p['L_struct'], p['L_struct'], straight_line_econ=True)
    cap = capex_series(scenario)
    rows = []
    for t in years_out:
        acc = econ = Vt = Bt = rent = 0.0
        for v, I in cap.items():
            if v > t:
                continue
            age_end = t - v + 0.5
            for share, asset in ((p['s'], srv), (1 - p['s'], stc)):
                e, a, Ve, Be, rn = year_flows(asset, age_end)
                econ += I * share * e; acc += I * share * a
                Vt += I * share * Ve; Bt += I * share * Be; rent += I * share * rn
        rows.append(dict(year=t, capex=cap[t], dep_acc=acc, dep_econ=econ, gap=econ - acc,
                         book_value=Bt, econ_value=Vt, value_gap=Bt - Vt, req_rent=rent))
    return rows, srv


def meta_check(params):
    """Model-implied 2025 depreciation reduction from moving Meta's server/network life
    from 5.0 to 5.5 years, as a function of the server+network share s_meta of capex.
    Straight-line, mid-year placement; vintages 2021-2024 (2020 and earlier ignored)."""
    out = {}
    for s_meta in [0.4, 0.5, 0.6, 0.7, 0.8]:
        delta = 0.0
        for v in range(2021, 2025):
            I = D.CAPEX['META'][v] * s_meta
            age = 2024 - v + 0.5         # age at 31 Dec 2024
            nbv = I * max(0.0, 1 - age / 5.0)
            if nbv <= 0:
                continue
            d_old = min(nbv, I / 5.0)
            rem_new = 5.5 - age
            d_new = min(nbv, nbv / rem_new)
            delta += d_old - d_new
        out[s_meta] = delta
    # share that reproduces the disclosed $2.92bn
    ss = np.linspace(0.3, 1.0, 701)
    vals = []
    for s_meta in ss:
        delta = 0.0
        for v in range(2021, 2025):
            I = D.CAPEX['META'][v] * s_meta; age = 2024 - v + 0.5
            nbv = I * max(0.0, 1 - age / 5.0)
            if nbv > 0:
                delta += min(nbv, I / 5.0) - min(nbv, nbv / (5.5 - age))
        vals.append(delta)
    vals = np.array(vals)
    s_star = float(ss[np.argmin(abs(vals - D.META_LIFE_CHANGE_EFFECT))])
    return dict(delta_by_share={str(k): v for k, v in out.items()}, implied_share=s_star,
                disclosed=D.META_LIFE_CHANGE_EFFECT)


def calibration(params):
    p = params
    srv = Asset(p['g'], p['c'], p['r'], p['Tmax'], p['L'])
    # implied first-year rent per H100-hour at different utilisations
    rent_hr = {u: srv.P * D.H100_ALLIN_COST / (8760 * u) for u in (0.6, 0.8, 1.0)}
    o = D.H100_OBS
    g_epoch = np.log(2) / np.array(D.EPOCH_DOUBLING_ML)
    implied = dict(
        epoch_ml_doubling=float(g_epoch[0]), epoch_ml_ci=[float(g_epoch[2]), float(g_epoch[1])],
        epoch_30pct=float(-np.log(1 - D.EPOCH_ANNUAL_DECLINE)),
        h100_peak2023_to_oct2025=float(np.log(o['peak_2023'] / o['sa_oct2025']) / 2.25),
        h100_peak2023_to_mar2026=float(np.log(o['peak_2023'] / o['sa_mar2026']) / 2.75),
        silicondata_sep24_jun25=float(np.log(o['sd_sep2024'] / o['sd_jun2025']) / (9.6 / 12)),
        semianalysis_oct25_mar26=float(np.log(o['sa_oct2025'] / o['sa_mar2026']) / (5 / 12)),
    )
    return dict(P=srv.P, econ_life=srv.T, rent_per_gpu_hour=rent_hr, implied_g=implied)


def main():
    p = dict(D.BASE)
    summ = {}
    for sc in D.SCENARIOS:
        rows, srv = run_fleet(p, sc)
        with open(os.path.join(OUT, f'fleet_{sc}.csv'), 'w') as f:
            keys = list(rows[0].keys())
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(f'{r[k]:.3f}' if k != 'year' else str(r[k]) for k in keys) + '\n')
        summ[sc] = {r['year']: {k: round(v, 2) for k, v in r.items() if k != 'year'} for r in rows}
    cal = calibration(p)
    cal['meta_check'] = meta_check(p)
    # low-obsolescence ("demand regime") and high-obsolescence variants
    var = {}
    for name, g in (('g_low_0.15', 0.15), ('g_base_0.35', 0.35), ('g_high_0.55', 0.55)):
        q = dict(p); q['g'] = g
        rows, srv = run_fleet(q, 'plateau')
        r26 = [r for r in rows if r['year'] == 2026][0]
        var[name] = dict(econ_life=srv.T, P=srv.P, gap2026=r26['gap'], dep_acc2026=r26['dep_acc'],
                         dep_econ2026=r26['dep_econ'], value_gap2026=r26['value_gap'],
                         req_rent2026=r26['req_rent'])
    cal['g_variants'] = var
    cal['cloud_q3_2026_annualised'] = 4 * sum(D.CLOUD_Q3_2026.values())
    json.dump(cal, open(os.path.join(OUT, 'calibration.json'), 'w'), indent=2)
    json.dump(summ, open(os.path.join(OUT, 'fleet_summary.json'), 'w'), indent=1)
    print(json.dumps(cal, indent=2))
    for sc in summ:
        print(sc)
        for y in (2024, 2025, 2026, 2027, 2028, 2029, 2030):
            print(y, summ[sc][y])


if __name__ == '__main__':
    main()
