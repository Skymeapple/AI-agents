"""Battery terminal voltage below the calibrated temperature range.

Experiment for HFM's "Battery Voltage Model for Cold Operating Conditions" track,
on the public LG 18650HG2 dataset (Kollmeyer et al., McMaster University, Mendeley
Data doi:10.17632/cp3473x7xv.3, CC-BY 4.0): one cell, drive cycles and
characterisation tests at 40, 25, 10, 0, -10 and -20 degC. HFM's logs mention LG
and Panasonic cells, so this is very likely one of its sources.

Download: https://data.mendeley.com/public-api/zip/cp3473x7xv/download/3
The same script runs unchanged on the Samsung INR21700 30T dataset from the same
laboratory (doi:10.17632/9xyvy2njj3.2, download .../zip/9xyvy2njj3/download/2),
which is used as an independent check.

Task as in HFM: calibration data are warmer than the assessment drive cycles.
Here the model is calibrated at 40, 25 and 10 degC and must predict voltage for
drive cycles at 0, -10 and -20 degC from the current and temperature profiles.

Model: open-circuit voltage from the 25 degC C/20 test, plus an ohmic resistance
and two RC pairs (time constants 4 s and 60 s). The three resistances are fitted
by least squares at each calibration temperature. What differs between methods is
how the resistances are carried to cold temperatures:

  hold coldest      use the 10 degC values unchanged (reference, skill 0)
  linear slope      straight line in temperature through the three calibration
                    temperatures (the "fitted thermal slope" HFM's agents tune)
  Arrhenius         ln R linear in 1/T (one activation energy per resistance), the
                    route of research/human-frontier-mission.md section 7.1
  Arrhenius, cold   the same form fitted only to 25 and 10 degC, the cold edge of the
  edge              calibrated range
  oracle            resistances fitted at the cold temperature itself (best this
                    model structure can do; not available in the task)
  Arrhenius +       a second structure: each resistance also varies with state of
  charge-dependent  charge (quadratic shape learned at the warm temperatures)
  oracle, charge-   that structure fitted at the cold temperature itself
  dependent

All methods use the measured cell temperature at each instant, so self-heating
during the cycle lowers the predicted resistance.

HFM's cells, cycles and score are private; numbers are NOT comparable.

Usage: python cold_voltage.py <dir-with-unzipped-LG-dataset>
"""
from __future__ import annotations

import glob
import io
import os
import sys

import numpy as np
import pandas as pd

TEMPS = {"40degC": 40, "25degC": 25, "10degC": 10, "0degC": 0, "n10degC": -10, "n20degC": -20}
CAL, COLD = ["40degC", "25degC", "10degC"], ["0degC", "n10degC", "n20degC"]
CYCLES = ("UDDS", "HWFET", "LA92", "US06", "Mixed1", "Mixed2", "Mixed3", "Mixed4", "Mixed5", "Mixed6", "Mixed7", "Mixed8")
TAUS = (4.0, 60.0)


def read(path):
    raw = open(path, encoding="latin-1").read().splitlines()
    h = next(i for i, l in enumerate(raw) if l.startswith("Time Stamp"))
    df = pd.read_csv(io.StringIO("\n".join([raw[h]] + raw[h + 2:])), usecols=["Prog Time", "Voltage", "Current", "Temperature", "Capacity"])
    t = pd.to_timedelta(df["Prog Time"]).dt.total_seconds().values
    out = pd.DataFrame(dict(t=t - t[0], v=df.Voltage.values, i=df.Current.values, temp=df.Temperature.values, ah=df.Capacity.values)).dropna()
    out = out[np.diff(out.t.values, prepend=-1) > 0]
    grid = np.arange(0, out.t.iloc[-1], 1.0)        # resample to 1 s
    return pd.DataFrame({c: np.interp(grid, out.t, out[c]) for c in ("v", "i", "temp", "ah")}).assign(t=grid)


def find(root, temp, name):
    # folders are named n10degC in the LG set and -10degC in the Samsung set
    hits = [f for t in (temp, temp.replace("n", "-")) for f in glob.glob(os.path.join(glob.escape(root), "**", t, f"*_{name}.csv"), recursive=True)]
    return hits[0] if hits else None


def ocv_curve(root):
    """Average of the C/20 discharge and charge branches at 25 degC, as a function of state of charge."""
    d = read(find(root, "25degC", "C20DisCh"))
    dis, chg = d[d.i < -0.05], d[d.i > 0.05]
    cap = dis.ah.max() - dis.ah.min()
    soc = np.linspace(0.02, 1.0, 200)
    s_dis = 1 - (dis.ah.iloc[0] - dis.ah) / cap
    s_chg = (chg.ah - chg.ah.iloc[0]) / (chg.ah.max() - chg.ah.iloc[0])
    v_dis = np.interp(soc, s_dis.values[::-1], dis.v.values[::-1])
    v_chg = np.interp(soc, s_chg.values, chg.v.values)
    return soc, 0.5 * (v_dis + v_chg), cap


def rc_response(i, r_scale, tau):
    """Voltage of one RC pair with resistance r_scale(t) (relative) and time constant tau, 1 s steps."""
    a = np.exp(-1.0 / tau)
    out = np.zeros(len(i))
    x = 0.0
    for k in range(len(i)):
        x = a * x + (1 - a) * i[k] * r_scale[k]
        out[k] = x
    return out


def features(d, soc_grid, ocv, cap, scale=None):
    """Columns multiply (R0, R1, R2). `scale` gives each resistance relative to 1 at every instant."""
    soc = 1 + (d.ah.values - d.ah.values[0]) / cap
    keep = soc > 0.1
    base = np.interp(soc, soc_grid, ocv)
    i = d.i.values
    if scale is None:
        scale = np.ones((3, len(i)))
    X = np.column_stack([i * scale[0], rc_response(i, scale[1], TAUS[0]), rc_response(i, scale[2], TAUS[1])])
    return X, d.v.values - base, keep


def main():
    root = sys.argv[1]
    soc_grid, ocv, cap = ocv_curve(root)
    print(f"capacity at 25 degC, C/20: {cap:.2f} Ah")
    data = {tn: {c: read(f) for c in CYCLES if (f := find(root, tn, c))} for tn in TEMPS}
    print("drive cycles per temperature: " + ", ".join(f"{TEMPS[t]} degC: {len(v)}" for t, v in data.items()))

    def fit(tn):
        Xs, ys = [], []
        for d in data[tn].values():
            X, y, keep = features(d, soc_grid, ocv, cap)
            Xs.append(X[keep]); ys.append(y[keep])
        r, *_ = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)
        return np.maximum(r, 1e-5)

    R = {tn: fit(tn) for tn in TEMPS}
    print("fitted resistances in milliohm (ohmic, 4 s pair, 60 s pair):")
    for tn in TEMPS:
        print(f"  {TEMPS[tn]:>4} degC: " + "  ".join(f"{1000 * x:6.1f}" for x in R[tn]))

    Tc = np.array([TEMPS[t] for t in CAL], float)
    Rc = np.array([R[t] for t in CAL])                       # (3 temps, 3 resistances)
    lin = [np.polyfit(Tc, Rc[:, j], 1) for j in range(3)]
    arr = [np.polyfit(1.0 / (Tc + 273.15), np.log(Rc[:, j]), 1) for j in range(3)]
    print("Arrhenius activation energies (kJ/mol): " + ", ".join(f"{8.314 * a[0] / 1000:.0f}" for a in arr))
    # the same form fitted only to the two coldest calibration temperatures (25 and 10 degC):
    # activation energies of battery processes are known to rise towards the cold, so the
    # trend at the cold edge of the calibrated range should extrapolate better than the average trend
    edge = [np.polyfit(1.0 / (Tc[1:] + 273.15), np.log(Rc[1:, j]), 1) for j in range(3)]
    print("cold-edge activation energies (kJ/mol): " + ", ".join(f"{8.314 * a[0] / 1000:.0f}" for a in edge))

    def r_of_t(method, j, temp):
        if method == "hold coldest":
            return np.full_like(temp, R["10degC"][j])
        if method == "linear slope":
            return np.maximum(np.polyval(lin[j], temp), R["10degC"][j] * 0.2)
        if method == "Arrhenius, cold edge":
            return np.exp(np.polyval(edge[j], 1.0 / (temp + 273.15)))
        return np.exp(np.polyval(arr[j], 1.0 / (temp + 273.15)))

    rows = []
    for tn in COLD:
        for cyc, d in data[tn].items():
            soc = 1 + (d.ah.values - d.ah.values[0]) / cap
            base = np.interp(soc, soc_grid, ocv)
            keep = soc > 0.1
            rec = dict(temp=TEMPS[tn], cycle=cyc)
            for method in ("hold coldest", "linear slope", "Arrhenius", "Arrhenius, cold edge"):
                rt = np.array([r_of_t(method, j, d.temp.values) for j in range(3)])
                X, _, _ = features(d, soc_grid, ocv, cap, scale=rt)   # resistances already included in the columns
                pred = base + X.sum(axis=1)
                rec[method] = 1000 * np.sqrt(np.mean((pred - d.v.values)[keep] ** 2))
            X, y, _ = features(d, soc_grid, ocv, cap)
            rec["oracle"] = 1000 * np.sqrt(np.mean((X @ R[tn] - y)[keep] ** 2))
            rows.append(rec)
    # --- second model structure: resistances that also depend on state of charge ---
    # R_j(T, soc) = R_j(T) * (1 + s_j1 z + s_j2 z^2), z = 1 - soc. The shape s is learned at the
    # calibration temperatures and assumed not to change with temperature; R_j(T) is Arrhenius.
    def soc_features(d, scale=None):
        soc = 1 + (d.ah.values - d.ah.values[0]) / cap
        z = 1 - soc
        i = d.i.values
        sc = np.ones((3, len(i))) if scale is None else scale
        cols = []
        for j, tau in enumerate((None,) + TAUS):
            for k in range(3):
                x = i * sc[j] * z ** k
                cols.append(x if tau is None else rc_response(x, np.ones(len(i)), tau))
        return np.column_stack(cols), d.v.values - np.interp(soc, soc_grid, ocv), soc > 0.1

    def fit_soc(tn):
        Xs, ys = zip(*[(X[k], y[k]) for X, y, k in (soc_features(d) for d in data[tn].values())])
        c, *_ = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)
        return c.reshape(3, 3)

    C = {tn: fit_soc(tn) for tn in TEMPS}
    base_cal = np.array([np.maximum(C[t][:, 0], 1e-5) for t in CAL])                      # (temps, 3)
    shape = np.median(np.array([C[t][:, 1:] / np.maximum(C[t][:, :1], 1e-5) for t in CAL]), axis=0)   # (3, 2)
    arr_soc = [np.polyfit(1.0 / (Tc + 273.15), np.log(base_cal[:, j]), 1) for j in range(3)]
    for rec, (tn, cyc) in zip(rows, [(tn, cyc) for tn in COLD for cyc in data[tn]]):
        d = data[tn][cyc]
        soc = 1 + (d.ah.values - d.ah.values[0]) / cap
        z = 1 - soc
        rt = np.array([np.exp(np.polyval(arr_soc[j], 1.0 / (d.temp.values + 273.15))) * (1 + shape[j, 0] * z + shape[j, 1] * z ** 2) for j in range(3)])
        X, _, _ = features(d, soc_grid, ocv, cap, scale=rt)
        keep = soc > 0.1
        rec["Arrhenius + charge-dependent"] = 1000 * np.sqrt(np.mean((np.interp(soc, soc_grid, ocv) + X.sum(axis=1) - d.v.values)[keep] ** 2))
        Xo, yo, _ = soc_features(d)
        rec["oracle, charge-dependent"] = 1000 * np.sqrt(np.mean((Xo @ C[tn].ravel() - yo)[keep] ** 2))
    r = pd.DataFrame(rows)
    methods = ["hold coldest", "linear slope", "Arrhenius", "Arrhenius, cold edge", "oracle", "Arrhenius + charge-dependent", "oracle, charge-dependent"]
    print("\nvoltage RMS error in millivolts on cold drive cycles (state of charge above 10%)")
    tab = r.groupby("temp")[methods].mean().sort_index(ascending=False)
    tab.loc["all"] = r[methods].mean()
    print(tab.round(1).to_string())
    print("\nskill against holding the 10 degC values (1 - error ratio), and cycles improved:")
    for m in methods[1:]:
        skill = 1 - r[m].mean() / r["hold coldest"].mean()
        print(f"  {m:<14} skill {skill:+.3f}   better on {(r[m] < r['hold coldest']).sum()}/{len(r)} cycles"
              + (f"   better than linear slope on {(r[m] < r['linear slope']).sum()}/{len(r)}" if m == "Arrhenius" else "")
              + (f"   better than three-point Arrhenius on {(r[m] < r['Arrhenius']).sum()}/{len(r)}" if m == "Arrhenius, cold edge" else ""))


if __name__ == "__main__":
    main()
