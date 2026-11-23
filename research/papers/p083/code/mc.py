"""Monte Carlo sensitivity of the water side effect of carbon-only temporal shifting.

For each draw we resample: per-fuel water factors (uniform between once-through
median and cooling-tower max, Macknick et al. 2011), hydro factor in {0,17,68} L/kWh,
OTH/UNK factor U[0,3.8], carbon signal (production vs consumption based), PUE
temperature slope, cooling-model parameters, flexible fraction phi and window length.
Run: python3 code/mc.py  -> results/mc_draws.csv, results/mc_summary.csv
"""
import os
import numpy as np
import pandas as pd

from factors import EWIF_LOW, EWIF_HIGH, HYDRO_ALT
from cooling import pue_t, wue_tower, wue_adiabatic
from analysis import to_mats, greedy_schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FUELS = ["COL", "GAS", "NUC", "OIL", "OTH", "UNK", "SUN", "WND", "WAT"]
N_DRAWS = 300


def draw(rng):
    ew = {f: rng.uniform(EWIF_LOW[f], EWIF_HIGH[f]) for f in FUELS}
    ew["UNK"] = ew["OTH"]
    ew["WAT"] = HYDRO_ALT[rng.integers(3)]
    return dict(ew=ew, ci=["ci_prod", "ci_cons"][rng.integers(2)], k=rng.uniform(0.004, 0.012),
                coc=rng.uniform(3, 6), eps_slope=rng.uniform(0.008, 0.016),
                t_set=rng.uniform(21, 27), eff=rng.uniform(0.7, 0.9), dT=rng.uniform(10, 15),
                phi=rng.uniform(0.1, 0.5), window=[24, 24, 72][rng.integers(3)])


def main():
    p = pd.read_pickle(os.path.join(RES, "panel_2023.pkl"))
    rng = np.random.default_rng(20261122)
    sites = list(dict.fromkeys(p.site))
    groups = {s: p[p.site == s] for s in sites}
    rows = []
    for i in range(N_DRAWS):
        d = draw(rng)
        for s in sites:
            df = groups[s]
            ewif = sum(df[f"sh_{f}"] * d["ew"][f] for f in FUELS)
            pue = pue_t(df["Twb"], k=d["k"])
            C = pue * df[d["ci"]]
            Woff = pue * ewif
            for arch in ["tower", "adiab"]:
                if arch == "tower":
                    Won = wue_tower(df["T"], pue, coc=d["coc"], eps_slope=d["eps_slope"])
                else:
                    Won = wue_adiabatic(df["T"], df["Twb"], t_set=d["t_set"], eff=d["eff"], dT_srv=d["dT"])
                m = to_mats(df, dict(C=C, Won=Won, Woff=Woff))
                Cm, On, Of = m["C"], m["Won"], m["Woff"]
                w = d["window"]
                nd = (Cm.shape[0] // (w // 24)) * (w // 24)
                Cm, On, Of = Cm[:nd], On[:nd], Of[:nd]
                if w != 24:
                    Cm, On, Of = (a.reshape(-1, w) for a in (Cm, On, Of))
                x = greedy_schedule(Cm, d["phi"], window=w)
                W = On + Of
                xs = [x]
                # water-neutral joint schedule: largest carbon weight with dW<=0
                Cn, Wn = Cm / Cm.mean(), W / W.mean()
                best = None
                for lam in np.linspace(1, 0, 41):
                    xl = greedy_schedule(lam * Cn + (1 - lam) * Wn, d["phi"], window=w)
                    if ((xl - 1) * W).sum() <= 0:
                        best = xl
                        break
                dC = ((x - 1) * Cm).sum() / Cm.sum()
                rows.append(dict(draw=i, site=s, arch=arch, ci=d["ci"], hydro=d["ew"]["WAT"], phi=d["phi"],
                                 window=w, dC=100 * dC,
                                 dW=100 * ((x - 1) * W).sum() / W.sum(),
                                 dWon_rel=100 * ((x - 1) * On).sum() / On.sum(),
                                 onsite_share=On.sum() / W.sum(),
                                 retained=(((best - 1) * Cm).sum() / Cm.sum()) / dC if best is not None and dC < 0 else np.nan))
        if i % 50 == 0:
            print("draw", i, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "mc_draws.csv"), index=False)
    S = D.groupby(["site", "arch"]).agg(
        P_dW_pos=("dW", lambda v: np.mean(v > 0)), dW_med=("dW", "median"),
        dW_p05=("dW", lambda v: np.quantile(v, 0.05)), dW_p95=("dW", lambda v: np.quantile(v, 0.95)),
        P_dWon_pos=("dWon_rel", lambda v: np.mean(v > 0)), dWon_med=("dWon_rel", "median"),
        dC_med=("dC", "median"), retained_med=("retained", "median"),
        retained_p05=("retained", lambda v: np.nanquantile(v, 0.05))).reset_index()
    S.to_csv(os.path.join(RES, "mc_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    # hydro split
    print(D.groupby(["hydro", "arch"]).dW.apply(lambda v: np.mean(v > 0)).round(3))


if __name__ == "__main__":
    main()
