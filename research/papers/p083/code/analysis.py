"""Main experiments: carbon/water co-variation, temporal shifting, Pareto frontiers,
test of the linear side-effect prediction, and accounting bias.

Run: python3 code/analysis.py   (after code/load_data.py)
Outputs in results/: *.csv, *.json, fig_*.png
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ARCH = {"tower": "wue_tower", "adiab": "wue_adiab"}


def daily_matrix(df, col):
    """Return (days x 24) matrix of col on local-date / local-hour grid (complete days only)."""
    m = df.pivot_table(index="local_date", columns="local_hour", values=col, aggfunc="mean")
    return m


def signals(df, arch, ci="ci_prod", ewif="ewif"):
    """Per-IT-kWh carbon C (g) and water W (L), with onsite/offsite split."""
    C = df["pue"] * df[ci]
    Won = df[ARCH[arch]]
    Woff = df["pue"] * df[ewif]
    return C, Won, Woff


def to_mats(df, series_dict):
    tmp = df[["local_date", "local_hour"]].copy()
    mats = {}
    for k, s in series_dict.items():
        tmp[k] = s.values
    piv = {k: tmp.pivot_table(index="local_date", columns="local_hour", values=k, aggfunc="mean")
           for k in series_dict}
    ok = np.ones(len(piv[list(piv)[0]]), bool)
    for k in piv:
        ok &= piv[k].notna().all(axis=1).values & (piv[k].shape[1] == 24)
    for k in piv:
        mats[k] = piv[k].values[ok]
    return mats


def greedy_schedule(score, phi, window=24):
    """Box-constrained LP per window: x = 1-phi + y, sum y = window*phi, 0<=y<=2*phi.
    Optimal: put y=2*phi on the window/2 lowest-score hours (Prop. 3)."""
    D, H = score.shape
    s = score.reshape(-1, window) if window != H else score
    order = np.argsort(s, axis=1, kind="stable")
    y = np.zeros_like(s)
    k = window // 2
    rows = np.arange(s.shape[0])[:, None]
    y[rows, order[:, :k]] = 2 * phi
    x = 1 - phi + y
    return x.reshape(D, H)


def evaluate(x, C, Won, Woff):
    return dict(C=(x * C).sum(), Won=(x * Won).sum(), Woff=(x * Woff).sum(), W=(x * (Won + Woff)).sum())


def frontier(C, Won, Woff, phi, lams, window=24):
    W = Won + Woff
    base = evaluate(np.ones_like(C), C, Won, Woff)
    Cn, Wn = C / C.mean(), W / W.mean()
    rows = []
    for lam in lams:
        x = greedy_schedule(lam * Cn + (1 - lam) * Wn, phi, window)
        r = evaluate(x, C, Won, Woff)
        rows.append(dict(lam=lam, dC=100 * (r["C"] / base["C"] - 1), dW=100 * (r["W"] / base["W"] - 1),
                         dWon=100 * (r["Won"] - base["Won"]) / base["W"],
                         dWoff=100 * (r["Woff"] - base["Woff"]) / base["W"]))
    return pd.DataFrame(rows), base


def within_day_stats(C, W):
    cs = C - C.mean(1, keepdims=True)
    ws = W - W.mean(1, keepdims=True)
    cov = (cs * ws).mean(1)
    rho = cov / (cs.std(1) * ws.std(1) + 1e-12)
    b = cov / (cs.var(1) + 1e-12)
    return rho, b, cov


def run_site(df, arch, phi=0.3, ci="ci_prod", ewif="ewif", lams=None):
    C, Won, Woff = signals(df, arch, ci, ewif)
    m = to_mats(df, dict(C=C, Won=Won, Woff=Woff))
    C, Won, Woff = m["C"], m["Won"], m["Woff"]
    W = Won + Woff
    if lams is None:
        lams = np.linspace(0, 1, 101)
    fr, base = frontier(C, Won, Woff, phi, lams)
    rho, b, cov = within_day_stats(C, W)
    rho_on, _, cov_on = within_day_stats(C, Won)
    rho_off, _, cov_off = within_day_stats(C, Woff)
    # Prop. 2 check: predicted dW = sum_d b_d dC_d for carbon-only schedule
    xC = greedy_schedule(C, phi)
    dCd = ((xC - 1) * C).sum(1)
    dWd = ((xC - 1) * W).sum(1)
    pred = (b * dCd).sum()
    real = dWd.sum()
    # Knee: largest lambda (most carbon weight) with dW <= 0
    c_only = fr.iloc[-1]
    feas = fr[fr.dW <= 0]
    knee = feas.iloc[-1] if len(feas) else None
    out = dict(
        days=int(C.shape[0]),
        rho_CW=float(np.nanmean(rho)), frac_days_neg=float(np.mean(rho < 0)),
        rho_C_Won=float(np.nanmean(rho_on)), rho_C_Woff=float(np.nanmean(rho_off)),
        cov_on=float(cov_on.mean()), cov_off=float(cov_off.mean()),
        onsite_share=float(Won.sum() / W.sum()),
        dC_carbon_only=float(c_only.dC), dW_carbon_only=float(c_only.dW),
        dWon_carbon_only=float(c_only.dWon), dWoff_carbon_only=float(c_only.dWoff),
        dWon_rel_carbon_only=float(c_only.dWon / (Won.sum() / W.sum())),
        dC_water_only=float(fr.iloc[0].dC), dW_water_only=float(fr.iloc[0].dW),
        knee_lam=float(knee.lam) if knee is not None else np.nan,
        knee_dC=float(knee.dC) if knee is not None else np.nan,
        knee_dW=float(knee.dW) if knee is not None else np.nan,
        pred_dW_pct=float(100 * pred / (W.sum())), real_dW_pct=float(100 * real / (W.sum())),
    )
    return out, fr


def accounting_bias(df, arch, amp=0.3, peak=15, ci="ci_prod"):
    """Relative error of annual-average-factor accounting for a diurnal load profile.
    Prop. 1: bias = rho(x,w) * CV_x * CV_w (population moments over all hours)."""
    C, Won, Woff = signals(df, arch, ci)
    h = df["local_hour"].values
    x = 1 + amp * np.cos(2 * np.pi * (h - peak) / 24.0)
    res = {}
    for name, w in [("C", C.values), ("Won", Won.values), ("W", (Won + Woff).values)]:
        true = (x * w).sum()
        approx = x.sum() * w.mean()
        bias = 100 * (approx / true - 1)  # % error of the average-factor estimate vs hourly truth
        rho = np.corrcoef(x, w)[0, 1]
        theory = rho * x.std() / x.mean() * w.std() / w.mean()  # (true-approx)/approx
        res[f"bias_{name}_pct"] = float(bias)
        res[f"theory_{name}_pct"] = float(100 * (1 / (1 + theory) - 1))
    # monthly-average accounting for onsite water
    d = pd.DataFrame(dict(x=x, w=Won.values, mth=df.index.month))
    mm = d.groupby("mth").apply(lambda g: g.x.sum() * g.w.mean(), include_groups=False).sum()
    res["bias_Won_monthly_pct"] = float(100 * (mm / (d.x * d.w).sum() - 1))
    return res


def main():
    p = pd.read_pickle(os.path.join(RES, "panel_2023.pkl"))
    sites = list(dict.fromkeys(p.site))
    rows, fronts, acc = [], [], []
    for arch in ARCH:
        for s in sites:
            df = p[p.site == s]
            for ci in ["ci_prod", "ci_cons"]:
                for phi in ([0.1, 0.3, 0.5] if ci == "ci_prod" else [0.3]):
                    o, fr = run_site(df, arch, phi=phi, ci=ci)
                    o.update(site=s, arch=arch, ci=ci, phi=phi)
                    rows.append(o)
                    if ci == "ci_prod" and phi == 0.3:
                        fr["site"], fr["arch"] = s, arch
                        fronts.append(fr)
            a = accounting_bias(df, arch)
            a.update(site=s, arch=arch)
            acc.append(a)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "shifting_summary.csv"), index=False)
    pd.concat(fronts).to_csv(os.path.join(RES, "frontiers.csv"), index=False)
    A = pd.DataFrame(acc)
    A.to_csv(os.path.join(RES, "accounting_bias.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["site", "arch", "dWon_rel_carbon_only", "rho_CW", "frac_days_neg", "rho_C_Won", "rho_C_Woff", "onsite_share",
            "dC_carbon_only", "dW_carbon_only", "dWon_carbon_only", "dWoff_carbon_only", "dC_water_only",
            "dW_water_only", "knee_lam", "knee_dC", "pred_dW_pct", "real_dW_pct"]
    print(R[(R.ci == "ci_prod") & (R.phi == 0.3)][cols].round(3).to_string())
    print(R[R.ci == "ci_cons"][["site", "arch", "rho_CW", "dC_carbon_only", "dW_carbon_only"]].round(3).to_string())
    print(A.round(3).to_string())


if __name__ == "__main__":
    main()
