"""Descriptive test against realised occupational employment change (BLS OEWS).

May-2021 and May-2024 OEWS estimates are built from disjoint sets of semi-annual survey
panels (each estimate pools six panels over three years), so 2021->2024 is the longest
panel-disjoint window available here. Annual changes (overlapping panels) are reported as
secondary descriptives only. Nothing here is causal.
"""
import os, pickle, json
import numpy as np, pandas as pd
from lib import R, rule_table

E = pd.read_csv(os.path.join(R, "emp_national.csv"), dtype={"soc": str}).set_index("soc")
E = E[E["has_tasks"]]
stats = pickle.load(open(os.path.join(R, "occ_stats.pkl"), "rb"))


def wls(y, X, w):
    """Weighted least squares with HC1 robust standard errors."""
    W = w / w.mean()
    Xw = X * np.sqrt(W)[:, None]; yw = y * np.sqrt(W)
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ Xw.T @ yw
    e = yw - Xw @ b
    n, k = X.shape
    meat = (Xw * e[:, None]).T @ (Xw * e[:, None])
    V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    r2 = 1 - (e ** 2).sum() / ((yw - np.average(y, weights=W) * np.sqrt(W)) ** 2).sum()
    return b, np.sqrt(np.diag(V)), r2


def design(df, cols, fe=True):
    X = [df[c].values for c in cols]
    names = list(cols)
    if fe:
        maj = df.index.str[:2]
        for m in sorted(set(maj))[1:]:
            X.append((maj == m).astype(float)); names.append("fe" + m)
    X.append(np.ones(len(df))); names.append("const")
    return np.column_stack(X), names


def within_r2(y, x, w, df):
    """R^2 of x after partialling out major-group FE (Frisch-Waugh), weighted."""
    Xf, _ = design(df, [], fe=True)
    def resid(v):
        b, _, _ = wls(v, Xf, w); return v - Xf @ b
    ry, rx = resid(y), resid(x)
    if np.allclose(rx, 0): return np.nan, np.nan
    b = np.sum(w * rx * ry) / np.sum(w * rx * rx)
    e = ry - b * rx
    return 1 - np.sum(w * e ** 2) / np.sum(w * ry ** 2), b


res = {}
periods = {"2021_2024": (2021, 2024), "2021_2022": (2021, 2022), "2022_2023": (2022, 2023), "2023_2024": (2023, 2024)}
rows = []
for meas in ["hum_beta", "gpt_beta"]:
    st = stats[(meas, "im")]
    for pname, (y0, y1) in periods.items():
        df = E[[f"emp{y0}", f"emp{y1}"]].dropna()
        df = df[(df[f"emp{y0}"] > 0) & (df[f"emp{y1}"] > 0)].join(st, how="inner")
        y = np.log(df[f"emp{y1}"] / df[f"emp{y0}"]).values
        w = df[f"emp{y0}"].values
        # standardise regressors (employment-weighted) for comparability
        for c in ["s", "a", "g", "prod", "bott", "sd"]:
            mu = np.average(df[c], weights=w); sdv = np.sqrt(np.average((df[c] - mu) ** 2, weights=w))
            df[c + "_z"] = (df[c] - mu) / sdv
        specs = {"M1_s_noFE": (["s_z"], False), "M2_s_FE": (["s_z"], True),
                 "M3_auto_aug_FE": (["a_z", "g_z"], True), "M4_s_bottleneck_FE": (["s_z", "bott_z"], True),
                 "M5_s_sd_FE": (["s_z", "sd_z"], True), "M6_oring_FE": (["prod_z"], True)}
        for sname, (cols, fe) in specs.items():
            X, names = design(df, cols, fe)
            b, se, r2 = wls(y, X, w)
            for c in cols:
                i = names.index(c)
                rows.append(dict(measure=meas, period=pname, spec=sname, var=c, coef=b[i], se=se[i],
                                 t=b[i] / se[i], r2=r2, n=len(df)))
reg = pd.DataFrame(rows)
reg.to_csv(os.path.join(R, "regressions.csv"), index=False)
print(reg.round(4).to_string())

# horse race of aggregation rules: within-major-group R^2 of each rule's predicted displacement
hr = []
for meas in ["hum_beta", "gpt_beta"]:
    st = stats[(meas, "im")]
    df = E[["emp2021", "emp2024"]].dropna()
    df = df[(df.emp2021 > 0) & (df.emp2024 > 0)].join(st, how="inner")
    y = np.log(df.emp2024 / df.emp2021).values; w = df.emp2021.values
    for name, fam, f in rule_table():
        d = np.asarray(f(df), float)
        if np.std(d) < 1e-12:
            continue
        r2, b = within_r2(y, d, w, df)
        hr.append(dict(measure=meas, rule=name, family=fam, within_r2=r2, slope=b))
hr = pd.DataFrame(hr).sort_values(["measure", "within_r2"], ascending=[True, False])
hr.to_csv(os.path.join(R, "rule_horse_race.csv"), index=False)
print(hr.groupby(["measure", "family"])["within_r2"].agg(["max", "median"]).round(4))
print(hr.groupby("measure").head(8).round(4).to_string())
