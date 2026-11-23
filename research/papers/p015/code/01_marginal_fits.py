"""Marginal (pooled-over-tasks) survival-family comparison on METR TH1.1.

For each agent: fit log-logistic (METR), Weibull, exponential, Lomax, log-normal and the
nonparametric completely-monotone class (mixtures of constant hazards), using equal task
weights. Report in-sample NLL, 5-fold task-grouped CV NLL, a short->long extrapolation test,
and implied 50/80/99% horizons.
"""
import sys
import numpy as np
import pandas as pd
from common import *

drop = "--drop-usage" in sys.argv
d = load_runs(drop_usage_limit=drop)
tag = "_dropusage" if drop else ""
rng = np.random.default_rng(1)
rows, extrap = [], []
names = list(FAMILIES) + ["CM mixture (NPMLE)"]

for alias, da in d.groupby("alias"):
    da = da.copy()
    n_i = da.groupby("task_id").y.transform("size")
    w = (1.0 / n_i).values
    T, y = da["T"].values, da["y"].values.astype(float)
    ntask = da.task_id.nunique()
    # in-sample
    fits = {}
    for nm in FAMILIES:
        th, nll = fit_family(nm, T, y, w)
        fits[nm] = th
    pi = fit_cm_mixture(T, y, w)
    # CV (task-grouped)
    tasks = da.task_id.unique()
    rng.shuffle(tasks)
    fold = {t: i % 5 for i, t in enumerate(tasks)}
    fid = da.task_id.map(fold).values
    cv = {nm: 0.0 for nm in names}
    for k in range(5):
        tr, te = fid != k, fid == k
        for nm in FAMILIES:
            th, _ = fit_family(nm, T[tr], y[tr], w[tr])
            cv[nm] += wnll(FAMILIES[nm][0](th, T[te]), y[te], w[te])
        p_ = fit_cm_mixture(T[tr], y[tr], w[tr], iters=1500)
        cv["CM mixture (NPMLE)"] += wnll(S_cm(p_, T[te]), y[te], w[te])
    for nm in names:
        if nm in FAMILIES:
            f = FAMILIES[nm][0]
            th = fits[nm]
            S = lambda TT, f=f, th=th: f(th, TT)
        else:
            S = lambda TT: S_cm(pi, TT)
        ins = wnll(S(T), y, w)
        r = dict(alias=alias, family=nm, ntask=ntask, nll_per_task=ins / ntask,
                 cv_nll_per_task=cv[nm] / ntask,
                 h50=horizon(S, 0.5), h80=horizon(S, 0.8), h99=horizon(S, 0.99),
                 P_8h=float(S(np.array([480.0]))[0]), P_40h=float(S(np.array([2400.0]))[0]),
                 P_160h=float(S(np.array([9600.0]))[0]))
        if nm == "Weibull":
            r["shape"] = float(np.exp(fits[nm][1]))
        if nm == "log-logistic (METR)":
            r["shape"] = float(np.exp(fits[nm][1]))  # beta in natural-log units
        if nm == "Lomax (gamma frailty)":
            r["shape"] = float(np.exp(fits[nm][1]))  # tail index a
        rows.append(r)
    # extrapolation: train on T < 120 min, test on T >= 240 min
    tr, te = T < 120, T >= 240
    if te.sum() > 0 and y[te].sum() > 0:
        obs = np.sum(w[te] * y[te]) / np.sum(w[te])
        for nm in names:
            if nm in FAMILIES:
                th, _ = fit_family(nm, T[tr], y[tr], w[tr])
                Ste = FAMILIES[nm][0](th, T[te])
            else:
                p_ = fit_cm_mixture(T[tr], y[tr], w[tr])
                Ste = S_cm(p_, T[te])
            extrap.append(dict(alias=alias, family=nm, n_test_tasks=len(np.unique(da.task_id.values[te])),
                               obs_mean=obs, pred_mean=np.sum(w[te] * Ste) / np.sum(w[te]),
                               test_nll_per_task=wnll(Ste, y[te], w[te]) / np.sum(w[te])))
    print(alias, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/marginal_fits{tag}.csv", index=False)
E = pd.DataFrame(extrap)
E.to_csv(f"{RES}/extrapolation{tag}.csv", index=False)

# summary tables
summ = R.groupby("family")[["nll_per_task", "cv_nll_per_task"]].mean()
wins = R.loc[R.groupby("alias").cv_nll_per_task.idxmin()].family.value_counts()
summ["cv_wins"] = wins.reindex(summ.index).fillna(0).astype(int)
es = E.groupby("family").apply(lambda g: pd.Series(dict(
    mean_test_nll=g.test_nll_per_task.mean(),
    mean_abs_err=np.mean(np.abs(g.pred_mean - g.obs_mean)),
    mean_signed_err=np.mean(g.pred_mean - g.obs_mean), n_agents=len(g))))
with open(f"{RES}/marginal_summary{tag}.txt", "w") as f:
    f.write("Mean per-task NLL across agents (equal task weights)\n")
    f.write(summ.to_string() + "\n\nExtrapolation (train T<120min, test T>=240min)\n")
    f.write(es.to_string() + "\n\n")
    sh = R[R.family.isin(["Weibull", "log-logistic (METR)", "Lomax (gamma frailty)"])].pivot(
        index="alias", columns="family", values="shape")
    f.write("Fitted shape parameters per agent\n" + sh.to_string() + "\n\n")
    f.write("Median ratios h80/h50 and h99/h50 by family\n")
    R["r80"] = R.h80 / R.h50
    R["r99"] = R.h99 / R.h50
    f.write(R.groupby("family")[["r80", "r99"]].median().to_string() + "\n")
print(open(f"{RES}/marginal_summary{tag}.txt").read())
