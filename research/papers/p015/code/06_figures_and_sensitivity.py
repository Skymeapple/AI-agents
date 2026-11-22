"""Main-text figures, Wald CIs for k_cond, capability trend, and projection sensitivity."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, special
from common import *

# re-use latent-class likelihood without re-running the script
src = open(f"{HERE}/04_robustness.py").read().split("rows = []")[0]
ns = {}
exec(src, ns)
nll_latent, best_of = ns["nll_latent"], ns["best_of"]

d = load_runs()
G = pd.read_csv(f"{RES}/frailty_glmm.csv")
M = pd.read_csv(f"{RES}/marginal_fits.csv")
P = pd.read_csv(f"{RES}/passk_validation.csv")


def num_hess(f, x, h=1e-3):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return H


rows, sens = [], []
for alias, da in d.groupby("alias"):
    tt = task_table(da)
    r = fit_glmm(tt, "cloglog", nested=True)
    H = num_hess(lambda th: glmm_nll(th, tt, "cloglog", True), r.x)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    h50 = M[(M.alias == alias) & (M.family == "log-logistic (METR)")].h50.iloc[0]
    rows.append(dict(alias=alias, k_cond=r.x[1], k_se=se[1], h50_marg=h50,
                     sigma_total=np.sqrt(np.exp(2 * r.x[2]) + np.exp(2 * r.x[3]))))
    # projection sensitivity at 40h for frontier agents
    if h50 > 200:
        rt = fit_glmm(tt, "cloglog", nested=False)
        sd = np.exp(rt.x[2])
        lc = best_of(nll_latent, [[rt.x[0] - 1.5 * sd, rt.x[0], rt.x[0] + 1.5 * sd, rt.x[1], 0, 0],
                                  [rt.x[0] - 3, rt.x[0] - 0.5, rt.x[0] + 3, 1.0, 0.5, -0.5]], (tt,))
        mus, kk = lc.x[:3], lc.x[3]
        pis = special.softmax(np.r_[0.0, lc.x[4:]])
        for T in [480.0, 2400.0, 9600.0]:
            lt = np.log(T / T_REF)
            p_g = task_success_draws(r.x, T, "cloglog", True, 200000)
            p_l = np.exp(-np.exp(np.clip(mus + kk * lt, -30, 6)))
            row = dict(alias=alias, T=T, frailty_gaussian=p_g.mean(), frailty_latent3=float(pis @ p_l),
                       latent3_frac_p_ge_0p8=float(pis[p_l >= 0.8].sum()))
            for fam in ["log-logistic (METR)", "Weibull", "exponential", "CM mixture (NPMLE)"]:
                col = {480.0: "P_8h", 2400.0: "P_40h", 9600.0: "P_160h"}[T]
                row["pooled_" + fam.split(" ")[0]] = M[(M.alias == alias) & (M.family == fam)][col].iloc[0]
            sens.append(row)
    print(alias, flush=True)

K = pd.DataFrame(rows)
K.to_csv(f"{RES}/kcond_with_se.csv", index=False)
SN = pd.DataFrame(sens)
SN.to_csv(f"{RES}/projection_sensitivity.csv", index=False)
c = np.corrcoef(np.log(K.h50_marg), K.k_cond)[0, 1]
c2 = np.corrcoef(np.log(K.h50_marg), np.log(K.sigma_total))[0, 1]
with open(f"{RES}/trend_and_sensitivity.txt", "w") as f:
    f.write(K.round(3).to_string(index=False) + "\n\n")
    f.write(f"corr(log h50, k_cond) = {c:.3f}; corr(log h50, log sigma_total) = {c2:.3f}; n={len(K)}\n")
    f.write(f"agents with k_cond - 1.96 se > 1: {(K.k_cond - 1.96 * K.k_se > 1).sum()}/{len(K)}\n\n")
    f.write("Projection sensitivity (P of success on a random task of length T from the METR-like task population)\n")
    f.write(SN.round(3).to_string(index=False) + "\n")
print(open(f"{RES}/trend_and_sensitivity.txt").read())

# Fig: k_marg vs k_cond
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
Kx = K.merge(G[["alias", "k_marg_cloglog"]], on="alias").sort_values("h50_marg")
y = np.arange(len(Kx))
ax[0].errorbar(Kx.k_cond, y, xerr=1.96 * Kx.k_se, fmt="o", color="#c0392b", ms=4, label="conditional (typical task)")
ax[0].plot(Kx.k_marg_cloglog, y, "s", color="#4a6fa5", ms=4, label="pooled (ignores heterogeneity)")
ax[0].axvline(1, color="k", lw=0.7, ls="--")
ax[0].set_yticks(y)
ax[0].set_yticklabels([a.replace(" (Inspect)", "") for a in Kx.alias], fontsize=7)
ax[0].set_xlabel("Weibull shape k  (k<1: declining hazard)")
ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[0].set_title("(a) pooled vs task-conditional shape")
# pass@k
Pm = P.groupby("k")[["emp", "pred_frailty", "pred_homog"]].mean()
ax[1].plot(Pm.index, Pm.emp, "ko-", label="observed (unbiased pass@k)")
ax[1].plot(Pm.index, Pm.pred_frailty, "o--", color="#c0392b", label="frailty model")
ax[1].plot(Pm.index, Pm.pred_homog, "s--", color="#4a6fa5", label="homogeneous-task model")
ax[1].set_xlabel("k (attempts)")
ax[1].set_ylabel("pass@k, mean over 20 agents")
ax[1].set_xticks([1, 2, 3, 4])
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) retries: frailty predicts, homogeneity overshoots")
# distribution of task-level p for best agent at 8h/40h/160h
J = pd.read_csv(f"{RES}/projections.csv")
best = "GPT-5.3-Codex"
th = fit_glmm(task_table(d[d.alias == best]), "cloglog", True).x
bins = np.linspace(0, 1, 21)
for T, ccol, lab in [(480, "#2e8b57", "8 h"), (2400, "#4a6fa5", "40 h"), (9600, "#c0392b", "160 h")]:
    p = task_success_draws(th, T, "cloglog", True, 100000)
    ax[2].hist(p, bins=bins, histtype="step", lw=2, color=ccol, density=True, label=f"{lab}: mean {p.mean():.2f}")
ax[2].set_yscale("log")
ax[2].set_xlabel("task-level success probability p_i")
ax[2].set_ylabel("density (log)")
ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title(f"(c) {best}: extrapolated p_i distribution")
plt.tight_layout()
plt.savefig(f"{RES}/fig_main.png", dpi=160)

# Fig: trend
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].errorbar(K.h50_marg, K.k_cond, yerr=1.96 * K.k_se, fmt="o", color="#c0392b", ms=4)
ax[0].set_xscale("log")
ax[0].axhline(1, color="k", ls="--", lw=0.7)
ax[0].set_xlabel("pooled 50% horizon (min)")
ax[0].set_ylabel("conditional Weibull shape k")
ax[1].plot(K.h50_marg, K.sigma_total, "o", color="#4a6fa5")
ax[1].set_xscale("log")
ax[1].set_xlabel("pooled 50% horizon (min)")
ax[1].set_ylabel("frailty s.d. (log-hazard units)")
plt.tight_layout()
plt.savefig(f"{RES}/fig_trend.png", dpi=160)
