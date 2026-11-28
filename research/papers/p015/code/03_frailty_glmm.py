"""Frailty decomposition: random-effects (task + family) GLMMs on replicate runs.

cloglog link: task-conditional cumulative hazard Lambda_i(T) = exp(mu + k log(T/60) + sf v_f + st u_i).
k is the Weibull shape for a task of fixed difficulty (u=v=0); k_marg is the shape obtained when
heterogeneity is ignored (sf=st=0), as in pooled Weibull fits. Proposition 3 predicts k_marg <= k.
Also fits the logit (METR) link for the conditional vs marginal logistic slope.
Saves per-agent parameters, likelihood-ratio tests, pass@k validation and long-task projections.
"""
import numpy as np
import pandas as pd
from math import comb
from common import *

d = load_runs()
rows, passk, proj = [], [], []
LENS = {"8h (1 work-day)": 480.0, "40h (1 work-week)": 2400.0, "160h (1 work-month)": 9600.0}
rng = np.random.default_rng(3)
GHx, GHw = np.polynomial.hermite_e.hermegauss(60)
GHw = GHw / GHw.sum()


def pass_at_k_emp(n, s, k):
    return 1 - comb(n - s, k) / comb(n, k)


for alias, da in d.groupby("alias"):
    tt = task_table(da)
    out = dict(alias=alias, ntask=len(tt), nruns=int(tt.n.sum()))
    for link in ["cloglog", "logit"]:
        full = fit_glmm(tt, link, nested=True)
        task_only = fit_glmm(tt, link, nested=False)
        r0 = optimize.minimize(glmm_fixed_only_nll, full.x[:2], args=(tt, link), method="Nelder-Mead",
                               options=dict(maxiter=2000, xatol=1e-7, fatol=1e-8))
        pre = "k" if link == "cloglog" else "b"
        out[f"{pre}_cond_{link}"] = full.x[1]
        out[f"{pre}_cond_taskonly_{link}"] = task_only.x[1]
        out[f"{pre}_marg_{link}"] = r0.x[1]
        out[f"mu_{link}"] = full.x[0]
        out[f"sig_task_{link}"] = np.exp(full.x[2])
        out[f"sig_fam_{link}"] = np.exp(full.x[3])
        out[f"nll_full_{link}"] = full.fun
        out[f"nll_taskonly_{link}"] = task_only.fun
        out[f"nll_fixed_{link}"] = r0.fun
        out[f"LR_RE_vs_fixed_{link}"] = 2 * (r0.fun - full.fun)
        if link == "cloglog":
            th_full, th_fixed = full.x, r0.x
    rows.append(out)

    # pass@k validation on tasks with >= 4 runs (predictions use fitted parameters only)
    mu, k, lst, lsf = th_full
    sd = np.sqrt(np.exp(lst) ** 2 + np.exp(lsf) ** 2)
    t4 = tt[tt.n >= 4]
    lt = np.log(t4["T"].values / T_REF)
    p_nodes = np.exp(-np.exp(np.clip(mu + k * lt[:, None] + sd * GHx[None, :], -30, 6)))  # [i, node]
    f_hom = np.exp(-np.exp(np.clip(th_fixed[0] + th_fixed[1] * lt, -30, 6)))
    for kk in [1, 2, 3, 4]:
        emp = np.mean([pass_at_k_emp(n, s, kk) for n, s in zip(t4.n, t4.s)])
        pred_re = np.mean(((1 - (1 - p_nodes) ** kk) * GHw[None, :]).sum(1))
        pred_hom = np.mean(1 - (1 - f_hom) ** kk)
        passk.append(dict(alias=alias, k=kk, ntask=len(t4), emp=emp, pred_frailty=pred_re, pred_homog=pred_hom))

    # projections to multi-day lengths (extrapolation beyond the 30h maximum task length)
    for lab, T in LENS.items():
        p = task_success_draws(th_full, T, "cloglog", True, ndraw=200000, rng=rng)
        f_h = float(np.exp(-np.exp(th_fixed[0] + th_fixed[1] * np.log(T / T_REF))))
        proj.append(dict(alias=alias, length=lab, T=T,
                         P_marginal_frailty=p.mean(),
                         P_typical_task=float(np.exp(-np.exp(mu + k * np.log(T / T_REF)))),
                         frac_tasks_p_ge_0p8=np.mean(p >= 0.8), frac_tasks_p_ge_0p5=np.mean(p >= 0.5),
                         frac_tasks_p_le_0p05=np.mean(p <= 0.05),
                         pass8_frailty=np.mean(1 - (1 - p) ** 8),
                         P_homog_weibull=f_h, pass8_homog=1 - (1 - f_h) ** 8))
    print(alias, {kk: round(v, 3) for kk, v in out.items() if kk.startswith(("k_", "sig_", "LR"))}, flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/frailty_glmm.csv", index=False)
P = pd.DataFrame(passk)
P.to_csv(f"{RES}/passk_validation.csv", index=False)
J = pd.DataFrame(proj)
J.to_csv(f"{RES}/projections.csv", index=False)

with open(f"{RES}/frailty_summary.txt", "w") as f:
    c = R[["alias", "k_marg_cloglog", "k_cond_taskonly_cloglog", "k_cond_cloglog", "sig_task_cloglog",
           "sig_fam_cloglog", "LR_RE_vs_fixed_cloglog", "b_marg_logit", "b_cond_logit"]]
    f.write(c.round(3).to_string(index=False) + "\n\n")
    f.write(f"median k_marg {R.k_marg_cloglog.median():.3f}; median k_cond {R.k_cond_cloglog.median():.3f}; "
            f"agents with k_cond > k_marg: {(R.k_cond_cloglog > R.k_marg_cloglog).sum()}/{len(R)}\n")
    f.write(f"median sigma_total {np.sqrt(R.sig_task_cloglog**2 + R.sig_fam_cloglog**2).median():.3f}\n")
    f.write(f"median LR stat RE vs fixed {R.LR_RE_vs_fixed_cloglog.median():.1f} (min {R.LR_RE_vs_fixed_cloglog.min():.1f})\n\n")
    P["err_frailty"] = P.pred_frailty - P.emp
    P["err_homog"] = P.pred_homog - P.emp
    f.write("pass@k validation: mean signed and abs error across agents\n")
    f.write(P.groupby("k")[["emp", "pred_frailty", "pred_homog", "err_frailty", "err_homog"]].mean().round(4).to_string() + "\n")
    f.write(P.assign(ae_f=P.err_frailty.abs(), ae_h=P.err_homog.abs()).groupby("k")[["ae_f", "ae_h"]].mean().round(4).to_string() + "\n\n")
    top = ["Claude Opus 4.6 (Inspect)", "GPT-5.3-Codex", "GPT-5.2", "Claude Opus 4.5 (Inspect)", "Gemini 3 Pro"]
    f.write("Projections for most capable agents (extrapolated beyond 30h max task length)\n")
    f.write(J[J.alias.isin(top)].round(3).to_string(index=False) + "\n")
print(open(f"{RES}/frailty_summary.txt").read())
