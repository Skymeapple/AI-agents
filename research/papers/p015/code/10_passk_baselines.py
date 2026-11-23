"""Held-out-task pass@m test against stronger baselines (addresses review r1), plus a nonparametric
check of the sharp H-brackets of Proposition 4.

Task folds are shared across agents (5-fold, task-grouped).  Models fitted on training tasks:
  homog      : pooled Weibull, independent retries (the original straw-man baseline)
  bb         : beta-binomial, no length dependence
  bb_T       : beta-binomial, logit mean linear in log T, constant overdispersion
  gauss      : Gaussian frailty, cloglog (the paper's model)
  logitnorm  : Gaussian frailty, logit link
  latent3    : 3-class latent frailty, cloglog
  crossed_T  : crossed agent x task model; prediction uses only T (integrates shared + agent effects)
  crossed_X  : crossed model; prediction also uses the OTHER 19 agents' outcomes on the held-out task
               (an IRT-style information set the single-agent models do not have)
"""
import numpy as np
import pandas as pd
import torch
from math import comb
from scipy import optimize, special
from common import *
from crossed import Crossed, build_arrays

MS = [1, 2, 3, 4]
d = load_runs()
agents, tasks, N, S, T = build_arrays(d)
LT = np.log(T / T_REF)
fam = d.groupby("task_id").task_family.first().reindex(tasks).values
rng = np.random.default_rng(7)
fold = np.empty(len(tasks), int)
fold[rng.permutation(len(tasks))] = np.arange(len(tasks)) % 5
xg, wg = np.polynomial.hermite_e.hermegauss(40)
wg = wg / wg.sum()


def bb_nll(th, s, n, lt, use_T):
    if use_T:
        a, b, lr = th
        m = special.expit(a - b * lt)
    else:
        a, lr = th
        m = np.full(len(s), special.expit(a))
    rho = special.expit(lr)
    al, be = m * (1 - rho) / rho, (1 - m) * (1 - rho) / rho
    return -np.sum(special.betaln(s + al, n - s + be) - special.betaln(al, be))


def bb_passm(th, lt, m_, use_T):
    if use_T:
        a, b, lr = th
        m = special.expit(a - b * lt)
    else:
        a, lr = th
        m = np.full(len(lt), special.expit(a))
    rho = special.expit(lr)
    al, be = m * (1 - rho) / rho, (1 - m) * (1 - rho) / rho
    return 1 - np.exp(special.betaln(al, be + m_) - special.betaln(al, be))


def nll_latent(th, lt, s, n, K=3):
    mus, k = th[:K], th[K]
    logits = np.concatenate([[0.0], th[K + 1:]])
    lpi = logits - special.logsumexp(logits)
    eta = mus[None, :] + k * lt[:, None]
    lp = -np.exp(np.clip(eta, -30, 6))
    lq = np.log(-np.expm1(lp) + 1e-300)
    return -special.logsumexp(s[:, None] * lp + (n - s)[:, None] * lq + lpi[None, :], axis=1).sum()


def nm(f, starts, args):
    best = None
    for s0 in starts:
        r = optimize.minimize(f, np.array(s0, float), args=args, method="Nelder-Mead",
                              options=dict(maxiter=4000, xatol=1e-5, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return best


def cll(eta):
    return np.exp(-np.exp(np.clip(eta, -30, 6)))


rows = []
for f in range(5):
    tr, te = fold != f, fold == f
    # ---- crossed model on training tasks (all agents)
    mc = Crossed(N[:, tr], S[:, tr], T[tr])
    thc, _ = mc.fit(mc.init())
    mu_c, k_c = thc[mc.sl["mu"]].numpy(), thc[mc.sl["k"]].numpy()
    lam_c, tau_c = np.exp(thc[mc.sl["loglam"]].numpy()), np.exp(thc[mc.sl["logtau"]].numpy())
    xu, wu = np.polynomial.hermite_e.hermegauss(25)
    wu = wu / wu.sum()
    xw, ww = np.polynomial.hermite_e.hermegauss(15)
    ww = ww / ww.sum()
    te_idx = np.where(te)[0]
    # p[a, i, u, w] for held-out tasks
    eta = (mu_c[:, None] + k_c[:, None] * LT[te_idx][None, :])[:, :, None, None] \
        + (lam_c[:, None] * xu[None, :])[:, None, :, None] + (tau_c[:, None] * xw[None, :])[:, None, None, :]
    P = cll(eta)
    s_, n_ = S[:, te_idx][:, :, None, None], N[:, te_idx][:, :, None, None]
    Lw = s_ * np.log(P + 1e-300) + (n_ - s_) * np.log1p(-P + 1e-15)
    Lai = special.logsumexp(Lw + np.log(ww), axis=3)  # [a, i, u]
    tot = Lai.sum(0)  # [i, u]
    for ai, alias in enumerate(agents):
        trm = tr & (N[ai] > 0)
        tem = te & (N[ai] >= 4)
        if tem.sum() == 0:
            continue
        tt = pd.DataFrame(dict(task_id=np.array(tasks)[trm], n=N[ai, trm], s=S[ai, trm], T=T[trm], family=fam[trm]))
        s_tr, n_tr, lt_tr = S[ai, trm], N[ai, trm], LT[trm]
        lt_te = LT[tem]
        r = fit_glmm(tt, "cloglog", nested=False)
        r0 = optimize.minimize(lambda th: glmm_nll(np.array([th[0], th[1], np.log(1e-6)]), tt, "cloglog", False),
                               r.x[:2], method="Nelder-Mead")
        rl = fit_glmm(tt, "logit", nested=False)
        sd = np.exp(r.x[2])
        lc = nm(nll_latent, [[r.x[0] - 1.5 * sd, r.x[0], r.x[0] + 1.5 * sd, r.x[1], 0, 0],
                             [r.x[0] - 3, r.x[0] - 0.5, r.x[0] + 3, 1.0, 0.5, -0.5]], (lt_tr, s_tr, n_tr))
        bb0 = nm(bb_nll, [[1.0, -1.0], [0.0, 0.0]], (s_tr, n_tr, lt_tr, False))
        bb1 = nm(bb_nll, [[1.0, 0.5, -1.0], [0.0, 1.0, 0.0]], (s_tr, n_tr, lt_tr, True))
        p_g = cll(r.x[0] + r.x[1] * lt_te[:, None] + sd * xg[None, :])
        p_l = special.expit(rl.x[0] - rl.x[1] * lt_te[:, None] - np.exp(rl.x[2]) * xg[None, :])
        p_h = cll(r0.x[0] + r0.x[1] * lt_te)
        lpi = np.concatenate([[0.0], lc.x[4:]])
        pi3 = np.exp(lpi - special.logsumexp(lpi))
        p_3 = cll(lc.x[:3][None, :] + lc.x[3] * lt_te[:, None])
        # crossed predictions
        cols = np.searchsorted(te_idx, np.where(tem)[0])
        Pa = P[ai][cols]  # [i, u, w]
        prior = np.log(wu)[None, :]
        post_X = tot[cols] - Lai[ai][cols] + prior  # leave this agent's own outcomes out
        post_X = np.exp(post_X - special.logsumexp(post_X, axis=1, keepdims=True))
        for j, (i_glob, ltj) in enumerate(zip(np.where(tem)[0], lt_te)):
            n_i, s_i = int(N[ai, i_glob]), int(S[ai, i_glob])
            for m_ in MS:
                g_uw = 1 - (1 - Pa[j]) ** m_
                g_u = (g_uw * ww[None, :]).sum(1)
                rows.append(dict(alias=alias, fold=f, m=m_, T=T[i_glob], task=tasks[i_glob],
                                 emp=1 - comb(n_i - s_i, m_) / comb(n_i, m_),
                                 homog=1 - (1 - p_h[j]) ** m_,
                                 bb=bb_passm(bb0.x, np.array([ltj]), m_, False)[0],
                                 bb_T=bb_passm(bb1.x, np.array([ltj]), m_, True)[0],
                                 gauss=((1 - (1 - p_g[j]) ** m_) * wg).sum(),
                                 logitnorm=((1 - (1 - p_l[j]) ** m_) * wg).sum(),
                                 latent3=((1 - (1 - p_3[j]) ** m_) * pi3).sum(),
                                 crossed_T=(g_u * wu).sum(),
                                 crossed_X=(g_u * post_X[j]).sum()))
    print("fold", f, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/passk_baselines.csv", index=False)
MODELS = ["homog", "bb", "bb_T", "gauss", "logitnorm", "latent3", "crossed_T", "crossed_X"]
A = R.groupby(["alias", "m"])[["emp"] + MODELS].mean().reset_index()
out = []
mean_tab = A.groupby("m")[["emp"] + MODELS].mean()
gain = mean_tab - mean_tab.loc[1]
mae = pd.DataFrame({mo: (A[mo] - A.emp).abs().groupby(A.m).mean() for mo in MODELS})
# task-level Brier-type score for the pass@m indicator expectation (squared error vs unbiased estimate)
sq = pd.DataFrame({mo: ((R[mo] - R.emp) ** 2).groupby(R.m).mean() for mo in MODELS})
L = R[R["T"] >= 240].groupby("m")[["emp"] + MODELS].mean()

# ---- nonparametric check of the sharp H-brackets (Prop. 4), in-sample by agent-stratum
EDGES = [0, 1, 4, 15, 60, 240, 960, 1e9]
br = []
for alias, da in d.groupby("alias"):
    tt = task_table(da)
    tt = tt[tt.n >= 4].copy()
    tt["bin"] = pd.cut(tt["T"], EDGES, right=False)
    for b, g in tt.groupby("bin", observed=True):
        mm = len(g)
        if mm < 8:
            continue
        p = g.s / g.n
        Sb = p.mean()
        if not (0.03 < Sb < 0.97):
            continue
        Pb2 = (p.sum() ** 2 - (p ** 2).sum()) / (mm * (mm - 1))
        M2 = (g.s * (g.s - 1) / (g.n * (g.n - 1))).mean()
        H = np.clip((M2 - Pb2) / (Sb * (1 - Sb)), 0, 1)
        Q = 1 - Sb
        for m_ in [3, 4]:
            emp = np.mean([1 - comb(int(n - s), m_) / comb(int(n), m_) for n, s in zip(g.n, g.s)])
            up = 1 - Q * (Q + H * Sb) ** (m_ - 1)
            rr = Sb + H * Q
            lo = (Sb / rr) * (1 - (1 - rr) ** m_)
            br.append(dict(alias=alias, stratum=str(b), m=m_, S=Sb, H=H, lo=lo, emp=emp, up=up,
                           homog=1 - Q ** m_, inside=(lo - 0.02 <= emp <= up + 0.02)))
B = pd.DataFrame(br)
B.to_csv(f"{RES}/passk_H_brackets.csv", index=False)

with open(f"{RES}/passk_baselines_summary.txt", "w") as fh:
    pd.set_option("display.width", 250)
    fh.write("Held-out tasks (5 shared task folds), mean over 20 agents of agent-level means\n")
    fh.write(mean_tab.round(4).to_string() + "\n\nGain over m=1\n" + gain.round(4).to_string() + "\n\n")
    fh.write("Mean absolute agent-level error\n" + mae.round(4).to_string() + "\n\n")
    fh.write("Mean squared task-level error vs unbiased pass@m estimate\n" + sq.round(4).to_string() + "\n\n")
    fh.write("Held-out tasks >= 4 h, pooled over task-agent pairs\n" + L.round(4).to_string() + "\n\n")
    fh.write("Sharp H-brackets of Prop. 4 (in-sample agent-strata):\n")
    for m_ in [3, 4]:
        Bm = B[B.m == m_]
        fh.write(f"  m={m_}: {len(Bm)} strata; observed inside [lo,up] (+-0.02 tol): {Bm.inside.mean():.3f}; "
                 f"mean width {np.mean(Bm.up - Bm.lo):.3f}; mean homog-minus-up {np.mean(Bm.homog - Bm.up):.3f}; "
                 f"sum obs gain {np.sum(Bm.emp - Bm.S):.2f} vs homog gain {np.sum(Bm.homog - Bm.S):.2f} vs upper-bracket gain {np.sum(Bm.up - Bm.S):.2f}\n")
print(open(f"{RES}/passk_baselines_summary.txt").read())
