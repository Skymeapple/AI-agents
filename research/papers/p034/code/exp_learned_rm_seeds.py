"""Exp 4b (revision): learned-RM study over 5 independent seeds and 8 held-out RM types.

Same generative setting as exp_learned_rm.py (same gold function), but for each seed the
training sets, network initialisations and output pool are redrawn. Held-out RM types:
  H1 same data A, new init     H2 fresh draw of A        H3 quadratic ridge on A
  H4 MLP on data B (decorrelated nuisance)               H8 small MLP (16,16) on A
  H5/H6/H7 MLP on a mixture of A and B with B-fraction 0.25 / 0.5 / 0.75 (dose-response)
Per seed: q-hat at the gold peak; proxy-only gap alarm (tau tuned on independent replicates
using the held-out RM with lowest q-hat, then transferred to all); covariance audit (leverage, m=20).
All statistics are computed causally along each replicate's checkpoint stream.
Run: python3 exp_learned_rm_seeds.py   (~5-8 min on 4 cores)
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription with multiprocessing

import json, os, time, warnings
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from rev_common import hh_cov, lev_probs

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DU, DH = 6, 6
W = np.random.default_rng(1).normal(0, 1, DU)  # identical gold to exp_learned_rm.py


def f_u(u):
    return 0.6 * u @ W / np.sqrt(DU) + np.sin(2 * u[:, 0]) * np.cos(u[:, 1]) + 0.5 * np.sin(u[:, 2] * u[:, 3]) + 0.3 * u[:, 4] * u[:, 5]


F_SD = np.std(f_u(np.random.default_rng(99).normal(0, 1, (100000, DU))))


def gold(X):
    u, h = X[:, :DU], X[:, DU:]
    return f_u(u) / F_SD - 1.0 * np.sum(np.maximum(np.abs(h) - 1.0, 0) ** 2, axis=1)


def sample_train(n, kind, rng, fracB=0.0):
    u = rng.normal(0, 1, (n, DU)); fu = f_u(u) / F_SD
    h = 0.25 * rng.normal(0, 1, (n, DH))
    h[:, :3] = 0.6 * np.tanh(fu)[:, None] + 0.15 * rng.normal(0, 1, (n, 3))
    if kind == "B":
        fracB = 1.0
    nb = int(round(fracB * n))
    if nb > 0:
        h[:nb] = rng.uniform(-1, 1, (nb, DH))
    X = np.hstack([u, h]); y = gold(X) + 0.3 * rng.normal(0, 1, n)
    return X, y


def mlp(seed, hid=(64, 64)):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hid, alpha=1e-3, max_iter=400,
                                                        early_stopping=True, random_state=seed))


def tilt_w(score, lam):
    w = np.exp(lam * (score - score.max())); return w / w.sum()


def run_seed(seed):
    t0 = time.time()
    base = 1000 * (seed + 1)
    XA, yA = sample_train(6000, "A", np.random.default_rng(base + 10))
    XA2, yA2 = sample_train(6000, "A", np.random.default_rng(base + 11))
    XB, yB = sample_train(6000, "B", np.random.default_rng(base + 12))
    mix = {f: sample_train(6000, "A", np.random.default_rng(base + 13 + i), fracB=f) for i, f in enumerate([0.25, 0.5, 0.75])}
    models = {"RM1": mlp(base + 0).fit(XA, yA),
              "H1 same data, new init": mlp(base + 1).fit(XA, yA),
              "H2 fresh draw of A": mlp(base + 2).fit(XA2, yA2),
              "H3 quadratic ridge on A": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)).fit(XA, yA),
              "H4 MLP on B": mlp(base + 3).fit(XB, yB),
              "H5 MLP on A/B mix fB=0.25": mlp(base + 4).fit(*mix[0.25]),
              "H6 MLP on A/B mix fB=0.5": mlp(base + 5).fit(*mix[0.5]),
              "H7 MLP on A/B mix fB=0.75": mlp(base + 6).fit(*mix[0.75]),
              "H8 small MLP(16,16) on A": mlp(base + 7, (16, 16)).fit(XA, yA)}
    fracB = {"H1 same data, new init": 0, "H2 fresh draw of A": 0, "H3 quadratic ridge on A": 0, "H4 MLP on B": 1.0,
             "H5 MLP on A/B mix fB=0.25": 0.25, "H6 MLP on A/B mix fB=0.5": 0.5, "H7 MLP on A/B mix fB=0.75": 0.75,
             "H8 small MLP(16,16) on A": 0}
    names = list(models); ho = names[1:]
    NP, P_EX = 200_000, 0.01
    Xg, _ = sample_train(NP, "A", np.random.default_rng(base + 20))
    n_ex = int(NP * P_EX); r_ = np.random.default_rng(base + 21)
    dirs = r_.normal(0, 1, (n_ex, DH)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Xe = np.hstack([r_.normal(0, 1, (n_ex, DU)), dirs * r_.uniform(2.0, 3.5, (n_ex, 1))])
    Xpool = np.vstack([Xg[: NP - n_ex], Xe]); Zpool = np.r_[np.zeros(NP - n_ex, bool), np.ones(n_ex, bool)]
    Gp = gold(Xpool)
    R = {k: m.predict(Xpool) for k, m in models.items()}
    for k in R:
        gk = R[k][~Zpool]; R[k] = (R[k] - gk.mean()) / gk.std() * Gp[~Zpool].std() + Gp[~Zpool].mean()
    r1 = R["RM1"]
    lams = np.round(np.arange(0, 8.0001, 0.25), 3)
    Gc = np.array([tilt_w(r1, l) @ Gp for l in lams]); i_s = int(np.argmax(Gc))
    w = tilt_w(r1, lams[i_s]); over1 = w @ ((r1 - Gp) * Zpool)
    qhat = {n: float(w @ ((R[n] - Gp) * Zpool) / over1) for n in ho}
    # replicate streams: per checkpoint, batch of 2000; gap for every held-out RM; cov audit m=20
    def stream(rng):
        gaps = np.zeros((len(lams), len(ho))); Cl = np.zeros(len(lams)); Cu = np.zeros(len(lams))
        for t, l in enumerate(lams):
            wt = tilt_w(r1, l); idx = np.minimum(np.searchsorted(np.cumsum(wt), rng.random(2000)), NP - 1)
            a = r1[idx]; g = Gp[idx]; h = a - a.mean()
            gaps[t] = [np.mean(a - R[n][idx]) for n in ho]
            Cl[t] = hh_cov(g, h, 20, lev_probs(h), rng)[0]
            Cu[t] = hh_cov(g, h, 20, np.full(2000, 1 / 2000), rng)[0]
        return gaps, Cl, Cu
    rng = np.random.default_rng(base + 30)
    tune_streams = [stream(rng) for _ in range(40)]
    eval_streams = [stream(rng) for _ in range(60)]

    def gap_stop(gap, tau, n_fit=4):
        b1, b0 = np.polyfit(lams[:n_fit], gap[:n_fit], 1); e = gap - (b0 + b1 * lams)
        hit = np.where((e > tau) & (np.arange(len(lams)) >= n_fit))[0]
        return int(hit[0]) if len(hit) else len(lams) - 1
    def first_neg(C):
        i = np.where(C < 0)[0]; return int(i[0]) if len(i) else len(C) - 1
    j_low = int(np.argmin([qhat[n] for n in ho]))
    taus = np.linspace(0.02, 2.0, 100)
    tr = [Gc.max() - np.mean([Gc[gap_stop(s[0][:, j_low], t)] for s in tune_streams]) for t in taus]
    tau = float(taus[int(np.argmin(tr))])
    out = dict(seed=seed, peak_gold=float(Gc.max()), lam_star=float(lams[i_s]), gold_at_8=float(Gc[-1]),
               qhat=qhat, fracB=fracB, tau=tau, tuned_on=ho[j_low], gap_regret={}, gap_regret_se={})
    for j, n in enumerate(ho):
        regs = np.array([Gc.max() - Gc[gap_stop(s[0][:, j], tau)] for s in eval_streams])
        out["gap_regret"][n] = float(regs.mean()); out["gap_regret_se"][n] = float(regs.std(ddof=1) / np.sqrt(len(regs)))
    for key, k in [("cov_lev", 1), ("cov_unif", 2)]:
        st = [first_neg(s[k]) for s in eval_streams]
        regs = np.array([Gc.max() - Gc[i] for i in st])
        out[key] = dict(regret=float(regs.mean()), se=float(regs.std(ddof=1) / np.sqrt(len(regs))),
                        labels_mean=float(np.mean([20 * (i + 1) for i in st])))
    def win_rb(C, w=3):
        Gint = np.concatenate([[0.0], np.cumsum(C[:-1] * np.diff(lams))])
        for t in range(w - 1, len(C)):
            if C[t - w + 1:t + 1].sum() < 0:
                return int(np.argmax(Gint[:t + 1])), t
        return int(np.argmax(Gint)), len(C) - 1
    st = [win_rb(s[1]) for s in eval_streams]
    regs = np.array([Gc.max() - Gc[i] for i, _ in st])
    out["cov_lev_win3_rollback"] = dict(regret=float(regs.mean()), se=float(regs.std(ddof=1) / np.sqrt(len(regs))),
                                        labels_mean=float(np.mean([20 * (t + 1) for _, t in st])))
    out["goodhart_decline_within_schedule"] = bool(Gc[-1] < Gc.max() - 1e-9)
    out["runtime_s"] = time.time() - t0
    return out


def main():
    t0 = time.time()
    with Pool(4) as p:
        runs = p.map(run_seed, range(5))
    ho = list(runs[0]["qhat"])
    q = np.array([[r["qhat"][n] for n in ho] for r in runs]); g = np.array([[r["gap_regret"][n] for n in ho] for r in runs])
    fb = np.array([[r["fracB"][n] for n in ho] for r in runs])
    summ = dict(qhat_mean={n: float(q[:, j].mean()) for j, n in enumerate(ho)},
                qhat_sd={n: float(q[:, j].std(ddof=1)) for j, n in enumerate(ho)},
                gap_regret_mean={n: float(g[:, j].mean()) for j, n in enumerate(ho)},
                gap_regret_sd_across_seeds={n: float(g[:, j].std(ddof=1)) for j, n in enumerate(ho)},
                spearman_qhat_vs_gap_regret_pooled=list(map(float, spearmanr(q.ravel(), g.ravel()))),
                spearman_fracB_vs_qhat_pooled=list(map(float, spearmanr(fb.ravel(), q.ravel()))),
                cov_lev_regret_mean=float(np.mean([r["cov_lev"]["regret"] for r in runs])),
                cov_lev_regret_range=[float(min(r["cov_lev"]["regret"] for r in runs)), float(max(r["cov_lev"]["regret"] for r in runs))],
                cov_lev_win3_rollback_regret_mean=float(np.mean([r["cov_lev_win3_rollback"]["regret"] for r in runs])),
                cov_lev_win3_rollback_regret_range=[float(min(r["cov_lev_win3_rollback"]["regret"] for r in runs)), float(max(r["cov_lev_win3_rollback"]["regret"] for r in runs))],
                cov_unif_regret_mean=float(np.mean([r["cov_unif"]["regret"] for r in runs])),
                cov_lev_labels_mean=float(np.mean([r["cov_lev"]["labels_mean"] for r in runs])),
                peak_gold_mean=float(np.mean([r["peak_gold"] for r in runs])),
                max_gap_regret=float(g.max()))
    # within-seed Spearman between qhat and gap regret (8 held-out RMs)
    summ["spearman_within_seed"] = [float(spearmanr(q[i], g[i])[0]) for i in range(len(runs))]
    res = dict(runs=runs, summary=summ, runtime_s=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "learned_rm_seeds_results.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
