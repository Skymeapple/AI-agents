"""Exp 3: audits in the exploit-mixture model.
(a) covariance identity check; (b) estimator RMSE by design/q; (c) sequential stopping
rules: covariance test (several designs), naive first-hack alarms, proxy-only gap alarm."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Params, closed_form, sample_tilted, gold_peak
from audit import cov_estimate, batch_cov

N_BATCH = 2000
SCHED = np.round(np.arange(0.0, 2.0001, 0.05), 3)
DESIGNS = [("uniform", False), ("leverage", False), ("uniform", True), ("leverage", True), ("disagree", True)]
DNAME = {("uniform", False): "COV-unif", ("leverage", False): "COV-lev", ("uniform", True): "COV-unif+CV",
         ("leverage", True): "COV-lev+CV", ("disagree", True): "COV-dis+CV"}


def true_cov(P, lam, eps=1e-5):
    return float((closed_form(P, lam + eps)["gold"] - closed_form(P, lam - eps)["gold"]) / (2 * eps))


def stop_cov(P, design, cv, m, z, rng):
    for lam in SCHED:
        b = sample_tilted(P, N_BATCH, lam, rng)
        est, se, _ = cov_estimate(b["gold"], b["R"][:, 0], b["R"][:, 1], m, design, cv, rng)
        if est + z * se < 0:
            return lam
    return SCHED[-1]


def stop_first_hack(P, m, strategy, rng):
    for lam in SCHED:
        b = sample_tilted(P, N_BATCH, lam, rng)
        if strategy == "random":
            idx = rng.choice(N_BATCH, m, replace=False)
        else:  # audit top-proxy samples
            idx = np.argsort(-b["R"][:, 0])[:m]
        if b["Z"][idx].any():
            return lam
    return SCHED[-1]


def gap_traj(P, rng):
    D = []
    for lam in SCHED:
        b = sample_tilted(P, N_BATCH, lam, rng)
        D.append(np.mean(b["R"][:, 0] - b["R"][:, 1]))
    return np.array(D)


def stop_gap(D, tau, n_fit=4):
    x = SCHED[:n_fit]; y = D[:n_fit]
    slope = np.sum(x * y) / np.sum(x * x)
    exc = D - slope * SCHED
    hit = np.where(exc > tau)[0]
    return SCHED[hit[0]] if len(hit) else SCHED[-1]


def run_exp3(rng, OUT):
    res = {}
    # (a) covariance identity: MC batch covariance vs finite-difference derivative of gold
    P = Params()
    chk = []
    for lam in [0.0, 0.3, 0.59, 0.8, 1.0, 1.5]:
        b = sample_tilted(P, 400_000, lam, rng)
        chk.append(dict(lam=lam, mc_cov=batch_cov(b["gold"], b["R"][:, 0]), dgold_dlam=true_cov(P, lam)))
    res["exp3a_cov_identity"] = chk
    print("Exp3a:", [(c["lam"], round(c["mc_cov"], 3), round(c["dgold_dlam"], 3)) for c in chk])

    # (b) estimator RMSE at the gold peak and just before/after, by design and q
    rmse = {}
    for q in [0.0, 0.5, 0.9, 1.0]:
        Q = Params(q=q)
        lam_s = gold_peak(Q)[0]
        for lam in [0.3, lam_s, 0.9]:
            errs = {DNAME[d]: [] for d in DESIGNS}
            for r in range(300):
                b = sample_tilted(Q, N_BATCH, lam, rng)
                c = batch_cov(b["gold"], b["R"][:, 0])
                for d in DESIGNS:
                    est, _, _ = cov_estimate(b["gold"], b["R"][:, 0], b["R"][:, 1], 20, d[0], d[1], rng)
                    errs[DNAME[d]].append(est - c)
            for k, e in errs.items():
                e = np.array(e)
                rmse[f"q={q}|lam={lam:.3f}|{k}"] = dict(bias=float(e.mean()), rmse=float(np.sqrt(np.mean(e ** 2))))
    res["exp3b_rmse_m20"] = rmse

    # (c) sequential stopping
    stop = {}
    REPS = 200
    for q in [0.0, 0.5, 0.9, 1.0]:
        Q = Params(q=q)
        lam_s, g_s, _ = gold_peak(Q)
        def summarize(lst):
            lst = np.array(lst); gs = closed_form(Q, lst)["gold"]
            return dict(mean_lam_stop=float(lst.mean()), mean_gold=float(gs.mean()), regret=float(g_s - gs.mean()),
                        frac_overshoot=float(np.mean(lst > lam_s + 1e-9)), p10_gold=float(np.percentile(gs, 10)))
        for m in [5, 20]:
            for d in DESIGNS:
                for z in [0.0, 1.0]:
                    stop[f"q={q}|{DNAME[d]}|z={z}|m={m}"] = summarize([stop_cov(Q, d[0], d[1], m, z, rng) for _ in range(REPS)])
            for s in ["random", "topproxy"]:
                stop[f"q={q}|first-hack-{s}|m={m}"] = summarize([stop_first_hack(Q, m, s, rng) for _ in range(REPS)])
        trajs = [gap_traj(Q, rng) for _ in range(REPS)]
        stop[f"_gaptraj_q={q}"] = trajs
        print("q", q, "done")
    # proxy-only gap alarm: tune tau on q=0, apply everywhere
    taus = np.linspace(0.02, 2.0, 100)
    Q0 = Params(q=0.0); g0s = gold_peak(Q0)[1]
    reg0 = [g0s - closed_form(Q0, np.array([stop_gap(D, t) for D in stop["_gaptraj_q=0.0"]]))["gold"].mean() for t in taus]
    tau_star = float(taus[int(np.argmin(reg0))])
    for q in [0.0, 0.5, 0.9, 1.0]:
        Q = Params(q=q); lam_s, g_s, _ = gold_peak(Q)
        lst = np.array([stop_gap(D, tau_star) for D in stop.pop(f"_gaptraj_q={q}")])
        gs = closed_form(Q, lst)["gold"]
        stop[f"q={q}|gap-alarm(proxy-only)"] = dict(mean_lam_stop=float(lst.mean()), mean_gold=float(gs.mean()),
                                                    regret=float(g_s - gs.mean()), frac_overshoot=float(np.mean(lst > lam_s + 1e-9)),
                                                    p10_gold=float(np.percentile(gs, 10)), tau=tau_star)
    res["exp3c_stopping"] = stop
    P = Params()
    res["exp3c_reference"] = dict(lam_star=float(gold_peak(P)[0]), gold_star=float(gold_peak(P)[1]),
                                  gold_at_end=float(closed_form(P, SCHED[-1])["gold"]), schedule_step=0.05, N_batch=N_BATCH)

    # figure: regret vs q for main rules (m=20)
    qs = [0.0, 0.5, 0.9, 1.0]
    rules = [("COV-unif|z=0.0|m=20", "cov test, uniform audit"), ("COV-lev|z=0.0|m=20", "cov test, leverage audit"), ("COV-lev+CV|z=0.0|m=20", "cov test, leverage+CV"),
             ("COV-dis+CV|z=0.0|m=20", "cov test, disagreement+CV"), ("first-hack-random|m=20", "first hack (random audit)"),
             ("first-hack-topproxy|m=20", "first hack (top-proxy audit)"), ("gap-alarm(proxy-only)", "proxy-only gap alarm (tuned at q=0)")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for key, lab in rules:
        ax.plot(qs, [stop[f"q={q}|{key}"]["regret"] for q in qs], marker="o", label=lab)
    ax.set_xlabel("shared vulnerability q"); ax.set_ylabel("gold regret at stopping (lower=better)")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_stopping_regret.png"), dpi=150); plt.close()
    return res
