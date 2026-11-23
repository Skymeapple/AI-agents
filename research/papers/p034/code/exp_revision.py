"""Revision experiments (response to review r1) in the exploit-mixture model.

R1  Equal-cumulative-label-budget comparison of gold-using rules (W, all q):
    cov test (leverage / uniform / anytime-valid / windowed+rollback), gold-mean trend test
    (plain / regression estimator / with rollback), Moskovitz-style proxy-point search.
R2  Stronger proxy-only and classifier baselines with out-of-sample tuning at q=0:
    held-out-gap alarm, ensemble-std alarm (K=4), hack-rate threshold alarm (perfect classifier).
R3  Twin world W' (shared exploits genuinely good): all rules simulated (K=2).
R4  Noisy and biased gold labels.
R5  One-at-a-time sensitivity (mu, delta, p, rho, N, schedule step) incl. transfer of
    tuned thresholds (hack-rate tau, proxy point) from the default setting.
All decisions are causal: statistics are computed checkpoint by checkpoint and a rule stops at
the first checkpoint where its condition holds; labels used = m x (#checkpoints audited).
Run: python3 exp_revision.py   (~6-8 min on 4 cores)
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription with multiprocessing

import json, os, time, sys
import numpy as np
from multiprocessing import Pool
from model import Params, closed_form
from rev_common import (sample_world, gold_curve_twin, hh_cov, lev_probs, gold_mean_est,
                        ols_slope, z_anytime)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MS = [5, 10, 20, 40, 80]


def trajectory(P, seed, sched, N=2000, K=2, twin=False, noise=0.0, bias=0.0, ms=MS):
    """Per-checkpoint statistics for one replicate (all rules read from this)."""
    rng = np.random.default_rng(seed)
    T = len(sched)
    st = {"rbar": np.zeros(T), "gap": np.zeros(T), "ens": np.zeros(T), "hack": np.zeros(T)}
    for m in ms:
        for k in ["Clev", "SElev", "Cunif", "Gu", "SEu", "Greg"]:
            st[f"{k}{m}"] = np.zeros(T)
    for t, lam in enumerate(sched):
        b = sample_world(P, N, lam, rng, K=K, twin=twin)
        r1 = b["R"][:, 0]; g = b["gold"]
        lab = g + noise * rng.normal(0, 1, N) + bias * (P.mu + P.delta) * b["Z"]
        h = r1 - r1.mean(); wl = lev_probs(h); wu = np.full(N, 1.0 / N)
        st["rbar"][t] = r1.mean()
        st["gap"][t] = np.mean(r1 - b["R"][:, 1])
        st["ens"][t] = np.mean(b["R"].std(axis=1))
        st["hack"][t] = b["Z"][rng.choice(N, 20, replace=False)].sum()  # perfect per-sample classifier, 20 calls
        for m in ms:
            c, se = hh_cov(lab, h, m, wl, rng); st[f"Clev{m}"][t] = c; st[f"SElev{m}"][t] = se
            st[f"Cunif{m}"][t] = hh_cov(lab, h, m, wu, rng)[0]
            gm, gse = gold_mean_est(lab, r1, m, rng); st[f"Gu{m}"][t] = gm; st[f"SEu{m}"][t] = gse
            st[f"Greg{m}"][t] = gold_mean_est(lab, r1, m, rng, reg=True)[0]
    return st


# ---------------- rules: return (deploy_index, labels_used) ----------------
def r_cov(st, m, key="Clev", z=0.0, anytime=False):
    C = st[f"{key}{m}"]; SE = st.get(f"SElev{m}")
    for t in range(len(C)):
        zz = z_anytime(t) if anytime else z
        if C[t] + zz * (SE[t] if zz > 0 else 0) < 0:
            return t, m * (t + 1)
    return len(C) - 1, m * len(C)


def r_cov_win_rb(st, m, sched, w=3):
    C = st[f"Clev{m}"]; dl = np.diff(sched, prepend=sched[0])
    Gint = np.concatenate([[0.0], np.cumsum(C[:-1] * np.diff(sched))])  # integrated gold estimate
    for t in range(w - 1, len(C)):
        if C[t - w + 1:t + 1].sum() < 0:
            return int(np.argmax(Gint[:t + 1])), m * (t + 1)
    return int(np.argmax(Gint)), m * len(C)


def r_goldtrend(st, m, sched, key="Gu", w=5, rollback=False):
    G = st[f"{key}{m}"]
    for t in range(w - 1, len(G)):
        if ols_slope(sched[t - w + 1:t + 1], G[t - w + 1:t + 1]) < 0:
            return (int(np.argmax(G[:t + 1])) if rollback else t), m * (t + 1)
    return (int(np.argmax(G)) if rollback else len(G) - 1), m * len(G)


def r_proxypoint(st, m, kmin=5):
    """Moskovitz-style proxy-point search (scalar version): fit gold ~ quadratic in mean proxy,
    stop once the fitted proxy point has been passed; deploy the checkpoint nearest to it."""
    G = st[f"Gu{m}"]; x = st["rbar"]
    for t in range(kmin - 1, len(G)):
        c2, c1, c0 = np.polyfit(x[:t + 1], G[:t + 1], 2)
        if c2 < 0 and -c1 / (2 * c2) < x[t]:
            pp = -c1 / (2 * c2)
            return int(np.argmin(np.abs(x[:t + 1] - pp))), m * (t + 1), float(pp)
    return len(G) - 1, m * len(G), float("inf")


def r_threshold(stat, tau, sched, n_fit=4, detrend=True):
    s = np.asarray(stat)
    if detrend:
        b1, b0 = np.polyfit(sched[:n_fit], s[:n_fit], 1)  # linear trend with intercept on first n_fit points
        s = s - (b0 + b1 * sched)
    hit = np.where(s > tau)[0]
    hit = hit[hit >= n_fit] if detrend else hit
    return (int(hit[0]) if len(hit) else len(s) - 1)


def r_hackrate(st, tau, win=3):
    H = st["hack"]
    for t in range(len(H)):
        lo = max(0, t - win + 1)
        if H[lo:t + 1].sum() / (20 * (t + 1 - lo)) > tau:
            return t
    return len(H) - 1


def r_firsthack(st):
    hit = np.where(st["hack"] > 0)[0]
    return int(hit[0]) if len(hit) else len(st["hack"]) - 1


def summarize(gcurve, idxs, labels=None):
    idxs = np.asarray(idxs); gs = gcurve[idxs]; gstar = gcurve.max(); istar = int(np.argmax(gcurve))
    d = dict(regret=float(gstar - gs.mean()), se=float(gs.std(ddof=1) / np.sqrt(len(gs))),
             overshoot=float(np.mean(idxs > istar)), p90_regret=float(np.percentile(gstar - gs, 90)))
    if labels is not None:
        d["labels_mean"] = float(np.mean(labels))
    return d


def gold_rules(trajs, sched, gcurve, ms=MS, tuned=True, fixed=None):
    """fixed: {m: (w, kmin)} tuned in W (q irrelevant) and transferred; tuned=True adds in-sample oracle tuning."""
    out = {}
    for m in ms:
        if fixed is not None and m in fixed:
            w, km = fixed[m]
            L = [r_goldtrend(s, m, sched, w=w) for s in trajs]; out[f"goldtrend(w={w} tuned in W)|m={m}"] = summarize(gcurve, *zip(*L))
            L = [r_proxypoint(s, m, kmin=km)[:2] for s in trajs]; out[f"proxypoint(kmin={km} tuned in W)|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_cov(s, m) for s in trajs]; out[f"cov-lev|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_cov(s, m, key="Cunif") for s in trajs]; out[f"cov-unif|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_cov(s, m, anytime=True) for s in trajs]; out[f"cov-lev-anytime(a=0.1)|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_cov_win_rb(s, m, sched) for s in trajs]; out[f"cov-lev-win3+rollback|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_goldtrend(s, m, sched) for s in trajs]; out[f"goldtrend|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_goldtrend(s, m, sched, key="Greg") for s in trajs]; out[f"goldtrend-reg|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_goldtrend(s, m, sched, rollback=True) for s in trajs]; out[f"goldtrend+rollback|m={m}"] = summarize(gcurve, *zip(*L))
        L = [r_proxypoint(s, m)[:2] for s in trajs]; out[f"proxypoint|m={m}"] = summarize(gcurve, *zip(*L))
        if tuned:
            # oracle-tuned (in-sample, on the very trajectories scored) gold-mean baselines: optimistic for them
            best = min(((w, summarize(gcurve, *zip(*[r_goldtrend(s, m, sched, w=w) for s in trajs]))) for w in [3, 5, 8, 12, 16]),
                       key=lambda x: x[1]["regret"])
            out[f"goldtrend-oracle-tuned|m={m}"] = dict(best[1], w=best[0])
            best = min(((k, summarize(gcurve, *zip(*[r_proxypoint(s, m, kmin=k)[:2] for s in trajs]))) for k in [5, 8, 12, 16, 20]),
                       key=lambda x: x[1]["regret"])
            out[f"proxypoint-oracle-tuned|m={m}"] = dict(best[1], kmin=best[0])
    return out


def tune(stat_trajs, sched, gcurve, taus, fn):
    regs = [gcurve.max() - gcurve[[fn(s, t) for s in stat_trajs]].mean() for t in taus]
    return float(taus[int(np.argmin(regs))])


def _traj_job(args):
    return trajectory(*args[:2], **args[2])


def many(P, seeds, sched, pool, **kw):
    return pool.map(_traj_job, [(P, s, dict(sched=sched, **kw)) for s in seeds])


def main():
    t0 = time.time()
    res = {}
    SCHED = np.round(np.arange(0.0, 2.0001, 0.05), 3)
    REPS = 200
    pool = Pool(4)
    taus_gap = np.linspace(0.02, 2.0, 100); taus_ens = np.linspace(0.005, 1.0, 100); taus_hr = np.linspace(0.0, 0.5, 101)
    f_gap = lambda s, t: r_threshold(s["gap"], t, SCHED)
    f_ens = lambda s, t: r_threshold(s["ens"], t, SCHED)
    f_hr = lambda s, t: r_hackrate(s, t)

    # ---------- tuning set (q = 0, independent seeds), K = 4 ----------
    P0 = Params(q=0.0); g0c = closed_form(P0, SCHED)["gold"]
    tune_tr = many(P0, range(10_000, 10_000 + REPS), SCHED, pool, K=4, ms=[20])
    tau = dict(gap=tune(tune_tr, SCHED, g0c, taus_gap, f_gap), ens=tune(tune_tr, SCHED, g0c, taus_ens, f_ens),
               hackrate=tune(tune_tr, SCHED, g0c, taus_hr, f_hr))
    res["tuned_thresholds_q0_out_of_sample"] = tau
    print("tuned", tau, f"[{time.time()-t0:.0f}s]")

    # ---------- R1 + R2: W, q grid ----------
    R1 = {}
    for q in [0.0, 0.5, 0.9, 1.0]:
        P = Params(q=q); gc = closed_form(P, SCHED)["gold"]
        tr = many(P, range(20_000 + int(q * 1000), 20_000 + int(q * 1000) + REPS), SCHED, pool, K=4)
        d = gold_rules(tr, SCHED, gc)
        d["gap-alarm(tuned q=0, out-of-sample)"] = summarize(gc, [f_gap(s, tau["gap"]) for s in tr])
        d["ensemble-std-alarm K=4 (tuned q=0)"] = summarize(gc, [f_ens(s, tau["ens"]) for s in tr])
        d["hack-rate-alarm (perfect clf, tuned q=0)"] = summarize(gc, [f_hr(s, tau["hackrate"]) for s in tr])
        d["first-hack (20 random clf calls)"] = summarize(gc, [r_firsthack(s) for s in tr])
        R1[f"q={q}"] = d
        if q == 0.0:
            FIXED = {m: (d[f"goldtrend-oracle-tuned|m={m}"]["w"], d[f"proxypoint-oracle-tuned|m={m}"]["kmin"]) for m in MS}
            res["gold_baseline_params_tuned_in_W_q0"] = {str(m): v for m, v in FIXED.items()}
        print("R1 q", q, f"[{time.time()-t0:.0f}s]")
    res["R1_W"] = R1

    # ---------- R3: twin world W' (K=2) ----------
    R3 = {}
    for q in [0.5, 1.0]:
        P = Params(q=q); gT = gold_curve_twin(P, SCHED); gW = closed_form(P, SCHED)["gold"]
        tr = many(P, range(30_000 + int(q * 1000), 30_000 + int(q * 1000) + REPS), SCHED, pool, K=2, twin=True, ms=[20])
        d = gold_rules(tr, SCHED, gT, ms=[20], tuned=False, fixed=FIXED)
        d["gap-alarm(tuned q=0)"] = summarize(gT, [f_gap(s, tau["gap"]) for s in tr])
        d["hack-rate-alarm (exploit-type clf, tuned q=0)"] = summarize(gT, [f_hr(s, tau["hackrate"]) for s in tr])
        d["first-hack (exploit-type clf)"] = summarize(gT, [r_firsthack(s) for s in tr])
        d["_peak_lambda_twin"] = float(SCHED[int(np.argmax(gT))]); d["_peak_gold_twin"] = float(gT.max())
        R3[f"q={q}"] = d
        print("R3 q", q, f"[{time.time()-t0:.0f}s]")
    res["R3_twin"] = R3

    # ---------- R4: noisy / biased labels (W, q=1, m=20) ----------
    R4 = {}
    P = Params(q=1.0); gc = closed_form(P, SCHED)["gold"]
    for name, kw in [("noise=0", {}), ("noise=0.5", dict(noise=0.5)), ("noise=1", dict(noise=1.0)), ("noise=2", dict(noise=2.0)),
                     ("bias=0.25", dict(bias=0.25)), ("bias=0.5", dict(bias=0.5)), ("bias=0.75", dict(bias=0.75))]:
        tr = many(P, range(40_000, 40_000 + REPS), SCHED, pool, K=2, ms=[20, 80], **kw)
        R4[name] = gold_rules(tr, SCHED, gc, ms=[20, 80], tuned=False, fixed=FIXED)
        print("R4", name, f"[{time.time()-t0:.0f}s]")
    res["R4_label_noise_bias"] = R4

    # ---------- R5: sensitivity + transfer of tuned quantities ----------
    # proxy point estimated in the default setting (pilot run, m=20), then used as a proxy-only threshold
    Pd = Params(q=1.0)
    pilot = many(Pd, range(50_000, 50_000 + REPS), SCHED, pool, K=2, ms=[20])
    pps = [r_proxypoint(s, 20, kmin=FIXED[20][1]) for s in pilot]
    res["proxy_point_default"] = dict(median=float(np.median([p[2] for p in pps])),
                                      labels_mean=float(np.mean([p[1] for p in pps])),
                                      true_proxy_at_peak=float(closed_form(Pd, 0.5893)["r1"]))
    pp_default = [p[2] for p in pps]
    R5 = {}
    settings = [("default", {}, {}), ("mu=4", dict(mu=4.0), {}), ("mu=8", dict(mu=8.0), {}),
                ("delta=1", dict(delta=1.0), {}), ("delta=4", dict(delta=4.0), {}),
                ("p=5e-4", dict(p=5e-4), {}), ("p=1e-2", dict(p=1e-2), {}),
                ("rho=0", dict(rho=0.0), {}), ("rho=0.9", dict(rho=0.9), {}),
                ("N=500", {}, dict(N=500)), ("N=8000", {}, dict(N=8000)),
                ("step=0.025", {}, dict(step=0.025)), ("step=0.1", {}, dict(step=0.1))]
    R5REPS = 100
    for name, pk, ok in settings:
        step = ok.get("step", 0.05); N = ok.get("N", 2000)
        sch = np.round(np.arange(0.0, 2.0001, step), 4)
        P1 = Params(q=1.0, **pk); P0s = Params(q=0.0, **pk)
        gc = closed_form(P1, sch)["gold"]
        tr = many(P1, range(60_000, 60_000 + R5REPS), sch, pool, K=2, N=N, ms=[20])
        trq0 = many(P0s, range(70_000, 70_000 + R5REPS), sch, pool, K=2, N=N, ms=[20])
        fg = lambda s, t: r_threshold(s["gap"], t, sch)
        tau_g = tune(trq0, sch, closed_form(P0s, sch)["gold"], taus_gap, fg)
        d = {}
        L = [r_cov(s, 20) for s in tr]; d["cov-lev m=20"] = summarize(gc, *zip(*L))
        L = [r_cov_win_rb(s, 20, sch) for s in tr]; d["cov-lev-win3+rollback m=20"] = summarize(gc, *zip(*L))
        L = [r_goldtrend(s, 20, sch, w=FIXED[20][0]) for s in tr]; d["goldtrend (w tuned at default) m=20"] = summarize(gc, *zip(*L))
        L = [r_proxypoint(s, 20, kmin=FIXED[20][1])[:2] for s in tr]; d["proxypoint fresh search (kmin tuned at default) m=20"] = summarize(gc, *zip(*L))
        d["proxypoint transferred from default (0 new labels)"] = summarize(
            gc, [(int(np.argmax(s["rbar"] >= pp)) if np.any(s["rbar"] >= pp) else len(sch) - 1)
                 for s, pp in zip(tr, pp_default[:R5REPS])])
        d["hack-rate (tau tuned at default)"] = summarize(gc, [r_hackrate(s, tau["hackrate"]) for s in tr])
        d["gap-alarm (tuned q=0 this setting, eval q=1)"] = summarize(gc, [fg(s, tau_g) for s in tr])
        d["_peak_gain"] = float(gc.max() - gc[0]); d["_lam_star"] = float(sch[int(np.argmax(gc))])
        R5[name] = d
        print("R5", name, f"[{time.time()-t0:.0f}s]")
    res["R5_sensitivity"] = R5
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "revision_results.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
