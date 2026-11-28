"""Experiment 2: honeypots of graded realism (Thm 3) and the realism-response curve.

Setting: GaussianShift(k=8, d'=2). The evaluation distribution is a mixture
E' = (1-rho) E + rho H_t with honeypots H_t = N(t mu, I) (t=1: perfect replica of
deployment; realism gap Delta = (1-t) d').
 (a) Train conditional defectors (width-16 MLP, annealed penalty) against E' and
     compare their (alpha', beta) with the exact mixture frontier and with the
     closed-form Renyi bound of Thm 3.
 (b) Realism-response curve: for schemers trained against E only, measure the
     defection rate on honeypots H_t as a function of t; Thm/Prop 4 predicts
     probit(rate) is linear in t with slope d'. Unconditional baselines are flat.
Run: OMP_NUM_THREADS=1 python3 exp2_honeypots.py   (~4 min on 4 cores)
"""
import json, os
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import GaussianShift, train_anneal, RESULTS
from theory import mixture_frontier, renyi_bound_mixture

DP = 2.0
RHOS = [0.02, 0.1]
TS = [0.0, 0.5, 0.75, 1.0]
LAMS = [2, 8, 32, 128]
SEEDS = [0, 1]


def env():
    return GaussianShift(8, DP, 0)


def job_a(args):
    rho, t, lam, s = args
    e = env()
    rng = np.random.default_rng(7919 * s + 101 * lam + int(1000 * t) + int(10000 * rho))
    mix = [(1 - rho, lambda n, r: e.sample(n, "E", r)), (rho, lambda n, r: e.sample_H(n, t, r))]
    m = train_anneal(e, 16, lam, rng, steps=3000, batch=4096, eval_mix=mix)
    r = np.random.default_rng(99 + s)
    n = 400000
    ae = m.prob(e.sample(n, "E", r)).mean()
    ah = m.prob(e.sample_H(n, t, r)).mean()
    b = m.prob(e.sample(n, "D", r)).mean()
    return dict(rho=rho, t=t, lam=lam, seed=s, alpha_E=float(ae), alpha_H=float(ah),
                alpha_mix=float((1 - rho) * ae + rho * ah), beta=float(b))


def job_b(args):
    lam, s = args
    e = env()
    rng = np.random.default_rng(31337 + 17 * lam + s)
    m = train_anneal(e, 16, lam, rng, steps=3000, batch=4096)
    r = np.random.default_rng(555 + s)
    tgrid = np.linspace(0, 1.2, 13)
    rates = [float(m.prob(e.sample_H(400000, t, r)).mean()) for t in tgrid]
    return dict(lam=lam, seed=s, t=tgrid.tolist(), rate=rates)


def main():
    jobs_a = [(rho, t, l, s) for rho in RHOS for t in TS for l in LAMS for s in SEEDS]
    jobs_b = [(l, s) for l in [16, 64, 256] for s in SEEDS]
    with Pool(4) as p:
        A = p.map(job_a, jobs_a)
        B = p.map(job_b, jobs_b)
    # theory for each learned point
    for q in A:
        comps = [(1 - q["rho"], 0.0), (q["rho"], q["t"] * DP)]
        a = max(q["alpha_mix"], 1e-9)
        q["frontier"] = float(mixture_frontier(DP, comps, a))
        q["renyi_bound"] = float(renyi_bound_mixture(a, [(1 - q["rho"], DP), (q["rho"], (1 - q["t"]) * DP)]))
        q["violates_frontier"] = bool(q["beta"] > q["frontier"] + 0.01)
    # summary table: frontier beta at alpha' = 1e-3 for each (rho, t), and no-honeypot reference
    al_ref = [1e-3, 1e-4]
    table = {}
    for a0 in al_ref:
        table[f"no_honeypot_alpha{a0}"] = float(norm.cdf(DP + norm.ppf(a0)))
        for rho in RHOS:
            for t in TS:
                comps = [(1 - rho, 0.0), (rho, t * DP)]
                table[f"rho{rho}_t{t}_alpha{a0}"] = dict(
                    frontier=float(mixture_frontier(DP, comps, a0)),
                    renyi=float(renyi_bound_mixture(a0, [(1 - rho, DP), (rho, (1 - t) * DP)])),
                    perfect_bound=min(1.0, a0 / rho))
    # probit slopes for response curves (fit on rates in (1e-6, 1-1e-6))
    slopes = []
    for q in B:
        t = np.array(q["t"]); r = np.array(q["rate"])
        ok = (r > 1e-6) & (r < 1 - 1e-6)
        sl, ic = np.polyfit(t[ok], norm.ppf(r[ok]), 1)
        q["probit_slope"], q["probit_intercept"] = float(sl), float(ic)
        slopes.append(sl)
    summ = dict(n_points=len(A), n_violations=int(sum(q["violates_frontier"] for q in A)),
                mean_eff=float(np.mean([q["beta"] / q["frontier"] for q in A if q["frontier"] > 1e-4])),
                renyi_valid=int(sum(q["beta"] <= q["renyi_bound"] + 1e-3 for q in A)),
                probit_slope_mean=float(np.mean(slopes)), probit_slope_sd=float(np.std(slopes)),
                true_dprime=DP)
    out = dict(A=A, B=B, table=table, summary=summ)
    json.dump(out, open(os.path.join(RESULTS, "exp2_honeypots.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)); print(json.dumps(table, indent=1))

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    al = np.logspace(-6, 0, 300)
    cols = {0.0: "#1f77b4", 0.5: "#2ca02c", 0.75: "#ff7f0e", 1.0: "#d62728"}
    for k, rho in enumerate(RHOS):
        a_ = ax[k]
        a_.plot(al, norm.cdf(DP + norm.ppf(al)), "k--", lw=1, label="no honeypots")
        for t in TS:
            comps = [(1 - rho, 0.0), (rho, t * DP)]
            a_.plot(al, mixture_frontier(DP, comps, al), "-", color=cols[t], lw=1.5,
                    label=f"t={t} (Δ={(1 - t) * DP:.1f})")
            a_.plot(al, renyi_bound_mixture(al, [(1 - rho, DP), (rho, (1 - t) * DP)]), ":",
                    color=cols[t], lw=1)
            sel = [q for q in A if q["rho"] == rho and q["t"] == t]
            a_.scatter([q["alpha_mix"] for q in sel], [q["beta"] for q in sel], color=cols[t], s=18)
        a_.set_xscale("log"); a_.set_xlim(1e-6, 1); a_.set_ylim(0, 1)
        a_.set_xlabel("caught rate on evaluation mixture α'")
        a_.set_title(f"({'ab'[k]}) honeypot share ρ={rho}  (solid: exact, dotted: Thm 3)")
        if k == 0:
            a_.set_ylabel("deployment defection rate β")
        a_.legend(fontsize=7)
    for q in B:
        ax[2].plot(q["t"], np.clip(q["rate"], 1e-7, 1), "-o", ms=3, alpha=0.8,
                   label=f"schemer λ={q['lam']} s{q['seed']}")
    ax[2].axhline(0.05, color="gray", ls="--", lw=1, label="unconditional misbehaviour (5%)")
    ax[2].set_yscale("log"); ax[2].set_xlabel("honeypot realism t (1 = deployment replica)")
    ax[2].set_ylabel("defection rate on honeypot")
    ax[2].set_title(f"(c) realism-response curve; probit slope {np.mean(slopes):.2f} (d'={DP})")
    ax[2].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig2_honeypots.png"), dpi=150)


if __name__ == "__main__":
    main()
