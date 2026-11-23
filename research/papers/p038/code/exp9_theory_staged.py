"""Revision r1 experiments.
E9a: Theorem 2 check -- survival probability (not only the mean) is monotone in the attempt
     budget; closed-form unlimited-retry survival (1-1/R_inf)/(1-h_s).
E9b: agent-level (try-by-try) validation of the staged / checkpointed model: offspring mean,
     geometric offspring law, survival = 1 - 1/R_ck, and the atomic-vs-checkpoint gap.
E9c: finite per-epoch try budget with progress carried across epochs (attempt time).
"""
import json, os, time
import numpy as np
from scipy.stats import chisquare
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import *
from model_ext import *

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(9)
t0 = time.time()
res = {}

# ------------------------------------------------------------ E9a
n_set = 2000; amax = 150
viol = 0; worst = 0.0; n_super = 0; lim_err = 0.0; dp_err = 0.0
for i in range(n_set):
    P = rng.uniform(0.005, 0.6); h_s = rng.uniform(0, 0.5); h_f = rng.uniform(0.001, 0.3)
    sigma = rng.uniform(0.3, 1.0); delta = rng.uniform(0, 0.3)
    Rinf = R_inf(P, h_s, h_f, sigma)
    qs = np.array([extinction_atomic(P, h_s, h_f, sigma, delta, a) for a in range(amax + 1)])
    inc = np.max(np.diff(qs))
    worst = max(worst, inc)
    if inc > 1e-9:
        viol += 1
    if Rinf > 1:
        n_super += 1
        u_cf = survival_atomic_unlimited(P, h_s, h_f, sigma)
        q_inf = extinction_atomic(P, h_s, h_f, sigma, delta, np.inf)
        lim_err = max(lim_err, abs((1 - q_inf) - u_cf))
    if i < 40:  # cross-check fast PGF against the DP pmf of model.py
        a = int(rng.integers(1, 30))
        pmf = offspring_pmf(P, h_s, h_f, sigma, delta, a)
        for s in [0.0, 0.3, 0.7, 0.95]:
            dp_err = max(dp_err, abs(pgf(pmf, s) - pgf_atomic(s, P, h_s, h_f, sigma, delta, a)))
res["E9a"] = dict(n_settings=n_set, a_range=[0, amax], n_supercritical_limit=n_super,
                  n_settings_with_q_increase=viol, max_increase_in_q=float(worst),
                  max_err_closed_form_unlimited_survival=float(lim_err),
                  max_err_fastPGF_vs_DP=float(dp_err))
print("E9a", res["E9a"])

# MC check of the closed form for unlimited retries (large budget a=400 ~ unlimited)
mc_rows = []
for (P, h_s, h_f, sigma) in [(0.1, 0.2, 0.03, 0.9), (0.05, 0.0, 0.02, 1.0), (0.3, 0.4, 0.1, 0.8)]:
    u = survival_atomic_unlimited(P, h_s, h_f, sigma)
    mc = simulate_population(P, h_s, h_f, sigma, 0.0, 400, reps=3000, cap=300, gens=100,
                             rng=int(rng.integers(1e9)))
    se = np.sqrt(mc * (1 - mc) / 3000)
    mc_rows.append(dict(P=P, h_s=h_s, h_f=h_f, sigma=sigma, R_inf=R_inf(P, h_s, h_f, sigma),
                        u_closed=u, one_minus_inv_R=1 - 1 / R_inf(P, h_s, h_f, sigma), mc=mc, se=se))
    print("E9a MC", mc_rows[-1])
res["E9a"]["mc_unlimited"] = mc_rows

# ------------------------------------------------------------ E9b
profiles = {
    "S1 p=(.3,.5,.6,.2) h=.05": (np.array([0.3, 0.5, 0.6, 0.2]), np.full(4, 0.05)),
    "S2 illustrative E4 p, h=.003": (np.array([0.01, 0.2, 0.5, 0.1]), np.full(4, 0.003)),
    "S3 p=.12x4, h=(.01,.05,.02,.1)": (np.full(4, 0.12), np.array([0.01, 0.05, 0.02, 0.1])),
    "S4 profile D p=.12x4, h=.05": (np.full(4, 0.12), np.full(4, 0.05)),
    "S5 p=(.3,.5,.6,.2), h=(.01,.1,.05,.02)": (np.array([0.3, 0.5, 0.6, 0.2]), np.array([0.01, 0.1, 0.05, 0.02])),
}
rows = []
for name, (p, h) in profiles.items():
    Rck = float(R_checkpoint(p, h))
    Pa, hfa = atomic_effective(p, h)
    Rat = Pa / ((1 - Pa) * hfa)
    row = dict(profile=name, p=p.tolist(), h=h.tolist(), R_ck_theory=Rck, R_atomic_theory=Rat,
               surv_ck_theory=float(survival_geometric(Rck)), surv_atomic_theory=float(survival_geometric(Rat)))
    for mode in ["checkpoint", "atomic"]:
        nrep = 20000 if mode == "checkpoint" else 4000
        alive, st, kids = staged_epoch(np.zeros(nrep, int), p, h, 1.0, mode, np.inf, rng)
        m = kids.mean(); se = kids.std() / np.sqrt(nrep)
        # geometric goodness of fit (bins 0..9, 10+)
        Rth = Rck if mode == "checkpoint" else Rat
        th = Rth / (1 + Rth)
        k = np.arange(10)
        probs = np.append((1 - th) * th ** k, th ** 10)
        obs = np.append(np.bincount(np.minimum(kids, 10), minlength=11)[:10],
                        (kids >= 10).sum())
        keep = probs * nrep >= 5
        exp_ = probs * nrep
        o2 = np.append(obs[keep], obs[~keep].sum()); e2 = np.append(exp_[keep], exp_[~keep].sum())
        if e2[-1] < 5:
            o2[-2] += o2[-1]; e2[-2] += e2[-1]; o2 = o2[:-1]; e2 = e2[:-1]
        pv = float(chisquare(o2, e2).pvalue) if len(o2) > 1 else float("nan")
        row[f"{mode}_mc_mean"] = float(m); row[f"{mode}_mc_se"] = float(se); row[f"{mode}_geom_chi2_p"] = pv
        reps = 1500 if mode == "checkpoint" else 600
        if Rth > 0.05:
            sv = staged_population(p, h, 1.0, mode, np.inf, reps=reps, cap=200, gens=200,
                                   rng=int(rng.integers(1e9)))
        else:
            sv = 0.0  # deeply subcritical: survival trivially 0 (checked by mean above)
        row[f"{mode}_mc_surv"] = sv; row[f"{mode}_mc_surv_se"] = float(np.sqrt(max(sv * (1 - sv), 1e-12) / reps))
    rows.append(row)
    print("E9b", {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()})
res["E9b"] = rows

# ------------------------------------------------------------ E9c finite budget, carried progress
p, h = profiles["S1 p=(.3,.5,.6,.2) h=.05"]
Rck = float(R_checkpoint(p, h))
budgets = [5, 10, 20, 40, 80, 160, np.inf]
fb = []
for b in budgets:
    sv = staged_population(p, h, 1.0, "checkpoint", b, delta=0.0, reps=1200, cap=200, gens=400,
                           rng=int(rng.integers(1e9)))
    fb.append(dict(budget=str(b), surv=sv, se=float(np.sqrt(sv * (1 - sv) / 1200))))
    print("E9c budget", b, sv)
res["E9c"] = dict(profile="S1", R_ck=Rck, surv_unlimited_theory=float(survival_geometric(Rck)),
                  note="progress kept across epochs; no background detection (delta=0)", rows=fb)
# with background detection per epoch delta=0.1
fb2 = []
for b in budgets:
    sv = staged_population(p, h, 1.0, "checkpoint", b, delta=0.1, reps=1200, cap=200, gens=400,
                           rng=int(rng.integers(1e9)))
    fb2.append(dict(budget=str(b), surv=sv, se=float(np.sqrt(sv * (1 - sv) / 1200))))
    print("E9c budget delta=.1", b, sv)
res["E9c"]["rows_delta0.1"] = fb2

json.dump(res, open(f"{OUT}/e9_theory_staged.json", "w"), indent=1, default=float)

# figure
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
th_m = []; mc_m = []; th_s = []; mc_s = []; se_s = []
for r in rows:
    for mode, key in [("checkpoint", "ck"), ("atomic", "atomic")]:
        th_m.append(r[f"R_{key}_theory"]); mc_m.append(r[f"{mode}_mc_mean"])
        th_s.append(r[f"surv_{key}_theory"]); mc_s.append(r[f"{mode}_mc_surv"]); se_s.append(r[f"{mode}_mc_surv_se"])
axs[0].loglog(th_m, mc_m, "o", color="#3b6ea5"); xx = [min(th_m) / 2, max(th_m) * 2]
axs[0].plot(xx, xx, "k--", lw=0.8); axs[0].set_xlabel("R from closed form"); axs[0].set_ylabel("mean children (try-level MC)")
axs[0].set_title("E9b: staged model, mean offspring")
axs[1].errorbar(th_s, mc_s, yerr=1.96 * np.array(se_s), fmt="o", color="#b5523b")
axs[1].plot([0, 1], [0, 1], "k--", lw=0.8); axs[1].set_xlabel("1 - 1/R (theory)"); axs[1].set_ylabel("survival (try-level MC)")
axs[1].set_title("E9b: survival")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e9_staged.png", dpi=150)
print("done", time.time() - t0)
