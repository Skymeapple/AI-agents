"""Experiment 1-3: verify closed forms, best-of-n analogue, audit detection & stopping.
Run: python3 exp_theory.py   (about 2-4 minutes on 4 CPUs)
"""
import json, os, time
import numpy as np
from scipy.stats import norm
from scipy.special import expit, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Params, closed_form, sample_base, sample_tilted, gold_peak

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
res = {}
t0 = time.time()

# ---------------- Exp 1a: closed form vs importance-weighted Monte Carlo -------------
P = Params()
base = sample_base(P, 4_000_000, rng)
lams = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
rows = []
for lam in lams:
    w = np.exp(lam * (base["R"][:, 0] - base["R"][:, 0].max()))
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    mc = dict(pi=np.sum(w * base["Z"]), gold=np.sum(w * base["gold"]),
              r1=np.sum(w * base["R"][:, 0]), r2=np.sum(w * base["R"][:, 1]),
              gap=np.sum(w * (base["R"][:, 0] - base["R"][:, 1])))
    cf = closed_form(P, lam)
    rows.append(dict(lam=float(lam), ess=float(ess), **{f"mc_{k}": float(v) for k, v in mc.items()},
                     **{f"cf_{k}": float(cf[k]) for k in mc}))
res["exp1a_IS_check"] = rows
maxerr = max(abs(r[f"mc_{k}"] - r[f"cf_{k}"]) for r in rows if r["ess"] > 1e4 for k in ["pi", "gold", "gap"])
res["exp1a_max_abs_err"] = float(maxerr)
print("Exp1a max abs err (pi,gold,gap):", maxerr, f"[{time.time()-t0:.0f}s]")
del base

# ---------------- Exp 1b: peak location & peak-prevalence identity -------------------
grid = []
for mu in [3.0, 4.0, 6.0, 8.0]:
    for delta in [1.0, 2.0, 4.0]:
        for p in [1e-4, 1e-3, 1e-2]:
            Q = Params(p=p, mu=mu, delta=delta)
            lam_s, g_s, pi_s = gold_peak(Q, np.linspace(0, 4, 40001))
            pred = Q.sg2 / (Q.mu * (Q.sg2 * lam_s + Q.delta))
            grid.append(dict(mu=mu, delta=delta, p=p, lam_star=float(lam_s), gold_star=float(g_s),
                             pi_star=float(pi_s), pi_star_pred=float(pred)))
res["exp1b_peak_identity"] = grid
res["exp1b_max_rel_err"] = float(max(abs(g["pi_star"] - g["pi_star_pred"]) / g["pi_star_pred"] for g in grid))
print("Exp1b peak-prevalence identity max rel err:", res["exp1b_max_rel_err"])

# ---------------- Fig 1: curves for q in {0, .5, 1} ---------------------------------
lam_grid = np.linspace(0, 2.0, 401)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for q, col in zip([0.0, 0.5, 1.0], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    Q = Params(q=q)
    cf = closed_form(Q, lam_grid)
    ax[1].plot(cf["kl"], cf["gap"], color=col, label=f"q={q}")
    ax[2].plot(cf["kl"], cf["gold_twin"], color=col, ls="--", label=f"twin world W', q={q}")
cf = closed_form(Params(), lam_grid)
ax[0].plot(cf["kl"], cf["gold"], "k", label="gold (any q)")
ax[0].plot(cf["kl"], cf["r1"], "r", label="optimised proxy r1")
ax[0].plot(cf["kl"], cf["pi"] * 5, "m:", label="exploit share x5")
ax[2].plot(cf["kl"], cf["gold"], "k", label="gold in W (any q)")
lam_s, g_s, pi_s = gold_peak(Params())
for a in ax:
    a.axvline(closed_form(Params(), lam_s)["kl"], color="grey", lw=0.8, ls=":")
    a.set_xlabel("KL(pi_lam || pi0)  [nats]")
ax[0].set_title("Exploit-mixture Goodhart curve"); ax[0].legend(fontsize=7); ax[0].set_ylim(-3, 15)
ax[1].set_title("Held-out proxy gap E[r1-r2]"); ax[1].legend(fontsize=7)
ax[2].set_title("Proxy-identical worlds: gold W vs W'"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_closed_form.png"), dpi=150); plt.close()

# ---------------- Exp 2: best-of-n analogue (numerical integral + MC check) ----------
def bon_curve(P, ns):
    t = np.linspace(-12, P.mu + 14, 40001); dt = t[1] - t[0]
    s = np.sqrt(P.v)
    f0 = norm.pdf(t, 0, s); f1 = norm.pdf(t, P.mu, s)
    F = (1 - P.p) * norm.cdf(t, 0, s) + P.p * norm.cdf(t, P.mu, s)
    out = []
    for n in ns:
        wn = n * np.exp((n - 1) * np.log(np.clip(F, 1e-300, 1)))
        pex = np.sum(wn * P.p * f1) * dt
        gold = np.sum(wn * ((1 - P.p) * f0 * (P.sg2 / P.v) * t - P.p * f1 * P.delta)) * dt
        out.append((pex, gold))
    return np.array(out)

ns = np.unique(np.round(np.logspace(0, 6, 61)).astype(int))
P = Params()
bon = bon_curve(P, ns)
# MC check at a few n
mc_check = []
for n in [10, 100, 1000]:
    reps = 4000
    pe, gd = [], []
    for _ in range(reps // 200):
        b = sample_base(P, n * 200, rng, K=1)
        r = b["R"][:, 0].reshape(200, n); i = np.argmax(r, axis=1)
        Z = b["Z"].reshape(200, n)[np.arange(200), i]; G = b["gold"].reshape(200, n)[np.arange(200), i]
        pe.append(Z.mean()); gd.append(G.mean())
    k = list(ns).index(n) if n in ns else None
    num = bon_curve(P, [n])[0]
    mc_check.append(dict(n=n, mc_pexploit=float(np.mean(pe)), num_pexploit=float(num[0]),
                         mc_gold=float(np.mean(gd)), num_gold=float(num[1])))
res["exp2_bon_mc_check"] = mc_check
kl_bon = np.log(ns) - (ns - 1) / ns
i_star = int(np.argmax(bon[:, 1]))
res["exp2_bon_peak"] = dict(n_star=int(ns[i_star]), gold_star=float(bon[i_star, 1]),
                            pexploit_at_peak=float(bon[i_star, 0]), kl_at_peak=float(kl_bon[i_star]))
cf = closed_form(P, lam_grid)
res["exp2_tilt_peak"] = dict(lam_star=float(lam_s), gold_star=float(g_s), pi_star=float(pi_s),
                             kl_at_peak=float(closed_form(P, lam_s)["kl"]))
print("Exp2 BoN:", res["exp2_bon_peak"], "tilt:", res["exp2_tilt_peak"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(kl_bon, bon[:, 1], label="best-of-n"); ax[0].plot(cf["kl"], cf["gold"], label="KL-tilting")
ax[0].set_xlabel("KL [nats]"); ax[0].set_ylabel("gold"); ax[0].legend(); ax[0].set_xlim(0, 14)
ax[1].plot(kl_bon, bon[:, 0], label="best-of-n"); ax[1].plot(cf["kl"], cf["pi"], label="KL-tilting")
ax[1].set_xlabel("KL [nats]"); ax[1].set_ylabel("exploit share of outputs"); ax[1].legend(); ax[1].set_xlim(0, 14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_bon_vs_tilt.png"), dpi=150); plt.close()

from exp3_audit import run_exp3
res.update(run_exp3(rng, OUT))

with open(os.path.join(OUT, "theory_results.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"done in {time.time()-t0:.0f}s")
