"""E4: Eviction of an identity item from a budgeted, recency-decayed salience memory
(Proposition 4). B slots; each step n new items with salience S~Exp(1); score at age a
is S*lam^a; keep top-B scores. The identity item has salience s_id. Optional rehearsal:
w.p. r per step the identity item is recalled and its age resets (MemoryBank-style).
Theory: threshold theta solves n * sum_a P(S > theta*lam^-a) = B; eviction age
a* = ln(s_id/theta)/ln(1/lam); with rehearsal, expected lifetime = ((1-r)^-m - 1)/r
with m = floor(a*)+1 consecutive non-recalls.
"""
import json, numpy as np
from pathlib import Path
from scipy.optimize import brentq
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(3)

def theta_of(n, B, lam):
    f = lambda th: n * sum(np.exp(-th * lam ** -a) for a in range(0, 5000)) - B
    return brentq(f, 1e-9, 1e3)

def simulate(n, B, lam, s_id, r=0.0, tmax=20000, warm=None):
    warm = warm or int(10 / (1 - lam)) + 50
    sal = np.empty(0); born = np.empty(0); ident = np.empty(0, bool)
    t_ins = None
    for t in range(warm + tmax):
        new_s = rng.exponential(1.0, n)
        sal = np.concatenate([sal, new_s]); born = np.concatenate([born, np.full(n, t)])
        ident = np.concatenate([ident, np.zeros(n, bool)])
        if t == warm:
            sal = np.append(sal, s_id); born = np.append(born, t); ident = np.append(ident, True); t_ins = t
        if t_ins is not None and r > 0 and rng.random() < r:
            born[ident] = t           # recall resets recency
        score = sal * lam ** (t - born)
        if len(score) > B:
            keep = np.argpartition(-score, B - 1)[:B]
            sal, born, ident = sal[keep], born[keep], ident[keep]
        if t_ins is not None and not ident.any():
            return t - t_ins
    return np.inf

res = dict(no_rehearsal=[], rehearsal=[])
n, B = 5, 100
for lam in [0.9, 0.97, 0.99]:
    th = theta_of(n, B, lam)
    for mult in [1, 2, 5, 10, 30, 100, 1000]:
        s_id = mult * th
        ages = [simulate(n, B, lam, s_id, tmax=5000) for _ in range(20)]
        a_star = np.log(s_id / th) / np.log(1 / lam)
        res["no_rehearsal"].append(dict(lam=lam, theta=th, s_over_theta=mult,
                                        mean_age=float(np.mean(ages)), sd_age=float(np.std(ages)),
                                        a_star=float(a_star)))
        print(f"lam={lam} theta={th:.3f} s/theta={mult:<5} sim_age={np.mean(ages):7.1f}±{np.std(ages):5.1f} theory={a_star:7.1f}", flush=True)

lam = 0.9; th = theta_of(n, B, lam)
for r in [0.1, 0.2, 0.3]:
    for mult in [2, 5, 10, 30]:
        s_id = mult * th
        a_star = np.log(s_id / th) / np.log(1 / lam)
        m = int(np.floor(a_star)) + 1
        pred = ((1 - r) ** -m - 1) / r
        TM = 60000
        lifes = np.array([simulate(n, B, lam, s_id, r=r, tmax=TM) for _ in range(30)])
        cens = int(np.isinf(lifes).sum())
        fin = lifes[np.isfinite(lifes)]
        ml = float(fin.mean()) if cens == 0 else None
        se = float(fin.std() / np.sqrt(len(fin))) if cens == 0 else None
        res["rehearsal"].append(dict(lam=lam, r=r, s_over_theta=mult, mean_life=ml, se_life=se,
                                     n_censored_at_60000=cens, pred=float(pred), m=m))
        print(f"r={r} s/theta={mult:<3} m={m:<3} sim_life={ml} se={se} censored={cens}/30 pred={pred:9.1f}", flush=True)
json.dump(res, open(OUT / "e4_eviction.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
for lam, c in zip([0.9, 0.97, 0.99], ["#1f77b4", "#d62728", "#2ca02c"]):
    sub = [x for x in res["no_rehearsal"] if x["lam"] == lam]
    m = [x["s_over_theta"] for x in sub]
    ax[0].errorbar(m, [x["mean_age"] for x in sub], yerr=[x["sd_age"] for x in sub], fmt="o", color=c, label=f"sim λ={lam}")
    ax[0].plot(m, [x["a_star"] for x in sub], "-", color=c, lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel("identity salience / threshold θ"); ax[0].set_ylabel("eviction age (steps)")
ax[0].set_title("lines: a* = ln(s/θ)/ln(1/λ)", fontsize=9); ax[0].legend(fontsize=7)
for r, c in zip([0.1, 0.2, 0.3], ["#9467bd", "#8c564b", "#e377c2"]):
    sub = [x for x in res["rehearsal"] if x["r"] == r and x["mean_life"] is not None]
    m = [x["s_over_theta"] for x in sub]
    ax[1].errorbar(m, [x["mean_life"] for x in sub], yerr=[x["se_life"] for x in sub], fmt="o", color=c, label=f"sim r={r}")
    ax[1].plot(m, [x["pred"] for x in sub], "-", color=c, lw=1)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("identity salience / θ (λ=0.9)")
ax[1].set_ylabel("expected lifetime (steps)"); ax[1].set_title("with recall-refresh; lines: run-length theory", fontsize=9)
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig4_eviction.png", dpi=150)
