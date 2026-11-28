"""E2: Redundancy ceiling under common-mode consolidation errors (Theorem 2).

Each identity fact (q-ary, true value x) is stored as R replicas. Per consolidation:
  - with prob gamma: common-mode event, all R replicas overwritten with ONE draw Y~pi
  - else: each replica independently overwritten w.p. p with an independent draw ~pi
Then self-reconciliation: all replicas set to the plurality value (ties -> keep the
pre-step consensus value). The consensus value is a q-state Markov chain; we estimate
its transition matrix by Monte Carlo, compute the fidelity curve exactly from
matrix powers, and read off the half-life of excess fidelity.
"""
import json, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(1)

q = 10
pi = np.full(q, 0.5 / (q - 1)); pi[0] = 0.5        # attractor: 'default' value 0
X_TRUE = 3                                           # idiosyncratic identity value
p = 0.05
N_MC = 400_000
Rs = [1, 3, 5, 7, 9, 15, 21, 31]
gammas = [0.0, 0.005, 0.01, 0.02, 0.05]

_cache = {}
def independent_part(R):
    """MC estimate of the consensus transition matrix when no common-mode event occurs."""
    if R in _cache: return _cache[R]
    T = np.zeros((q, q))
    for s in range(q):
        reps = np.full((N_MC, R), s)
        ind = rng.random((N_MC, R)) < p
        yind = rng.choice(q, (N_MC, R), p=pi)
        reps = np.where(ind, yind, reps)
        counts = np.zeros((N_MC, q), dtype=np.int32)
        for v in range(q):
            counts[:, v] = (reps == v).sum(1)
        mx = counts.max(1)
        # tie-break: keep s if s is among the maxima, else lowest index among maxima
        keep = counts[:, s] == mx
        newv = np.where(keep, s, counts.argmax(1))
        T[s] = np.bincount(newv, minlength=q) / N_MC
    _cache[R] = T
    return T

def transition_matrix(R, gamma):
    # common-mode event: all replicas equal one draw Y~pi, so the new consensus is Y (exact)
    return gamma * np.outer(np.ones(q), pi) + (1 - gamma) * independent_part(R)

def curve_and_halflife(T, x, tmax=20000):
    c = np.zeros(q); c[x] = 1.0
    # stationary
    w, V = np.linalg.eig(T.T); st = np.real(V[:, np.argmin(np.abs(w - 1))]); st /= st.sum()
    cinf = st[x]
    target = cinf + (1 - cinf) / 2
    # doubling search with matrix powers
    t = 0; P = c.copy(); fid = [1.0]
    step_mats = {}
    # simple iteration (chains are small)
    while t < tmax:
        P = P @ T; t += 1
        if t <= 400: fid.append(P[x])
        if P[x] <= target:
            return t, cinf, fid
    return np.inf, cinf, fid

res = []
for g in gammas:
    for R in Rs:
        T = transition_matrix(R, g)
        e_R = 1 - T[X_TRUE, X_TRUE]
        t12, cinf, fid = curve_and_halflife(T, X_TRUE)
        # analytic quantities
        floor_err = g * (1 - pi[X_TRUE])
        from scipy.stats import binom
        indep_err_ub = binom.sf(np.ceil(R / 2) - 1, R, p * (1 - pi[X_TRUE]))  # P(#non-x >= R/2)
        ceiling = np.log(2) / -np.log(1 - g) if g > 0 else np.inf
        lb_time_to_half = float(np.log(2) / -np.log1p(-indep_err_ub)) if g == 0 else None
        res.append(dict(gamma=g, R=R, indep_err_upper=float(indep_err_ub),
                        lb_steps_until_fidelity_half=lb_time_to_half, err_per_step=e_R, err_floor=floor_err,
                        err_upper=floor_err + (1 - g) * indep_err_ub,
                        half_life=float(t12), c_inf=float(cinf),
                        half_life_ceiling=float(ceiling),
                        fid=fid[:401:20]))
        print(f"gamma={g:<6} R={R:<3} err/step={e_R:.2e} floor={floor_err:.2e} "
              f"t1/2={t12} ceiling={ceiling:.1f} c_inf={cinf:.3f}", flush=True)
json.dump(res, open(OUT / "e2_redundancy.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
cols = plt.cm.viridis(np.linspace(0, 0.9, len(gammas)))
for g, cc in zip(gammas, cols):
    sub = [r for r in res if r["gamma"] == g]
    if g == 0:
        ax[0].semilogy(Rs, [r["lb_steps_until_fidelity_half"] for r in sub], "s--", color=cc,
                       label="γ=0 (analytic lower bound)")
    else:
        ax[0].semilogy(Rs, [r["half_life"] for r in sub], "o-", color=cc, label=f"γ={g}")
    if g > 0:
        ax[0].axhline(sub[0]["half_life_ceiling"], color=cc, ls=":", lw=1)
    okk = [(r["R"], r["err_per_step"]) for r in sub if r["err_per_step"] > 0]
    ax[1].semilogy([a for a, b in okk], [b for a, b in okk], "o-", color=cc)
    if g == 0: ax[1].semilogy(Rs, [r["indep_err_upper"] for r in sub], "s--", color=cc, lw=1)
    if g > 0: ax[1].axhline(sub[0]["err_floor"], color=cc, ls=":", lw=1)
ax[0].set_xlabel("replicas R (self-reconciled)"); ax[0].set_ylabel("identity half-life (steps)")
ax[0].legend(fontsize=7); ax[0].set_title("dotted: Thm 2 ceiling ln2/-ln(1-γ)", fontsize=9)
ax[1].set_xlabel("replicas R"); ax[1].set_ylabel("per-step loss prob. from correct state")
ax[1].set_title("dotted: floor γ(1-π(x)); γ=0 MC shown only where >0", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig2_redundancy.png", dpi=150)
