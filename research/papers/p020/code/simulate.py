"""Simulations: (1) estimator validation for authorship information I(X;A);
(2) novelty-competence identity  N + C = I(X;A) - E_i KL(r_i || p_i).
Outputs -> ../results/sim_*.json, ../results/fig_sim_*.png
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from infodecomp import plugin_mi, excess_mi, H_counts

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)


def H(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def zipf(K, s):
    w = 1.0 / np.arange(1, K + 1) ** s
    return w / w.sum()

# ---------------------------------------------------------------- Sim 1
t0 = time.time()
K = 26
base = zipf(K, 1.1)
concs = [0.3, 1, 3, 10, 30, 100, np.inf]
designs = [(220, 10), (55, 10), (220, 5)]
REPS = 60
sim1 = []
for (n, k) in designs:
    for c in concs:
        for r in range(REPS):
            if np.isinf(c):
                P = np.tile(base, (n, 1))
            else:
                P = rng.dirichlet(c * K * base, size=n)
            pbar = P.mean(0)
            I_true = H(pbar) - np.mean([H(p) for p in P])
            x = np.concatenate([rng.choice(K, size=k, p=p) for p in P])
            a = np.repeat(np.arange(n), k)
            res = excess_mi(x, a, K, n_perm=100, rng=rng)
            sim1.append(dict(n=n, k=k, conc=float(c) if np.isfinite(c) else "inf",
                             I_true=I_true, H_true=H(pbar), I_plugin=res["I"],
                             I_star=res["I_star"], p=res["p"],
                             pi_true=I_true / H(pbar), pi_star=res["pi_star"]))
json.dump(sim1, open(os.path.join(OUT, "sim1_estimator.json"), "w"), indent=1)

# summary table
summ = []
for (n, k) in designs:
    for c in concs:
        cc = float(c) if np.isfinite(c) else "inf"
        rows = [s for s in sim1 if s["n"] == n and s["k"] == k and s["conc"] == cc]
        it = np.array([s["I_true"] for s in rows]); ip = np.array([s["I_plugin"] for s in rows])
        ist = np.array([s["I_star"] for s in rows]); pv = np.array([s["p"] for s in rows])
        summ.append(dict(n=n, k=k, conc=cc, I_true=it.mean(), plugin_bias=(ip - it).mean(),
                         star_bias=(ist - it).mean(), star_rmse=float(np.sqrt(((ist - it) ** 2).mean())),
                         reject_rate_005=float((pv < 0.05).mean())))
json.dump(summ, open(os.path.join(OUT, "sim1_summary.json"), "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (n, k) in zip(axes, designs):
    rows = [s for s in sim1 if s["n"] == n and s["k"] == k]
    it = np.array([s["I_true"] for s in rows])
    ax.scatter(it, [s["I_plugin"] for s in rows], s=8, alpha=.5, c="#b0b0b0", label="plug-in $\\hat I$")
    ax.scatter(it, [s["I_star"] for s in rows], s=8, alpha=.6, c="#1f5fa8", label="permutation-corrected $I^*$")
    m = max(it.max(), 0.1) * 1.1
    ax.plot([0, m], [0, m], "k--", lw=.8)
    ax.set_title(f"n={n} agents, k={k} ideas each")
    ax.set_xlabel("true I(X;A) [bits]")
axes[0].set_ylabel("estimate [bits]")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sim1_estimator.png"), dpi=160); plt.close(fig)
print("sim1 done", round(time.time() - t0, 1), "s")

# ---------------------------------------------------------------- Sim 2
K2 = 200
p = zipf(K2, 1.0)
logp = np.log2(p)
temp_route = []
for T in np.linspace(0.6, 3.0, 25):
    r = p ** (1 / T); r /= r.sum()
    N = H(r); C = float((r * logp).sum()); KL = float((r * (np.log2(r) - logp)).sum())
    temp_route.append(dict(T=T, N=N, C=C, I=0.0, KL=KL, identity_resid=N + C - (0 - KL)))
persp_route = []
nag = 50
Z = rng.standard_normal((nag, K2))
for beta in np.linspace(0, 3.0, 25):
    Pi = p[None, :] * np.exp(beta * Z); Pi /= Pi.sum(1, keepdims=True)
    pbar = Pi.mean(0)
    Hi = np.array([H(q) for q in Pi])
    N = H(pbar); C = float(-Hi.mean()); I = N - Hi.mean()
    persp_route.append(dict(beta=beta, N=N, C=C, I=I, KL=0.0, identity_resid=N + C - I))
# ensemble route: M copies of tilted models, each sampled at temperature T=1
ens_route = []
for M in [1, 2, 4, 8, 16, 32]:
    Pi = p[None, :] * np.exp(1.5 * Z[:M]); Pi /= Pi.sum(1, keepdims=True)
    pbar = Pi.mean(0); Hi = np.array([H(q) for q in Pi])
    ens_route.append(dict(M=M, N=H(pbar), C=float(-Hi.mean()), I=H(pbar) - Hi.mean(), log2M=np.log2(M)))
json.dump(dict(temp=temp_route, persp=persp_route, ens=ens_route),
          open(os.path.join(OUT, "sim2_identity.json"), "w"), indent=1)
print("max |identity residual|:", max(abs(d["identity_resid"]) for d in temp_route + persp_route))

fig, ax = plt.subplots(figsize=(5.2, 4))
ax.plot([d["N"] for d in temp_route], [d["C"] for d in temp_route], "-o", ms=3, c="#c0392b",
        label="temperature route (one agent, $T$ 0.6→3)")
ax.plot([d["N"] for d in persp_route], [d["C"] for d in persp_route], "-s", ms=3, c="#1f5fa8",
        label="perspective route (50 agents, tilt β 0→3)")
ax.plot([d["N"] for d in ens_route], [d["C"] for d in ens_route], "-^", ms=4, c="#27ae60",
        label="ensemble of M tilted models (M=1…32)")
nn = np.linspace(4, 8, 10); ax.plot(nn, -nn, "k:", lw=.8, label="N + C = 0 (I = 0, no KL)")
ax.set_xlabel("population novelty N = E[-log₂ r̄(X)]  [bits]")
ax.set_ylabel("competence C = E[log₂ p_i(X)]  [bits]")
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sim2_identity.png"), dpi=160); plt.close(fig)
print("sim2 done")

# ---------------------------------------------------------------- Sim 3
# Within-list dependence breaks the i.i.d.-response permutation null; replicate lists + list-level
# permutation restore validity.  Lists are generated sequentially: a category already in the list
# has its weight multiplied by psi (psi<1 repetition avoidance, psi>1 persistence).
from infodecomp import list_perm_mi
K3, n3, k3 = 26, 120, 10


def gen_list(p, psi, k, r_):
    w = p.copy(); out = []
    for _ in range(k):
        c = r_.choice(len(p), p=w / w.sum()); out.append(c); w[c] *= psi
    return out

sim3 = []
for hetero in (False, True):
    for psi in (0.2, 0.5, 1.0, 2.0, 4.0):
        for r in range(20):
            P = rng.dirichlet(3 * K3 * base, size=n3) if hetero else np.tile(base, (n3, 1))
            Pb = P.mean(0); I_true = H(Pb) - np.mean([H(q) for q in P])
            # one list per agent -> naive i.i.d.-response permutation
            xs = [gen_list(P[i], psi, k3, rng) for i in range(n3)]
            x = np.concatenate(xs); a = np.repeat(np.arange(n3), k3)
            naive = excess_mi(x, a, K3, n_perm=100, rng=rng)
            # two replicate lists per agent -> list-level permutation of agent labels
            xs2 = xs + [gen_list(P[i], psi, k3, rng) for i in range(n3)]
            x2 = np.concatenate(xs2); lid = np.repeat(np.arange(2 * n3), k3)
            agent_of_list = np.concatenate([np.arange(n3), np.arange(n3)])
            rep = list_perm_mi(x2, lid, agent_of_list, K3, n_perm=100, rng=rng)
            sim3.append(dict(hetero=hetero, psi=psi, I_true=I_true, naive_I_star=naive["I_star"],
                             naive_p=naive["p"], rep_I_star=rep["I_star"], rep_p=rep["p"]))
S3 = {}
for hetero in (False, True):
    for psi in (0.2, 0.5, 1.0, 2.0, 4.0):
        rows = [s for s in sim3 if s["hetero"] == hetero and s["psi"] == psi]
        S3[f"hetero={hetero},psi={psi}"] = {k: float(np.mean([s[k] for s in rows])) for k in rows[0] if k not in ("hetero", "psi")}
        S3[f"hetero={hetero},psi={psi}"].update(naive_reject=float(np.mean([s["naive_p"] < .05 for s in rows])),
                                                rep_reject=float(np.mean([s["rep_p"] < .05 for s in rows])))
json.dump(S3, open(os.path.join(OUT, "sim3_list_dependence.json"), "w"), indent=1)
for k_, v in S3.items():
    print(k_, {a: round(b, 3) for a, b in v.items()})
