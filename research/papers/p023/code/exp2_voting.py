"""Exp 2: plurality-vote conflict resolution vs team size, real data vs two surrogates.
S0 real answers; S1 wrong answers scrambled (removes shared wrong attractors, keeps shared difficulty);
S2 independent (keeps only each agent's marginal accuracy/abstention).
Also computes effective team size N_eff = N / (1 + (N-1) rho_bar) per team."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, plurality, surrogate_scramble_wrong, surrogate_independent

d = load()
A, y, nopt, models = d["A"], d["y"], d["nopt"], d["models"]
C = (A == y[:, None])
acc = C.mean(0)
rho = np.corrcoef(C.T.astype(float))
rng = np.random.default_rng(1)
S1 = surrogate_scramble_wrong(A, y, nopt, rng)
S2 = surrogate_independent(A, y, nopt, rng)

pools = {
    "all46": np.arange(len(models)),
    "peer_0.60-0.80": np.nonzero((acc >= 0.60) & (acc <= 0.80))[0],
    "peer_0.35-0.55": np.nonzero((acc >= 0.35) & (acc <= 0.55))[0],
}
TEAMS = 150
res = {}
for pname, pool in pools.items():
    sizes = sorted(set([s for s in [1, 3, 5, 7, 9, 11, 15, 21, 31] if s < len(pool)] + [len(pool)]))
    rows = []
    for N in sizes:
        r0, r1, r2, mem_mean, mem_best, neff, rbar = [], [], [], [], [], [], []
        nt = 1 if N == len(pool) else TEAMS
        for t in range(nt):
            team = pool if N == len(pool) else rng.choice(pool, N, replace=False)
            r0.append((plurality(A[:, team], rng) == y).mean())
            r1.append((plurality(S1[:, team], rng) == y).mean())
            r2.append((plurality(S2[:, team], rng) == y).mean())
            mem_mean.append(acc[team].mean()); mem_best.append(acc[team].max())
            if N > 1:
                rb = rho[np.ix_(team, team)][np.triu_indices(N, 1)].mean()
                rbar.append(rb); neff.append(N / (1 + (N - 1) * rb))
            else:
                rbar.append(np.nan); neff.append(1.0)
        f = lambda v: [float(np.mean(v)), float(np.std(v))]
        rows.append(dict(N=N, real=f(r0), scrambled=f(r1), independent=f(r2), member_mean=f(mem_mean),
                         member_best=f(mem_best), frac_teams_vote_beats_best=float(np.mean(np.array(r0) > np.array(mem_best))),
                         neff=f(neff), rho_bar=[float(np.nanmean(rbar)), float(np.nanstd(rbar))]))
        print(pname, N, {k: (round(v[0], 4) if isinstance(v, list) else v) for k, v in rows[-1].items()})
    res[pname] = dict(members=[models[i] for i in pool], rows=rows)

# Decomposition of the shortfall at the largest odd team size of each pool
decomp = {}
for pname, r in res.items():
    last = r["rows"][-1]
    ind, scr, real = last["independent"][0], last["scrambled"][0], last["real"][0]
    decomp[pname] = dict(N=last["N"], independent=ind, scrambled=scr, real=real,
                         loss_shared_difficulty=ind - scr, loss_shared_wrong_answers=scr - real,
                         share_wrong_answers=(scr - real) / (ind - real) if ind > real else None)
res["decomposition"] = decomp
json.dump(res, open("../results/exp2_voting.json", "w"), indent=2)
print(json.dumps(decomp, indent=2))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (pname, r) in zip(axs, [(k, v) for k, v in res.items() if k != "decomposition"]):
    N = [row["N"] for row in r["rows"]]
    for key, lab, c in [("independent", "S2 independent (Condorcet)", "C2"), ("scrambled", "S1 shared difficulty only", "C1"),
                        ("real", "S0 real answers", "C0"), ("member_best", "best team member", "k"), ("member_mean", "mean team member", "gray")]:
        mu = np.array([row[key][0] for row in r["rows"]]); sd = np.array([row[key][1] for row in r["rows"]])
        ls = "--" if key.startswith("member") else "-"
        ax.plot(N, mu, ls, marker="o", ms=3, c=c, label=lab)
        ax.fill_between(N, mu - sd, mu + sd, color=c, alpha=.12)
    ax.set_xscale("log"); ax.set_xlabel("team size N (random teams from pool)"); ax.set_ylabel("accuracy")
    ax.set_title(f"pool {pname} ({len(r['members'])} agents)")
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("../results/fig2_voting.png", dpi=140)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
for pname, r in [(k, v) for k, v in res.items() if k != "decomposition"]:
    ax.plot([row["N"] for row in r["rows"]], [row["neff"][0] for row in r["rows"]], marker="o", label=pname)
ax.plot([1, 46], [1, 46], "k:", lw=1, label="N_eff = N")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("nominal team size N"); ax.set_ylabel("effective team size N_eff")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("../results/fig3_neff.png", dpi=140)
