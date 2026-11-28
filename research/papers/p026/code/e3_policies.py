"""E3: Persistence policies compared by Monte Carlo, with closed-form predictions
(Proposition 1 for rewrite/rehearsal, Theorem 3 for verified anchors).

Per-fact model (q-ary): each consolidation step a stored value is overwritten w.p. p
(by a draw from the attractor pi); with prob h the environment re-states the true
value (rehearsal). Policies differ in what else happens.
"""
import json, numpy as np
from pathlib import Path
from common import stationary_fidelity, fidelity_curve
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(2)

q = 10
pi = np.full(q, 0.5 / (q - 1)); pi[0] = 0.5
p, TMAX, N = 0.02, 2000, 4000
FACTS = {"idiosyncratic (x=3, pi(x)=0.056)": 3, "default-matching (x=0, pi(x)=0.5)": 0}

def run(policy, x, **kw):
    R = kw.get("R", 1); gamma = kw.get("gamma", 0.0); h = kw.get("h", 0.0)
    T = kw.get("T", None); eps = kw.get("eps", 0.0); gen_anchor = kw.get("gen_anchor", False)
    reps = np.full((N, R), x)
    anchor = np.full(N, x)
    fid = np.empty(TMAX + 1); fid[0] = 1.0
    for t in range(1, TMAX + 1):
        prev = reps[:, 0].copy()
        cm = rng.random(N) < gamma
        y = rng.choice(q, N, p=pi)
        reps[cm] = y[cm, None]
        ind = (~cm)[:, None] & (rng.random((N, R)) < p)
        reps = np.where(ind, rng.choice(q, (N, R), p=pi), reps)
        if R > 1:   # plurality reconciliation; ties keep the previous consensus value
            counts = np.stack([(reps == v).sum(1) for v in range(q)], 1)
            mx = counts.max(1)
            keep = counts[np.arange(N), prev] == mx
            cons = np.where(keep, prev, counts.argmax(1))
            reps = np.repeat(cons[:, None], R, 1)
        if h > 0:
            reh = rng.random(N) < h
            reps[reh] = x
        if gen_anchor:   # 'anchor' that is itself a regenerated summary (unverified)
            a_ind = rng.random(N) < p
            anchor = np.where(a_ind, rng.choice(q, N, p=pi), anchor)
        if T is not None and t % T == 0:
            ok = rng.random(N) >= eps         # verified read succeeds (hash ok)
            reps[ok] = anchor[ok, None]
        fid[t] = (reps[:, 0] == x).mean()
    return fid

def anchor_theory(p, pi_x, h, T, eps):
    lam = (1 - h) * (1 - p); cs = stationary_fidelity(p, pi_x, h)
    u = (1 - eps + eps * cs * (1 - lam ** T)) / (1 - eps * lam ** T)
    avg = cs + (u - cs) * (1 - lam ** T) / (T * (1 - lam))
    mn = cs + (u - cs) * lam ** (T - 1)
    return dict(post=u, time_avg=avg, min=mn)

policies = {
    "P0 rewrite only":                    dict(),
    "P1 rewrite + rehearsal h=0.005":     dict(h=0.005),
    "P2 R=7 self-reconciled, gamma=0.01": dict(R=7, gamma=0.01),
    "P3 R=7 self-reconciled, gamma=0":    dict(R=7, gamma=0.0),
    "P4 verified anchor, T=20, eps=0.01": dict(T=20, eps=0.01),
    "P5 verified anchor, T=1, eps=0.01":  dict(T=1, eps=0.01),
    "P6 unverified generative anchor, T=20": dict(T=20, gen_anchor=True),
}
storage = {"P0": 1, "P1": 1, "P2": 7, "P3": 7, "P4": 2, "P5": 2, "P6": 2}  # copies of k*log2(q) bits
out = {}
for fname, x in FACTS.items():
    for pname, kw in policies.items():
        fid = run(pname, x, **kw)
        tail = fid[TMAX // 2:].mean()
        th = None
        if pname.startswith(("P0", "P1")):
            h = kw.get("h", 0.0)
            th = dict(c_inf=stationary_fidelity(p, pi[x], h),
                      fid_t100=float(fidelity_curve(100, p, pi[x], h)))
        if pname.startswith(("P4", "P5")):
            th = anchor_theory(p, pi[x], 0.0, kw["T"], kw["eps"])
        out[f"{fname} | {pname}"] = dict(fact=fname, policy=pname, fid_t100=float(fid[100]),
                                          fid_t2000=float(fid[-1]), tail_avg=float(tail),
                                          storage_copies=storage[pname[:2]], theory=th,
                                          curve=fid.tolist())
        print(f"{fname[:14]:14} {pname:40} t100={fid[100]:.3f} t2000={fid[-1]:.3f} "
              f"tail_avg={tail:.3f} theory={th}", flush=True)
json.dump(out, open(OUT / "e3_policies.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
W = 20  # plot a trailing 20-step moving average (one anchor period) to avoid sawtooth aliasing
cols = plt.cm.tab10(np.arange(len(policies)))
for i, (fname, x) in enumerate(FACTS.items()):
    for c, pname in zip(cols, policies):
        cv = np.array(out[f"{fname} | {pname}"]["curve"])
        ma = np.convolve(cv, np.ones(W) / W, mode="valid")
        ax[i].plot(np.arange(W - 1, TMAX + 1)[1:], ma[1:], color=c, label=pname, lw=1.3)
    ax[i].set_xscale("log"); ax[i].set_title(fname, fontsize=9); ax[i].set_xlabel("consolidation steps")
ax[0].set_ylabel("P(identity fact intact), 20-step avg"); ax[1].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig3_policies.png", dpi=150)
