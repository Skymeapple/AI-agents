"""Figure 1 from results/theory_sim.json. The Theorem-1 bound on A/Delta is floor(R_ub)
(strict), where R_ub = (2B - log(T-1)) / log((1-eps)/eps); recomputed here from (B,T,eps)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); res = json.load(open(os.path.join(OUT, "theory_sim.json")))
fig, ax = plt.subplots(1, 3, figsize=(14, 4)); cols = {10**3: "#1f77b4", 10**5: "#ff7f0e", 10**7: "#2ca02c"}
for T in cols:
    rows = [r for r in res["E1"] if r["T"] == T]; Bs = [r["B"] for r in rows]
    ub = [max(np.floor((2*r["B"] - np.log(T-1)) / np.log((1-r["eps"])/r["eps"])), 0) for r in rows]
    ax[0].plot(Bs, ub, "s-", c=cols[T], label=f"Thm 1 bound, T=1e{int(np.log10(T))}")
    ax[0].plot(Bs, [max(r["A_ach_over_Delta"], 0) for r in rows], "o--", c=cols[T], label=f"Prop 2 construction, T=1e{int(np.log10(T))}")
ax[0].set_xlabel("logit range budget B (logits in [-B,B])"); ax[0].set_ylabel("revision-robust horizon A/Δ"); ax[0].set_title("(a) horizon: upper bound vs achieved (ε=0.05)"); ax[0].legend(fontsize=7)
fr = res["E1b_frontier"]
ax[1].plot([f["cold"] for f in fr], [f["hot"] for f in fr], "o-", label="bounded-logit head, sweep of recency slope λ")
ax[1].plot([1], [1], "r*", ms=13, label="delta-rule memory (orthogonal keys, M≤d)")
ax[1].set_xlabel("accuracy: cold key (written once, age~U[0,T))"); ax[1].set_ylabel("accuracy: hot key (revised every ~32)"); ax[1].set_title("(b) recency–retention frontier, T=4096, B=10"); ax[1].legend(fontsize=7)
d = res["E1c_delta"]
for R in [1, 8, 32]:
    rows = [r for r in d if r["R"] == R]; ax[2].plot([r["M"]/r["d"] for r in rows], [r["acc"] for r in rows], "o-", label=f"revisions per key = {R}")
ax[2].set_xscale("log"); ax[2].set_xticks([0.125,0.25,0.5,1,2,4]); ax[2].set_xticklabels(["1/8","1/4","1/2","1","2","4"]); ax[2].minorticks_off(); ax[2].set_xlabel("distinct keys M / key dim d (d=64, random keys)"); ax[2].set_ylabel("latest-value accuracy"); ax[2].set_title("(c) delta-rule: capacity-limited, revision-robust"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_theory.png"), dpi=150); print("ok")
