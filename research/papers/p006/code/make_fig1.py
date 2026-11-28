"""Figure 1 (revision r1) from results/theory_sim.json and results/frontier_lp.json."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
res = json.load(open(os.path.join(OUT, "theory_sim.json"))); fr = json.load(open(os.path.join(OUT, "frontier_lp.json")))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cols = {10**3: "#1f77b4", 10**5: "#ff7f0e", 10**7: "#2ca02c"}
for T, c in cols.items():
    rows = [r for r in res["E1"] if r["T"] == T]; Bs = [r["B"] for r in rows]
    ax[0].plot(Bs, [max(r["A_ub_over_Delta_strict"], 0) for r in rows], "s-", c=c, label=f"Thm 1 bound, T=1e{int(np.log10(T))}")
    ax[0].plot(Bs, [max(r["staircase_A_over_Delta"], 0) for r in rows], "^:", c=c, ms=7, mfc="none", label=f"staircase head (Prop 2'), T=1e{int(np.log10(T))}")
    ax[0].plot(Bs, [max(r["A_ach_over_Delta"], 0) for r in rows], "o--", c=c, alpha=0.6, label=f"linear head (Prop 2), T=1e{int(np.log10(T))}")
ax[0].set_xlabel("logit budget B (logits in [-B, B])"); ax[0].set_ylabel("revision-robust horizon A/Δ")
ax[0].set_title("(a) horizon: bound vs constructions (ε=0.05, Δ=16)", fontsize=10); ax[0].legend(fontsize=6)
T = fr["T"]; Dl = fr["Delta"]
cm = plt.get_cmap("viridis")
for i, f in enumerate(fr["frontiers"]):
    c = cm(i / (len(fr["frontiers"]) - 1))
    pts = [p for p in f["lp"] if p["C"] >= 0]
    x = [p["A_h"] / Dl for p in pts]; y = [(p["C"] + 1) / T for p in pts]
    ax[1].step(x, y, where="post", c=c, lw=2, marker="o", ms=3, label=f"B={f['B']}: optimum over all age-score heads (LP)")
    lin = {}
    for p in f["linear"]:
        if p["A_h"] >= 0: lin[p["A_h"]] = max(lin.get(p["A_h"], -1), p["C"])
    ax[1].plot([a / Dl for a in lin], [(lin[a] + 1) / T for a in lin], ".", c=c, ms=4, alpha=0.6)
    if f["A_h_thm1_max"] >= 0: ax[1].axvline((f["A_h_thm1_max"] + 1) / Dl, c=c, ls=":", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel("hot horizon A_h/Δ (revised facts, spacing ≥ Δ=32)")
ax[1].set_ylabel("cold horizon (C+1)/T (never-revised facts)")
ax[1].set_title(f"(b) recency–retention frontier, T={T}, ε={fr['eps']}\n(dots: linear-recency heads; dotted: Thm-1 limit)", fontsize=9)
ax[1].legend(fontsize=6, loc="lower left")
d = res["E1c_delta"]
for R in [1, 8, 32]:
    rows = [r for r in d if r["R"] == R]
    ax[2].errorbar([r["M"] / r["d"] for r in rows], [r["acc"] for r in rows], yerr=[1.96 * r["se"] for r in rows], marker="o", capsize=2, label=f"revisions per key = {R}")
ax[2].set_xscale("log"); ax[2].set_xticks([0.125, 0.25, 0.5, 1, 2, 4]); ax[2].set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4"]); ax[2].minorticks_off()
ax[2].set_xlabel("distinct keys M / key dim d (d=64, random keys)"); ax[2].set_ylabel("latest-value accuracy")
ax[2].set_title("(c) delta rule: capacity-limited, not revision-limited", fontsize=10); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_theory.png"), dpi=150); print("ok")
