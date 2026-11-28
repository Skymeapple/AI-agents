"""Produce figures and tables from E1 and E3 outputs (run from code/)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

R = "../results"
os.makedirs(f"{R}/figs", exist_ok=True)
out = {}


def np_frontier(U, q):
    """minimum ASR at utility U for binary symmetric provenance flag (Theorem 2)"""
    U = np.asarray(U, float)
    if q >= 1.0:
        return np.zeros_like(U)
    return np.where(U <= q, (1 - q) * U / q, (1 - q) + q * (U - q) / (1 - q))


# ---------------- E1 ----------------
e1 = json.load(open(f"{R}/e1_tradeoff.json"))
rows = []
for r in e1:
    fr = float(np_frontier([r["U"]], r["q"])[0])
    n = r["n_eval"]
    se = np.sqrt(max(r["ASR"] * (1 - r["ASR"]), 1 / n) / n + max(r["U"] * (1 - r["U"]), 1 / n) / n)
    rows.append(dict(q=r["q"], w=r["w"], seed=r["seed"], U=r["U"], ASR=r["ASR"], TVbound=r["U"] - (2 * r["q"] - 1),
                     NP=fr, gap=r["ASR"] - fr, z=(r["ASR"] - fr) / se, acc_CLEAN=r["acc_CLEAN"], acc_INJ=r["acc_INJ"]))
gaps = np.array([x["gap"] for x in rows]); zs = np.array([x["z"] for x in rows])
out["E1"] = dict(n_models=len(rows), min_gap=float(gaps.min()), max_gap=float(gaps.max()),
                 median_abs_gap=float(np.median(np.abs(gaps))), min_z=float(zs.min()),
                 n_violate_TV_bound=int(sum(x["ASR"] < x["TVbound"] - 2.5 * 0.0065 for x in rows)),
                 n_below_NP_more_than_2se=int((zs < -2).sum()),
                 min_acc_CLEAN=float(min(x["acc_CLEAN"] for x in rows)), min_acc_INJ=float(min(x["acc_INJ"] for x in rows)),
                 rows=rows)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
uu = np.linspace(0, 1, 201)
for i, q in enumerate([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ax[0].plot(uu, np_frontier(uu, q), color=cols[i], lw=1.5, label=f"q={q}")
    pts = [x for x in rows if x["q"] == q]
    ax[0].scatter([x["U"] for x in pts], [x["ASR"] for x in pts], color=cols[i], edgecolor="k", s=36, zorder=3)
ax[0].set_xlabel("utility U (follow legitimate in-data instruction)")
ax[0].set_ylabel("attack success rate (follow injected instruction)")
ax[0].set_title("E1: trained models vs. Neyman-Pearson frontier")
ax[0].legend(fontsize=8, frameon=False)
# continuous score illustration: Gaussian provenance score with separation d'
for i, d in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
    t = np.linspace(-8, 12, 2000)
    U = 1 - norm.cdf(t - d); A = 1 - norm.cdf(t)
    ax[1].plot(U, A, color=plt.cm.magma(0.15 + 0.17 * i), label=f"d'={d:g}")
ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
ax[1].set_xlabel("utility U"); ax[1].set_ylabel("min. achievable ASR")
ax[1].set_title("Frontier for a Gaussian provenance score (analytic)")
ax[1].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f"{R}/figs/fig1_frontier.png", dpi=160); plt.close()

# ---------------- E3 ----------------
p3 = f"{R}/e3_adaptive.json"
if os.path.exists(p3):
    e3 = json.load(open(p3))
    systems = ["UNDEF", "INBAND", "OOB", "GATED"]
    KS = [1, 2, 4, 8]
    tab = {}
    for s in systems:
        rs = [r for r in e3 if r["system"] == s]
        keys = [k for k in rs[0] if not k.startswith(("adaptQ", "examples")) and k not in ("system", "seed")]
        tab[s] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])), [r[k] for r in rs]) for k in keys}
        # median queries-to-success over all successful episodes, pooled over seeds
        for g in ["G1", "G2"]:
            for k in KS:
                q = np.concatenate([np.array(r[f"adaptQ_{g}_k{k}"]) for r in rs])
                fin = q[np.isfinite(q)]
                tab[s][f"medQ_{g}_k{k}"] = float(np.median(fin)) if len(fin) else None
        tab[s]["examples_G1_k4"] = rs[0].get("examples_G1_k4", [])
    out["E3"] = tab
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    mk = dict(UNDEF="o", INBAND="s", OOB="^", GATED="D")
    for j, g in enumerate(["G1", "G2"]):
        for s in systems:
            m = [tab[s][f"adaptASR_{g}_k{k}"][0] for k in KS]
            sd = [tab[s][f"adaptASR_{g}_k{k}"][1] for k in KS]
            ax[j].errorbar(KS, m, yerr=sd, marker=mk[s], capsize=3, label=s)
        ax[j].set_xscale("log", base=2); ax[j].set_xticks(KS); ax[j].set_xticklabels(KS)
        ax[j].set_ylim(-0.03, 1.03)
        ax[j].set_xlabel("attacker-controlled tokens k")
        ax[j].set_ylabel("adaptive ASR (greedy coordinate search)")
        ax[j].set_title({"G1": "G1: hijack a fixed task", "G2": "G2: forge a delegation"}[g])
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{R}/figs/fig2_adaptive.png", dpi=160); plt.close()

json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps({k: (v if k != "E1" else {kk: vv for kk, vv in v.items() if kk != "rows"}) for k, v in out.items()},
                 indent=1, default=str)[:6000])
