"""Figure 4 from results/exp2_summary.json: success vs integration exposure delta, by policy."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
S = json.load(open(os.path.join(OUT, "exp2_summary.json")))["sweep"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
D = [0.0, 0.1, 0.3, 0.6, 0.9]
for a, g in zip(ax, ["numpy", "scipy", "setuptools"]):
    for pol, col in [("uniform", "#1f4e79"), ("sink-aware", "#e67e22"), ("reach", "#c0392b")]:
        y = [S[g][f"delta={d},b=0.02,{pol}"]["success"] for d in D]
        e = [2 * S[g][f"delta={d},b=0.02,{pol}"]["se"] for d in D]
        t = [S[g][f"delta={d},b=0.02,{pol}"]["prop4"] for d in D]
        a.errorbar(D, y, yerr=e, color=col, marker="o", ms=4, lw=1.2, capsize=2, label=pol + " (simulation, kappa=0.3)")
        a.plot(D, t, color=col, ls=":", lw=1, label=pol + " (Prop. 4, kappa=0)")
    a.set_title(f"{g} dependency DAG (b=0.02)"); a.set_xlabel("integration exposure delta"); a.set_ylabel("P(project correct)")
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_dag_policies.png"), dpi=150)
