"""Figure 6: terminal skill vs guardrail strength kappa (S3 curves from sensitivity.json)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
C = json.load(open(os.path.join(OUT, "sensitivity.json")))["S3"]["curves"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
cols = {0.1: "#C8553D", 0.3: "#2D7DD2", 0.5: "#3BB273"}
for a, beta in zip(ax, [0.5, 1.0]):
    for x in C:
        if x["beta"] == beta and x["c"] in (0.005, 0.01):
            a.plot(x["kappa"], x["sT"], color=cols[x["s0"]], ls="-" if x["c"] == 0.005 else "--", lw=2,
                   label=f"s0={x['s0']}, c={x['c']}")
    a.set_title(f"beta={beta} (q=0.9, w=0.3)", fontsize=10); a.set_xlabel("retained engagement kappa (guardrail strength)")
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("terminal unassisted skill s_T"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_guardrail.png"), dpi=160)
