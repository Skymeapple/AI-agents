"""Bar chart of feasible security cases per policy (from permission_analysis.json)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = json.load(open(os.path.join(RES, "permission_analysis.json")))
pol = ["full_suite", "suite_union", "readonly_split", "task_tool_LP", "task_arg_LP_realizable", "task_arg_LP"]
lab = ["full suite", "suite union", "read-only split", "per-task tools", "per-task args\n(realizable)", "per-task args\n(oracle)"]
suites = list(d["suites"])
x = np.arange(len(pol)); w = 0.18
fig, ax = plt.subplots(figsize=(8, 3.6))
cols = ["#2a6fb0", "#c2571a", "#2e8b57", "#8a3fa0"]
for j, s in enumerate(suites):
    v = [d["suites"][s][p] / d["suites"][s]["cases"] for p in pol]
    ax.bar(x + (j - 1.5) * w, v, w, label=s, color=cols[j])
tot = [d["totals"][p] / d["totals"]["cases"] for p in pol]
ax.plot(x, tot, "k_", ms=22, mew=2, label="all suites")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("fraction of security cases feasible")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper right")
ax.grid(axis="y", alpha=.3); ax.set_ylim(0, 1.12)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_policies.png"), dpi=160)
print([round(t, 4) for t in tot])
