"""Figure 6: definitional share of arrival-date variance by definition menu and by non-definitional uncertainty set.
Usage: python3 fig_decomposition.py  (after decompose.py for 1-1 and 1-0)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import RESULTS

menus = [("all24", "all 24"), ("defensible18 (drop p=+1, min-family)", "drop p=+1, min (18)"),
         ("recommended8 (pooled,geo,q25,q10 x q=.5,.8)", "recommended (8)"),
         ("threshold_only pooled q=.5,.8", "pooled, q=.5 vs .8")]
configs = [("task", "task resampling"), ("task+time", "+ human-time error"),
           ("task+window", "+ window choice"), ("all (time+loso+dates+window+form)", "all sources")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, v in zip(axes, ["1-1", "1-0"]):
    try:
        D = json.load(open(f"{RESULTS}/decomp_{v}.json"))["decomposition"]
    except FileNotFoundError:
        continue
    w = 0.2
    for j, (c, cl) in enumerate(configs):
        x = np.arange(len(menus)) + (j - 1.5) * w
        s = [D[c][m]["share"] for m, _ in menus]
        ax.bar(x, s, w, label=cl, color=plt.cm.Blues(0.35 + 0.18 * j))
        if c == "task":   # nested-bootstrap 90% interval (only computed for task resampling)
            try:
                N = json.load(open(f"{RESULTS}/nested_share_{v}.json"))
                lo = [N[m]["lo"] for m, _ in menus]; hi = [N[m]["hi"] for m, _ in menus]
                ax.errorbar(x, s, yerr=[np.clip(np.array(s) - lo, 0, None), np.clip(np.array(hi) - s, 0, None)],
                            fmt="none", ecolor="k", lw=0.8, capsize=2)
            except FileNotFoundError:
                pass
    ax.set_xticks(np.arange(len(menus))); ax.set_xticklabels([ml for _, ml in menus], fontsize=8, rotation=15)
    ax.set_title(f"METR TH{v.replace('-', '.')}", fontsize=10); ax.set_ylim(0, 1)
axes[0].set_ylabel("definitional share of arrival-date variance")
axes[0].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(f"{RESULTS}/fig_decomposition.png", dpi=160)
print("saved")
