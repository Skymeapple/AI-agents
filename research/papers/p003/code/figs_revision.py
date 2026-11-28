"""Figures for the round-1 revision: fig8 (audit AUCs with 95% CIs) and fig11 (real judges)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "results")

d = json.load(open(os.path.join(R, "exp8.json")))
a = d["auc"]["all120"]
keys = ["AGG-256", "AGG-B", "LOOK-B", "TAIL-B", "ON-20", "ON-50", "ON-100", "X0-GAP", "H0/J"]
lab = ["aggregate\n256", "aggregate\n2816", "look-ahead\n2816", "tail-aware\n2816", "on-policy\n≤20 steps",
       "on-policy\n≤50 steps", "on-policy\n≤100 steps", "oracle\nx0 gap", "oracle\nh0/J"]
col = ["C0"] * 4 + ["C1"] * 3 + ["C2"] * 2
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
v = np.array([a[k] for k in keys])
ax[0].bar(range(len(keys)), v[:, 0], color=col, alpha=.8)
ax[0].errorbar(range(len(keys)), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="k", capsize=3)
ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylim(0.5, 1.01)
ax[0].set_ylabel("AUC for eventual collapse (95% bootstrap CI)")
ax[0].set_title("E8: audits on 120 configurations (blue offline, orange on-policy, green oracle)", fontsize=9)
td = np.array([r["t_dec"] for r in d["runs"]]); td = td[np.isfinite(td)]
ax[1].hist(td, bins=np.arange(0, 400, 10), color="gray")
for w in [20, 50, 100]: ax[1].axvline(w, color="C1", ls="--")
ax[1].set_xlabel("decision step (exploit > 0.5 or accuracy > 0.5)"); ax[1].set_ylabel("runs")
ax[1].set_title("Most races are decided before step 100", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig8_audits.png"), dpi=150)

d = json.load(open(os.path.join(R, "exp11.json")))
doms = ["knowledge-humanities/social/bio", "knowledge-STEM", "reasoning", "math", "code"]
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, rj in d["judges"].items():
    y = [rj[x]["J_pair"][0] for x in doms]
    ax.plot(range(5), y, "-", color="C3" if k.startswith("reward_model") else "gray", alpha=.35, lw=1)
med = [d["pooled"][x]["median_J_pair_all_judges"] for x in doms]
ax.plot(range(5), med, "ko-", lw=2.5, label="median over 33 judges")
ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(5))
ax.set_xticklabels(["knowledge:\nhumanities/social/bio", "knowledge:\nSTEM", "reasoning", "math", "code"], fontsize=8)
ax.set_ylabel("J_pair = 2·P(prefer correct) − 1"); ax.legend(fontsize=8)
ax.set_title("E11: within-prompt discrimination of real judges (JudgeBench; red = reward models)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig11_real_judges.png"), dpi=150)
