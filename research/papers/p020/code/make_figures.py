"""Figures for the empirical results (reads ../results/*.csv)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TASK = {1: "Verbal fluency (animals)", 2: "AUT: brick", 3: "AUT: paperclip"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig: list-divergence ladder
d = pd.read_csv(os.path.join(R, "D_list_divergence_ladder.csv"))
d = d[(d.k == 10) & (d.g == 5)]
order = ["LLM: replicates, T=0", "LLM: replicates, T in 0.1-1.0", "LLM: replicates, T=1.0",
         "LLM: same model, g temperatures", "LLM: same T, g models", "LLM: g models, any T",
         "Humans: g different people"]
labels = ["1 model, T=0\n(replicates)", "1 model, T 0.1–1\n(replicates)", "1 model, T=1\n(replicates)",
          "1 model,\n5 temperatures", "5 models,\nsame T", "5 models,\nany T", "5 different\npeople"]
cols = ["#bfc9d9"] * 3 + ["#8fa6c7", "#4f73a8", "#2a4f86", "#c0392b"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
for j, task in enumerate((1, 2, 3)):
    for i, level in enumerate(("category", "idea_str")):
        ax = axes[i, j]
        s = d[(d.task == task) & (d.level == level)].set_index("structure").loc[order]
        y = s["mean"].to_numpy()
        err = np.vstack([y - s.q025.to_numpy(), s.q975.to_numpy() - y])
        ax.bar(range(len(order)), y, color=cols, yerr=err, capsize=2, error_kw=dict(lw=.7))
        ax.axhline(np.log2(5), ls=":", c="k", lw=.7)
        ax.set_title(f"{TASK[task]} — {'semantic category' if level == 'category' else 'response string'}", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("divergence among 5 lists of 10\n(plug-in JSD, bits)")
for ax in axes[1]:
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=60, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_ladder.png"), dpi=170); plt.close(fig)

# ---- Fig: temperature
t = pd.read_csv(os.path.join(R, "C_temperature_sweep.csv"))
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for task, mk in ((2, "o"), (3, "s")):
    tt = t[t.task == task]
    axes[0].plot(tt.temp, tt.replicate_floor, mk + "-", ms=3, label=TASK[task])
    axes[1].plot(tt.temp, tt.I_model_star, mk + "-", ms=3)
    axes[2].plot(tt.temp, tt.G_within, mk + "-", ms=3)
    axes[3].plot(tt.temp, tt.orig_mean, mk + "-", ms=3)
for ax, ttl in zip(axes, ["run-to-run divergence among 5\nreplicates of one model (bits)",
                          "between-model information\n(list-permutation corrected, bits)",
                          "within-list category entropy\n(bits, 10 responses)",
                          "mean originality rating\n(dataset scores, 1–5)"]):
    ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("sampling temperature T")
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_temperature.png"), dpi=170); plt.close(fig)

# ---- Fig: peer-normed originality
p = pd.read_csv(os.path.join(R, "E_peer_normed_originality_agents.csv"))
conds = [("Human", "human peers (100)", "Humans vs\nhuman norm"),
         ("LLM", "human peers (100)", "LLM vs\nhuman norm"),
         ("LLM", "LLM-ensemble peers (100)", "LLM vs\nLLM-ensemble norm"),
         ("LLM", "same-model peers (<=54)", "LLM vs\nsame-model norm"),
         ("Human", "human peers (54)", "Humans vs\nhuman norm (54)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
w = 0.38
for ti, task in enumerate((2, 3)):
    for ai, (metric, lab) in enumerate((("surprisal", "category surprisal under norm group (bits)"),
                                       ("unique", "share of responses absent from norm group"))):
        ax = axes[ai]
        vals = [p[(p.task == task) & (p["pop"] == a) & (p.norm == b)][metric] for a, b, _ in conds]
        m = [v.mean() for v in vals]; se = [v.sem() for v in vals]
        ax.bar(np.arange(len(conds)) + (ti - .5) * w, m, w, yerr=se, capsize=2,
               color=["#c0392b", "#4f73a8", "#4f73a8", "#4f73a8", "#e8a196"], alpha=1 if ti == 0 else .55,
               label=TASK[task])
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[2] for c in conds], fontsize=7)
axes[0].set_ylim(3, 5.3)
axes[1].legend(frameon=False, fontsize=7.5, title="dark = brick, light = paperclip", title_fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_peer_norm.png"), dpi=170); plt.close(fig)
print("figures written")
