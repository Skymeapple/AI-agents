"""Figures/tables added in revision 1: E2 partial-mimicry figure + summary, and Figure 2 redrawn with
sample s.d. (ddof=1) and pooled Wilson 95% intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results"
d = json.load(open(f"{R}/e2_partial.json")); rows = d["rows"]
summ = {}
for q in [0.5, 0.8]:
    for e in [0.1, 0.25, 0.5]:
        for reg in ["STATIC", "ROBUST"]:
            allr = [r for r in rows if r["q"] == q and r["eps_eval"] == e and r["regime"] == reg]
            rs = [r for r in allr if 0.05 < r["U"] < 0.99]      # non-degenerate operating points
            g = np.array([r["gap_T5"] for r in rs]); cg = np.array([r["content_gain"] for r in rs])
            opt = np.array([r["frontier_sonly"] - r["frontier_T5"] for r in rs])
            summ[f"q{q}_eps{e}_{reg}"] = dict(
                n_models=len(allr), n_nondegenerate=len(rs),
                median_gap_to_T5=float(np.median(g)), max_gap_to_T5=float(g.max()),
                median_content_gain=float(np.median(cg)), min_content_gain=float(cg.min()),
                frac_worse_than_s_only=float((cg < 0).mean()), median_optimal_content_gain=float(np.median(opt)),
                median_gmax=float(np.median([r["gmax"] for r in rs])),
                min_gap_all=float(min(r["gap_T5"] for r in allr)))
summ["min_gap_to_T5_over_all_rows"] = float(min(r["gap_T5"] for r in rows))
summ["formula_vs_lp_maxerr"] = d["formula_vs_lp_maxerr"]
summ["TV_content"] = d["config"]["TV_content"]
json.dump(summ, open(f"{R}/e2_summary.json", "w"), indent=1)

fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), sharey="row")
for i, q in enumerate([0.5, 0.8]):
    cv = d["curves"][str(q)]
    for j, e in enumerate([0.1, 0.25, 0.5]):
        a = ax[i, j]
        a.plot(cv["U"], cv["sonly"], "k--", lw=1.2, label="s-only frontier f_K (Thm 3, full mimicry)")
        a.plot(cv["U"], cv[f"eps{e}"], "C2-", lw=2, label=f"Thm 5 frontier, eps={e}")
        a.plot(cv["U"], cv["eps0.0"], "C7:", lw=1.2, label="static frontier (eps=0)")
        for reg, mk, col in [("STATIC", "o", "C3"), ("ROBUST", "^", "C0")]:
            rs = [r for r in rows if r["q"] == q and r["eps_eval"] == e and r["regime"] == reg]
            a.scatter([r["U"] for r in rs], [r["ASR_eps"] for r in rs], marker=mk, color=col, edgecolor="k", s=30,
                      zorder=3, label=f"{reg}-trained models")
        a.set_title(f"q={q} ({'no signal' if q == 0.5 else 'noisy flag'}), eps={e}")
        a.set_xlabel("utility U"); a.set_xlim(0, 1)
        if j == 0: a.set_ylabel("worst-case ASR_eps (exact)")
ax[0, 0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{R}/figs/fig3_partial.png", dpi=150); plt.close()

# Figure 2 redrawn with ddof=1 and Wilson intervals
st = json.load(open(f"{R}/stats_r1.json"))["E3"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); KS = [1, 2, 4, 8]
mk = dict(UNDEF="o", INBAND="s", OOB="^", GATED="D")
for j, g in enumerate(["G1", "G2"]):
    for s in ["UNDEF", "INBAND", "OOB", "GATED"]:
        m = np.array([st[f"{s}_{g}_k{k}"]["mean"] for k in KS])
        lo = np.array([st[f"{s}_{g}_k{k}"]["wilson95_pooled"][0] for k in KS]); hi = np.array([st[f"{s}_{g}_k{k}"]["wilson95_pooled"][1] for k in KS])
        sd = np.array([st[f"{s}_{g}_k{k}"]["sd_ddof1"] for k in KS])
        ax[j].errorbar(KS, m, yerr=sd, marker=mk[s], capsize=3, label=s)
        ax[j].fill_between(KS, lo, hi, alpha=0.15)
    ax[j].set_xscale("log", base=2); ax[j].set_xticks(KS); ax[j].set_xticklabels(KS); ax[j].set_ylim(-0.03, 1.03)
    ax[j].set_xlabel("attacker-controlled tokens k"); ax[j].set_ylabel("adaptive ASR")
    ax[j].set_title({"G1": "G1: hijack a fixed task", "G2": "G2: forge a delegation"}[g] + "\n(bars: seed s.d., ddof=1; band: pooled Wilson 95%)", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{R}/figs/fig2_adaptive.png", dpi=160); plt.close()
print(json.dumps(summ, indent=1))
