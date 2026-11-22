"""Summarise MLP results: mean log-ratio over seeds, crossover width, figure."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = json.load(open(os.path.join(OUT, "mlp_results.json")))
hs = np.array(R["hs"]); ks, kp = R["ks"], R["kp"]
summ = {}
ns = sorted(set(r["n"] for r in R["runs"])); Ms = sorted(set(r["M"] for r in R["runs"]))
fig, axs = plt.subplots(1, len(Ms), figsize=(4 * len(Ms), 3.3), sharey=True)
for ax, M in zip(axs, Ms):
    for c, n in zip(plt.cm.viridis(np.linspace(0, .85, len(ns))), ns):
        rr = [r for r in R["runs"] if r["n"] == n and r["M"] == M]
        lr = np.log(np.array([np.array(r["Lu"]) / np.array(r["Ls"]) for r in rr]))
        m, se = lr.mean(0), lr.std(0, ddof=1) / np.sqrt(len(rr))
        Lu = np.mean([r["Lu"] for r in rr], 0); Ls = np.mean([r["Ls"] for r in rr], 0)
        peak = int(np.argmax(m))
        cross = next((int(hs[j]) for j in range(peak, len(hs)) if m[j] < 0), None)
        summ[f"n={n},M={M}"] = dict(ratio_mean=np.exp(m).tolist(), log_ratio_se=se.tolist(),
                                    Lu=Lu.tolist(), Ls=Ls.tolist(), peak_h=int(hs[peak]),
                                    peak_ratio=float(np.exp(m[peak])), h_cross=cross,
                                    ks_plus_Mkp=ks + M * kp, min_ratio=float(np.exp(m.min())),
                                    n_seeds=len(rr))
        ax.errorbar(hs, np.exp(m), yerr=np.exp(m) * se, fmt="o-", ms=3, lw=1, color=c, label=f"n={n}")
    ax.axhline(1, color="k", ls="--", lw=.8)
    ax.axvline(ks, color="gray", ls=":", lw=1); ax.axvline(ks + M * kp, color="gray", ls="-.", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("hidden width h")
    ax.set_xticks(hs); ax.set_xticklabels([str(h) for h in hs], fontsize=7)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title(f"M={M} (dotted k_s, dash-dot k_s+M k_p)", fontsize=8)
axs[0].set_ylabel("L_unified / L_specialist (test)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figF_mlp.png"), dpi=150)
json.dump(summ, open(os.path.join(OUT, "mlp_summary.json"), "w"), indent=1)
for k, v in summ.items():
    print(k, "peak h", v["peak_h"], "peak ratio %.2f" % v["peak_ratio"], "h_cross", v["h_cross"],
          "min %.2f" % v["min_ratio"], "ratios", np.round(v["ratio_mean"], 2).tolist())
