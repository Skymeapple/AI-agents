"""Analysis 1: per-agent 2-parameter logistic vs 3-parameter (lapse) psychometric fits.

Outputs results/fits.csv, results/fits_bootstrap.json, results/fig1_curves.png
"""
import json
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

B = int(os.environ.get("NBOOT", 300))
QS = [0.5, 0.8, 0.9, 0.95, 0.99]
rng = np.random.default_rng(0)
rows = load_runs()
agents = sorted(RELEASE, key=date_float)

out, boot = [], {}
for al in agents:
    T = task_table(rows, al)
    x, n, s, fam = T["x"], T["n"], T["s"], T["fam"]
    f2 = fit(x, n, s, lapse=False)
    f3 = fit(x, n, s, lapse=True)
    LR = max(0.0, 2 * (f2["nll"] - f3["nll"]))
    p_lrt = 0.5 * chi2.sf(LR, 1) if LR > 0 else 1.0  # boundary-corrected (chi-bar-square)
    # profile-likelihood CI for lambda
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 0.25, 80)])
    prof = np.array([fit(x, n, s, fixed_lam=g)["nll"] for g in grid])
    ok = grid[2 * (prof - prof.min()) <= chi2.ppf(0.95, 1)]
    lam_lo, lam_hi = ok.min(), ok.max()
    # equal-task-weight robustness (weights rescaled to mean 1)
    w = T["w"] / T["w"].mean()
    f3w = fit(x, n, s, w=w, lapse=True)
    f2w = fit(x, n, s, w=w, lapse=False)
    rec = dict(agent=short(al), date=RELEASE[al], tasks=len(x), runs=int(n.sum()),
               succ_rate=float(s.sum() / n.sum()),
               a2=f2["a"], b2=f2["b"], a3=f3["a"], b3=f3["b"], lam=f3["lam"],
               lam_lo=lam_lo, lam_hi=lam_hi, LR=LR, p_lrt=p_lrt,
               dAIC=2 * (f3["nll"] - f2["nll"]) + 2, lam_w=f3w["lam"])
    for q in QS:
        rec[f"H{int(q*100)}_2pl"] = horizon(f2, q)
        rec[f"H{int(q*100)}_3pl"] = horizon(f3, q)
        rec[f"H{int(q*100)}_2pl_w"] = horizon(f2w, q)
        rec[f"H{int(q*100)}_3pl_w"] = horizon(f3w, q)
    # family-cluster bootstrap
    bl = {k: [] for k in ["lam"] + [f"H{int(q*100)}_{m}" for q in QS for m in ("2pl", "3pl")]}
    for b in range(B):
        idx = cluster_bootstrap_indices(fam, rng)
        g2 = fit(x[idx], n[idx], s[idx], lapse=False)
        g3 = fit(x[idx], n[idx], s[idx], lapse=True)
        bl["lam"].append(g3["lam"])
        for q in QS:
            bl[f"H{int(q*100)}_2pl"].append(horizon(g2, q))
            bl[f"H{int(q*100)}_3pl"].append(horizon(g3, q))
    for k, v in bl.items():
        v = np.array(v, dtype=float)
        rec[f"{k}_bfrac_defined"] = float(np.mean(np.isfinite(v)))
        vv = np.where(np.isfinite(v), v, 0.0)  # undefined horizon == 0 minutes
        rec[f"{k}_blo"], rec[f"{k}_bhi"] = np.percentile(vv, [2.5, 97.5]).tolist()
    boot[short(al)] = {k: np.array(v, dtype=float).tolist() for k, v in bl.items()}
    out.append(rec)
    print(f"{short(al):28s} lam={f3['lam']:.4f} [{lam_lo:.4f},{lam_hi:.4f}] p={p_lrt:.3g} "
          f"H50 2pl={rec['H50_2pl']:.1f} 3pl={rec['H50_3pl']:.1f}  H99 2pl={rec['H99_2pl']:.3f} "
          f"3pl={rec['H99_3pl']:.3f} def={rec['H99_3pl_bfrac_defined']:.2f}", flush=True)

keys = list(out[0].keys())
with open(os.path.join(RESULTS, "fits.csv"), "w") as fh:
    fh.write(",".join(keys) + "\n")
    for r in out:
        fh.write(",".join(str(r[k]) for k in keys) + "\n")
json.dump(boot, open(os.path.join(RESULTS, "fits_bootstrap.json"), "w"))

# ---- Figure 1: empirical success vs length with 2PL and 3PL curves for 6 agents
sel = ["GPT-4o (Inspect)", "Claude 3.7 Sonnet (Inspect)", "o3 (Inspect)",
       "Claude Opus 4.5 (Inspect)", "Gemini 3 Pro", "GPT-5.3-Codex"]
fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
xs = np.linspace(-6, 11.5, 300)
edges = np.array([-6, -3, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11.5])
for ax, al in zip(axs.ravel(), sel):
    T = task_table(rows, al)
    x, n, s = T["x"], T["n"], T["s"]
    f2 = fit(x, n, s, lapse=False); f3 = fit(x, n, s, lapse=True)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if n[m].sum() == 0:
            continue
        k, N = s[m].sum(), n[m].sum()
        cl = clopper_pearson(N - k, N)
        ax.errorbar((lo + hi) / 2, 1 - k / N, yerr=[[1 - k / N - cl[0]], [cl[1] - (1 - k / N)]],
                    fmt="o", color="k", ms=4, capsize=2)
    ax.plot(xs, 1 - p_model([f2["a"], f2["b"]], xs, lapse=False), color="#1f77b4", label="2-param logistic")
    ax.plot(xs, 1 - (1 - f3["lam"]) * expit(f3["a"] - f3["b"] * xs), color="#d62728", label="3-param (lapse)")
    ax.axhline(0.01, ls=":", color="gray"); ax.axhline(0.05, ls=":", color="gray")
    ax.set_yscale("log"); ax.set_ylim(5e-4, 1.2)
    ax.set_title(f"{short(al)}  (lapse={f3['lam']:.3f})", fontsize=10)
    ticks = [-5, -2, 0, 2, 4, 6, 8, 10]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{2**t:g}" if t >= 0 else f"{2**t*60:.0f}s" for t in ticks], fontsize=8)
for ax in axs[1]:
    ax.set_xlabel("human task length (minutes)")
for ax in axs[:, 0]:
    ax.set_ylabel("failure rate (log scale)")
axs[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("Failure rate vs task length (METR TH1.1 runs; bins with 95% Clopper-Pearson CIs)")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig1_curves.png"), dpi=150)
print("done")
