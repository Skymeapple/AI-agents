"""Analysis 3: does high-reliability performance track capability over model releases?

* log2 H50 (2PL, from a1) vs release date -> doubling time
* log short-task failure rate vs release date -> halving time (all tasks / excluding
  consensus-hard tasks)
* task-paired cluster bootstrap for successive releases from the same developer
Outputs results/trends.json, results/paired_tests.csv, results/fig4_trends.png
"""
import json, csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

rows = load_runs()
agents = sorted(RELEASE, key=date_float)
tabs = {a: task_table(rows, a) for a in agents}
fits = {r["agent"]: r for r in csv.DictReader(open(os.path.join(RESULTS, "fits.csv")))}
CONS = {"blackbox/apron", "blackbox/beach"}  # failed in all 46 frontier runs (see a2 output)
rng = np.random.default_rng(2)


def short_fail(al, TH, drop=()):
    T = tabs[al]
    m = (T["minutes"] < TH) & ~np.isin(T["task"], list(drop))
    return T["n"][m].sum() - T["s"][m].sum(), T["n"][m].sum()


def ols(x, y):
    r = stats.linregress(x, y)
    return dict(slope=r.slope, se=r.stderr, intercept=r.intercept, r2=r.rvalue ** 2, p=r.pvalue)


d = np.array([date_float(a) for a in agents])
res = {}
h50 = np.array([float(fits[short(a)]["H50_2pl"]) for a in agents])
o = ols(d, np.log2(h50)); o["doubling_months"] = 12 / o["slope"]
o["doubling_ci"] = [12 / (o["slope"] + 1.96 * o["se"]), 12 / (o["slope"] - 1.96 * o["se"])]
res["H50"] = o
for TH in (4, 15):
    for tag, drop in (("all", ()), ("excl_consensus", CONS)):
        fr = []
        for a in agents:
            F, N = short_fail(a, TH, drop)
            fr.append((F + 0.5) / (N + 1))  # continuity-corrected for log
        fr = np.array(fr)
        o = ols(d, np.log2(fr)); o["halving_months"] = -12 / o["slope"]
        # recent subset (released 2025-04 onwards, i.e. "frontier")
        m = d >= 2025.25
        o2 = ols(d[m], np.log2(fr[m])); o["recent_slope"] = o2["slope"]; o["recent_se"] = o2["se"]; o["recent_p"] = o2["p"]
        o["recent_H50_slope"] = ols(d[m], np.log2(h50[m]))["slope"]
        o["rates"] = fr.tolist()
        res[f"fail_T{TH}_{tag}"] = o
        print(f"T<{TH} {tag}: halving={o['halving_months']:.1f} mo (slope {o['slope']:.3f}+-{o['se']:.3f}); "
              f"since 2025-04: slope {o2['slope']:.3f}+-{o2['se']:.3f} (p={o2['p']:.2f}) vs H50 slope {o['recent_H50_slope']:.3f}")
print(f"H50 doubling: {res['H50']['doubling_months']:.1f} months CI {res['H50']['doubling_ci']}")
json.dump(res, open(os.path.join(RESULTS, "trends.json"), "w"), indent=1)

# paired tests (same tasks), cluster bootstrap over task families
pairs = [("Claude Opus 4.5 (Inspect)", "Claude Opus 4.6 (Inspect)"),
         ("GPT-5.1-Codex-Max (Inspect)", "GPT-5.3-Codex"),
         ("GPT-5 (Inspect)", "GPT-5.2"),
         ("o3 (Inspect)", "GPT-5 (Inspect)"),
         ("Claude 4.1 Opus (Inspect)", "Claude Opus 4.5 (Inspect)")]
out = []
for a, b in pairs:
    for TH in (4, 15):
        A, Bt = tabs[a], tabs[b]
        common = sorted(set(A["task"][A["minutes"] < TH]) & set(Bt["task"][Bt["minutes"] < TH]))
        ia = np.array([np.where(A["task"] == t)[0][0] for t in common])
        ib = np.array([np.where(Bt["task"] == t)[0][0] for t in common])
        fa = 1 - A["s"][ia] / A["n"][ia]; fb = 1 - Bt["s"][ib] / Bt["n"][ib]
        fam = A["fam"][ia]
        diff = fb.mean() - fa.mean()
        bs = []
        for _ in range(5000):
            idx = cluster_bootstrap_indices(fam, rng)
            bs.append(fb[idx].mean() - fa[idx].mean())
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p2 = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
        H50a, H50b = float(fits[short(a)]["H50_2pl"]), float(fits[short(b)]["H50_2pl"])
        out.append(dict(older=short(a), newer=short(b), T=TH, tasks=len(common), fail_older=fa.mean(),
                        fail_newer=fb.mean(), diff=diff, lo=lo, hi=hi, p_boot=p2, H50_older=H50a, H50_newer=H50b))
        print(f"{short(a)} -> {short(b)} T<{TH}: fail {fa.mean():.4f} -> {fb.mean():.4f} diff {diff:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] p~{p2:.3f}; H50 {H50a:.0f} -> {H50b:.0f} min")
with open(os.path.join(RESULTS, "paired_tests.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

# Figure 4
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].semilogy(d, h50, "o", color="k")
xx = np.linspace(d.min(), d.max(), 50)
axs[0].semilogy(xx, 2 ** (res["H50"]["intercept"] + res["H50"]["slope"] * xx), color="#1f77b4",
                label=f"doubling {res['H50']['doubling_months']:.1f} mo")
axs[0].set_ylabel("50% horizon, 2PL (human minutes)"); axs[0].set_xlabel("release date"); axs[0].legend()
axs[0].set_title("Capability: 50% time horizon")
for TH, c in ((4, "#2ca02c"), (15, "#d62728")):
    r = res[f"fail_T{TH}_all"]
    axs[1].semilogy(d, r["rates"], "o", color=c, label=f"tasks < {TH} min (halving {r['halving_months']:.1f} mo)")
    axs[1].semilogy(xx, 2 ** (r["intercept"] + r["slope"] * xx), color=c, alpha=0.6)
    r2 = res[f"fail_T{TH}_excl_consensus"]
    if TH == 15:
        axs[1].semilogy(d, r2["rates"], "x", color=c, alpha=0.7, label="tasks < 15 min, excl. 2 consensus-hard")
axs[1].axhline(0.01, ls=":", color="gray")
axs[1].set_ylabel("run failure rate on short tasks"); axs[1].set_xlabel("release date")
axs[1].legend(fontsize=7); axs[1].set_title("Reliability in the short-task regime")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig4_trends.png"), dpi=150)
