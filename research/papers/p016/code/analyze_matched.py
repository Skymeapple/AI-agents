"""Aggregate mlpm_*.json (R1 revision) -> results/mlpm_summary.json, fig6_mlp_pareto.png, fig7_mlp_short_lin.png"""
import json, glob, os, numpy as np, collections
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "mlpm_*_[0-9]*.json")))]
QS = ["0.5", "0.25", "0.1", "0.05"]


def tag(r):
    if r["method"] == "gd": return f"GD c={r['c']}"
    if r["method"] == "adam": return f"Adam lr={r['lr']}"
    return f"GGN-flow lr={r['lr']}"


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return [float(x.mean()) if n else None, None, None, n]
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h), n]

out = {"matched": {}, "paired_vs_GDc1": {}, "short": {}, "lin": {}}
for scen in ["perm", "split"]:
    Js = sorted([j for j in J if j["scenario"] == scen], key=lambda j: j["seed"])
    for rho in [0.1, 0.3]:
        by = collections.defaultdict(dict)
        for j in Js:
            for r in j["runs"]:
                if r["rho"] == rho: by[tag(r)][j["seed"]] = r
        for t, d in by.items():
            row = {"G_raw": ci([r["G"] for r in d.values()]), "L2_final": ci([r["L2_final"] for r in d.values()])}
            for q in QS:
                vals = [r["matched"][q]["G"] for r in d.values() if r["matched"][q]]
                steps = [r["matched"][q]["step"] for r in d.values() if r["matched"][q]]
                row[f"G@{q}"] = ci(vals) if vals else None; row[f"step@{q}"] = ci(steps) if steps else None
            out["matched"][f"{scen}|{rho}|{t}"] = row
        base = by["GD c=1.0"]
        for t, d in by.items():
            if t == "GD c=1.0": continue
            row = {}
            seeds = sorted(set(d) & set(base))
            for key in ["raw"] + QS:
                diffs = []; rel = []
                for s in seeds:
                    if key == "raw": a, b = d[s]["G"], base[s]["G"]
                    else:
                        if not (d[s]["matched"][key] and base[s]["matched"][key]): continue
                        a, b = d[s]["matched"][key]["G"], base[s]["matched"][key]["G"]
                    diffs.append(a - b); rel.append(a / b - 1)
                if diffs:
                    row[key] = dict(diff=ci(diffs), rel=ci(rel), n_lower=int(sum(x < 0 for x in diffs)),
                                    p_paired_t=float(stats.ttest_1samp(diffs, 0).pvalue) if len(diffs) > 1 else None)
            out["paired_vs_GDc1"][f"{scen}|{rho}|{t}"] = row
    # short-time collapse: dL1(k)/(1-rho)^2 relative to rho=0, and linearised/true
    sh = collections.defaultdict(list)
    for j in Js:
        ref = np.array([s for s in j["short"] if s["rho"] == 0.0][0]["dL1"])
        for s in j["short"]:
            a = np.array(s["dL1"]); al = np.array(s["dL1_lin"])
            sh[s["rho"]].append(dict(norm=(a[1:] / (1 - s["rho"]) ** 2) / ref[1:], lin=al[1:] / a[1:]))
    for rho, lst in sh.items():
        N = np.array([x["norm"] for x in lst]); Lr = np.array([x["lin"] for x in lst])
        out["short"][f"{scen}|{rho}"] = {f"k={k}": dict(norm_ratio=ci(N[:, k - 1]), lin_over_true=ci(Lr[:, k - 1]))
                                         for k in [1, 2, 3, 5, 10, 20, 30]}
    # linearisation over the full budget
    li = collections.defaultdict(list)
    for j in Js:
        for l in j["lin"]:
            idx = np.array(l["idx"]); L = np.array(l["L1c"]); Ll = np.array(l["L1c_lin"]); G = l["G"]
            prev = [r for r in j["runs"] if r["rho"] == l["rho"] and r["method"] == "gd" and r["c"] == 1.0]
            if prev: G = prev[0]["G"]; l["t_peak"] = prev[0]["t_peak"]   # exact values from the full-resolution run
            bad = np.where(np.abs(L - Ll) > 0.1 * G)[0]
            li[l["rho"]].append(dict(ratio=l["G_lin"] / G, tp=l["t_peak"], tpl=l["t_peak_lin"],
                                     diverge=int(idx[bad[0]]) if len(bad) else None, G=G, Glin=l["G_lin"]))
    for rho, lst in li.items():
        out["lin"][f"{scen}|{rho}"] = dict(Glin_over_G=ci([x["ratio"] for x in lst]), G=ci([x["G"] for x in lst]),
                                           G_lin=ci([x["Glin"] for x in lst]),
                                           t_peak=ci([x["tp"] for x in lst]), t_peak_lin=ci([x["tpl"] for x in lst]),
                                           diverge_step=ci([x["diverge"] for x in lst if x["diverge"] is not None]),
                                           n_never_diverge=sum(x["diverge"] is None for x in lst))
json.dump(out, open(os.path.join(R, "mlpm_summary.json"), "w"), indent=1)

def fmt(c): return "NA" if c is None else (f"{c[0]:.4f} [{c[1]:.4f},{c[2]:.4f}] n={c[3]}" if c[1] is not None else f"{c[0]:.4f} n={c[3]}")
for k, v in out["matched"].items(): print("M", k, {kk: fmt(vv) for kk, vv in v.items()})
for k, v in out["paired_vs_GDc1"].items():
    print("P", k, {kk: (fmt(vv["rel"]), vv["n_lower"], None if vv["p_paired_t"] is None else round(vv["p_paired_t"], 5)) for kk, vv in v.items()})
for k, v in out["short"].items(): print("S", k, {kk: (fmt(vv["norm_ratio"]), fmt(vv["lin_over_true"])) for kk, vv in v.items()})
for k, v in out["lin"].items(): print("L", k, {kk: (fmt(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})

# Fig 6: forgetting-vs-learning Pareto curves (running max of L1-L1(0) vs L2/L2(0)), mean over seeds on a common L2 grid
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
grid = np.geomspace(0.02, 1, 120)
for p, (scen, rho) in enumerate([("perm", 0.1), ("perm", 0.3), ("split", 0.1), ("split", 0.3)]):
    for t in ["GD c=1.0", "GD c=0.3", "GD c=0.1", "Adam lr=0.001", "Adam lr=0.0003", "GGN-flow lr=0.2"]:
        curves = []
        for j in J:
            if j["scenario"] != scen: continue
            for r in j["runs"]:
                if r["rho"] == rho and tag(r) == t:
                    L1 = np.array(r["L1c"]); L2 = np.array(r["L2c"]) / r["L2_0"]; runmax = np.maximum.accumulate(L1 - L1[0])
                    y = np.full(len(grid), np.nan)
                    for i, q in enumerate(grid):
                        k = np.where(L2 <= q)[0]
                        if len(k): y[i] = runmax[k[0]]
                    curves.append(y)
        if not curves: continue
        C = np.array(curves); ok = ~np.isnan(C).any(0)
        ax[p].plot(grid[ok], C[:, ok].mean(0), ls="--" if "GD" in t else ("-." if "Adam" in t else "-"), label=t)
    ax[p].set_xscale("log"); ax[p].invert_xaxis(); ax[p].set_xlabel("new-task progress L2/L2(0)")
    ax[p].set_ylabel("forgetting so far: max L1 − L1(0)"); ax[p].set_title(f"{scen}, ρ={rho} (mean of {len(curves)} seeds)")
    ax[p].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig6_mlp_pareto.png"), dpi=150); plt.close()

# Fig 7: short-time collapse + linearisation
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for scen, mk in [("perm", "o"), ("split", "s")]:
    for rho in [0.0, 0.01, 0.1, 0.3, 0.5]:
        v = out["short"][f"{scen}|{rho}"]; ks = [1, 2, 3, 5, 10, 20, 30]
        ax[0 if scen == "perm" else 1].plot(ks, [v[f"k={k}"]["norm_ratio"][0] for k in ks], mk + "-", label=f"ρ={rho}")
for a_, s in zip(ax[:2], ["perm", "split"]):
    a_.set_xscale("log"); a_.set_xlabel("step k (common η)"); a_.set_ylabel("ΔL1(k)/(1−ρ)²  ÷  ΔL1(k)|ρ=0")
    a_.set_title(f"{s}: multi-step short-time collapse"); a_.legend(fontsize=7); a_.axhline(1, color="gray", lw=0.6)
j = [j for j in J if j["scenario"] == "split" and j["seed"] == 0][0]
for l in j["lin"]:
    if l["rho"] in (0.01, 0.1, 0.3):
        ln, = ax[2].plot(np.array(l["idx"]) + 1, l["L1c"], label=f"network ρ={l['rho']}")
        ax[2].plot(np.array(l["idx"]) + 1, l["L1c_lin"], "--", color=ln.get_color(), label=f"linearised ρ={l['rho']}")
ax[2].set_xscale("log"); ax[2].set_xlabel("GD step + 1"); ax[2].set_ylabel("L1"); ax[2].set_title("split seed 0: network vs NTK-linearised GD")
ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig7_mlp_short_lin.png"), dpi=150); plt.close()
