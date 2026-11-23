"""Aggregate mlps_*.json -> results/mlps_summary.json, fig9_mlp_speedlimit.png"""
import json, glob, os, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "mlps_*_[0-9]*.json")))]
S = {"runs": []}
by = collections.defaultdict(dict)
for j in J:
    for r in j["runs"]:
        if r["rho"] == 0: continue
        by[(j["scenario"], r["method"], j["seed"])][r["rho"]] = r
        S["runs"].append(dict(scen=j["scenario"], seed=j["seed"], method=r["method"], rho=r["rho"], G=r["G"], T=r["T"],
                              censored=r["censored"], steps=r["steps_run"], k_peak=r["k_peak"],
                              Gq01=r["matched"].get("0.1", {}).get("G") if r["matched"].get("0.1") else None,
                              final_ex=r["final_ex"], Fref=j["Fref"]))
S["slopes"] = {}
for (scen, m, seed), d in sorted(by.items()):
    for lo, hi, tag in [(0.001, 0.03, "small"), (0.03, 0.3, "large"), (0.001, 0.3, "all")]:
        xs = [x for x in sorted(d) if lo <= x <= hi and d[x]["T"] is not None]
        if len(xs) >= 3 and len(xs) == len([x for x in d if lo <= x <= hi]):
            S["slopes"][f"{scen}|{m}|{seed}|{tag}"] = float(np.polyfit(np.log(xs), np.log([d[x]["T"] for x in xs]), 1)[0])
agg = collections.defaultdict(list)
for k, v in S["slopes"].items():
    scen, m, seed, tag = k.split("|"); agg[f"{scen}|{m}|{tag}"].append(v)
S["slope_summary"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v), vals=v) for k, v in sorted(agg.items())}
# depth ratios momentum / GD, per rho (paired by seed)
dr = collections.defaultdict(list)
for (scen, m, seed), d in by.items():
    if m == "GD": continue
    g = by.get((scen, "GD", seed), {})
    for x in d:
        if x in g: dr[f"{scen}|{m}|{x}"].append(d[x]["G"] / g[x]["G"])
S["depth_ratio_vs_GD"] = {k: dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)), n=len(v)) for k, v in sorted(dr.items())}
tr = collections.defaultdict(list)
for (scen, m, seed), d in by.items():
    if m == "GD": continue
    g = by.get((scen, "GD", seed), {})
    for x in d:
        if x in g and d[x]["T"] and g[x]["T"]: tr[f"{scen}|{m}|{x}"].append(d[x]["T"] / g[x]["T"])
S["T_ratio_vs_GD"] = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(tr.items())}
json.dump(S, open(os.path.join(R, "mlps_summary.json"), "w"), indent=1)
for k, v in S["slope_summary"].items(): print(k, v)
for k, v in S["depth_ratio_vs_GD"].items(): print("depth", k, v)
for k, v in S["T_ratio_vs_GD"].items(): print("T", k, v)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
col = {"GD": "k", "HB_0.9": "b", "HB_tuned": "c"}
for p, scen in enumerate(["perm", "split"]):
    for (sc, m, seed), d in sorted(by.items()):
        if sc != scen: continue
        xs = sorted(x for x in d if d[x]["T"] is not None)
        ax[p].loglog(xs, [d[x]["T"] for x in xs], "o-", color=col[m], alpha=0.6, ms=3,
                     label=m if seed == min(s for (s2, m2, s) in by if s2 == scen and m2 == m) else None)
    xr = np.array([1e-3, 0.3])
    ax[p].loglog(xr, 30 * (xr / 0.3) ** -1.0 * 10, "k:", lw=0.8, label="slope −1")
    ax[p].loglog(xr, 30 * (xr / 0.3) ** -0.5 * 3, "c:", lw=0.8, label="slope −1/2")
    ax[p].set_xlabel("ρ"); ax[p].set_ylabel("recovery time T (steps)"); ax[p].set_title(f"MLP {scen}"); ax[p].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig9_mlp_speedlimit.png"), dpi=140)
