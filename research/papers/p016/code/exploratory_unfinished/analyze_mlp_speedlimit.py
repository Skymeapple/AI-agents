"""Aggregate mlps_*.json (GD runs) -> results/mlps_summary.json, fig9_mlp_recovery.png
Recovery time T (steps) as defined in mlp_speedlimit.py; local log-log exponents between consecutive rho, per (scenario, seed)."""
import json, glob, os, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "mlps_*_[0-9]*.json")))]
S = {"meta": dict(HID=J[0]["HID"], N_PER=J[0]["N_PER"], P=J[0]["P"], n_files=len(J)), "runs": [], "local_exponents": {}, "T_table": {}}
by = collections.defaultdict(dict)
for j in J:
    for r in j["runs"]:
        if r["method"] != "GD": continue
        if r["rho"] == 0:
            S["runs"].append(dict(scen=j["scenario"], seed=j["seed"], rho=0.0, G=r["G"], L10=r["L10"], final_L2=r["final_L2"])); continue
        by[(j["scenario"], j["seed"])][r["rho"]] = r
        S["runs"].append(dict(scen=j["scenario"], seed=j["seed"], rho=r["rho"], G=r["G"], G_over_Fref=r["G"] / j["Fref"], T=r["T"],
                              censored=r["censored"], steps=r["steps_run"], k_peak=r["k_peak"], lr=r["lr"],
                              final_excess_over_Fref=r["final_ex"] / j["Fref"], final_L2_over_L20=r["final_L2"] / r["L20"], L10=r["L10"]))
for (scen, seed), d in sorted(by.items()):
    xs = sorted(d, reverse=True)
    S["T_table"][f"{scen}|{seed}"] = {str(x): d[x]["T"] for x in xs}
    for a, b in zip(xs, xs[1:]):
        if d[a]["T"] and d[b]["T"]:
            S["local_exponents"][f"{scen}|{seed}|{a}->{b}"] = float(np.log(d[b]["T"] / d[a]["T"]) / np.log(b / a))
agg = collections.defaultdict(list)
for k, v in S["local_exponents"].items():
    scen, seed, seg = k.split("|"); agg[f"{scen}|{seg}"].append(v)
S["local_exponent_summary"] = {k: dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)), n=len(v)) for k, v in agg.items()}
dep = collections.defaultdict(list)
for r in S["runs"]:
    if r["rho"] > 0: dep[f"{r['scen']}|{r['rho']}"].append(r["G_over_Fref"])
S["depth_over_norePlay"] = {k: dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)), n=len(v)) for k, v in dep.items()}
S["n_censored"] = int(sum(bool(r.get("censored")) for r in S["runs"]))
json.dump(S, open(os.path.join(R, "mlps_summary.json"), "w"), indent=1)
for k, v in S["T_table"].items(): print("T", k, v)
for k, v in S["local_exponent_summary"].items(): print("exp", k, {a: round(b, 3) for a, b in v.items()})
for k, v in S["depth_over_norePlay"].items(): print("depth", k, {a: round(b, 3) for a, b in v.items()})
print("censored", S["n_censored"], "max final excess/Fref", max(r["final_excess_over_Fref"] for r in S["runs"] if r["rho"] > 0),
      "max phase-1 L1", max(r["L10"] for r in S["runs"]))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for p, scen in enumerate(["perm", "split"]):
    for (sc, seed), d in sorted(by.items()):
        if sc != scen: continue
        xs = sorted(x for x in d if d[x]["T"]); ax[p].loglog(xs, [d[x]["T"] for x in xs], "o-", ms=3, label=f"seed {seed}")
    xr = np.array([0.01, 0.3]); ax[p].loglog(xr, 200 / xr, "k:", lw=0.8, label="slope −1")
    ax[p].set_xlabel("ρ"); ax[p].set_ylabel("recovery time T (GD steps)"); ax[p].set_title(f"over-parameterised MLP, {scen}"); ax[p].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig9_mlp_recovery.png"), dpi=140)
