"""Condense results/exp2_sim.json + exp2_structure.json into the tables quoted in the paper
(results/exp2_summary.json)."""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
st = json.load(open(os.path.join(OUT, "exp2_structure.json")))
r = json.load(open(os.path.join(OUT, "exp2_sim.json")))
main = {}
for x in r:
    if "sweep" in x["graph"]: continue
    g = x["graph"].split("|")
    key = x["policy"] if len(g) == 1 else g[1] + ":" + x["policy"]
    main.setdefault(g[0], {})[key] = x
keys = ["uniform", "reach", "sink-aware", "combined", "null:uniform", "kappa0:uniform", "kappa0:sink-aware"]
tab = {g: {k: dict(success=d[k]["success"], se=round(d[k]["success_se"], 4), prop4=round(d[k]["success_prop4"], 4),
                   cost=round(d[k]["cost"], 4), latent=round(d[k]["latent"], 4)) for k in keys} for g, d in main.items()}
agg = dict(
    mean_success={k: float(np.mean([d[k]["success"] for d in main.values()])) for k in keys},
    mean_cost={k: float(np.mean([d[k]["cost"] for d in main.values()])) for k in keys},
    n_graphs=len(main),
    sink_beats_uniform=int(sum(d["sink-aware"]["success"] > d["uniform"]["success"] for d in main.values())),
    sink_beats_uniform_kappa0=int(sum(d["kappa0:sink-aware"]["success"] > d["kappa0:uniform"]["success"] for d in main.values())),
    reach_below_uniform=int(sum(d["reach"]["success"] < d["uniform"]["success"] for d in main.values())),
    null_above_real=int(sum(d["null:uniform"]["success"] > d["uniform"]["success"] for d in main.values())),
    prop4_kappa0_z={k: [round((d[k]["success"] - d[k]["success_prop4"]) / max(d[k]["success_se"], 1e-9), 2) for d in main.values()]
                    for k in ["kappa0:uniform", "kappa0:sink-aware"]},
    prop4_max_abs_err={k: float(max(abs(d[k]["success"] - d[k]["success_prop4"]) for d in main.values())) for k in keys if "null" not in k},
    prop4_mean_signed_err={k: float(np.mean([d[k]["success"] - d[k]["success_prop4"] for d in main.values()])) for k in keys},
    structure_means={f: float(np.mean([s[f] for s in st.values()])) for f in ["frac_sinks", "gini_reach", "neff_frac"]},
    structure_null_means={f: float(np.mean([s["null"][f] for s in st.values()])) for f in ["frac_sinks", "gini_reach", "neff_frac"]},
    sinks_real_gt_null=int(sum(s["frac_sinks"] > s["null"]["frac_sinks"] for s in st.values())),
    neff_real_gt_null=int(sum(s["neff_frac"] > s["null"]["neff_frac"] for s in st.values())),
)
z = np.array(agg["prop4_kappa0_z"]["kappa0:uniform"] + agg["prop4_kappa0_z"]["kappa0:sink-aware"])
agg["prop4_kappa0_z_summary"] = dict(n=int(z.size), mean=float(z.mean()), sd=float(z.std(ddof=1)), max_abs=float(abs(z).max()),
                                     frac_within_2=float((abs(z) < 2).mean()))
sweep = {}
for x in r:
    if "sweep" in x["graph"]:
        g = x["graph"].split("|")[0]
        sweep.setdefault(g, {})[f"delta={x['delta']},b={x['b']},{x['policy']}"] = dict(
            success=x["success"], se=round(x["success_se"], 4), prop4=round(x["success_prop4"], 4), cost=round(x["cost"], 4))
json.dump(dict(aggregate=agg, per_graph=tab, sweep=sweep, structure=st), open(os.path.join(OUT, "exp2_summary.json"), "w"), indent=1)
print(json.dumps(agg, indent=1))
for g, d in tab.items():
    print(g, {k: (v["success"], v["prop4"], v["cost"]) for k, v in d.items()})
for g in sweep:
    for k, v in sweep[g].items(): print(g, k, v)
