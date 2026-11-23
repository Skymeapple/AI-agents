"""Summaries of the revision-1 results used in the paper (writes ../results/r1_tables.json)."""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = lambda f: json.load(open(os.path.join(R, f)))
out = {}
# ---- R1: family winners in calibrated games (target), B=32 and 16
cal = L("r1_r1_calibrated.json")["table"]
win = {}
for r in cal:
    if r["target"]["chain"]["value"] < 1e-3 and r["target"]["graph"]["value"] < 1e-3:
        key_skip = True
    t = r["target"]
    best = max(t, key=lambda f: t[f]["value"])
    k = f'B={r["B"]},rho={r["rho"]},rho_s={r["rho_s_mode"]}'
    w = win.setdefault(k, {"n": 0, "chain": 0, "chain-r": 0, "graph": 0, "hybrid": 0,
                           "graph_ge_chain_r": 0, "cross_ge_chain_r": 0, "nontrivial": 0})
    w["n"] += 1; w[best] += 1
    w["graph_ge_chain_r"] += t["graph"]["value"] >= t["chain-r"]["value"] - 1e-12
    w["cross_ge_chain_r"] += max(t["graph"]["value"], t["hybrid"]["value"]) >= t["chain-r"]["value"] - 1e-12
    w["nontrivial"] += max(v["value"] for v in t.values()) >= 0.05
out["r1_winners_target"] = win
sw = {}
for r in cal:
    s = r["strict"]; best = max(s, key=lambda f: s[f]["value"])
    k = f'B={r["B"]},rho={r["rho"]},rho_s={r["rho_s_mode"]}'
    w = sw.setdefault(k, {"n": 0, "chain": 0, "chain-r": 0, "graph": 0, "hybrid": 0})
    w["n"] += 1; w[best] += 1
out["r1_winners_strict"] = sw
# by game, B=32, eq: ranges
rng = {}
for r in cal:
    if r["B"] != 32:
        continue
    k = f'{r["game"]}|Delta={int(r["Delta"])}|rho={r["rho"]}|{r["rho_s_mode"]}'
    t = r["target"]
    e = rng.setdefault(k, {f: [] for f in t})
    for f in t:
        e[f].append(t[f]["value"])
out["r1_ranges_B32"] = {k: {f: [min(v), max(v)] for f, v in e.items()} for k, e in rng.items()}
# houdini-steeper-like (Backdoor) maximum target
out["r1_backdoor_max_target"] = {str(D): max(max(v["value"] for v in r["target"].values())
                                            for r in cal if r["game"] == "Backdoor Code" and r["Delta"] == D)
                                 for D in (400.0, 800.0, 1600.0)}
# anchor use among graph winners
an = {}
for r in cal:
    if r["B"] != 32 or r["rho_s_mode"] != "eq":
        continue
    g = r["target"]["graph"]
    a = an.setdefault(r["game"], {"n": 0, "anchor": 0, "full": 0})
    a["n"] += 1; a["anchor"] += bool(g["anchor"]); a["full"] += bool(g["full"])
out["r1_graph_anchor_full"] = an
# ---- R7 rho_s crossover
r7 = L("r1_r7_rho_s.json")
cross = {}
for r in r7:
    if r["B"] != 32:
        continue
    k = f'{r["regime"]}|Delta={int(r["Delta"])}|rho={r["rho"]}'
    t = r["target"]
    cross.setdefault(k, []).append(dict(rho_s=r["rho_s"], chain_r=t["chain-r"]["value"],
                                        graph=t["graph"]["value"], hybrid=t["hybrid"]["value"],
                                        chain=t["chain"]["value"],
                                        strict_chain_r=r["strict"]["chain-r"]["value"],
                                        strict_graph=r["strict"]["graph"]["value"]))
out["r7_crossover"] = cross
# ---- R3 robustness
r3 = L("r1_r3_robustness.json")
rob = []
for e in r3:
    row = dict(game=e["game"], Delta=e["Delta"], rho=e["rho"], c=e["c"], t=e["t"])
    for obj, fams in e["objectives"].items():
        best = max(fams, key=lambda f: fams[f]["value"])
        b = fams[best]
        row[obj] = dict(best_family=best, n=b["n"], k=b["k"], r=b["r"], anchor=b["anchor"],
                        full=b["full"], value=b["value"], target=b["target"], pfa=b["pfa"],
                        chain=fams["chain"]["value"], chain_r=fams["chain-r"]["value"],
                        graph=fams["graph"]["value"], hybrid=fams.get("hybrid", {}).get("value"))
    rob.append(row)
out["r3"] = rob
# aggregate: share of settings where a cross-generation design (graph/hybrid) is best
agg = {}
for row in rob:
    for obj in [k for k in row if k.startswith("kappa") or k.startswith("pfa")]:
        if row[obj]["value"] < 0.02:      # skip degenerate settings (all designs fail)
            continue
        for keyname, cond in (("all", True), (f"c={row['c']}", True), (f"t={row['t']}", True)):
            k = f"{obj}|{keyname}"
            a = agg.setdefault(k, {"n": 0, "cross_best": 0, "chain_r_best": 0, "chain_best": 0,
                                   "k_ge_3_best": 0})
            a["n"] += 1
            f = row[obj]["best_family"]
            a["cross_best"] += f in ("graph", "hybrid")
            a["chain_r_best"] += f == "chain-r"
            a["chain_best"] += f == "chain"
            a["k_ge_3_best"] += (f in ("graph", "hybrid") and (row[obj]["k"] >= 3 or row[obj]["full"]))
out["r3_agg"] = agg
json.dump(out, open(os.path.join(R, "r1_tables.json"), "w"), indent=1)
print(json.dumps(out["r1_winners_target"], indent=0))
print(json.dumps(out["r1_winners_strict"], indent=0))
print(out["r1_backdoor_max_target"], out["r1_graph_anchor_full"])
for k, v in cross.items():
    print(k, " ".join(f'{x["rho_s"]}:{x["chain_r"]:.3f}/{x["graph"]:.3f}/{x["hybrid"]:.3f}' for x in v))
for k, v in agg.items():
    if "|all" in k: print(k, v)
