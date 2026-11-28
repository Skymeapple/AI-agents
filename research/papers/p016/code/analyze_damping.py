"""Aggregate damping-scaling runs -> results/mlpdamp_summary.json"""
import json, glob, os, numpy as np, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
g = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(R, "mlpdamp_*_[0-9].json"))):
    o = json.load(open(f))
    for r in o["runs"]:
        L1 = np.array(r["L1"]); g[(o["scenario"], r["rho"], r["delta"])].append(
            dict(G=float(L1.max() - L1[0]), L1_final_inc=float(L1[-1] - L1[0]), L2_final=float(r["L2"][-1]),
                 A1_min=float(min(r["A1"])), A1_0=float(r["A1"][0])))
out = {}
for k, ms in sorted(g.items()):
    out["|".join(map(str, k))] = {kk: [float(np.mean([m[kk] for m in ms])), float(np.std([m[kk] for m in ms])), len(ms)] for kk in ms[0]}
    print(k, {kk: round(v[0], 4) for kk, v in out["|".join(map(str, k))].items()})
json.dump(out, open(os.path.join(R, "mlpdamp_summary.json"), "w"), indent=1)
