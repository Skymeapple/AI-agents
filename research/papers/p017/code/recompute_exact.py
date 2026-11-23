"""Recompute Acc*_k with the exact finite-time prior (fix for review M4) for every saved run,
and re-derive k_eff and the truncation-test excess against it. No model is retrained: the
test data are regenerated deterministically from the same seeds as run_experiment.py."""
import json, os, glob
import numpy as np
from world import TorusWorld, exact_curve
import run_experiment as R

RES = R.RES
out = {}
for mode in ["unique", "balanced2", "landmark"]:
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        w = TorusWorld(5, mode, seed)
        w.sample(14000, R.T, rng); w.sample(6000, R.T, rng)
        te_toks, te_pos, te_mv, te_ob = w.sample(2000, R.T, rng)
        ce = exact_curve(w, te_mv, te_ob, R.EVAL_T)
        ct = {t: exact_curve(w, te_mv, te_ob, [t], kmax=16) for t in R.TRUNC_T}
        ctm = np.mean([ct[t] for t in R.TRUNC_T], 0).tolist()
        raw = json.load(open(os.path.join(RES, f"raw_{mode}_s{seed}.json")))
        rec = {"exact_curve": ce, "exact_curve_trunc_t": ctm, "uniform_curve": raw["suffix_curve"], "models": {}}
        for name, r in raw["models"].items():
            keff = next((k for k, a in enumerate(ce) if a >= r["probe_acc"] - 1e-12), None)
            m = {"probe_acc": r["probe_acc"], "k_eff_exact": keff, "k_eff_uniform": r["k_eff"]}
            if "trunc" in r:
                m["trunc_excess"] = {k: r["trunc"][k]["trunc_probe_acc"] - ctm[int(k)] for k in r["trunc"]}
            rec["models"][name] = m
        out[f"{mode}_s{seed}"] = rec
        print(mode, seed, [round(ce[k], 3) for k in [1, 2, 4, 8, 12, 16, 24]],
              {n: (v["k_eff_exact"], v["k_eff_uniform"]) for n, v in rec["models"].items()},
              "max trunc excess", max([max(v["trunc_excess"].values()) for v in rec["models"].values() if "trunc_excess" in v]))
# dynamics
for f in sorted(glob.glob(os.path.join(RES, "dyn_*.json"))):
    d = json.load(open(f))
    key = f"{d['mode']}_s{d['seed']}"
    ce = out[key]["exact_curve"]
    out[key]["dyn_k_eff_exact"] = [(c["epoch"], c["probe_acc"], next((k for k, a in enumerate(ce) if a >= c["probe_acc"] - 1e-12), None)) for c in d["checkpoints"]]
    print("dyn", key, out[key]["dyn_k_eff_exact"])
json.dump(out, open(os.path.join(RES, "exact_ceiling.json"), "w"), indent=1)
