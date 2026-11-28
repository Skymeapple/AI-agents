"""R1 revision audit of the main MLP sweep (results/mlp_<scen>_<seed>.json, 4 seeds):
 - per-run half-recovery steps for GD (range over individual runs, not seed means);
 - paired (same-seed) relative gap reduction vs GD for Adam and the GGN variants, with t-based 95% CIs (n=4);
 - multi-step short-time collapse for GD: [L1(k)-L1(0)] / (eta^2 (1-rho)^2 g2'G1 g2 / 2) for k=1..10
   (eta differs slightly across rho because eta = 1/lambda_max(G_rho(theta_0)); the normalisation removes it to second order).
Output: ../results/mlp_audit_r1.json"""
import json, glob, os, numpy as np, collections
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def ci(x):
    x = np.asarray(x, float); n = len(x); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), n=n, min=float(x.min()), max=float(x.max()))
def tag(r): return f"ggn(d={r['kw']['delta']},lr={r['kw']['lr']})" if r["method"] == "ggn" else r["method"]
runs = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(R, "mlp_*_[0-9]*.json"))):
    o = json.load(open(f))
    for r in o["runs"]:
        L1 = np.array(r["L1"]); G = L1.max() - L1[0]; ip = int(L1.argmax())
        half = np.where(L1[ip:] - L1[0] <= 0.5 * G)[0]
        d = dict(G=float(G), t_half=int(ip + half[0]) if len(half) else None, t_peak=ip)
        if r["method"] == "gd":
            d["short"] = ((L1[1:11] - L1[0]) / (r["lr"] ** 2 * r["quad_pred_coef"])).tolist()
        runs[(o["scenario"], tag(r), r["rho"])][o["seed"]] = d
out = {"t_half_gd": {}, "paired_rel_change_vs_gd": {}, "short_time_gd": {}}
for (scen, t, rho), d in sorted(runs.items()):
    key = f"{scen}|{t}|{rho}"
    if t == "gd":
        th = [d[s]["t_half"] for s in sorted(d)]
        out["t_half_gd"][key] = dict(per_seed=th, n_censored=sum(x is None for x in th))
        S = np.array([d[s]["short"] for s in sorted(d)])
        out["short_time_gd"][key] = {f"k={k}": dict(mean=float(S[:, k - 1].mean()), min=float(S[:, k - 1].min()), max=float(S[:, k - 1].max())) for k in [1, 2, 3, 5, 10]}
    else:
        base = runs[(scen, "gd", rho)]
        rel = [d[s]["G"] / base[s]["G"] - 1 for s in sorted(d)]
        c = ci(rel); c["p_paired_t"] = float(stats.ttest_1samp([d[s]["G"] - base[s]["G"] for s in sorted(d)], 0).pvalue)
        c["n_lower"] = int(sum(x < 0 for x in rel)); out["paired_rel_change_vs_gd"][key] = c
json.dump(out, open(os.path.join(R, "mlp_audit_r1.json"), "w"), indent=1)
for k, v in out["t_half_gd"].items(): print("T", k, v)
for k, v in out["paired_rel_change_vs_gd"].items(): print("P", k, {a: round(b, 4) for a, b in v.items()})
for k, v in out["short_time_gd"].items(): print("S", k, {a: (round(b["mean"], 3), round(b["min"], 3), round(b["max"], 3)) for a, b in v.items()})
