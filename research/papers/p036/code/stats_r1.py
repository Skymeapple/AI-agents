"""Revised statistics for E1 and E3 (response to review round 1).
E3: sample s.d. (ddof=1) over seeds, a two-sided 95% t-interval over the 3 seed means, and a 95% Wilson
    interval on the pooled episodes (all seeds).  Pairwise INBAND-vs-UNDEF G1 comparison per seed.
E1: per-model one-sided test of 'ASR below the frontier' using Agresti-Coull standard errors for both U and
    ASR (delta method on the frontier slope), replacing the ad hoc 1/n floor; the Theorem-1 check uses the
    same per-model SE instead of a fixed tolerance."""
import json
import numpy as np
from scipy.stats import t as tdist, norm

R = "../results"
e3 = json.load(open(f"{R}/e3_adaptive.json"))
out = {"E3": {}, "E1": {}}


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


for s in ["UNDEF", "INBAND", "OOB", "GATED"]:
    rs = [r for r in e3 if r["system"] == s]
    for g in ["G1", "G2"]:
        for k in [1, 2, 4, 8]:
            vals = np.array([r[f"adaptASR_{g}_k{k}"] for r in rs])
            ns = np.array([len(r[f"adaptQ_{g}_k{k}"]) for r in rs])
            xs = np.array([np.isfinite(np.array(r[f"adaptQ_{g}_k{k}"], float)).sum() for r in rs])
            assert np.allclose(xs / ns, vals)
            sd = float(vals.std(ddof=1)); m = float(vals.mean())
            hw = float(tdist.ppf(0.975, len(vals) - 1) * sd / np.sqrt(len(vals)))
            out["E3"][f"{s}_{g}_k{k}"] = dict(mean=m, sd_ddof1=sd, t95=[m - hw, m + hw], pooled_x=int(xs.sum()),
                                             pooled_n=int(ns.sum()), wilson95_pooled=wilson(xs.sum(), ns.sum()))
# INBAND vs UNDEF on G1, per seed
cmp = {}
for k in [1, 2, 4, 8]:
    a = {r["seed"]: r[f"adaptASR_G1_k{k}"] for r in e3 if r["system"] == "INBAND"}
    b = {r["seed"]: r[f"adaptASR_G1_k{k}"] for r in e3 if r["system"] == "UNDEF"}
    cmp[f"k{k}"] = dict(diff_per_seed=[a[s] - b[s] for s in sorted(a)], n_seeds_INBAND_higher=int(sum(a[s] > b[s] for s in a)))
out["E3"]["INBAND_minus_UNDEF_G1"] = cmp

# ---------------- E1 ----------------
e1 = json.load(open(f"{R}/e1_tradeoff.json"))


def frontier(U, q):
    if q >= 1: return 0.0, 0.0
    if U <= q: return (1 - q) * U / q, (1 - q) / q
    return (1 - q) + q * (U - q) / (1 - q), q / (1 - q)


rows = []
for r in e1:
    n = r["n_eval"]
    pa = (r["ASR"] * n + 2) / (n + 4); pu = (r["U"] * n + 2) / (n + 4)
    se_a = np.sqrt(pa * (1 - pa) / (n + 4)); se_u = np.sqrt(pu * (1 - pu) / (n + 4))
    fr, slope = frontier(r["U"], r["q"])
    se = np.sqrt(se_a ** 2 + (slope * se_u) ** 2)
    z = (r["ASR"] - fr) / se
    tvb = r["U"] - (2 * r["q"] - 1); z_tv = (r["ASR"] - tvb) / np.sqrt(se_a ** 2 + se_u ** 2)
    rows.append(dict(q=r["q"], w=r["w"], seed=r["seed"], U=r["U"], ASR=r["ASR"], frontier=fr, gap=r["ASR"] - fr,
                     z=float(z), p_one_sided=float(norm.cdf(z)), z_TV=float(z_tv)))
zs = np.array([x["z"] for x in rows]); ztv = np.array([x["z_TV"] for x in rows])
out["E1"] = dict(min_z=float(zs.min()), n_z_below_minus2=int((zs < -2).sum()), min_z_TV=float(ztv.min()),
                 n_TV_z_below_minus2=int((ztv < -2).sum()),
                 n_saturated=int(sum(1 for x in rows if x["U"] in (0.0, 1.0) or x["ASR"] in (0.0,))),
                 note="Agresti-Coull SEs; z<-2 would indicate a point significantly below the frontier", rows=rows)
json.dump(out, open(f"{R}/stats_r1.json", "w"), indent=1)
for k, v in out["E3"].items():
    if k.startswith(("UNDEF_G1", "INBAND_G1", "OOB_G1", "GATED_G1")) or "minus" in k:
        print(k, v)
print({k: v for k, v in out["E1"].items() if k != "rows"})
