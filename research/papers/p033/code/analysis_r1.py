"""Revision analyses that read existing outputs only (no training):
 - Table 1 with between-seed uncertainty and medians (exp1_frontier.json)
 - per-seed values for the XOR lambda=16 example (exp1_frontier.json)
 - exact incentive-collapse deployment rates (exp2_honeypots.json)
 - probit slope: SD and a t-based 95% CI over the 6 policies (exp2_honeypots.json)
Run: python3 analysis_r1.py
"""
import json, os
import numpy as np
from scipy.stats import t as tdist
from common import RESULTS


def main():
    e1 = json.load(open(os.path.join(RESULTS, "exp1_frontier.json")))
    pts = e1["points"]
    table = {}
    for env, widths in [("gaussian", [0, 16]), ("xor", [0, 2, 8, 32])]:
        for h in widths:
            sel = [p for p in pts if p["env"] == env and p["width"] == h and p["alpha"] > 1e-4]
            eff = np.array([p["efficiency"] for p in sel])
            seed_means = [np.mean([p["efficiency"] for p in sel if p["seed"] == s]) for s in [0, 1, 2]]
            rng = np.random.default_rng(0)
            boot = [np.mean(rng.choice(eff, len(eff))) for _ in range(5000)]
            table[f"{env}_w{h}"] = dict(n=len(sel), n_excluded=sum(
                1 for p in pts if p["env"] == env and p["width"] == h and p["alpha"] <= 1e-4),
                mean=float(eff.mean()), median=float(np.median(eff)), min=float(eff.min()),
                seed_means=[float(x) for x in seed_means], sd_between_seeds=float(np.std(seed_means, ddof=1)),
                boot95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                n_below_0p7=int(np.sum(eff < 0.7)))
    xor16 = [dict(width=p["width"], seed=p["seed"], alpha=p["alpha"], beta=p["beta"], frontier=p["frontier_beta"])
             for p in pts if p["env"] == "xor" and p["lam"] == 16 and p["width"] in (0, 8)]
    e2 = json.load(open(os.path.join(RESULTS, "exp2_honeypots.json")))
    collapse = [dict(rho=q["rho"], lam=q["lam"], seed=q["seed"], lam_rho=q["lam"] * q["rho"], beta=q["beta"],
                     alpha_mix=q["alpha_mix"]) for q in e2["A"] if q["t"] == 1.0]
    slopes = np.array([q["probit_slope"] for q in e2["B"]])
    n = len(slopes)
    ci = tdist.ppf(0.975, n - 1) * slopes.std(ddof=1) / np.sqrt(n)
    out = dict(table1=table, xor_lam16=xor16, incentive_collapse=collapse,
               probit=dict(n=n, mean=float(slopes.mean()), sd_pop=float(slopes.std()),
                           sd_sample=float(slopes.std(ddof=1)),
                           ci95=[float(slopes.mean() - ci), float(slopes.mean() + ci)]))
    json.dump(out, open(os.path.join(RESULTS, "analysis_r1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
