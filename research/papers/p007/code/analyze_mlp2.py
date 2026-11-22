"""Summarise exp_mlp2.py (round-2 ReLU study) under two comparison conventions.

fixed width : L_u(h) / L_s(h), both networks of width exactly h (round-1 table)
oracle width: min_{h'<=h} L_u(h') / min_{h'<=h} L_s(h')  -- the convention of the
              theory (oracle rank rho <= r), to which Corollary 2 applies.
Robust crossover: smallest width after the peak from which the seed-mean
fixed-width ratio stays below 1 at every larger width (None if never).
95% CIs by bootstrap over seeds.  Gamma: Corollary 2's formula for the teacher's
(d, q, k_s, k_p, M) (only a formal analogue: q < k_s + k_p here).
"""
import json, os, glob
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
runs, meta = [], None
for f in sorted(glob.glob(os.path.join(R, "mlp2_results_n*.json"))):
    j = json.load(open(f)); runs += j["runs"]; meta = j
hs = np.array(meta["hs"]); d, q, ks, kp = meta["d"], meta["q"], meta["ks"], meta["kp"]
rng = np.random.default_rng(0)


def gamma(M):
    k, ru = ks + kp, ks + M * kp
    return ru * (d + M * q - ru) / (M * k * (d + q - k))


def stay_below(r):
    peak = int(np.argmax(r))
    for j in range(peak, len(r)):
        if np.all(r[j:] < 1):
            return int(hs[j])
    return None


out = {}
for n in sorted(set(r["n"] for r in runs)):
    for M in [2, 4, 8]:
        rr = [r for r in runs if r["n"] == n and r["M"] == M]
        if not rr:
            continue
        Lu = np.array([r["Lu"] for r in rr]); Ls = np.array([r["Ls"] for r in rr])
        lf = np.log(Lu / Ls)
        lo = np.log(np.minimum.accumulate(Lu, 1) / np.minimum.accumulate(Ls, 1))
        fixed, orc = np.exp(lf.mean(0)), np.exp(lo.mean(0))
        bf, bo, bc = [], [], []
        for b in range(2000):
            ii = rng.integers(0, len(rr), len(rr))
            bf.append(np.exp(lf[ii].mean(0)).min()); bo.append(np.exp(lo[ii].mean(0))[-1])
            bc.append(stay_below(np.exp(lf[ii].mean(0))) or np.nan)
        peak = int(np.argmax(fixed))
        out[f"n={n},M={M}"] = dict(
            n_seeds=len(rr), fixed_ratio=fixed.round(3).tolist(), oracle_ratio=orc.round(3).tolist(),
            peak_h=int(hs[peak]), peak_ratio=float(fixed[peak]),
            fixed_min=float(fixed.min()), fixed_min_h=int(hs[fixed.argmin()]),
            fixed_min_ci95=np.percentile(bf, [2.5, 97.5]).tolist(),
            oracle_at_hmax=float(orc[-1]), oracle_at_hmax_ci95=np.percentile(bo, [2.5, 97.5]).tolist(),
            crossover_stay_below=stay_below(fixed),
            crossover_ci95=np.nanpercentile(bc, [2.5, 97.5]).tolist() if np.any(~np.isnan(bc)) else None,
            ratio_at_h_eq_ks=float(fixed[list(hs).index(ks)]) if ks in hs else None,
            ks_plus_Mkp=ks + M * kp, gamma_formula=gamma(M), gamma_d_inf=1 - ks / (ks + kp) * (1 - 1 / M),
            lr_u_choice_frac_3e-3=float(np.mean([np.array(r["lr_u"]) == 3e-3 for r in rr])))
json.dump(out, open(os.path.join(R, "mlp2_summary.json"), "w"), indent=1, default=float)
for k, v in out.items():
    print(k, {a: (np.round(b, 3).tolist() if isinstance(b, (list, float)) else b) for a, b in v.items()})
