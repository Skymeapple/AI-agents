"""Round-2 analyses of existing results (no new simulation except cheap df formulas).

1. Theory-vs-simulation relative error of the loss RATIO across ALL experiments
   (A, B, C, D, E2), by n.
2. Exp C: accuracy of the competition/synergy BOUNDARY, not just cell signs.
3. Ratio definitions: mean of per-run ratios vs ratio of mean losses (Exp A).
4. Corollary 2 for the actual MLP configuration, and the fixed-width (no rank
   selection) df prediction, versus the ReLU results under both conventions.
5. Finite-sample interaction below k_s (Exp A, Exp C) vs df prediction.
Writes ../results/analysis_r2.json
"""
import json, os
import numpy as np
from common import make_task, theory_losses

R = os.path.join(os.path.dirname(__file__), "..", "results")
J = lambda f: json.load(open(os.path.join(R, f)))
out = {}

# ---------------------------------------------------------------- 1. errors
lin = J("linear_results.json"); pw = J("powerlaw_results.json")
err = {}
for n, runs in lin["A"]["runs"].items():
    e = [abs(o["sim_ratio"] - o["th_ratio"]) / o["th_ratio"] for o in runs]
    err[f"A n={n}"] = dict(max=max(e), median=float(np.median(e)), cells=len(e))
for n, runs in lin["B"]["runs"].items():
    e = [abs(o["sim_ratio"] - o["th_ratio"]) / o["th_ratio"] for o in runs]
    err[f"B n={n}"] = dict(max=max(e), median=float(np.median(e)), cells=len(e))
Cs, Ct = np.array(lin["C"]["log_ratio_sim"]), np.array(lin["C"]["log_ratio_theory"])
e = np.abs(np.exp(Cs - Ct) - 1).ravel()
err["C n=1000"] = dict(max=float(e.max()), median=float(np.median(e)), cells=int(e.size))
for o in lin["D"]:
    err[f"D n={o['n']} (r=24)"] = dict(max=abs(o["sim_ratio"] - o["th_ratio"]) / o["th_ratio"],
                                      median=None, cells=1)
for n, v in pw["E2"]["runs"].items():
    e = [abs(s - t) / t for s, t in zip(v["sim"], v["theory"])]
    err[f"E2 n={n}"] = dict(max=max(e), median=float(np.median(e)), cells=len(e),
                            plateau_sim=v["sim"][-1], plateau_theory=v["theory"][-1])
out["ratio_errors"] = err

# ---------------------------------------------------------------- 2. Exp C boundary
th = np.degrees(lin["C"]["thetas"]); rs = lin["C"]["rs"]
near = np.abs(Ct) < 0.1
out["C_boundary"] = dict(
    cells_total=int(Cs.size), cells_near_boundary_abs_logratio_lt_0p1=int(near.sum()),
    sign_agree_near_boundary=int((np.sign(Cs) == np.sign(Ct))[near].sum()),
    # for each r: smallest theta (grid) at which the sign is competition
    first_competition_theta_per_r={str(r): dict(
        sim=next((float(th[i]) for i in range(len(th)) if Cs[i, j] > 0), None),
        theory=next((float(th[i]) for i in range(len(th)) if Ct[i, j] > 0), None))
        for j, r in enumerate(rs)},
    synergy_r_threshold_theta0=dict(
        sim=next((r for j, r in enumerate(rs) if j > 0 and Cs[0, j] < 0 and rs[j] > 8), None),
        theory=next((r for j, r in enumerate(rs) if j > 0 and Ct[0, j] < 0 and rs[j] > 8), None)))

# ---------------------------------------------------------------- 3. ratio definitions
rd = {}
for n, runs in lin["A"]["runs"].items():
    a = np.array([o["sim_ratio"] for o in runs]); b = np.array([o["sim_u"] / o["sim_s"] for o in runs])
    rd[n] = dict(max_rel_diff=float(np.max(np.abs(a - b) / b)))
out["ratio_definition_check_A"] = rd

# ---------------------------------------------------------------- 4. Corollary 2 vs MLP
def gamma(d, q, ks, kp, M):
    k = ks + kp; ru = ks + M * kp
    if ru > min(d, M * q):
        return None
    return ru * (d + M * q - ru) / (M * k * (d + q - k))
mlp = J("mlp_results.json"); hs = np.array(mlp["hs"])
c2 = {}
for M in [2, 4, 8]:
    ent = dict(gamma_mlp_config=gamma(32, 8, 6, 6, M), gamma_bound_d_inf=1 - 0.5 * (1 - 1 / M))
    for n in [256, 1024, 4096]:
        rr = [r for r in mlp["runs"] if r["n"] == n and r["M"] == M]
        Lu = np.array([r["Lu"] for r in rr]); Ls = np.array([r["Ls"] for r in rr])
        fixed = np.exp(np.log(Lu / Ls).mean(0))
        orc = np.exp(np.log(np.minimum.accumulate(Lu, 1) / np.minimum.accumulate(Ls, 1)).mean(0))
        best = np.exp(np.log(Lu.min(1) / Ls.min(1)).mean())
        c2[f"M={M},n={n}"] = dict(fixed_width_min=float(fixed.min()), fixed_width_argmin_h=int(hs[fixed.argmin()]),
                                  oracle_width_ratio_by_h=orc.round(3).tolist(),
                                  best_over_all_widths_ratio=float(best), **ent)
out["corollary2_vs_mlp_r1data"] = c2

# fixed-width df prediction (linear model, Exp A setup): ratio when both models
# are forced to use rank exactly r (no oracle rank selection)
d, q, ks, kp, M = 64, 16, 4, 4, 4
Ws = make_task(d, q, ks, kp, M, rng=0)
G = sum(W.T @ W for W in Ws); eu = np.sort(np.linalg.eigvalsh(G))[::-1]
es = np.sort(np.linalg.eigvalsh(Ws[0].T @ Ws[0]))[::-1]
fw = {}
for r in [8, 12, 16, 20, 24]:
    Lu = (eu[r:].sum() + r * (d + M * q - r) / 1000) / M
    Ls = es[r:].sum() + min(r, q) * (d + q - min(r, q)) / 1000
    tu, ts = theory_losses(Ws, 1.0, 1000, r)
    fw[str(r)] = dict(fixed_rank_ratio=float(Lu / Ls), oracle_rank_ratio=float(tu / ts))
out["fixed_vs_oracle_rank_df_expA_n1000"] = fw

# ---------------------------------------------------------------- 5. below k_s
bk = {}
for n, runs in lin["A"]["runs"].items():
    bk[n] = [dict(r=r, sim=o["sim_ratio"], theory=o["th_ratio"]) for r, o in zip(lin["A"]["rs"], runs) if r <= 4]
out["below_ks_expA"] = bk
json.dump(out, open(os.path.join(R, "analysis_r2.json"), "w"), indent=1, default=float)
for k, v in out.items():
    print("==", k); print(json.dumps(v, default=float)[:3000])
