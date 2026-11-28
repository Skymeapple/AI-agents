"""Aggregate linear_speedlimit outputs (parts) -> results/linear_speedlimit_summary.json + fig8_speedlimit.png"""
import json, os, glob, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
parts = sorted(glob.glob(os.path.join(R, "ls_part_*.json")))
A, B, C = [], [], []
for p in parts:
    o = json.load(open(p)); A += o["A"]; B += o["B"]; C += o["C"]
extra = os.path.join(R, "ls_hbrho.json")
if os.path.exists(extra): B += json.load(open(extra))["B"]
S = {"n_seeds": len(set(r["seed"] for r in B))}
# ---- A: bound checks
S["A_total_rows"] = len(A); S["A_viol_thm"] = sum(r["viol_thmA"] for r in A); S["A_viol_thm2"] = sum(r["viol_thmA2"] for r in A)
S["A_markov_all_ok"] = all(r["markov_ok"] for r in A if r["method"] != "CG")
S["A_markov_CG_ok"] = all(r["markov_ok"] for r in A if r["method"] == "CG")
S["A_Cmax"] = {m: float(max(r["Cmax"] for r in A if r["method"] == m)) for m in sorted(set(r["method"] for r in A))}
S["A_steps_checked"] = int(sum(r["steps"] for r in A))
# depth at the moment the new task is learned (L2 <= 1e-3 L2(0))
g = collections.defaultdict(list)
for r in A: g[(r["method"], r["rho"])].append(r)
S["A_learned"] = {}
for (m, rho), rs in sorted(g.items()):
    ok = [r for r in rs if r["kL"] is not None]
    if not ok: continue
    S["A_learned"][f"{m}|{rho}"] = dict(n=len(ok), kL_mean=float(np.mean([r["kL"] for r in ok])),
        L1_over_F=float(np.mean([r["L1_at_kL_over_F"] for r in ok])), L1_over_F_min=float(min(r["L1_at_kL_over_F"] for r in ok)),
        depth_over_F=float(np.mean([r["depth_to_kL_over_F"] for r in ok])),
        bound_over_F=float(np.mean([r["bound2_at_kL_over_F"] for r in ok])), sigma=float(np.mean([r["sigma_at_kL"] for r in ok])),
        phi=float(np.mean([r["phi_at_kL"] for r in ok])), g2_over_2F=float(np.mean([r["g2_over_2F"] for r in ok])))
# ---- B: recovery-time exponents
gb = collections.defaultdict(dict)
for r in B: gb[(r["method"], r["seed"])][r["rho"]] = r
S["B"] = {}
methods = sorted(set(r["method"] for r in B))
for m in methods:
    seeds = sorted(set(s for (mm, s) in gb if mm == m))
    rhos = sorted(gb[(m, seeds[0])])
    T = np.array([[gb[(m, s)][x]["T_eps"] for x in rhos] for s in seeds])
    small = [i for i, x in enumerate(rhos) if x <= 1e-3]
    slopes = []
    for row in T:
        if np.all(np.isfinite(row[small])) and np.all(row[small] > 0):
            slopes.append(np.polyfit(np.log(np.array(rhos)[small]), np.log(row[small]), 1)[0])
    rhoT = (T * np.array(rhos)[None, :]); sqT = (T * np.sqrt(np.array(rhos))[None, :])
    S["B"][m] = dict(rhos=rhos, T_mean=np.nanmean(np.where(np.isfinite(T), T, np.nan), 0).tolist(),
                     T_min=T.min(0).tolist(), T_max=T.max(0).tolist(), n_censored=int((~np.isfinite(T)).sum()),
                     slope_rho_le_1e-3_mean=float(np.mean(slopes)) if slopes else None,
                     slope_sd=float(np.std(slopes, ddof=1)) if len(slopes) > 1 else None, n_slopes=len(slopes),
                     rhoT_mean=np.nanmean(rhoT, 0).tolist(), sqrtrhoT_mean=np.nanmean(sqT, 0).tolist())
# a-priori lower bounds on T (Theorem 1c) for GD (sigma <= k lamS/L) and bounded methods C=1 (sigma <= 2 k^2 lamS/L)
lb = collections.defaultdict(list)
for r in A:
    if r["method"] != "GD": continue
    # threshold on sigma once fast modes are learned (phi -> 0), eps = 0.1
    thr = (r["g_rho"] - np.sqrt(0.1) * r["sqrt2F"]) / (r["sqrt_l1max"] * r["nPS"])
    lb[r["rho"]].append(dict(GD=thr * r["L"] / r["lamS"], C1=np.sqrt(thr * r["L"] / (2 * r["lamS"])), thr=thr))
S["T_lower_bounds"] = {str(k): dict(GD=float(np.mean([x["GD"] for x in v])), C1=float(np.mean([x["C1"] for x in v])),
                                    sigma_threshold=float(np.mean([x["thr"] for x in v]))) for k, v in sorted(lb.items())}
# ---- C: peak depth (overshoot) and depth at matched progress
gc = collections.defaultdict(list)
for r in C: gc[(r["method"], r["rho"])].append(r)
S["C"] = {}
for (m, rho), rs in sorted(gc.items()):
    d = dict(peak_over_F_mean=float(np.mean([r["G_over_F_window"] for r in rs])), peak_over_F_sd=float(np.std([r["G_over_F_window"] for r in rs], ddof=1)))
    for q in ["0.01", "0.001"]:
        v = [r[f"Gq{q}"] for r in rs if r[f"Gq{q}"] is not None]; kk = [r[f"kq{q}"] for r in rs if r[f"kq{q}"] is not None]
        d[f"depth_at_q{q}"] = float(np.mean(v)) if v else None; d[f"k_q{q}"] = float(np.mean(kk)) if kk else None
    S["C"][f"{m}|{rho}"] = d
json.dump(S, open(os.path.join(R, "linear_speedlimit_summary.json"), "w"), indent=1)

# ---- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
sty = {"GD": "k-o", "HB_0.9": "b-s", "HB_rho": "c-^", "HB_tuned": "c--^", "Nesterov": "g-D", "Cheb": "r-v", "CG": "m:x"}
for m in methods:
    b = S["B"][m]
    if m == "HB_tuned": continue
    lab = f"{m} (slope {b['slope_rho_le_1e-3_mean']:.2f})" if b["slope_rho_le_1e-3_mean"] is not None else m
    ax[0].loglog(b["rhos"], b["T_mean"], sty.get(m, "-"), label=lab, ms=4)
xs = sorted(float(k) for k in S["T_lower_bounds"])
ax[0].loglog(xs, [S["T_lower_bounds"][str(x) if str(x) in S["T_lower_bounds"] else repr(x)]["GD"] for x in xs], "k:", label="Thm 1 lower bound, GD")
ax[0].loglog(xs, [S["T_lower_bounds"][str(x) if str(x) in S["T_lower_bounds"] else repr(x)]["C1"] for x in xs], "r:", label="Thm 1 lower bound, |p|<=1")
ax[0].set_xlabel("replay weight ρ"); ax[0].set_ylabel("steps to gap-free state $T_{0.1}$"); ax[0].legend(fontsize=7)
ax[0].set_title("(a) recovery time: GD ∝ 1/ρ, accelerated ∝ ρ^{-1/2}")
rho0 = 1e-3; ms = ["GD", "GD_large", "GD_warmup", "HB_0.9", "Nesterov", "Cheb", "CG"]
vals = [S["C"].get(f"{m}|{rho0}", {}).get("peak_over_F_mean", np.nan) for m in ms]
vq = [S["C"].get(f"{m}|{rho0}", {}).get("depth_at_q0.001", np.nan) for m in ms]
xx = np.arange(len(ms)); ax[1].bar(xx - 0.2, vq, 0.4, label="depth until new task learned ($L_2\\leq10^{-3}L_2(0)$)")
ax[1].bar(xx + 0.2, vals, 0.4, label="peak over first 2·10^4 steps")
ax[1].axhline(1, color="k", ls=":"); ax[1].set_xticks(xx); ax[1].set_xticklabels(ms, rotation=30, fontsize=7)
ax[1].set_ylabel("$L_1/F_\\infty$"); ax[1].set_title(f"(b) depth by method, ρ={rho0}"); ax[1].legend(fontsize=7)
for m, c in [("GD", "k"), ("Nesterov", "g"), ("Cheb", "r"), ("HB_0.9", "b")]:
    pts = [(r["sigma_at_kL"], r["L1_at_kL_over_F"]) for r in A if r["method"] == m and r["kL"] is not None]
    if pts: ax[2].scatter(*zip(*pts), s=10, c=c, label=m)
ax[2].set_xscale("log"); ax[2].set_xlabel("slip σ at the step the new task is learned"); ax[2].set_ylabel("$L_1/F_\\infty$ at that step")
ax[2].set_title("(c) learned fast ⇒ forgot ≈ $F_\\infty$"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig8_speedlimit.png"), dpi=140)
print(json.dumps({k: S[k] for k in ["n_seeds", "A_viol_thm", "A_viol_thm2", "A_markov_all_ok", "A_markov_CG_ok", "A_Cmax", "A_steps_checked"]}, indent=1))
for m in methods: print(m, S["B"][m]["slope_rho_le_1e-3_mean"], S["B"][m]["slope_sd"], S["B"][m]["n_censored"])
