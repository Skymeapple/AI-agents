"""Aggregate MLP runs -> results/mlp_summary.json, fig3_mlp_curves.png, fig4_mlp_scaling.png"""
import json, glob, os, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
files = sorted(glob.glob(os.path.join(R, "mlp_*_[0-9]*.json")))
runs = []
for f in files:
    o = json.load(open(f))
    for r in o["runs"]:
        r["scenario"] = o["scenario"]; r["seed"] = o["seed"]; r["base"] = o["base"]; runs.append(r)

def tag(r):
    if r["method"] == "ggn": return f"ggn(δ={r['kw']['delta']},lr={r['kw']['lr']})"
    return r["method"]

def metrics(r):
    L1 = np.array(r["L1"]); L2 = np.array(r["L2"]); A1 = np.array(r["A1"])
    G = L1.max() - L1[0]; ip = int(L1.argmax())
    half = np.where(L1[ip:] - L1[0] <= 0.5 * G)[0]
    t_half = int(ip + half[0]) if len(half) else None
    d1 = L1[1] - L1[0]
    return dict(G=float(G), t_peak=ip, t_half=t_half, L1_final_inc=float(L1[-1] - L1[0]),
                L2_final=float(L2[-1]), L2_0=float(L2[0]), A1_0=float(A1[0]), A1_min=float(A1.min()),
                A1_final=float(A1[-1]), A2_final=float(r["A2"][-1]), dL1_step1=float(d1),
                quad_pred_step1=float(r["lr"] ** 2 * r["quad_pred_coef"]) if r["method"] == "gd" else None,
                steps=len(L1) - 1)

summary = collections.defaultdict(list)
for r in runs:
    summary[(r["scenario"], tag(r), r["rho"])].append(metrics(r))
out = {}
for k, ms in sorted(summary.items()):
    agg = {}
    for key in ms[0]:
        vals = [m[key] for m in ms if m[key] is not None]
        if vals: agg[key] = [float(np.mean(vals)), float(np.std(vals)), len(vals)]
    agg["n_seeds"] = len(ms); agg["n_censored_half_recovery"] = sum(m["t_half"] is None for m in ms)
    out["|".join(map(str, k))] = agg
json.dump(out, open(os.path.join(R, "mlp_summary.json"), "w"), indent=1)

# print table
for k, a in out.items():
    print(k, {kk: (round(v[0], 4) if isinstance(v, list) else v) for kk, v in a.items()})

# Fig 3: GD curves
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for j, scen in enumerate(["perm", "split"]):
    for rho in [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]:
        rs = [r for r in runs if r["scenario"] == scen and r["method"] == "gd" and r["rho"] == rho]
        if not rs: continue
        L = np.mean([np.array(r["L1"]) for r in rs], 0)
        ax[0, j].plot(np.arange(len(L)) + 1, L, label=f"ρ={rho}")
        A = np.mean([np.array(r["A1"]) for r in rs], 0)
        st = [0] + rs[0]["acc_steps"]
        ax[1, j].plot(np.array(st) + 1, A, label=f"ρ={rho}")
    ax[0, j].set_xscale("log"); ax[1, j].set_xscale("log")
    ax[0, j].set_title(f"{scen}: old-task train loss (GD, mean of seeds)"); ax[0, j].set_ylabel("L1")
    ax[1, j].set_title(f"{scen}: old-task test accuracy"); ax[1, j].set_xlabel("phase-2 step + 1")
    ax[0, j].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig3_mlp_curves.png"), dpi=150); plt.close()

# Fig 4: scaling: G(rho)/G(0) and t_half vs rho (GD); method comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for scen, mk in [("perm", "o-"), ("split", "s-")]:
    rhos = [0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
    G0 = np.mean([metrics(r)["G"] for r in runs if r["scenario"] == scen and r["method"] == "gd" and r["rho"] == 0.0])
    Gs = [np.mean([metrics(r)["G"] for r in runs if r["scenario"] == scen and r["method"] == "gd" and r["rho"] == x]) / G0 for x in rhos]
    ax[0].semilogx(rhos, Gs, mk, label=scen)
    ms = [metrics(r) for r in runs if r["scenario"] == scen and r["method"] == "gd"]
    ax[1].loglog([m["quad_pred_step1"] for m in ms], [m["dL1_step1"] for m in ms], mk[0], alpha=0.6, label=scen)
ax[0].set_xlabel("ρ"); ax[0].set_ylabel("G(ρ)/G(0)  (GD)"); ax[0].set_title("(a) gap depth vs replay weight"); ax[0].legend()
lim = [5e-3, 0.1]; ax[1].loglog(lim, lim, "k--", lw=0.8, label="y = x")
ax[1].set_xlabel("predicted ½η²(1−ρ)² g₂ᵀG₁g₂"); ax[1].set_ylabel("measured ΔL1 after step 1 (GD)")
ax[1].set_title("(b) one-step law (Thm 2), all GD runs"); ax[1].legend()
meths = ["gd", "adam", "ggn(δ=0.01,lr=1.0)", "ggn(δ=0.001,lr=1.0)", "ggn(δ=0.001,lr=0.2)"]
w = 0.15
for i, m in enumerate(meths):
    vals = []; errs = []; labels = []
    for scen in ["perm", "split"]:
        for rho in [0.01, 0.1, 0.3]:
            ms = summary.get((scen, m, rho), [])
            vals.append(np.mean([x["G"] for x in ms]) if ms else np.nan)
            errs.append(np.std([x["G"] for x in ms]) if ms else 0); labels.append(f"{scen}\nρ={rho}")
    ax[2].bar(np.arange(len(vals)) + (i - 2) * w, vals, w, yerr=errs, label=m)
ax[2].set_xticks(np.arange(len(labels))); ax[2].set_xticklabels(labels, fontsize=7)
ax[2].set_ylabel("gap depth G = max L1 − L1(0)"); ax[2].set_title("(c) optimiser comparison"); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig4_mlp_scaling.png"), dpi=150); plt.close()
