"""Make figures and summary tables from ../results/*.json."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
C = dict(g="#2a78d6", m="#eb6834", c3="#1baf7a", c4="#eda100", ink="#52514e")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6",
                     "lines.linewidth": 2, "lines.markersize": 5})
L = lambda f: json.load(open(os.path.join(RES, f)))
summary = {}


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return np.nanmean(v), np.nanstd(v) / np.sqrt(max(1, np.sum(~np.isnan(v))))


def auc(x, y):
    """P(X>Y)+0.5P(X=Y) for grounded-trained X vs mimic-trained Y."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None]) + 0.5 * (x[:, None] == y[None])).mean())


# ---------------- Figure 1: E1 ----------------
E1 = L("E1_alpha_sweep.json"); th = L("theory.json")
al = sorted(set(r["alpha"] for r in E1))
sel = lambda a, l: [r for r in E1 if r["alpha"] == a and r["lam"] == l]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for lam, col, nm in ((1.0, C["g"], "self-trained"), (0.0, C["m"], "corpus-trained (mimic)")):
    m = [agg(sel(a, lam), "acc_self")[0] for a in al]
    t = [agg(sel(a, lam), "agree_typical")[0] for a in al]
    ax[0].plot(al, m, "-o", color=col, label=f"{nm}: acc. vs own state")
    ax[0].plot(al, t, "--s", color=col, mfc="white", label=f"{nm}: agreement w/ typical report")
ta = [r["alpha"] for r in th["prop2"]]; tc = [r["mimic_ceiling_theory"] for r in th["prop2"]]
ax[0].plot(ta, tc, ":", color=C["ink"], lw=1.5, label="Prop. 2 mimic ceiling")
ax[0].set(xlabel="privileged-variance fraction α", ylabel="rate", title="(a) Observational statistics")
ax[0].legend(fontsize=6.5, loc="lower left")
for key, ls, mk, lab in (("ISG_true", "-", "o", "oracle state direction"),
                         ("ISG_dom", "--", "^", "difference-of-means direction"),
                         ("ISG_res", ":", "s", "residualized direction (ours)")):
    for lam, col in ((1.0, C["g"]), (0.0, C["m"])):
        ax[1].plot(al, [agg(sel(a, lam), key)[0] for a in al], ls=ls, marker=mk, color=col,
                   label=f"{'self' if lam else 'mimic'}: {lab}")
ax[1].set(xlabel="α", ylabel="ISG", title="(b) Interventional grounding (steering) score")
ax[1].legend(fontsize=6, loc="center right")
stats = [("agree_typical", "agreement w/ typical", C["ink"], "-."), ("acc_self", "acc. vs own state (white-box)", C["c4"], "-"),
         ("DA", "idiosyncratic-trial acc. (DA)", C["c3"], "-"), ("ISG_dom", "ISG diff-of-means", C["m"], "--"),
         ("ISG_res", "ISG residualized", C["g"], ":")]
summary["E1_auc"] = {}
for key, lab, col, ls in stats:
    aucs = []
    for a in al:
        x = [r[key] for r in sel(a, 1.0)]; y = [r[key] for r in sel(a, 0.0)]
        aucs.append(np.nan if np.isnan(x).any() else auc(x, y))
    summary["E1_auc"][key] = dict(zip(map(str, al), aucs))
    ax[2].plot(al, aucs, ls=ls, marker="o", color=col, label=lab)
ax[2].axhline(0.5, color="#999", lw=1)
ax[2].set(xlabel="α", ylabel="AUC (self-trained > mimic), 8 vs 8 seeds", title="(c) Which statistic separates them?", ylim=(-0.03, 1.03))
ax[2].legend(fontsize=6.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_alpha_sweep.png"), dpi=170); plt.close(fig)

summary["E1_table"] = []
for a in al:
    for lam in (1.0, 0.0):
        rr = sel(a, lam)
        summary["E1_table"].append(dict(alpha=a, lam=lam, **{k: [round(v, 3) for v in agg(rr, k)] for k in
                                       ("acc_self", "agree_typical", "DA", "ISG_true", "ISG_dom", "ISG_res", "cos_dom_true", "cos_res_true")}))

# ---------------- Figure 2: ridge theory + MLP weight-decay sweep ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
P4 = th["prop4"]
for rho, col in ((1e-3, C["g"]), (1e-2, C["c3"]), (1e-1, C["m"])):
    rr = [r for r in P4 if r["rho"] == rho]
    a = [r["alpha"] for r in rr]
    ax[0].plot(a, [r["w2_mimic_th"] for r in rr], "-", color=col, label=f"mimic, ρ={rho:g}")
    ax[0].plot(a, [r["w2_mimic_sm"] for r in rr], "o", color=col, ms=3.5)
    ax[0].plot(a, [r["w2_self_th"] for r in rr], "--", color=col, lw=1.2)
    ax[1].plot(a, [r["dom_mimic_th"] for r in rr], "-", color=col, label=f"ρ={rho:g}: diff-of-means dir.")
    ax[1].plot(a, [r["w2_mimic_th"] for r in rr], ":", color=col, label=f"ρ={rho:g}: residual dir.")
ax[0].set(xlabel="α", ylabel="weight on state z (effect of do(z))", title="(a) Prop. 4: ridge reporter (lines theory,\n dots n=2e5 sample; dashed = self-trained)")
ax[0].legend(fontsize=6.5)
ax[1].set(xlabel="α", ylabel="mimic's steering effect", title="(b) Prop. 5: mimic false positive by direction")
ax[1].legend(fontsize=6)
E4 = L("E4_wd_sweep.json")
a4 = sorted(set(r["alpha"] for r in E4))
for wd, col in ((1e-4, C["g"]), (1e-3, C["c3"]), (1e-2, C["m"])):
    for lam, ls in ((0.0, "-"), (1.0, "--")):
        v = [agg([r for r in E4 if r["wd"] == wd and r["alpha"] == a and r["lam"] == lam], "ISG_true")[0] for a in a4]
        ax[2].plot(a4, v, ls=ls, marker="o", color=col, label=f"{'mimic' if lam == 0 else 'self'}, wd={wd:g}")
ax[2].set(xlabel="α", ylabel="ISG (oracle direction)", title="(c) MLP reporters: weight decay sweep")
ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_ridge_theory_wd.png"), dpi=170); plt.close(fig)
summary["E4_table"] = [dict(wd=wd, alpha=a, lam=lam, ISG_true=round(agg([r for r in E4 if r["wd"] == wd and r["alpha"] == a and r["lam"] == lam], "ISG_true")[0], 3))
                       for wd in (1e-4, 1e-3, 1e-2) for a in a4 for lam in (0.0, 1.0)]

# ---------------- Figure 3: E2 lambda sweep vs Prop 3 ----------------
E2 = L("E2_lambda_sweep.json"); P3 = th["prop3"]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
summary["E2_table"] = []
for a, col in zip((0.2, 0.4, 0.6, 0.8), (C["g"], C["c3"], C["c4"], C["m"])):
    pt = [r for r in P3 if r["alpha"] == a]
    ax[0].plot([r["lam"] for r in pt], [r["DA_theory"] for r in pt], "-", color=col, lw=1.5, label=f"α={a} theory")
    lams = sorted(set(r["lam"] for r in E2))
    em = [agg([r for r in E2 if r["alpha"] == a and r["lam"] == l], "DA") for l in lams]
    ax[0].errorbar(lams, [e[0] for e in em], yerr=[e[1] for e in em], fmt="o", color=col, ms=4, capsize=2)
    ax[1].plot(lams, [agg([r for r in E2 if r["alpha"] == a and r["lam"] == l], "ISG_res")[0] for l in lams], "-o", color=col, label=f"α={a}")
    for l, e in zip(lams, em):
        thv = min(pt, key=lambda r: abs(r["lam"] - l))["DA_theory"]
        summary["E2_table"].append(dict(alpha=a, lam=l, DA_emp=round(e[0], 3), DA_theory=round(thv, 3),
                                        ISG_res=round(agg([r for r in E2 if r["alpha"] == a and r["lam"] == l], "ISG_res")[0], 3),
                                        **{k: round(agg([r for r in E2 if r["alpha"] == a and r["lam"] == l], k)[0], 3)
                                           for k in ("mae_q_mixture", "mae_q_typical", "mae_q_self")}))
ax[0].set(xlabel="fraction of self-labels λ", ylabel="DA: acc. on idiosyncratic trials", title="(a) Prop. 3 (lines) vs trained MLPs (dots)")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="λ", ylabel="ISG (residualized direction)", title="(b) Interventional grounding vs λ")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_lambda.png"), dpi=170); plt.close(fig)
d = [abs(r["DA_emp"] - r["DA_theory"]) for r in summary["E2_table"]]
summary["E2_mean_abs_dev"] = float(np.mean(d)); summary["E2_max_abs_dev"] = float(np.max(d))
summary["E2_mixture_best_fraction"] = float(np.mean([r["mae_q_mixture"] <= min(r["mae_q_typical"], r["mae_q_self"]) + 1e-9 for r in summary["E2_table"]]))
summary["E2_mae_mixture_range"] = [min(r["mae_q_mixture"] for r in summary["E2_table"]), max(r["mae_q_mixture"] for r in summary["E2_table"])]

# ---------------- Figure 4: E3 self-evidence budget ----------------
E3 = L("E3_budget.json")
kap = lambda a: np.arccos(1 - a ** 2) / np.pi
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
pts = []
for n in sorted(set(r["n"] for r in E3)):
    for a in sorted(set(r["alpha"] for r in E3)):
        g = [r["ISG_true"] for r in E3 if r["n"] == n and r["alpha"] == a and r["lam"] == 1.0]
        m = [r["ISG_true"] for r in E3 if r["n"] == n and r["alpha"] == a and r["lam"] == 0.0]
        gd = [r["acc_self"] for r in E3 if r["n"] == n and r["alpha"] == a and r["lam"] == 1.0]
        md = [r["acc_self"] for r in E3 if r["n"] == n and r["alpha"] == a and r["lam"] == 0.0]
        B = n * 4 * kap(a)
        pts.append(dict(n=n, alpha=a, budget=B, auc_isg=auc(g, m), dISG=float(np.mean(g) - np.mean(m)),
                        auc_acc=auc(gd, md), bound=float(1 - (1 - kap(a)) ** (n * 4))))
summary["E3_table"] = pts
B = np.array([p["budget"] for p in pts])
for key, col, lab in (("auc_isg", C["g"], "ISG (oracle dir.)"), ("auc_acc", C["c4"], "acc. vs own state")):
    ax[0].scatter(B, [abs(p[key] - 0.5) for p in pts], color=col, s=14, label=f"|AUC−0.5|, {lab}")
bb = np.logspace(-2, 3.5, 200)
ax[0].plot(bb, 1 - np.exp(-bb), color=C["ink"], lw=1.2, ls="--", label="Prop. 6 bound on |AUC−0.5| (≤ TV ≤ 1−(1−κ)^{nK})")
ax[0].axhline(0.5, color="#999", lw=1); ax[0].set(xscale="log", xlabel="expected # disagreeing labels  nKκ", ylabel="|AUC − 0.5| (max 0.5)", title="(a) Self-evidence budget")
ax[0].legend(fontsize=6.5)
ax[1].scatter(B, [p["dISG"] for p in pts], c=np.log10([p["n"] for p in pts]), cmap="viridis", s=16)
ax[1].set(xscale="log", xlabel="nKκ", ylabel="mean ISG(self) − ISG(mimic)", title="(b) Interventional gap (colour = log10 n)")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_budget.png"), dpi=170); plt.close(fig)

# ---------------- E5 idiosyncratic deterministic variant ----------------
E5 = L("E5_idio.json")
summary["E5_table"] = []
for a in sorted(set(r["alpha"] for r in E5)):
    for lam in (1.0, 0.0):
        rr = [r for r in E5 if r["alpha"] == a and r["lam"] == lam]
        summary["E5_table"].append(dict(alpha=a, lam=lam, **{k: round(agg(rr, k)[0], 3) for k in
                                       ("acc_self", "agree_typical", "DA", "ISG_true", "ISG_dom", "ISG_res")}))
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k in ("E1_auc", "E2_mean_abs_dev", "E2_max_abs_dev")}, indent=1))
