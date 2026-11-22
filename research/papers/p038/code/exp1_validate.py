"""E1: validate closed-form mean (Thm 1), exact-PGF extinction, and attempt-level MC.
E2: phase diagram in (P, h_f) and the bang-bang attempt-budget property (Prop 2)."""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import *

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(0)
t0 = time.time()

# ---------------- E1: validation over random parameter draws
rows = []
for i in range(14):
    P = rng.uniform(0.02, 0.4); h_s = rng.uniform(0, 0.3); h_f = rng.uniform(0.005, 0.2)
    sigma = rng.uniform(0.5, 1.0); delta = rng.uniform(0.0, 0.2); a = int(rng.integers(2, 12))
    pmf = offspring_pmf(P, h_s, h_f, sigma, delta, a)
    m_cf = mean_offspring(P, h_s, h_f, sigma, delta, a)
    m_pmf = float(np.dot(np.arange(len(pmf)), pmf))
    q = extinction_prob(pmf)
    surv_mc = simulate_population(P, h_s, h_f, sigma, delta, a, reps=3000, cap=500,
                                  gens=200, rng=int(rng.integers(1e9)))
    se = np.sqrt(max(surv_mc * (1 - surv_mc), 1e-9) / 3000)
    rows.append(dict(P=P, h_s=h_s, h_f=h_f, sigma=sigma, delta=delta, a=a, m_closed=m_cf,
                     m_pmf=m_pmf, surv_exact=1 - q, surv_mc=surv_mc, mc_se=se))
    print(f"E1 {i}: m={m_cf:.4f}/{m_pmf:.4f}  surv exact={1-q:.4f} mc={surv_mc:.4f}±{se:.4f}")
maxdm = max(abs(r["m_closed"] - r["m_pmf"]) for r in rows)
z = [abs(r["surv_exact"] - r["surv_mc"]) / max(r["mc_se"], 1e-3) for r in rows]
json.dump(dict(rows=rows, max_abs_mean_diff=maxdm, max_z=max(z), mean_abs_z=float(np.mean(z))),
          open(f"{OUT}/e1_validation.json", "w"), indent=1)
print("max |m_closed-m_pmf| =", maxdm, " max z =", max(z))

fig, ax = plt.subplots(figsize=(4, 4))
se = np.array([r["mc_se"] for r in rows])
ax.errorbar([r["surv_exact"] for r in rows], [r["surv_mc"] for r in rows], yerr=1.96 * se,
            fmt="o", ms=4, color="#3b6ea5")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("survival probability (exact PGF)"); ax.set_ylabel("survival (attempt-level MC)")
ax.set_title("E1: theory vs simulation"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_e1_validation.png", dpi=150)

# ---------------- E2a: bang-bang in attempt budget
fig, ax = plt.subplots(figsize=(5, 3.4))
res_bb = {}
for (P, h_f, lab) in [(0.05, 0.02, "R_inf=2.6"), (0.05, 0.06, "R_inf=0.88"), (0.02, 0.01, "R_inf=2.0"),
                      (0.02, 0.04, "R_inf=0.51")]:
    ms = [mean_offspring(P, 0.0, h_f, 1.0, 0.1, a) for a in range(0, 201)]
    res_bb[lab] = dict(P=P, h_f=h_f, R_inf=R_inf(P, 0, h_f, 1.0), m=ms[::20])
    ax.plot(range(201), ms, label=f"P={P}, h_f={h_f} ({lab})")
ax.axhline(1, color="k", lw=0.7, ls=":")
ax.set_xlabel("attempt budget per epoch a"); ax.set_ylabel("mean offspring m(a)")
ax.set_title("E2a: m(a) is monotone; limit is R_inf"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e2a_bangbang.png", dpi=150)
json.dump(res_bb, open(f"{OUT}/e2a_bangbang.json", "w"), indent=1)

# ---------------- E2b: phase diagram R_inf over (P, h_f), with exact survival prob for a=20
Ps = np.logspace(-3, np.log10(0.5), 120); Hs = np.logspace(-3, 0, 120)
PP, HH = np.meshgrid(Ps, Hs)
R = R_inf(PP, 0.0, HH, 1.0)
surv = np.zeros((30, 30))
Ps2 = np.logspace(-3, np.log10(0.5), 30); Hs2 = np.logspace(-3, 0, 30)
for i, h in enumerate(Hs2):
    for j, p in enumerate(Ps2):
        pmf = offspring_pmf(p, 0.0, h, 1.0, 0.05, 50)
        surv[i, j] = 1 - extinction_prob(pmf)
np.savez(f"{OUT}/e2b_phase.npz", Ps=Ps2, Hs=Hs2, surv=surv)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
c = axs[0].contourf(PP, HH, np.log10(R), levels=np.linspace(-3, 3, 25), cmap="RdBu_r")
axs[0].contour(PP, HH, R, levels=[1], colors="k", linewidths=1.5)
p10 = pass_at_k_to_p1(0.5, 10)
axs[0].axvline(p10, color="#444", ls="--", lw=1)
axs[0].text(p10 * 1.08, 1.5e-3, "pass@10 = 50%", fontsize=7, rotation=90)
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("per-attempt success P"); axs[0].set_ylabel("detection prob. per failed attempt h_f")
axs[0].set_title("log10 R_inf (black: R_inf = 1)"); plt.colorbar(c, ax=axs[0])
c2 = axs[1].pcolormesh(Ps2, Hs2, surv, cmap="magma", shading="auto")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); plt.colorbar(c2, ax=axs[1])
axs[1].contour(PP, HH, R, levels=[1], colors="w", linewidths=1)
axs[1].set_xlabel("per-attempt success P"); axs[1].set_title("survival prob. of 1 replica (a=50, delta=0.05)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e2b_phase.png", dpi=150)

# key numbers: critical h_f for given P (h_s=0, sigma=1): h_f* = P/(1-P)
crit = {f"{p:.3f}": p / (1 - p) for p in [0.001, 0.01, p10, 0.1, 0.3]}
json.dump(dict(p1_from_pass10_50=p10, critical_hf=crit), open(f"{OUT}/e2b_keynumbers.json", "w"), indent=1)
print("p1 from pass@10=0.5:", p10, crit)
print("done in", time.time() - t0, "s")
