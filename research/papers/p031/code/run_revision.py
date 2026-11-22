"""Round-1 revision analyses. Run: python3 run_revision.py  (about 3-5 min)
A. simulator validation against brute-force enumeration (any rho, kappa, gamma, f, R)
B. ensemble floor (Theorem 1, common-mode form): property tests; counterexamples to the old d0 claim
C. anchored optimal rate vs the rho-floor
D. optimal retry budget for S - c*F
E. sensitivity of the Table-2 conclusions
F. step-size-dependent persistence gamma(Delta)
-> ../results/revision_results.json, fig6_gamma_delta.png, fig7_rho_floor.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from bos_model import (Params, chain_exact, anchored_exact, anchored_beta, brute_force, simulate,
                       chain_two_state, mu, det, eps_step, optimal_rate)

OUT = "../results/"
R_INF = 10**9
base = Params()
res = {}
rng = np.random.default_rng(2027)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def floor_bound(p, n, beta_lo):
    c0 = np.log((1 - (1 - beta_lo) * p.mu0) / (1 - p.mu0))
    return np.exp(-p.lam * beta_lo * p.G - n * c0)


# ---------------------------------------------------------------- A. validation
val = []
for t in range(40):
    pol = "anchored" if t % 2 == 0 else "chain"
    p = Params(G=float(rng.uniform(100, 3000)), mu0=float(10 ** rng.uniform(-3, -1)), d0=float(rng.uniform(0.5, 0.97)),
               k=float(rng.uniform(0.3, 2)), gamma=float(rng.choice([1.0, rng.uniform(0, 1)])),
               kappa=float(rng.choice([1.0, rng.uniform(0, 1)])), rho=float(rng.uniform(0, 1)),
               f=float(rng.choice([0.0, 0.02, 0.1])), R=int(rng.choice([1, 2, 4, R_INF])),
               lam=float(10 ** rng.uniform(-4, -2.5)))
    n = int(rng.integers(2, 7))
    bf = brute_force(n, p, pol)
    mc = simulate(n, p, pol, reps=200_000, seed=t)
    se = [np.sqrt(max(x, 1e-6) * (1 - x) / 200_000) for x in bf]
    z = max(abs(a - b) / s for a, b, s in zip(bf, mc, se))
    rec = dict(policy=pol, n=n, rho=p.rho, kappa=p.kappa, gamma=p.gamma, f=p.f,
               R=("inf" if p.R == R_INF else p.R), brute=list(map(float, bf)), mc=list(map(float, mc)),
               max_abs=float(max(abs(a - b) for a, b in zip(bf, mc))), max_z=float(z))
    if pol == "chain":
        rec["chain_exact_maxabs"] = float(max(abs(a - b) for a, b in zip(bf, chain_exact(n, p))))
    if pol == "anchored" and p.kappa == 1.0 and p.gamma == 1.0:
        rec["anchored_exact_maxabs"] = float(max(abs(a - b) for a, b in zip(bf, anchored_exact(n, p))))
    val.append(rec)
res["validation_bruteforce"] = dict(
    configs=len(val), max_abs_mc=max(v["max_abs"] for v in val), max_z_mc=max(v["max_z"] for v in val),
    max_abs_chain_exact=max([v.get("chain_exact_maxabs", 0) for v in val]),
    max_abs_anchored_exact=max([v.get("anchored_exact_maxabs", 0) for v in val]),
    n_anchored_exact_checked=sum("anchored_exact_maxabs" in v for v in val), rows=val)
print("A", {k: v for k, v in res["validation_bruteforce"].items() if k != "rows"}, flush=True)

# ---------------------------------------------------------------- B. ensemble floor property tests
viol_new, viol_rho, viol_old, worst_new, trials = 0, 0, 0, np.inf, 1500
old_examples = []
for t in range(trials):
    p = Params(G=float(rng.uniform(100, 4000)), lam=float(10 ** rng.uniform(-4.5, -2.5)), mu0=float(10 ** rng.uniform(-4, -1)),
               d0=float(rng.uniform(0.3, 0.99)), k=float(rng.uniform(0.2, 3)), rho=float(rng.uniform(0, 1)),
               f=float(rng.choice([0.0, 0.01, 0.05])), R=int(rng.choice([1, 2, 5, R_INF])))
    n = int(rng.integers(1, 60))
    S = anchored_exact(n, p)[0]
    beta_lo = float(anchored_beta(n, p).min())
    b_new = floor_bound(p, n, beta_lo)
    b_rho = floor_bound(p, n, p.rho * (1 - p.d0))
    b_old = floor_bound(p, n, 1 - p.d0)
    worst_new = min(worst_new, np.log(b_new) - np.log(max(S, 1e-300)))
    viol_new += S > b_new * (1 + 1e-9)
    viol_rho += S > b_rho * (1 + 1e-9)
    if S > b_old * (1 + 1e-9):
        viol_old += 1
        if len(old_examples) < 5:
            old_examples.append(dict(rho=p.rho, n=n, G=p.G, d0=p.d0, S=float(S), old_bound=float(b_old)))
# small-n brute force with kappa<1 (honesty randomisation) and chain/anchored
viol_bf = 0
for t in range(300):
    pol = "anchored" if t % 2 == 0 else "chain"
    p = Params(G=float(rng.uniform(100, 3000)), lam=float(10 ** rng.uniform(-4, -2.5)), mu0=float(10 ** rng.uniform(-3, -1)),
               d0=float(rng.uniform(0.3, 0.97)), k=float(rng.uniform(0.3, 2)), rho=float(rng.uniform(0, 1)),
               kappa=float(rng.uniform(0, 1)), f=float(rng.choice([0.0, 0.05])), R=int(rng.choice([1, 3, R_INF])))
    n = int(rng.integers(1, 6))
    S = brute_force(n, p, pol)[0]
    beta_lo = float(anchored_beta(n, p).min()) if pol == "anchored" else 1 - float(det(p.G / n, p))
    viol_bf += S > floor_bound(p, n, beta_lo) * (1 + 1e-9)
# Table-2 configuration
pT = replace(base, f=0.02, R=R_INF)
tab2 = {}
for rho in [0.0, 0.44, 1.0]:
    q = replace(pT, rho=rho)
    S = anchored_exact(22, q)[0]
    bl = float(anchored_beta(22, q).min())
    tab2[str(rho)] = dict(S=float(S), beta_min=bl, floor_common_mode=float(floor_bound(q, 22, bl)),
                          floor_rho=float(floor_bound(q, 22, rho * (1 - q.d0))), old_d0_floor=float(floor_bound(q, 22, 1 - q.d0)))
res["ensemble_floor"] = dict(trials_anchored_dp=trials, violations_common_mode=int(viol_new),
                             violations_rho_form=int(viol_rho), min_log_slack=float(worst_new),
                             violations_of_old_d0_claim=int(viol_old), old_claim_counterexamples=old_examples,
                             trials_bruteforce_kappa=300, violations_bruteforce=int(viol_bf), table2_config=tab2)
print("B", {k: v for k, v in res["ensemble_floor"].items() if k not in ("old_claim_counterexamples",)}, flush=True)


# ---------------------------------------------------------------- C. anchored optimal rate vs rho floor
def anchored_S_inf(n, p):
    """P(S) for anchored, gamma=1, R=inf: product over steps of x/(x+y) on the aligned lineage."""
    m = mu(p.G / n, p)
    beta = anchored_beta(n, p)
    j = np.arange(n) + 1
    nofp = p.rho * (1 - p.f) + (1 - p.rho) * (1 - p.f) ** j
    x = (1 - m) * nofp
    y = m * beta
    return float(np.prod(x / (x + y)))


rows = []
for mu0 in [0.01, 1e-4]:
    for rho in [0.0, 0.1, 0.25, 0.44, 0.7, 1.0]:
        p = replace(base, G=4000.0, rho=rho, mu0=mu0)
        ns = np.unique(np.geomspace(1, 1500 if mu0 < 1e-3 else 400, 60).astype(int))
        Ss = [anchored_S_inf(int(n), p) for n in ns]
        i = int(np.argmax(Ss))
        rows.append(dict(mu0=mu0, rho=rho, best_n=int(ns[i]), S=float(Ss[i]), rate_over_lam=float(-np.log(Ss[i]) / p.G / p.lam),
                         rho_floor_over_lam=float(rho * (1 - p.d0)), chain_rate_over_lam=float(optimal_rate(p)[0] / p.lam)))
res["anchored_rate"] = rows
print("C", rows, flush=True)
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for mu0, c, mk in [(0.01, "#2a6f97", "o"), (1e-4, "#c44e52", "s")]:
    rr = [r for r in rows if r["mu0"] == mu0]
    ax.plot([r["rho"] for r in rr], [r["rate_over_lam"] for r in rr], mk + "-", c=c, ms=4, label=f"anchored, best n (mu0={mu0:g})")
rh = np.linspace(0, 1, 50)
ax.plot(rh, rh * (1 - base.d0), "k:", label="common-mode floor rho(1-d0)")
ax.axhline(1 - base.d0, ls="--", c="gray", lw=0.8, label="old claim: 1-d0 for any ensemble")
ax.set_xlabel("dependence rho"); ax.set_ylabel("loss rate / lambda (G=4000)")
ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(OUT + "fig7_rho_floor.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- D. optimal retry budget
pD = replace(base, f=0.02)
Rs = list(range(1, 31)) + [R_INF]
opt_R = []
tables = {}
for name, fn, rho in [("chain", chain_exact, 1.0), ("anchored_rho1", anchored_exact, 1.0),
                      ("anchored_rho0.44", anchored_exact, 0.44), ("anchored_rho0", anchored_exact, 0.0)]:
    out = [fn(22, replace(pD, rho=rho, R=R)) for R in Rs]
    tables[name] = [dict(R=("inf" if R == R_INF else R), S=float(o[0]), F=float(o[1]), A=float(o[2])) for R, o in zip(Rs, out)]
    for c in [1, 3, 10, 30, 100]:
        U = [o[0] - c * o[1] for o in out]
        i = int(np.argmax(U))
        opt_R.append(dict(policy=name, c=c, R_opt=("inf" if Rs[i] == R_INF else Rs[i]), S=float(out[i][0]),
                          F=float(out[i][1]), A=float(out[i][2]), U=float(U[i]), U_Rinf=float(U[-1])))
# per-flag likelihood ratio at the baseline step (anchored, lineage aligned vs compromised at lag 1)
lr = {}
for rho in [1.0, 0.44, 0.0]:
    p = replace(pD, rho=rho)
    n, i = 22, 11
    delta = p.G / n
    m = mu(delta, p)
    dj = det((i + 1 - np.arange(i + 1)) * delta, p)
    beta_A = rho * np.min(1 - dj) + (1 - rho) * np.prod(1 - dj)
    nofp = rho * (1 - p.f) + (1 - rho) * (1 - p.f) ** (i + 1)
    phi_A = 1 - (1 - m) * nofp - m * beta_A
    djc = dj[:-1]  # parent (lag 1) is misaligned and silent
    beta_C = rho * np.min(1 - djc) + (1 - rho) * np.prod(1 - djc)
    phi_C = 1 - beta_C
    lr[str(rho)] = dict(phi_aligned=float(phi_A), phi_compromised=float(phi_C), LR_per_flag=float(phi_C / phi_A))
res["retry_policy"] = dict(n=22, G=1000, f=0.02, optimal=opt_R, tables=tables, likelihood_ratio_step11=lr)
print("D", opt_R, lr, flush=True)

# ---------------------------------------------------------------- E. sensitivity of Table-2 conclusions
sens = []
for d0 in [0.7, 0.9, 0.97]:
    for k in [0.5, 1.0, 2.0]:
        for mu0 in [0.002, 0.01, 0.05]:
            for f in [0.005, 0.02, 0.05]:
                p0 = replace(base, d0=d0, k=k, mu0=mu0, f=f)
                n_s = int(np.argmax([chain_exact(n, replace(p0, f=0.0))[0] for n in range(1, 151)]) + 1)
                rec = dict(d0=d0, k=k, mu0=mu0, f=f, n=n_s)
                F_inf = {}
                for rho in [0.0, 0.44, 1.0]:
                    outs = [anchored_exact(n_s, replace(p0, rho=rho, R=R)) for R in [1, 2, 3, 5, 10, 20, R_INF]]
                    Fs = [o[1] for o in outs]
                    rec[f"F_monotone_R_rho{rho}"] = bool(all(b >= a - 1e-12 for a, b in zip(Fs, Fs[1:])))
                    rec[f"rho{rho}"] = dict(S5=outs[3][0], F5=outs[3][1], A5=outs[3][2], Sinf=outs[-1][0], Finf=outs[-1][1], S1=outs[0][0])
                    F_inf[rho] = outs[-1][1]
                ch = chain_exact(n_s, replace(p0, R=R_INF))
                rec["chain_Finf"] = ch[1]
                rec["anch_rho1_equals_chain_at_Rinf"] = bool(abs(F_inf[1.0] - ch[1]) < 1e-9)
                rec["F_increasing_in_rho"] = bool(F_inf[0.0] <= F_inf[0.44] <= F_inf[1.0])
                rec["frac_benefit_lost_at_044"] = float((F_inf[0.44] - F_inf[0.0]) / (F_inf[1.0] - F_inf[0.0]))
                r = rec["rho0.44"]
                rec["F_reduction_R5_vs_inf_rho044"] = float(r["Finf"] / max(r["F5"], 1e-12))
                rec["S_cost_R5_rho044"] = float(r["Sinf"] - r["S5"])
                sens.append(rec)
summ = dict(
    configs=len(sens),
    F_monotone_in_R_all=all(s[f"F_monotone_R_rho{r}"] for s in sens for r in [0.0, 0.44, 1.0]),
    anch_rho1_equals_chain_all=all(s["anch_rho1_equals_chain_at_Rinf"] for s in sens),
    F_increasing_in_rho_all=all(s["F_increasing_in_rho"] for s in sens),
    frac_benefit_lost_044_quantiles=list(np.quantile([s["frac_benefit_lost_at_044"] for s in sens], [0, 0.25, 0.5, 0.75, 1]).astype(float)),
    F_reduction_R5_quantiles=list(np.quantile([s["F_reduction_R5_vs_inf_rho044"] for s in sens], [0, 0.25, 0.5, 0.75, 1]).astype(float)),
    S_cost_R5_quantiles=list(np.quantile([s["S_cost_R5_rho044"] for s in sens], [0, 0.25, 0.5, 0.75, 1]).astype(float)),
    A5_rho044_quantiles=list(np.quantile([s["rho0.44"]["A5"] for s in sens], [0, 0.25, 0.5, 0.75, 1]).astype(float)),
    S1_rho0_quantiles=list(np.quantile([s["rho0.0"]["S1"] for s in sens], [0, 0.25, 0.5, 0.75, 1]).astype(float)),
)
res["table2_sensitivity"] = dict(summary=summ, rows=sens)
print("E", summ, flush=True)

# ---------------------------------------------------------------- F. gamma(Delta)
d_ref = optimal_rate(base)[1]  # 45.4 Elo
param = {
    "const_0.99": lambda D: 0.99 + 0 * D,
    "const_0.999": lambda D: 0.999 + 0 * D,
    # recovery per unit capability: 1-gamma = 1-exp(-beta*Delta); matched to 1-gamma=0.01 / 0.001 at Delta=45.4
    "percap_0.01": lambda D, b=-np.log(0.99) / d_ref: np.exp(-b * D),
    "percap_0.001": lambda D, b=-np.log(0.999) / d_ref: np.exp(-b * D),
    # saturating: 1-gamma = (1-g_inf)(1-exp(-Delta/Dc)), g_inf=0.9, Dc=2000 (1-gamma=0.0022 at 45 Elo)
    "sat_0.9_2000": lambda D: 1 - 0.1 * (1 - np.exp(-D / 2000.0)),
}
dgrid = np.geomspace(0.5, 20000, 4000)
Gs = [400, 1000, 4000, 16000, 64000, 256000]
gam = {}
for name, gf in param.items():
    pis, Ls = [], []
    for D in dgrid:
        p = replace(base, G=D)  # one step of size D
        e = eps_step(D, p); a = (1 - mu(D, p)) * (1 - gf(D))
        pis.append(a / (a + e)); Ls.append(D / (a + e))
    pis = np.array(pis); i = int(np.argmax(pis))
    rec = dict(pi_star=float(pis[i]), delta_pi_star=float(dgrid[i]), relax_len_at_pi_star=float(Ls[i]),
               pi_small_delta=float(pis[0]), by_G=[])
    for G in Gs:
        p = replace(base, G=float(G))
        nmax = int(min(G / 0.5, 200000))
        ns = np.unique(np.geomspace(1, nmax, 1500).astype(int))
        vals = np.array([chain_two_state(int(n), p, gf) for n in ns])
        j = int(np.argmax(vals))
        rec["by_G"].append(dict(G=G, best_n=int(ns[j]), best_delta=float(G / ns[j]), S=float(vals[j]),
                                S_gamma1=float(np.exp(-G * optimal_rate(base)[0]))))
    gam[name] = rec
# check two-state closed form against the DP
chk = []
for name in ["percap_0.01", "sat_0.9_2000", "const_0.99"]:
    for n in [5, 40, 300]:
        p = replace(base, G=1000.0)
        g = float(param[name](p.G / n))
        chk.append(abs(chain_two_state(n, p, param[name]) - chain_exact(n, replace(p, gamma=g))[0]))
res["gamma_delta"] = dict(params=gam, closed_form_vs_dp_maxabs=float(max(chk)), delta_ref=float(d_ref))
print("F", json.dumps(gam, indent=0)[:3000], max(chk), flush=True)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
cols = ["#2a6f97", "#64b5cd", "#c44e52", "#dd8452", "#55a868"]
for c, (name, gf) in zip(cols, param.items()):
    pis = []
    for D in dgrid:
        p = replace(base, G=D); e = eps_step(D, p); a = (1 - mu(D, p)) * (1 - gf(D)); pis.append(a / (a + e))
    ax[0].plot(dgrid, pis, c=c, label=name)
    ax[1].plot(Gs, [r["S"] for r in gam[name]["by_G"]], "o-", c=c, ms=3, label=name)
ax[1].plot(Gs, [r["S_gamma1"] for r in gam["const_0.99"]["by_G"]], "k--", label="gamma=1")
ax[0].set_xscale("log"); ax[0].set_xlabel("step size Delta (Elo)"); ax[0].set_ylabel("stationary P(aligned) pi(Delta)")
ax[1].set_xscale("log"); ax[1].set_xlabel("capability gap G (Elo)"); ax[1].set_ylabel("best P(S) over n")
ax[0].legend(fontsize=6, frameon=False); ax[1].set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(OUT + "fig6_gamma_delta.png", dpi=200); plt.close(fig)

json.dump(res, open(OUT + "revision_results.json", "w"), indent=1, default=float)
print("done")
