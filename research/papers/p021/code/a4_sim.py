"""Analysis 4: simulation study of what a METR-sized suite can and cannot certify.

Generative model (calibrated to Claude Opus 4.5 on METR TH1.1):
   p_i = (1 - lamT_i) * (1 - lamR) * sigmoid(a - b x_i + u_i),  u_i ~ N(0, sigma^2)
   lamT_i ~ Bernoulli(piT): task-level lapse ("broken / consensus-hard" task, p_i = 0)
   lamR: run-level lapse (independent per run)
We then fit the practitioner models (2PL / 3PL binomial, ignoring task effects) and ask:
 S1  How large is the 2PL extrapolated H99 when a ceiling exists (H99 undefined)?
     What is the power of the lapse LRT, under the current design, 4x more short tasks,
     or 4x more runs on the same tasks?
 S2  Coverage of a run-level Clopper-Pearson interval vs a task-cluster bootstrap for the
     short-task failure rate.
Outputs results/sim_S1.csv, results/sim_S2.csv, results/sigma_calibration.json, fig5_sim.png
"""
import json, csv
import numpy as np
from scipy.stats import chi2
from scipy.special import roots_hermitenorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

R = int(os.environ.get("NSIM", 200))
rng = np.random.default_rng(3)
rows = load_runs()
REF = "Claude Opus 4.5 (Inspect)"
T = task_table(rows, REF)
x0, n0, s0, fam0 = T["x"], T["n"], T["s"], T["fam"]

# ---- calibrate random-intercept SD by marginal ML (Gauss-Hermite) ----
gh_x, gh_w = roots_hermitenorm(40); gh_w = gh_w / gh_w.sum()


def marg_nll(th):
    a, b, ls = th
    sig = np.exp(ls)
    eta = a - b * x0[:, None] + sig * gh_x[None, :]
    p = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    lik = np.exp(s0[:, None] * np.log(p) + (n0 - s0)[:, None] * np.log1p(-p)) @ gh_w
    return -np.sum(np.log(lik + 1e-300))


from scipy.optimize import minimize
best = min((minimize(marg_nll, np.array(st), method="Nelder-Mead", options=dict(maxiter=4000))
            for st in [(3, 0.6, 0.0), (5, 1.0, 0.7), (2, 0.4, -1)]), key=lambda r: r.fun)
A, Bs, SIG = best.x[0], best.x[1], float(np.exp(best.x[2]))
json.dump(dict(agent=short(REF), a=A, b=Bs, sigma=SIG, nll=best.fun),
          open(os.path.join(RESULTS, "sigma_calibration.json"), "w"), indent=1)
print(f"calibration: a={A:.3f} b={Bs:.3f} sigma={SIG:.3f}")


def design(kind):
    x, n, fam = x0.copy(), n0.copy(), fam0.copy()
    if kind == "4x_short_tasks":
        m = 2 ** x < 15
        x = np.concatenate([x] + [x[m]] * 3); n = np.concatenate([n] + [n[m]] * 3)
        fam = np.concatenate([fam] + [np.char.add(fam[m], f"_rep{r}") for r in range(3)])
    elif kind == "4x_runs":
        n = n * 4
    return x, n, fam


def simulate(x, n, piT, lamR):
    u = rng.normal(0, SIG, len(x))
    broken = rng.random(len(x)) < piT
    p = (1 - broken) * (1 - lamR) * expit(A - Bs * x + u)
    return rng.binomial(n, p)


# ---------------- S1 ----------------
configs = [("none", 0.0, 0.0), ("run 1%", 0.0, 0.01), ("run 2%", 0.0, 0.02),
           ("task 1%", 0.01, 0.0), ("task 2%", 0.02, 0.0), ("task 1% + run 1%", 0.01, 0.01)]
designs = ["current", "4x_short_tasks", "4x_runs"]
S1 = []
for dname in designs:
    x, n, fam = design(dname)
    for cname, piT, lamR in configs:
        h99, h95, rej, lam_hat, sfail = [], [], [], [], []
        for r in range(R):
            s = simulate(x, n, piT, lamR)
            f2 = fit(x, n, s, lapse=False); f3 = fit(x, n, s, lapse=True)
            LR = max(0, 2 * (f2["nll"] - f3["nll"]))
            rej.append(0.5 * chi2.sf(LR, 1) < 0.05 if LR > 0 else False)
            h99.append(horizon(f2, 0.99)); h95.append(horizon(f2, 0.95)); lam_hat.append(f3["lam"])
            m = 2 ** x < 15
            sfail.append(1 - s[m].sum() / n[m].sum())
        rec = dict(design=dname, config=cname, piT=piT, lamR=lamR, ceiling=1 - (1 - piT) * (1 - lamR) if True else 0,
                   H99_2pl_median=float(np.nanmedian(h99)), H99_2pl_q10=float(np.nanpercentile(h99, 10)),
                   H99_2pl_q90=float(np.nanpercentile(h99, 90)), H95_2pl_median=float(np.nanmedian(h95)),
                   lrt_power=float(np.mean(rej)), lam_hat_median=float(np.median(lam_hat)),
                   short_fail_mean=float(np.mean(sfail)))
        S1.append(rec)
        print(f"{dname:15s} {cname:18s} H99_2pl med={rec['H99_2pl_median']:.3f}min [{rec['H99_2pl_q10']:.3f},{rec['H99_2pl_q90']:.3f}] "
              f"LRT power={rec['lrt_power']:.2f} lam_hat={rec['lam_hat_median']:.4f} shortfail={rec['short_fail_mean']:.4f}", flush=True)
with open(os.path.join(RESULTS, "sim_S1.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(S1[0].keys())); w.writeheader(); w.writerows(S1)

# ---------------- S2: CI coverage for short-task failure rate ----------------
S2 = []
m15 = 2 ** x0 < 15
for cname, piT, lamR in configs:
    # true expected short-task failure rate (run-weighted), by Monte Carlo over u, broken
    xs, ns = x0[m15], n0[m15]
    U = rng.normal(0, SIG, (20000, len(xs)))
    ptrue = (1 - piT) * (1 - lamR) * expit(A - Bs * xs + U)
    true_fail = 1 - np.sum(ptrue.mean(0) * ns) / ns.sum()
    cov_cp = cov_bs = 0; w_cp = []; w_bs = []
    for r in range(R):
        s = simulate(xs, ns, piT, lamR)
        F, N = (ns - s).sum(), ns.sum()
        lo, hi = clopper_pearson(F, N)
        cov_cp += lo <= true_fail <= hi; w_cp.append(hi - lo)
        bs = []
        for b in range(400):
            idx = cluster_bootstrap_indices(fam0[m15], rng)
            bs.append(1 - s[idx].sum() / ns[idx].sum())
        blo, bhi = np.percentile(bs, [2.5, 97.5])
        cov_bs += blo <= true_fail <= bhi; w_bs.append(bhi - blo)
    rec = dict(config=cname, true_fail=true_fail, cov_clopper_pearson=cov_cp / R, cov_cluster_boot=cov_bs / R,
               width_cp=float(np.mean(w_cp)), width_boot=float(np.mean(w_bs)))
    S2.append(rec)
    print(f"S2 {cname:18s} true={true_fail:.4f} cov CP={rec['cov_clopper_pearson']:.2f} boot={rec['cov_cluster_boot']:.2f} "
          f"width CP={rec['width_cp']:.4f} boot={rec['width_boot']:.4f}", flush=True)
with open(os.path.join(RESULTS, "sim_S2.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(S2[0].keys())); w.writeheader(); w.writerows(S2)

# ---------------- figure ----------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
cn = [c[0] for c in configs]
for j, dname in enumerate(designs):
    v = [r["lrt_power"] for r in S1 if r["design"] == dname]
    axs[0].bar(np.arange(len(cn)) + (j - 1) * 0.27, v, width=0.27, label=dname)
axs[0].set_xticks(range(len(cn))); axs[0].set_xticklabels(cn, rotation=30, fontsize=8)
axs[0].set_ylabel("power of lapse LRT (alpha=0.05)"); axs[0].legend(fontsize=8)
axs[0].axhline(0.05, ls=":", color="gray"); axs[0].set_title("Detecting a reliability ceiling")
v = [r["H99_2pl_median"] * 60 for r in S1 if r["design"] == "current"]
lo = [r["H99_2pl_q10"] * 60 for r in S1 if r["design"] == "current"]
hi = [r["H99_2pl_q90"] * 60 for r in S1 if r["design"] == "current"]
axs[1].errorbar(range(len(cn)), v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="o", capsize=3, color="k")
axs[1].set_xticks(range(len(cn))); axs[1].set_xticklabels(cn, rotation=30, fontsize=8)
axs[1].set_ylabel("2PL-extrapolated 99% horizon (seconds)")
axs[1].set_title("2PL reports a finite H99 even when none exists")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig5_sim.png"), dpi=150)
print("done")
