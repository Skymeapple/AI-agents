"""Exp 3: (a) heterogeneous latent-defect rates (Proposition 3) and the ratio of
reliability horizons; (b) verifier diversity under correlated blind spots
(Proposition 5).  Writes results/exp3_hetero_diversity.json and fig2/fig3."""
import json, os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import mixture_success, diverse_blindspot, ceiling_q, horizon

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(3)
out = {}

# ---- (a) horizon ratios -------------------------------------------------
def mix_horizon(level, alpha, beta):
    f = lambda N: float(mixture_success(N, alpha, beta)) - level
    return brentq(f, 1e-9, 1e12)

def gamma_ratio(level, alpha):
    """Small-q (Gamma) limit: S(N) = (1+N/beta)^(-alpha) -> N_x/N_50, free of beta."""
    return (level ** (-1 / alpha) - 1) / (0.5 ** (-1 / alpha) - 1)

const = {str(l): float(np.log(l) / np.log(0.5)) for l in [0.8, 0.9, 0.99]}
rows = []
for alpha in [0.5, 1.0, 2.0, 4.0, 16.0, 1e3]:
    row = dict(alpha=alpha)
    for qbar in [0.01, 0.05]:
        beta = alpha * (1 - qbar) / qbar
        n50 = mix_horizon(0.5, alpha, beta)
        row[f"qbar={qbar}"] = dict(N50=n50, **{f"N{int(l*100)}/N50": mix_horizon(l, alpha, beta) / n50
                                               for l in [0.8, 0.9, 0.99]},
                                    N50_times_qbar=n50 * qbar,
                                    S_at_10xN50=float(mixture_success(10 * n50, alpha, beta)))
    row["gamma_limit"] = {f"N{int(l*100)}/N50": float(gamma_ratio(l, alpha)) for l in [0.8, 0.9, 0.99]}
    rows.append(row)
# alpha that reproduces a given N80/N50 ratio (Gamma limit)
inv = {}
for name, target in [("1/4", 0.25), ("1/5", 0.2), ("1/6", 1 / 6)]:
    inv[name] = float(brentq(lambda a: gamma_ratio(0.8, a) - target, 0.05, 50))
# Monte Carlo check of Prop. 3
mc = []
for alpha, qbar, N in [(1.0, 0.02, 20), (1.0, 0.02, 100), (2.0, 0.05, 40), (0.5, 0.01, 200)]:
    beta = alpha * (1 - qbar) / qbar
    q = rng.beta(alpha, beta, size=200_000)
    succ = (rng.random((200_000, 1)) < (1 - q[:, None]) ** N).mean()  # exact conditional success
    succ2 = (rng.binomial(N, q) == 0).mean()                          # explicit defect draws
    mc.append(dict(alpha=alpha, qbar=qbar, N=N, mc=float(succ2), mc_cond=float(succ),
                   theory=float(mixture_success(N, alpha, beta)), homogeneous=float((1 - qbar) ** N)))
out["horizon_ratios"] = dict(constant_hazard=const, mixture=rows, alpha_for_N80_over_N50=inv, mc_check=mc)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
N = np.logspace(0, 4, 300)
qbar = 0.02
ax[0].loglog(N, (1 - qbar) ** N, "k--", label="homogeneous (constant hazard)")
for alpha, col in zip([0.5, 1, 2, 4], ["#c0392b", "#e67e22", "#2e86c1", "#1f4e79"]):
    ax[0].loglog(N, mixture_success(N, alpha, alpha * (1 - qbar) / qbar), color=col, label=f"Beta mixture, alpha={alpha}")
ax[0].set_ylim(1e-4, 1.2); ax[0].set_xlabel("project size N (modules)"); ax[0].set_ylabel("P(project correct)")
ax[0].set_title("(a) Heterogeneity turns exponential decay into a power law"); ax[0].legend(frameon=False, fontsize=8)
al = np.logspace(-0.5, 2, 200)
ax[1].semilogx(al, [gamma_ratio(0.8, a) for a in al], color="#1f4e79")
ax[1].axhline(const["0.8"], color="k", ls="--", lw=0.8); ax[1].text(20, const["0.8"] + 0.008, "constant hazard: 0.322", fontsize=8)
ax[1].axhspan(1 / 6, 1 / 4, color="#e67e22", alpha=0.25); ax[1].text(0.35, 0.255, "reported range 4-6x shorter (Kwa et al., 2025)", fontsize=8)
ax[1].set_xlabel("heterogeneity shape alpha (small = heterogeneous)"); ax[1].set_ylabel("N$_{80}$ / N$_{50}$")
ax[1].set_title("(b) Ratio of 80% to 50% horizon")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_heterogeneity.png"), dpi=150)

# ---- (b) verifier diversity -------------------------------------------
p = 0.9
div = []
for b1, conc in [(0.2, None), (0.2, 10.0), (0.2, 1.0), (0.2, 0.25), (0.05, None), (0.05, 1.0)]:
    row = dict(b1=b1, concentration=conc, by_m={})
    for m in [1, 2, 3, 5, 10, 30]:
        bm = b1 ** m if conc is None else float(diverse_blindspot(m, b1 * conc, (1 - b1) * conc))
        row["by_m"][str(m)] = dict(b_m=bm, N50_ceiling=float(horizon(ceiling_q(p, bm))))
    div.append(row)
# MC check of b_m for one correlated case
a_, b_ = 0.2 * 1.0, 0.8 * 1.0
th = rng.beta(a_, b_, size=400_000)
mcdiv = {str(m): dict(mc=float((rng.random((400_000, m)) < th[:, None]).all(1).mean()),
                      theory=float(diverse_blindspot(m, a_, b_))) for m in [1, 2, 3, 5]}
out["diversity"] = dict(p=p, rows=div, mc_check_conc1=mcdiv)

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ms = np.arange(1, 31)
ax.loglog(ms, horizon(ceiling_q(p, 0.2 ** ms)), "k--", label="independent blind spots")
for conc, col in zip([10.0, 1.0, 0.25], ["#2e86c1", "#e67e22", "#c0392b"]):
    ax.loglog(ms, horizon(ceiling_q(p, diverse_blindspot(ms, 0.2 * conc, 0.8 * conc))), color=col,
              label=f"correlated, concentration {conc}")
ax.set_ylim(10, 1e7); ax.set_xlabel("number of verifiers m"); ax.set_ylabel("horizon ceiling N$_{50}$ (c=1)")
ax.set_title("Verifier diversity (p=0.9, b$_1$=0.2)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_diversity.png"), dpi=150)

json.dump(out, open(os.path.join(OUT, "exp3_hetero_diversity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
