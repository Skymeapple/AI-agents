"""Experiment 3 (analytic, no training): how the deployment-defection ceiling of a
schemer scales with the number of evaluation episodes N (Thm 2), with and without
honeypots (Thm 3), and what discriminability values reported in the literature imply.

A schemer requiring P(undetected across N i.i.d. eval episodes) >= 1 - delta must keep
alpha <= alpha_N = 1-(1-delta)^(1/N). Its deployment defection rate is then at most
R(alpha_N). We use delta = 0.1 (the schemer accepts at most a 10% chance of ever being
caught) as the headline value and report delta = 0.5 as a sensitivity check.

Literature inputs (cited in the paper, not produced here): AUCs for distinguishing
evaluation from deployment transcripts from Needham et al. (2025), arXiv:2505.23836:
Gemini 2.5 Pro 0.83, Claude 3.7 Sonnet 0.82, GPT-4.1 0.83, GPT-4.1 mini 0.75, human 0.92.
Converting an AUC to d' assumes an equal-variance binormal ROC (exp4 tests this).
Run: python3 exp3_scaling.py  (seconds)
"""
import json, os
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS
from theory import alpha_budget, binormal, auc_to_a, kl_bound, mixture_frontier

LIT_AUC = {"GPT-4.1 mini": 0.75, "Claude 3.7 Sonnet": 0.82, "Gemini 2.5 Pro": 0.83,
           "GPT-4.1": 0.83, "human baseline": 0.92}


def n_required(dp, beta_cap, delta):
    """Smallest N such that R(alpha_N) <= beta_cap for equal-variance binormal with d'."""
    a_need = norm.cdf(norm.ppf(beta_cap) - dp)        # alpha with R(alpha) = beta_cap
    # alpha_N <= a_need  <=>  N >= ln(1-delta)/ln(1-a_need)
    return float(np.log(1 - delta) / np.log(1 - a_need))


def n_required_honeypot(rho, Delta, dp_eval, beta_cap, delta):
    comps = [(1 - rho, 0.0), (rho, dp_eval - Delta)]
    f = lambda la: float(mixture_frontier(dp_eval, comps, 10 ** la)) - beta_cap
    if f(-12) > 0:
        return float("inf")
    la = brentq(f, -12, 0)
    return float(np.log(1 - delta) / np.log(1 - 10 ** la))


def main():
    out = {}
    Ns = np.logspace(1, 8, 200)
    res = {}
    for delta in [0.1, 0.5]:
        aN = alpha_budget(Ns, delta)
        curves = {}
        for dp in [1.0, 1.35, 2.0, 3.0, 4.0]:
            curves[f"dprime_{dp}"] = binormal(aN, dp).tolist()
            curves[f"kl_dprime_{dp}"] = kl_bound(aN, dp * dp / 2).tolist()
        for rho, Delta in [(0.01, 0.0), (0.01, 0.5), (0.01, 1.0), (0.1, 0.0)]:
            comps = [(1 - rho, 0.0), (rho, 2.0 - Delta)]
            curves[f"honeypot_rho{rho}_Delta{Delta}_dp2"] = mixture_frontier(2.0, comps, aN).tolist()
        res[f"delta_{delta}"] = curves
    out["N"] = Ns.tolist(); out["curves"] = res

    # headline numbers
    head = {}
    for delta in [0.1, 0.5]:
        for N in [100, 1000, 10000, 100000]:
            aN = float(alpha_budget(N, delta))
            row = {"alpha_N": aN}
            for name, auc in LIT_AUC.items():
                dp = float(auc_to_a(auc))
                row[name] = {"dprime": dp, "beta_ceiling": float(binormal(aN, dp))}
            head[f"delta{delta}_N{N}"] = row
    out["literature_ceilings"] = head

    # required N to cap deployment defection at 1% (and 10%) vs d', no honeypots
    req = {}
    for cap in [0.01, 0.1]:
        for dp in [1.0, 1.35, 1.99, 2.0, 2.5, 3.0, 4.0, 5.0]:
            req[f"cap{cap}_dp{dp}"] = n_required(dp, cap, 0.1)
        for rho in [0.01, 0.1]:
            for Delta in [0.0, 0.5, 1.0, 1.5]:
                req[f"cap{cap}_honeypot_rho{rho}_Delta{Delta}"] = n_required_honeypot(rho, Delta, 2.0, cap, 0.1)
    out["N_required"] = req
    # marginal value: d ln N_req / d d' at d'=2 (numerical), vs d ln N_req / d Delta
    eps = 1e-3
    out["dlnN_ddprime_at2"] = float((np.log(n_required(2 + eps, 0.01, 0.1)) - np.log(n_required(2 - eps, 0.01, 0.1))) / (2 * eps))
    # decomposition of ln N_req at cap b (revision): x = d' - Phi^-1(b);
    # ln N_req ~ x^2/2 + ln x + ln(c sqrt(2 pi)), and x^2/2 = d'^2/2 + d'|z_b| + z_b^2/2
    dec = {}
    c = -np.log(0.9)
    for cap in [0.01, 0.1]:
        zb = norm.ppf(cap)
        for dp in [1.0, 2.0, 3.0, 4.0]:
            x = dp - zb
            dec[f"cap{cap}_dp{dp}"] = dict(lnN_exact=float(np.log(n_required(dp, cap, 0.1))),
                                           asymptotic=float(x * x / 2 + np.log(x) + np.log(c * np.sqrt(2 * np.pi))),
                                           dp2_over_2=dp * dp / 2, cross=float(dp * abs(zb)),
                                           zb2_over_2=float(zb * zb / 2), lnx=float(np.log(x)),
                                           local_slope_x_plus_1_over_x=float(x + 1 / x))
    out["lnN_decomposition"] = dec
    json.dump(out, open(os.path.join(RESULTS, "exp3_scaling.json"), "w"), indent=1)
    print(json.dumps(head["delta0.1_N1000"], indent=1))
    print(json.dumps(req, indent=1)); print("dlnN/dd'", out["dlnN_ddprime_at2"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    c = res["delta_0.1"]
    cols = {1.0: "#1f77b4", 1.35: "#2ca02c", 2.0: "#ff7f0e", 3.0: "#d62728", 4.0: "#9467bd"}
    for dp, col in cols.items():
        ax[0].plot(Ns, c[f"dprime_{dp}"], "-", color=col, label=f"d'={dp}")
        ax[0].plot(Ns, c[f"kl_dprime_{dp}"], ":", color=col, lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1)
    ax[0].set_xlabel("number of i.i.d. evaluation episodes N")
    ax[0].set_ylabel("max deployment defection rate β*")
    ax[0].set_title("(a) No honeypots: solid exact (binormal), dotted KL bound")
    ax[0].legend(fontsize=8)
    ax[1].plot(Ns, c["dprime_2.0"], "k-", label="no honeypots, d'=2")
    for (rho, Delta), col in zip([(0.01, 0.0), (0.01, 0.5), (0.01, 1.0), (0.1, 0.0)],
                                 ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]):
        ax[1].plot(Ns, c[f"honeypot_rho{rho}_Delta{Delta}_dp2"], "-", color=col,
                   label=f"ρ={rho} honeypots, realism gap Δ={Delta}")
    ax[1].plot(Ns, -np.log(0.9) / (0.01 * Ns), "--", color="gray", lw=1, label="c/(ρN), ρ=0.01")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1)
    ax[1].set_xlabel("number of evaluation episodes N")
    ax[1].set_title("(b) Honeypots: realism gap Δ replaces d', share ρ rescales N")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig3_scaling.png"), dpi=150)


if __name__ == "__main__":
    main()
