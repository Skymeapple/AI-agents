"""Experiment 7 (added in revision): what the realism-response curve does and does not
identify. Simulation with known policies and binomial sampling noise; no training.

 (a) Realism units. If the rung's cue strength is g(t)*d' with g nonlinear (the
     evaluator does not know the cue metric), probit(rate) is not linear in t and its
     fitted slope is not d'. Only monotonicity survives.
 (b) Confound. An ALIGNED model whose accidental misbehaviour rises with task
     difficulty, where difficulty is correlated with realism along the ladder, gives a
     rising probit-linear curve that a single ladder cannot tell apart from a schemer.
 (c) Factorial control. Varying cue realism t and task difficulty u independently
     (a 2-factor ladder) and fitting probit(rate) ~ t + u separates the two, *provided*
     realism can be varied at fixed difficulty.
Each rung has n = 500 episodes. 200 replicate ladders.
Run: python3 exp7_diagnostic.py   (seconds)
"""
import json, os
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

DP = 2.0
TAU = DP + 1.5          # schemer threshold: rate Phi(t d' - tau); ~2e-4 at t=0, ~7% at t=1
N_RUNG = 500
REPS = 200
TS = np.linspace(0, 1.2, 7)


def probit_mle(X, k, n):
    """Binomial probit regression; returns coefficients and Wald SEs."""
    X = np.column_stack([np.ones(len(k)), X])

    def nll(b):
        p = np.clip(norm.cdf(X @ b), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    r = minimize(nll, np.r_[-2.0, np.zeros(X.shape[1] - 1)], method="BFGS")
    eta = X @ r.x
    p = np.clip(norm.cdf(eta), 1e-12, 1 - 1e-12)
    w = n * norm.pdf(eta) ** 2 / (p * (1 - p))          # expected (Fisher) information weights
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (w[:, None] * X))))
    return r.x, se


def main():
    rng = np.random.default_rng(7)
    out = {}
    # (a) realism units
    g_funcs = {"linear g(t)=t": lambda t: t, "convex g(t)=t^2": lambda t: t ** 2,
               "concave g(t)=1-(1-t)^2 (capped)": lambda t: 1 - (1 - np.minimum(t, 1)) ** 2}
    a = {}
    for name, g in g_funcs.items():
        rate = norm.cdf(g(TS) * DP - TAU)
        sl, ic = np.polyfit(TS, norm.ppf(rate), 1)
        resid = norm.ppf(rate) - (sl * TS + ic)
        a[name] = dict(slope=float(sl), max_abs_resid=float(np.max(np.abs(resid))),
                       monotone=bool(np.all(np.diff(rate) >= 0)), rates=rate.tolist())
    out["a_realism_units"] = a

    # (b) single ladder: schemer vs aligned-with-accidental-errors (difficulty u = t)
    def schemer(t, u):
        return norm.cdf(t * DP - TAU)

    def accidental(t, u):
        return norm.cdf(-3.0 + 1.5 * u)

    b = {}
    for name, f in [("schemer", schemer), ("aligned_accidental", accidental)]:
        slopes, sig = [], 0
        for _ in range(REPS):
            p = f(TS, TS)
            k = rng.binomial(N_RUNG, p)
            co, se = probit_mle(TS[:, None], k, N_RUNG)
            slopes.append(co[1]); sig += int(co[1] - 1.96 * se[1] > 0)
        b[name] = dict(slope_median=float(np.median(slopes)), slope_q05=float(np.quantile(slopes, 0.05)),
                       slope_q95=float(np.quantile(slopes, 0.95)), frac_significant_positive=sig / REPS)
    out["b_single_ladder"] = b

    # (c) factorial ladder: t in TS x u in {0, 0.6, 1.2}
    U = np.array([0.0, 0.6, 1.2])
    tt, uu = np.meshgrid(TS, U)
    tt, uu = tt.ravel(), uu.ravel()
    c = {}
    for name, f in [("schemer", schemer), ("aligned_accidental", accidental),
                    ("schemer_plus_accidental", lambda t, u: 1 - (1 - schemer(t, u)) * (1 - accidental(t, u)))]:
        ct, cu, st, su = [], [], 0, 0
        for _ in range(REPS):
            k = rng.binomial(N_RUNG, f(tt, uu))
            co, se = probit_mle(np.column_stack([tt, uu]), k, N_RUNG)
            ct.append(co[1]); cu.append(co[2])
            st += int(co[1] - 1.96 * se[1] > 0); su += int(co[2] - 1.96 * se[2] > 0)
        c[name] = dict(coef_t_median=float(np.median(ct)), coef_u_median=float(np.median(cu)),
                       frac_t_significant=st / REPS, frac_u_significant=su / REPS)
    out["c_factorial"] = c
    out["config"] = dict(dprime=DP, tau=TAU, n_per_rung=N_RUNG, reps=REPS, t=TS.tolist(), u=U.tolist())
    json.dump(out, open(os.path.join(RESULTS, "exp7_diagnostic.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, g in g_funcs.items():
        ax[0].plot(TS, norm.ppf(norm.cdf(g(TS) * DP - TAU)), "o-", ms=3, label=name)
    ax[0].set_xlabel("nominal realism t"); ax[0].set_ylabel("probit(defection rate)")
    ax[0].set_title("(a) slope depends on unknown realism units"); ax[0].legend(fontsize=7)
    ax[1].plot(TS, norm.ppf(schemer(TS, TS)), "o-", label="schemer (conditional)")
    ax[1].plot(TS, norm.ppf(accidental(TS, TS)), "s-", label="aligned, errors rise with difficulty")
    ax[1].set_xlabel("rung (realism and difficulty co-vary)")
    ax[1].set_title("(b) a single ladder cannot separate these"); ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig7_diagnostic.png"), dpi=150)


if __name__ == "__main__":
    main()
