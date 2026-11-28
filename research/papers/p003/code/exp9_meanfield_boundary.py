"""E9 (revision): derive the takeover boundary J*(h0) from the mean-field (population)
dynamics instead of fitting a power law to noisy bisection points.  Response to review W4.

Mean-field GRPO: theta <- theta + lr * E_p[ (f - fbar)/sigma * grad log p ],
sigma = sqrt(fbar(1-fbar)) (population limit of the group std for Bernoulli rewards),
computed exactly by enumerating all 1,024 strings.  Collapse = exploit mass > 0.5 at
step 3000 (same horizon and learning rate as E5/E7).  J* is found by bisection to 1e-4.

Also computes:
  * the initial-velocity (1-D) boundary: the J at which the initial drift of the x0 logit
    changes sign (exploit direction vs correct direction);
  * local log-log slopes of the boundary and the saturation limit J -> eta - beta.
Compared against the finite-sample bisection points of E7a (results/exp7.json)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import exp4_sequence as E

R = os.path.join(os.path.dirname(__file__), "..", "results")
BETA, ETA, LR, STEPS = 0.15, 0.95, 0.5, 3000
ALL = np.arange(len(E.X))


def set_suffix(ps):
    E.P0 = np.array([0.6] * 5 + [0.5] * 3 + [ps] * 2)
    return float(E.Factorised().probs() @ E.EXPLOIT)


def mf_final_exploit(cls, J, steps=STEPS):
    f = E.verifier(BETA + J, BETA, ETA); pol = cls()
    GL = None
    for s in range(steps):
        p = pol.probs(); fb = p @ f; sg = np.sqrt(max(fb * (1 - fb), 1e-12))
        w = p * (f - fb) / sg
        if isinstance(pol, E.Factorised):
            d = w @ (E.X - E.sig(pol.th)[None, :])
        else:
            d = w @ pol.grad_logp(ALL)
        d = LR * d
        if hasattr(pol, "step"): pol.step(d)
        else: pol.th += d
        if s % 200 == 0:
            ex = p @ E.EXPLOIT
            if ex > 0.9: return 1.0
    return float(pol.probs() @ E.EXPLOIT)


def mf_boundary(cls, tol=1e-4):
    lo, hi = 0.005, 0.799  # J > eta - beta = 0.8 makes the exploit no better than correct
    if mf_final_exploit(cls, hi) > 0.5: return hi
    if mf_final_exploit(cls, lo) <= 0.5: return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mf_final_exploit(cls, mid) > 0.5: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def x0_drift(J):
    f = E.verifier(BETA + J, BETA, ETA); p = E.Factorised().probs(); fb = p @ f
    q0 = E.sig(E.Factorised().th)[0]
    return float(p @ ((f - fb) * (E.X[:, 0] - q0)))  # >0: x0 pushed towards 1 (correct)


if __name__ == "__main__":
    ps_grid = [0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.225, 0.25, 0.275, 0.30, 0.33, 0.36, 0.40]
    rows = []
    for ps in ps_grid:
        h0 = set_suffix(ps)
        row = {"suffix_p": ps, "h0": h0}
        for pname, cls in [("factorised", E.Factorised), ("AR1", E.AR1)]:
            row[f"Jstar_mf_{pname}"] = mf_boundary(cls)
        try:
            row["Jstar_init_velocity"] = brentq(x0_drift, 1e-4, 0.8) if x0_drift(1e-4) < 0 < x0_drift(0.8) else None
        except ValueError:
            row["Jstar_init_velocity"] = None
        rows.append(row); print(row, flush=True)
    emp = json.load(open(os.path.join(R, "exp7.json")))["boundary"]
    out = {"rows": rows, "empirical_E7a": emp}
    # local slopes and comparison
    for pname in ["factorised", "AR1"]:
        h = np.array([r["h0"] for r in rows]); j = np.array([r[f"Jstar_mf_{pname}"] for r in rows])
        ok = (j > 0.006) & (j < 0.798)
        lh, lj = np.log(h[ok]), np.log(j[ok])
        out[f"local_slopes_mf_{pname}"] = [float(x) for x in np.diff(lj) / np.diff(lh)]
        e = [r for r in emp if r["policy"] == pname]
        jm = np.interp(np.log([r["h0"] for r in e]), np.log(h), j)
        out[f"empirical_vs_mf_{pname}"] = [{"h0": r["h0"], "J_emp": r["J_star"], "J_mf": float(x),
                                            "diff": float(r["J_star"] - x)} for r, x in zip(e, jm)]
        ej = np.array([r["J_star"] for r in e]); eh = np.array([r["h0"] for r in e])
        out[f"local_slopes_empirical_{pname}"] = [float(x) for x in np.diff(np.log(ej)) / np.diff(np.log(eh))]
        out[f"mean_abs_diff_emp_mf_{pname}"] = float(np.mean(np.abs(ej - jm)))
    json.dump(out, open(os.path.join(R, "exp9.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    h = [r["h0"] for r in rows]
    for k, pname in enumerate(["factorised", "AR1"]):
        ax[0].plot(h, [r[f"Jstar_mf_{pname}"] for r in rows], f"C{k}-", label=f"mean-field ({pname})")
        e = [r for r in emp if r["policy"] == pname]
        ax[0].plot([r["h0"] for r in e], [r["J_star"] for r in e], f"C{k}o", label=f"finite-sample bisection ({pname})")
    iv = [(r["h0"], r["Jstar_init_velocity"]) for r in rows if r["Jstar_init_velocity"]]
    ax[0].plot(*zip(*iv), "k:", label="initial x0-drift sign change")
    ax[0].axhline(ETA - BETA, color="gray", lw=0.6, ls="--"); ax[0].text(h[0], 0.81, "η−β", fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("exploit base mass h0")
    ax[0].set_ylabel("critical J*"); ax[0].legend(fontsize=7); ax[0].set_title("Takeover boundary")
    for k, pname in enumerate(["factorised", "AR1"]):
        hh = np.array(h); j = np.array([r[f"Jstar_mf_{pname}"] for r in rows]); ok = (j > 0.006) & (j < 0.798)
        mid = np.sqrt(hh[ok][1:] * hh[ok][:-1])
        ax[1].semilogx(mid, out[f"local_slopes_mf_{pname}"], f"C{k}-o", label=pname)
    ax[1].set_xlabel("h0"); ax[1].set_ylabel("local slope d log J*/d log h0"); ax[1].legend()
    ax[1].set_title("The boundary is curved (no single exponent)")
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig9_meanfield_boundary.png"), dpi=150)
