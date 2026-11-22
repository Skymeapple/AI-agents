"""Numerical check of the attention-budget theorem (Theorem 3) and of the cost of
fatigue-blind escalation design, on random heterogeneous instances.

Model: m permission atoms. Atom i: need prob p_i, value of a needed-but-denied
atom v_i, attack-exposure harm r_i = lambda_i * h_i if standing-granted,
malicious-request rate lambda_i. Options: G (grant), D (deny), E (escalate).
  loss = sum_D p_i v_i + sum_G r_i + sum_E [kappa*w_i + (1 - d(n)) r_i],
  w_i = p_i + lambda_i (prompts generated), n = sum_E w_i,
  d(n) = d0 * exp(-n / n0)  (human denial prob. of a malicious request).
Exact optimum by enumeration of all 3^m assignments (m = 10).
Outputs results/attention_budget.json and results/fig_attention_budget.png
"""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng = np.random.default_rng(7)
m = 10
A = np.array(list(itertools.product(range(3), repeat=m)), dtype=np.int8)   # 0=G 1=D 2=E
isG, isD, isE = (A == 0), (A == 1), (A == 2)

def instance():
    p = rng.beta(0.6, 2.0, m)
    lam = rng.uniform(0.01, 0.2, m)
    h = rng.lognormal(0.0, 1.5, m)
    r = lam * h
    v = rng.lognormal(0.0, 1.0, m)
    kappa = rng.uniform(0.005, 0.05)
    return p, lam, r, v, kappa

def losses(p, lam, r, v, kappa, d0, n0):
    w = p + lam
    n = isE @ w
    d = d0 * np.exp(-n / n0) if np.isfinite(n0) else d0 * np.ones_like(n)
    base = isD @ (p * v) + isG @ r + kappa * n + isE @ r
    return base - d * (isE @ r), n, w

def blind_choice(p, lam, r, v, kappa, d0):
    w = p + lam
    costs = np.stack([r, p * v, kappa * w + (1 - d0) * r], 1)
    return costs.argmin(1)

def idx_of(choice):
    return int(sum(int(c) * 3 ** (m - 1 - j) for j, c in enumerate(choice)))

d0 = 0.9
ANCKER_N0 = 1.0 / -np.log(0.7)   # 30% drop in acceptance per additional alert (Ancker et al. 2017)
n0s = [0.5, 1.0, 2.0, ANCKER_N0, 4.0, 8.0, np.inf]
N_INST = 300
out = {"m": m, "d0": d0, "n_instances": N_INST, "ancker_n0": ANCKER_N0, "by_n0": {}}
insts = [instance() for _ in range(N_INST)]
for n0 in n0s:
    rec = dict(nstar=[], bound=[], bound_ok=[], blind_regret=[], blind_load=[], escall_vs_grantall=[],
               opt_vs_noesc=[], frac_E=[])
    for (p, lam, r, v, kappa) in insts:
        L, n, w = losses(p, lam, r, v, kappa, d0, n0)
        k = int(np.argmin(L))
        noesc = L[~isE.any(1)].min()
        b = blind_choice(p, lam, r, v, kappa, d0); kb = idx_of(b)
        escall = L[idx_of([2] * m)]; grantall = L[idx_of([0] * m)]
        wmax = w[isE[k]].max() if isE[k].any() else 0.0
        rec["nstar"].append(float(n[k])); rec["bound"].append(float(n0 + wmax) if np.isfinite(n0) else np.inf)
        rec["bound_ok"].append(bool(n[k] < n0 + wmax + 1e-12) if np.isfinite(n0) else True)
        rec["blind_regret"].append(float(L[kb] / L[k] - 1)); rec["blind_load"].append(float(n[kb]))
        rec["escall_vs_grantall"].append(bool(escall > grantall))
        rec["opt_vs_noesc"].append(float(1 - L[k] / noesc)); rec["frac_E"].append(float(isE[k].mean()))
    key = "inf" if not np.isfinite(n0) else f"{n0:.3g}"
    out["by_n0"][key] = {
        "n0": None if not np.isfinite(n0) else n0,
        "bound_holds_frac": float(np.mean(rec["bound_ok"])),
        "nstar_mean": float(np.mean(rec["nstar"])), "nstar_max": float(np.max(rec["nstar"])),
        "nstar_over_n0_max": float(np.max(np.array(rec["nstar"]) / n0)) if np.isfinite(n0) else None,
        "blind_load_mean": float(np.mean(rec["blind_load"])),
        "blind_regret_median": float(np.median(rec["blind_regret"])),
        "blind_regret_p90": float(np.percentile(rec["blind_regret"], 90)),
        "escalate_all_worse_than_grant_all_frac": float(np.mean(rec["escall_vs_grantall"])),
        "opt_saving_vs_best_no_escalation_median": float(np.median(rec["opt_vs_noesc"])),
        "frac_atoms_escalated_mean": float(np.mean(rec["frac_E"])),
    }
    print(key, out["by_n0"][key])
json.dump(out, open(os.path.join(RES, "attention_budget.json"), "w"), indent=1)

# figure: optimal vs fatigue-blind load against n0
ks = [k for k in out["by_n0"] if k != "inf"]
x = [out["by_n0"][k]["n0"] for k in ks]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(x, [out["by_n0"][k]["nstar_mean"] for k in ks], "o-", label="optimal load n* (mean)", color="#2a6fb0")
ax[0].plot(x, [out["by_n0"][k]["nstar_max"] for k in ks], "s--", label="optimal load n* (max)", color="#2a6fb0", alpha=.6)
ax[0].plot(x, [out["by_n0"][k]["blind_load_mean"] for k in ks], "^-", label="fatigue-blind load (mean)", color="#c2571a")
ax[0].plot(x, x, "k:", label="n = n0")
ax[0].axvline(ANCKER_N0, color="gray", lw=.8); ax[0].text(ANCKER_N0, 0.2, " Ancker-calibrated n0", fontsize=7, color="gray")
ax[0].set_xlabel("fatigue scale n0 (prompts per task)"); ax[0].set_ylabel("approval prompts per task")
ax[0].legend(frameon=False, fontsize=7); ax[0].grid(alpha=.3)
ax[1].plot(x, [100 * out["by_n0"][k]["blind_regret_median"] for k in ks], "o-", label="median", color="#c2571a")
ax[1].plot(x, [100 * out["by_n0"][k]["blind_regret_p90"] for k in ks], "s--", label="90th percentile", color="#c2571a", alpha=.6)
ax[1].set_xlabel("fatigue scale n0"); ax[1].set_ylabel("excess loss of fatigue-blind design (%)")
ax[1].legend(frameon=False, fontsize=7); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_attention_budget.png"), dpi=160)
