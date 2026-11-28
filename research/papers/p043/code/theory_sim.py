"""Numerical check of Propositions 1-3 (linear-Gaussian model of probe transfer).

h = c*u + d*v + e*u_B + eps,  eps ~ N(0, I_d)  (whitened coordinates)
train: honest (c=0,d=0,e=0) vs instructed lie (c=1,d=1,e=0)
test : trigger-B lie (c=0,d=1,e=1) and cue-only (c=1,d=0,e=0)
Prediction (Prop. 1, LDA limit, u ⟂ v ⟂ u_B):
  AUROC_ID  = Phi(||u+v||/sqrt2)
  AUROC_B   = Phi(rho * ||u+v||/sqrt2),   rho = ||v||^2/||u+v||^2
  AUROC_cue = Phi((1-rho) * ||u+v||/sqrt2)
Prop. 3: with cue-only negatives added to training, the probe recovers v and
  AUROC_B -> Phi(||v||/sqrt2), AUROC_cue -> 0.5.
Prop. 4 (mismatch): belief code B and output code O; decoder trained where t=o.
  In a lie context the decoder's argmax is the truth iff ||B|| > ||O||.
"""
import json, os
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(0)
dim, n = 64, 4000


def auc(neg, pos):
    return roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])


def sample(n, c, d, e, u, v, uB):
    return rng.standard_normal((n, dim)) + c * u + d * v + e * uB


rows = []
S = 3.0  # ||u+v||  -> AUROC_ID = Phi(3/sqrt2) = 0.983
for rho in np.linspace(0.05, 0.95, 10):
    nv = np.sqrt(rho) * S; nu = np.sqrt(1 - rho) * S
    u = np.zeros(dim); u[0] = nu
    v = np.zeros(dim); v[1] = nv
    uB = np.zeros(dim); uB[2] = 2.0
    Xh = sample(n, 0, 0, 0, u, v, uB); Xa = sample(n, 1, 1, 0, u, v, uB)
    Xb = sample(n, 0, 1, 1, u, v, uB); Xc = sample(n, 1, 0, 0, u, v, uB)
    Xh2 = sample(n, 0, 0, 0, u, v, uB); Xa2 = sample(n, 1, 1, 0, u, v, uB)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.r_[Xh, Xa], np.r_[np.zeros(n), np.ones(n)])
    s = clf.decision_function
    # with controls (cue-only negatives)
    Xc_tr = sample(n, 1, 0, 0, u, v, uB)
    clf2 = LogisticRegression(C=1.0, max_iter=2000).fit(np.r_[Xh, Xc_tr, Xa],
                                                        np.r_[np.zeros(2 * n), np.ones(n)])
    s2 = clf2.decision_function
    rows.append({"rho": float(rho),
                 "ID_emp": auc(s(Xh2), s(Xa2)), "ID_pred": float(norm.cdf(S / np.sqrt(2))),
                 "B_emp": auc(s(Xh2), s(Xb)), "B_pred": float(norm.cdf(rho * S / np.sqrt(2))),
                 "cue_emp": auc(s(Xh2), s(Xc)), "cue_pred": float(norm.cdf((1 - rho) * S / np.sqrt(2))),
                 "B_ctrl_emp": auc(s2(Xh2), s2(Xb)), "B_ctrl_pred": float(norm.cdf(nv / np.sqrt(2))),
                 "cue_ctrl_emp": auc(s2(Xh2), s2(Xc))})

# Prop 4: belief/output code competition
rows4 = []
K = 10
for ratio in [0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0, 4.0]:
    Bc = rng.standard_normal((K, dim)); Bc /= np.linalg.norm(Bc, axis=1, keepdims=True)
    Oc = rng.standard_normal((K, dim)); Oc /= np.linalg.norm(Oc, axis=1, keepdims=True)
    b_norm = 2.0; o_norm = ratio * b_norm
    t = rng.integers(0, K, n)
    Hh = b_norm * Bc[t] + o_norm * Oc[t] + 0.5 * rng.standard_normal((n, dim))
    dec = LogisticRegression(C=1.0, max_iter=3000).fit(Hh, t)
    t2 = rng.integers(0, K, n); o2 = (t2 + 1) % K
    Hl = b_norm * Bc[t2] + o_norm * Oc[o2] + 0.5 * rng.standard_normal((n, dim))
    th = rng.integers(0, K, n)
    Hh2 = b_norm * Bc[th] + o_norm * Oc[th] + 0.5 * rng.standard_normal((n, dim))
    lp_l = dec.predict_log_proba(Hl)[np.arange(n), o2]
    lp_h = dec.predict_log_proba(Hh2)[np.arange(n), th]
    rows4.append({"O_over_B": ratio, "frac_argmax_truth_in_lie": float((dec.predict(Hl) == t2).mean()),
                  "mismatch_auroc": auc(-lp_h, -lp_l)})

json.dump({"prop1_3": rows, "prop4": rows4}, open(os.path.join(RES, "theory_sim.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
r = np.array([x["rho"] for x in rows])
for key, lab, col in [("B", "trigger-B lie vs honest", "#1f77b4"), ("cue", "cue-only vs honest", "#d62728"),
                      ("B_ctrl", "trigger-B, probe w/ controls", "#2ca02c")]:
    ax[0].plot(r, [x[key + "_emp"] for x in rows], "o", color=col, label=lab + " (sim)")
    if key + "_pred" in rows[0]:
        ax[0].plot(r, [x[key + "_pred"] for x in rows], "-", color=col, alpha=.7)
ax[0].plot(r, [x["cue_ctrl_emp"] for x in rows], "s", color="#9467bd", label="cue-only, probe w/ controls (sim)")
ax[0].axhline(rows[0]["ID_pred"], color="k", ls=":", label="in-distribution AUROC")
ax[0].set_xlabel(r"act share $\rho=\|v\|^2/\|u+v\|^2$"); ax[0].set_ylabel("AUROC")
ax[0].set_title("Prop. 1/3: probe transfer (lines = closed form)"); ax[0].legend(fontsize=7)
q = [x["O_over_B"] for x in rows4]
ax[1].plot(q, [x["mismatch_auroc"] for x in rows4], "o-", label="mismatch AUROC")
ax[1].plot(q, [x["frac_argmax_truth_in_lie"] for x in rows4], "s--", label="decoder reads truth in lie ctx")
ax[1].axvline(1.0, color="k", ls=":")
ax[1].set_xlabel(r"output-code / belief-code norm $\|O\|/\|B\|$"); ax[1].set_title("Prop. 4: belief vs output code")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_theory.png"), dpi=150)
print(json.dumps(rows, indent=0)); print(json.dumps(rows4, indent=0))
