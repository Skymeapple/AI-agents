"""E1: Exact check of Theorem 1 (contraction bound on identity recoverability).

A memory store with S states evolves under a random consolidation kernel
K = (1-p) * Q + p * 1 nu^T, where Q is an arbitrary random stochastic matrix
(the 'content-preserving' part of the rewrite, which may itself scramble) and nu
is a regeneration distribution independent of the current memory (Doeblin
component). Identity X is uniform over Qn identities, each mapped to a distinct
initial memory state. We compute EXACTLY (matrix powers) the Bayes accuracy of
recovering X from M_t, the mutual information I(X;M_t), and the Dobrushin
coefficient of K, and compare to the bounds of Theorem 1.
"""
import json, numpy as np
from pathlib import Path
rng = np.random.default_rng(0)
OUT = Path(__file__).resolve().parents[1] / "results"

def dobrushin(K):
    S = K.shape[0]
    return max(0.5 * np.abs(K[i] - K[j]).sum() for i in range(S) for j in range(S))

def mi_bits(P_m_given_x):
    Qn = P_m_given_x.shape[0]
    pm = P_m_given_x.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(P_m_given_x > 0, P_m_given_x * np.log2(P_m_given_x / pm), 0.0)
    return r.sum() / Qn

rows = []
for trial, (S, Qn, p, conc, keep) in enumerate([(40, 8, 0.05, 0.3, 0.9), (40, 8, 0.02, 0.1, 0.9),
                                           (60, 16, 0.1, 1.0, 0.9), (30, 4, 0.01, 0.05, 1.0)]):
    Q = rng.dirichlet(np.full(S, conc), size=S)
    # make Q 'mostly preserving': mix with identity so memory is usually kept
    # (keep=1.0: pure keep-or-regenerate channel, where the bound is nearly tight)
    Q = keep * np.eye(S) + (1 - keep) * Q
    nu = rng.dirichlet(np.ones(S))
    K = (1 - p) * Q + p * np.outer(np.ones(S), nu)
    delta = dobrushin(K)
    init = rng.choice(S, Qn, replace=False)
    P = np.zeros((Qn, S)); P[np.arange(Qn), init] = 1
    for t in range(0, 301):
        if t % 10 == 0:
            acc = P.max(0).sum() / Qn
            rows.append(dict(trial=trial, S=S, Qn=Qn, p=p, delta=delta, t=t,
                             bayes_acc=acc,
                             acc_bound=1 / Qn + (1 - 1 / Qn) * delta ** t,
                             acc_bound_doeblin=1 / Qn + (1 - 1 / Qn) * (1 - p) ** t,
                             mi_bits=mi_bits(P),
                             mi_bound=np.log2(Qn) * delta ** t))
        P = P @ K
viol = [r for r in rows if r["bayes_acc"] > r["acc_bound"] + 1e-12 or r["mi_bits"] > r["mi_bound"] + 1e-9]
json.dump(dict(rows=rows, n_violations=len(viol)), open(OUT / "e1_contraction.json", "w"), indent=1)
print("violations:", len(viol))
for tr in range(4):
    sub = [r for r in rows if r["trial"] == tr]
    print(f"trial {tr}: delta={sub[0]['delta']:.4f} (1-p={1-sub[0]['p']:.3f}) "
          f"t=100 acc={sub[10]['bayes_acc']:.4f} bound={sub[10]['acc_bound']:.4f} "
          f"MI={sub[10]['mi_bits']:.4f} bound={sub[10]['mi_bound']:.4f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for tr in range(4):
    sub = [r for r in rows if r["trial"] == tr]
    t = [r["t"] for r in sub]
    lab = f"S={sub[0]['S']},|X|={sub[0]['Qn']},p={sub[0]['p']}"
    ax[0].plot(t, [r["bayes_acc"] for r in sub], color=cols[tr], label=lab)
    ax[0].plot(t, [r["acc_bound"] for r in sub], "--", color=cols[tr], lw=1)
    ax[1].semilogy(t, np.maximum([r["mi_bits"] for r in sub], 1e-8), color=cols[tr])
    ax[1].semilogy(t, [r["mi_bound"] for r in sub], "--", color=cols[tr], lw=1)
ax[0].set_xlabel("consolidation steps t"); ax[0].set_ylabel("Bayes accuracy of recovering X")
ax[1].set_xlabel("consolidation steps t"); ax[1].set_ylabel("I(X; M_t) [bits]")
ax[0].legend(fontsize=7); ax[0].set_title("solid: exact, dashed: Thm 1 bound", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig1_contraction.png", dpi=150)
