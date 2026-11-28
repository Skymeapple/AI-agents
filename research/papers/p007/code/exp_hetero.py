"""Round 3 (review_r2 W2): interaction below k_s when feature strengths differ across modalities.

Prop. 1 assumes equal-strength features with orthonormal heads, which gives a population tie for
r <= k_s.  The ReLU teacher of Exp. F uses Gaussian heads A_m, B_m (entries N(0, 1/(k_s+k_p))),
so each modality weights the shared features differently.  Lemma 1 holds for any W_m, and Ky Fan
gives sum_m top_r(G_m) >= top_r(sum_m G_m): the population interaction is >= 0 at every r, with
equality only if the modalities agree on their top-r subspace.

(1) Linear analogue of the ReLU teacher (d=64, q=8, k_s=6, k_p in {6, 0}, M in {2,4,8}):
    W_m = A_m^T S^T + B_m^T P_m^T.  Population ratio L_u/L_s at r = 2, 4, 6 (ratio of mean losses
    over 400 teacher draws), and Monte Carlo at finite data (sigma=0.5, n=4096, orthogonal-design
    model, oracle rank; 20 draws) with the df heuristic for the same draws.
(2) Same with orthonormal heads (the paper's Sec. 3 model) as the control: tie at r <= k_s.
Writes ../results/hetero_results.json   (~1 min)
"""
import json, os
import numpy as np
from common import orth, theory_losses, sim_losses

d, q, ks = 64, 8, 6
RS = [2, 4, 6]


def teacher(M, kp, heads, rng):
    Q = orth(rng.standard_normal((d, ks + M * kp)))
    S = Q[:, :ks]; Ws = []
    for m in range(M):
        P = Q[:, ks + m * kp: ks + (m + 1) * kp]
        if heads == "gaussian":
            H = rng.standard_normal((q, ks + kp)) / np.sqrt(ks + kp)
        else:  # orthonormal columns need q >= k; use q_eff = max(q, k) rows then it is the Sec. 3 model
            H = orth(rng.standard_normal((max(q, ks + kp), ks + kp)))
        Ws.append(H @ np.concatenate([S, P], 1).T)
    return Ws


def pop(Ws, r):
    eu = np.sort(np.linalg.eigvalsh(sum(W.T @ W for W in Ws)))[::-1]
    Lu = eu[r:].sum() / len(Ws)
    Ls = np.mean([np.sort(np.linalg.eigvalsh(W.T @ W))[::-1][r:].sum() for W in Ws])
    return Lu, Ls


out = {}
for heads in ["gaussian", "orthonormal"]:
    for kp in [6, 0]:
        for M in [2, 4, 8]:
            rng = np.random.default_rng(0)
            P = np.array([[pop(Ws, r) for r in RS] for Ws in (teacher(M, kp, heads, rng) for _ in range(400))])  # [draw, r, 2]
            mu, ms = P[:, :, 0].mean(0), P[:, :, 1].mean(0)  # both are 0 (up to rounding) at r = k_s when k_p = 0
            ent = dict(pop_ratio=[float(u / s_) if s_ > 1e-9 else None for u, s_ in zip(mu, ms)],
                       pop_Lu=P[:, :, 0].mean(0).tolist(), pop_Ls=P[:, :, 1].mean(0).tolist())
            if heads == "gaussian":
                sim, th = [], []
                for _ in range(20):
                    Ws = teacher(M, kp, heads, rng)
                    sim.append([sim_losses(Ws, 0.25, 4096, r, rng) for r in RS]); th.append([theory_losses(Ws, 0.25, 4096, r) for r in RS])
                sim, th = np.array(sim), np.array(th)
                ent["sim_ratio_n4096"] = (sim[:, :, 0].mean(0) / sim[:, :, 1].mean(0)).tolist()
                ent["df_ratio_n4096"] = (th[:, :, 0].mean(0) / th[:, :, 1].mean(0)).tolist()
            out[f"{heads},kp={kp},M={M}"] = ent
            print(heads, "kp", kp, "M", M, "pop", [None if x is None else round(x, 3) for x in ent["pop_ratio"]],
                  "| sim n=4096", np.round(ent.get("sim_ratio_n4096", []), 3).tolist(), "df", np.round(ent.get("df_ratio_n4096", []), 3).tolist(), flush=True)
json.dump(dict(d=d, q=q, ks=ks, rs=RS, results=out), open(os.path.join(os.path.dirname(__file__), "..", "results", "hetero_results.json"), "w"), indent=1)
