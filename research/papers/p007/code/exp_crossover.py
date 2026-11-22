"""Data-limited crossover: for fixed per-modality data n, find the capacity r_x
above which the unified model beats specialists (ratio < 1), for a power-law
spectrum; compare with the prediction r_x ~ unified oracle rank rho_u*(n),
which scales as n^{1/alpha}.  Run: OMP_NUM_THREADS=1 python3 exp_crossover.py"""
import json, os, time
import numpy as np
from common import orth

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(1)
d, q, M, phi, scale = 512, 96, 4, 0.5, 25.0
Ks = Kp = 48
res = {}
t0 = time.time()
for alpha in [1.5, 2.0]:
    shared = (np.arange(1, Ks + 1) / phi) ** (-alpha)
    priv = (np.arange(1, Kp + 1) / (1 - phi)) ** (-alpha)
    basis = orth(rng.standard_normal((d, Ks + M * Kp)))
    S = basis[:, :Ks]; Ws = []
    for m in range(M):
        P = basis[:, Ks + m * Kp: Ks + (m + 1) * Kp]
        AB = orth(rng.standard_normal((q, Ks + Kp)))
        sv = np.sqrt(scale * np.concatenate([shared, priv]))
        Ws.append(AB @ (sv[:, None] * np.concatenate([S, P], 1).T))
    G = sum(W.T @ W for W in Ws)
    eu = np.sort(np.clip(np.linalg.eigvalsh(G), 0, None))[::-1]
    rows = []
    for n in np.unique(np.logspace(np.log10(1000), np.log10(256000), 9).astype(int)):
        # predicted: unified oracle rank in df model
        tot = eu.sum(); cum = np.cumsum(eu)
        risk = [tot] + [tot - cum[k - 1] + k * (d + M * q - k) / n for k in range(1, 241)]
        rho_u = int(np.argmin(risk))
        # simulated ratio over r grid (3 reps), oracle rank per model
        rgrid = list(range(2, 241, 2))
        ratios = np.zeros(len(rgrid))
        for rep in range(3):
            Wh = [W + rng.standard_normal(W.shape) / np.sqrt(n) for W in Ws]
            _, vec = np.linalg.eigh(sum(w.T @ w for w in Wh)); vec = vec[:, ::-1]
            # errors for every rank k (incremental projections)
            eu_k = np.array([sum(np.sum((w @ vec[:, :k] @ vec[:, :k].T - W) ** 2) for w, W in zip(Wh, Ws)) / M
                             for k in range(0, 241)])
            es_k = np.zeros(97)
            for w, W in zip(Wh, Ws):
                u_, s_, vt = np.linalg.svd(w, full_matrices=False)
                es_k += np.array([np.sum(((u_[:, :k] * s_[:k]) @ vt[:k] - W) ** 2) for k in range(0, 97)]) / M
            for j, r in enumerate(rgrid):
                ratios[j] += eu_k[: r + 1].min() / es_k[: min(r, 96) + 1].min() / 3
        # crossover: smallest r beyond the competition peak with ratio < 1
        peak = int(np.argmax(ratios))
        below = [rgrid[j] for j in range(peak, len(rgrid)) if ratios[j] < 1]
        rx = below[0] if below else None
        rows.append(dict(n=int(n), rho_u_pred=rho_u, r_cross_sim=rx, max_ratio=float(ratios.max()),
                         plateau_ratio=float(ratios[-1])))
        print(alpha, rows[-1], round(time.time() - t0), flush=True)
    ns = np.array([r["n"] for r in rows if r["r_cross_sim"]]); rx = np.array([r["r_cross_sim"] for r in rows if r["r_cross_sim"]])
    pr = np.array([r["rho_u_pred"] for r in rows if r["r_cross_sim"]])
    slope_sim = np.polyfit(np.log(ns), np.log(rx), 1)[0]
    slope_pred = np.polyfit(np.log(ns), np.log(pr), 1)[0]
    res[str(alpha)] = dict(rows=rows, slope_sim=slope_sim, slope_pred=slope_pred,
                           slope_asymptotic=1 / alpha)
    print("alpha", alpha, "slope sim", slope_sim, "pred", slope_pred)
json.dump(res, open(os.path.join(OUT, "crossover_results.json"), "w"), indent=1, default=float)
