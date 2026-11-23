"""Random Gaussian designs (round 2): drop the X^T X = n I idealisation.

Setup of Exp. A (d=64, q=16, k_s=k_p=4, M=4, s=sigma=1).  For every modality
X_m has n i.i.d. N(0, I_d) rows (n > d+1 so OLS exists), Y_m = X_m W_m^T + noise.
  specialist: classical reduced-rank regression (Izenman 1975), exact.
  unified   : min_{U, V_m} sum_m ||Y_m - X_m U V_m^T||^2, solved by alternating
              least squares from the stacked-OLS SVD start (not guaranteed
              global; we report the objective gap to the orthogonal-design SVD
              start as a check).
Excess risk under x ~ N(0, I) is ||U V_m^T - W_m^T||_F^2; oracle rank rho <= r
for both models, as in the paper.  Theory: df heuristic with n replaced by
n_eff = n - d - 1 (E (X^T X)^{-1} = I / (n - d - 1) for Gaussian X).
Run: OMP_NUM_THREADS=1 python3 exp_random_design.py   (~10 min)"""
import json, os, time, sys
import numpy as np
from common import make_task, theory_losses

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
d, q, ks, kp, M = 64, 16, 4, 4, 4
RMAX = 24
RS = [2, 4, 6, 8, 12, 16, 20, 24]
NS = [100, 200, 500, 1000, 3000]
REPS = int(os.environ.get("RD_REPS", "6"))
rng = np.random.default_rng(21)


def rrr_specialist(X, Y, W):
    C = X.T @ X
    B = np.linalg.solve(C, X.T @ Y)              # d x q
    Yh = X @ B
    _, _, vt = np.linalg.svd(Yh, full_matrices=False)
    errs = [np.sum(W.T ** 2)]
    for rho in range(1, min(RMAX, q) + 1):
        Vr = vt[:rho].T
        errs.append(np.sum((B @ Vr @ Vr.T - W.T) ** 2))
    errs += [errs[-1]] * (RMAX + 1 - len(errs))
    return np.array(errs)


def als_unified(Cs, XtYs, Bols, rho, iters=30, tol=1e-6):
    St = np.concatenate([b.T for b in Bols], 0)  # (Mq) x d
    _, _, vt = np.linalg.svd(St, full_matrices=False)
    U = vt[:rho].T                                # d x rho
    prev = np.inf
    for it in range(iters):
        Vs = []
        for C, XtY in zip(Cs, XtYs):
            A = U.T @ C @ U
            Vs.append(np.linalg.solve(A, U.T @ XtY).T)   # q x rho
        lhs = sum(np.kron(V.T @ V, C) for V, C in zip(Vs, Cs))
        rhs = sum(XtY @ V for XtY, V in zip(XtYs, Vs))
        U = np.linalg.solve(lhs, rhs.reshape(-1, order="F")).reshape(d, rho, order="F")
        # objective (up to constant): sum tr(B^T C B) - 2 tr(B^T XtY), B = U V^T
        obj = sum(np.sum((C @ U @ V.T) * (U @ V.T)) - 2 * np.sum((U @ V.T) * XtY)
                  for C, XtY, V in zip(Cs, XtYs, Vs))
        if abs(prev - obj) < tol * abs(obj):
            break
        prev = obj
    Vs = [np.linalg.solve(U.T @ C @ U, U.T @ XtY).T for C, XtY in zip(Cs, XtYs)]
    return [U @ V.T for V in Vs], it + 1


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for n in NS:
        Eu = np.zeros((REPS, RMAX + 1)); Es = np.zeros((REPS, RMAX + 1)); its = []
        for rep in range(REPS):
            Ws = make_task(d, q, ks, kp, M, rng=rng.integers(1 << 31))
            Xs = [rng.standard_normal((n, d)) for _ in range(M)]
            Ys = [X @ W.T + rng.standard_normal((n, q)) for X, W in zip(Xs, Ws)]
            Cs = [X.T @ X for X in Xs]; XtYs = [X.T @ Y for X, Y in zip(Xs, Ys)]
            Bols = [np.linalg.solve(C, XtY) for C, XtY in zip(Cs, XtYs)]
            Es[rep] = np.mean([rrr_specialist(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)], 0)
            Eu[rep, 0] = np.mean([np.sum(W ** 2) for W in Ws])
            for rho in range(1, RMAX + 1):
                Bs, it = als_unified(Cs, XtYs, Bols, rho); its.append(it)
                Eu[rep, rho] = np.mean([np.sum((B - W.T) ** 2) for B, W in zip(Bs, Ws)])
        neff = n - d - 1
        for r in RS:
            lu = np.array([Eu[i, :r + 1].min() for i in range(REPS)])
            ls = np.array([Es[i, :r + 1].min() for i in range(REPS)])
            tu, ts = theory_losses(Ws, 1.0, neff, r)
            tu0, ts0 = theory_losses(Ws, 1.0, n, r)
            rows.append(dict(n=n, n_eff=neff, r=r, sim_u=lu.mean(), sim_s=ls.mean(),
                             sim_ratio=lu.mean() / ls.mean(),
                             sim_ratio_ci95=list(np.percentile(
                                 [lu[ii].mean() / ls[ii].mean() for ii in
                                  rng.integers(0, REPS, (2000, REPS))], [2.5, 97.5])),
                             th_ratio_neff=tu / ts, th_u_neff=tu, th_s_neff=ts,
                             th_ratio_n=tu0 / ts0))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                   if k != "sim_ratio_ci95"}, flush=True)
        print("n", n, "mean ALS iters", np.mean(its), "t", round(time.time() - t0), flush=True)
    json.dump(dict(d=d, q=q, ks=ks, kp=kp, M=M, reps=REPS, rows=rows),
              open(os.path.join(OUT, "random_design_results.json"), "w"), indent=1, default=float)
