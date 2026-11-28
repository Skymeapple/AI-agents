"""Round 3 (review_r2 W5): three width conventions in the linear shared-core model.

Exp. A configuration (d=64, q=16, k_s=k_p=4, M=4, s=sigma=1), orthogonal-design / SNR model.
For each capacity r and each n, both models are fitted by truncated SVD and the rank is set by
  oracle     : rho <= r minimising the TRUE risk (the convention of Sec. 4, optimistic);
  validation : rho <= r minimising the loss on an independent validation estimate
               W_val = W + sigma/sqrt(n_val) Z', n_val = n/4 (the unified model picks one rho on
               the summed validation loss, each specialist picks its own);
  fixed      : rho = min(r, q) for a specialist, rho = r for the unified model (no selection).
Reports the unified/specialist loss ratio (ratio of mean losses over repetitions) with a
bootstrap 95% CI over repetitions, and the df-heuristic prediction for the oracle and fixed
conventions.  Writes ../results/valrank_results.json.   ~1 min.
"""
import json, os
import numpy as np
from common import make_task, theory_losses

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "valrank_results.json")
d, q, ks, kp, M = 64, 16, 4, 4, 4
NS, RS, REPS, VAL_FRAC = [200, 1000, 5000], list(range(2, 25)), 40, 0.25


def one(Ws, n, rng):
    """Return arrays [conv, r] of unified and specialist losses (per modality)."""
    sd, sdv = np.sqrt(1.0 / n), np.sqrt(1.0 / (VAL_FRAC * n))
    What = [W + sd * rng.standard_normal(W.shape) for W in Ws]
    Wval = [W + sdv * rng.standard_normal(W.shape) for W in Ws]
    rmax = max(RS)
    # unified: errors for every rank 0..rmax
    _, vec = np.linalg.eigh(sum(Wh.T @ Wh for Wh in What)); vec = vec[:, ::-1]
    tu, vu = np.zeros(rmax + 1), np.zeros(rmax + 1)
    for rho in range(rmax + 1):
        P = vec[:, :rho] @ vec[:, :rho].T
        for Wh, W, Wv in zip(What, Ws, Wval):
            tu[rho] += np.sum((Wh @ P - W) ** 2); vu[rho] += np.sum((Wh @ P - Wv) ** 2)
    ts, vs = np.zeros((M, q + 1)), np.zeros((M, q + 1))
    for m, (Wh, W, Wv) in enumerate(zip(What, Ws, Wval)):
        u_, sv, vt = np.linalg.svd(Wh, full_matrices=False)
        for rho in range(q + 1):
            A = (u_[:, :rho] * sv[:rho]) @ vt[:rho]
            ts[m, rho] = np.sum((A - W) ** 2); vs[m, rho] = np.sum((A - Wv) ** 2)
    Lu, Ls = np.zeros((3, len(RS))), np.zeros((3, len(RS)))
    for j, r in enumerate(RS):
        rs_ = min(r, q)
        Lu[0, j] = tu[: r + 1].min(); Ls[0, j] = sum(ts[m, : rs_ + 1].min() for m in range(M))
        Lu[1, j] = tu[np.argmin(vu[: r + 1])]; Ls[1, j] = sum(ts[m, np.argmin(vs[m, : rs_ + 1])] for m in range(M))
        Lu[2, j] = tu[r]; Ls[2, j] = sum(ts[m, rs_] for m in range(M))
    return Lu / M, Ls / M


def df_fixed(Ws, n, r):
    eu = np.sort(np.linalg.eigvalsh(sum(W.T @ W for W in Ws)))[::-1]
    es = np.sort(np.linalg.eigvalsh(Ws[0].T @ Ws[0]))[::-1]
    rs_ = min(r, q)
    return ((eu[r:].sum() + r * (d + M * q - r) / n) / M) / (es[rs_:].sum() + rs_ * (d + q - rs_) / n)


res = dict(config=dict(d=d, q=q, ks=ks, kp=kp, M=M, reps=REPS, val_frac=VAL_FRAC), rs=RS, runs={})
rng = np.random.default_rng(0)
for n in NS:
    LU, LS = [], []
    for rep in range(REPS):
        Ws = make_task(d, q, ks, kp, M, rng=rng.integers(1 << 31))
        lu, ls = one(Ws, n, rng); LU.append(lu); LS.append(ls)
    LU, LS = np.array(LU), np.array(LS)  # [rep, conv, r]
    ratio = LU.mean(0) / LS.mean(0)
    brng = np.random.default_rng(1)
    bs = np.array([(lambda i: LU[i].mean(0) / LS[i].mean(0))(brng.integers(0, REPS, REPS)) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    ent = {}
    for c, name in enumerate(["oracle", "validation", "fixed"]):
        win = [r for r, x in zip(RS, ratio[c]) if x > 1]
        ent[name] = dict(ratio=ratio[c].tolist(), ci_lo=lo[c].tolist(), ci_hi=hi[c].tolist(),
                         Lu=LU.mean(0)[c].tolist(), Ls=LS.mean(0)[c].tolist(),
                         competition_r=win, peak_r=int(RS[int(np.argmax(ratio[c]))]))
    ent["df_oracle"] = [float(np.divide(*theory_losses(Ws, 1.0, n, r))) for r in RS]
    ent["df_fixed"] = [float(df_fixed(Ws, n, r)) for r in RS]
    res["runs"][str(n)] = ent
    for name in ["oracle", "validation", "fixed"]:
        e = ent[name]
        print(n, name, "comp r:", e["competition_r"][:1], e["competition_r"][-1:], "peak", e["peak_r"],
              "| r=4 %.3f r=8 %.3f r=16 %.3f r=20 %.3f r=24 %.3f [%.3f,%.3f]" % (
                  e["ratio"][2], e["ratio"][6], e["ratio"][14], e["ratio"][18], e["ratio"][22], e["ci_lo"][22], e["ci_hi"][22]))
    print(n, "df oracle r=20,24: %.3f %.3f | df fixed r=20,24: %.3f %.3f" % (ent["df_oracle"][18], ent["df_oracle"][22], ent["df_fixed"][18], ent["df_fixed"][22]))
json.dump(res, open(OUT, "w"), indent=1)
