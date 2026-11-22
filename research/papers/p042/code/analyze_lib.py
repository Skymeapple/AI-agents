"""Shared fitting utilities (imported by robustness.py)."""
import os, glob, json
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results"); RAW = os.path.join(R, "raw")
THR = float(os.environ.get("THR", 0.9))
TMIN = int(os.environ.get("TMIN", 100_000))
SUF = os.environ.get("SUF", "")
rng = np.random.default_rng(0)


def load(name):
    z = np.load(os.path.join(RAW, name + ".npz"))
    d = {k: z[k] for k in z.files}; d["cfg"] = json.loads(str(d["cfg"])); return d


def sig(z): return 1 / (1 + np.exp(-z))


def nll_beta(th, lp, lT, y):
    """P(rec) = q * sigmoid((lp + b*lT - c)/s);  b=1: occurrence-count threshold; b=0: frequency threshold."""
    b, c, ls, lq = th
    P = sig(lq) * sig((lp + b * lT - c) / np.exp(ls))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean()


def fit_beta(lp, lT, y, fix_b=None):
    best = None
    for b0 in ([fix_b] if fix_b is not None else [0.0, 0.5, 1.0, 1.5]):
        for c0 in np.linspace(-1, 4, 6):
            f = (lambda th: nll_beta([fix_b, *th], lp, lT, y)) if fix_b is not None else (lambda th: nll_beta(th, lp, lT, y))
            x0 = [c0, np.log(0.3), 3.0] if fix_b is not None else [b0, c0, np.log(0.3), 3.0]
            r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
    th = [fix_b, *best.x] if fix_b is not None else list(best.x)
    return dict(beta=float(th[0]), log10_mstar=float(th[1]), width=float(np.exp(th[2])), ceiling=float(sig(th[3])),
                nll=float(best.fun), n=int(len(y)))


def stack(d, tmin=TMIN, tmax=np.inf, xcount=None, thr=None):
    thr = THR if thr is None else thr
    lp, lT, y = [], [], []
    for j, T in enumerate(d["T"]):
        if T < tmin or T > tmax: continue
        lp.append(np.log10(d["p"] if xcount is None else xcount)); lT.append(np.full(len(d["p"]), np.log10(T)))
        y.append((d["maxcos"][j] >= thr).astype(float))
    return map(np.concatenate, (lp, lT, y))


def boot_beta(lp, lT, y, nf, B=40):
    """bootstrap over features (resample feature indices, keep all their checkpoints)."""
    nck = len(y) // nf; bs = []
    for _ in range(B):
        idx = rng.integers(0, nf, nf)
        sel = (np.arange(nck)[:, None] * nf + idx[None, :]).ravel()
        bs.append(fit_beta(lp[sel], lT[sel], y[sel])["beta"])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


