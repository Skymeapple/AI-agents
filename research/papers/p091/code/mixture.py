"""Audit-error / memorisation mixture model for responses on items whose original key was flagged wrong.

Y[m,i] = 1 if model m chose the ORIGINAL (flagged) key on item i.  theta[m] = standardised ability.
Item class B ('audit error', prior pi):  P(Y=1) = sig(a_i + softplus(b_i) * theta)        (key is actually right: tracks ability)
Item class A ('audit correct'):          P(Y=1) = lam(theta) + (1-lam(theta)) * sig(c_i - softplus(d_i) * theta)
lam(theta) = sig(g0 + g1*theta) is a memorisation curve common to all class-A items (H0: lam == 0).
Fitted by EM; M-step uses L-BFGS per item and for (g0,g1)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sig
sp = lambda x: np.logaddexp(0, x)
EPS = 1e-9
def _pB(p, th): return sig(p[0] + sp(p[1]) * th)
def _pA(p, th, lam): return lam + (1 - lam) * sig(p[0] - sp(p[1]) * th)
def _ll(P, y, w): return (w * (y * np.log(P + EPS) + (1 - y) * np.log(1 - P + EPS))).sum()
def fit(Y, V, th, memorisation=True, n_iter=60, seed=0, init=None, tol=1e-6, boundsB=((-12, 12), (-12, 6)), lam_fixed=None):
    M, I = Y.shape
    rng = np.random.default_rng(seed)
    pB = np.tile([0.0, 0.0], (I, 1)); pA = np.tile([-1.0, 0.0], (I, 1))
    g = np.array([-6.0, 0.0]); pi = 0.2
    # init responsibilities: positive empirical slope -> class B
    r = np.zeros(I)
    for i in range(I):
        v = V[:, i]
        with np.errstate(all='ignore'): cc = np.corrcoef(th[v], Y[v, i])[0, 1]
        r[i] = 0.8 if (np.isfinite(cc) and cc > 0.2) else 0.1
    pB[:, 0] = np.clip(pB[:, 0], boundsB[0][0], boundsB[0][1]); pB[:, 1] = np.clip(pB[:, 1], boundsB[1][0], boundsB[1][1])
    if init is not None:
        r, pA, pB = init['r'].copy(), init['pA'].copy(), init['pB'].copy(); g = np.array([-12.0, 0.0])
    prev = -np.inf
    for it in range(n_iter):
        lam = (lam_fixed if lam_fixed is not None else sig(g[0] + g[1] * th)) if memorisation else np.zeros(M)
        # M-step items
        for i in range(I):
            v = V[:, i]; y = Y[v, i]; t = th[v]
            f = lambda p: -_ll(_pB(p, t), y, np.ones_like(y))
            pB[i] = minimize(f, pB[i], method='L-BFGS-B', bounds=list(boundsB)).x
            f = lambda p: -_ll(_pA(p, t, lam[v]), y, np.ones_like(y))
            pA[i] = minimize(f, pA[i], method='L-BFGS-B', bounds=[(-12, 12), (-12, 6)]).x
        if memorisation and lam_fixed is None:
            def fg(gg):
                l = sig(gg[0] + gg[1] * th)
                tot = 0.0
                for i in range(I):
                    v = V[:, i]
                    tot += (1 - r[i]) * _ll(_pA(pA[i], th[v], l[v]), Y[v, i], np.ones(v.sum()))
                return -tot
            g = minimize(fg, g, method='L-BFGS-B', bounds=[(-12, 8), (-6, 6)]).x
            lam = sig(g[0] + g[1] * th)
        pi = np.clip(r.mean(), 1e-3, 1 - 1e-3)
        # E-step
        llB = np.array([_ll(_pB(pB[i], th[V[:, i]]), Y[V[:, i], i], np.ones(V[:, i].sum())) for i in range(I)])
        llA = np.array([_ll(_pA(pA[i], th[V[:, i]], lam[V[:, i]]), Y[V[:, i], i], np.ones(V[:, i].sum())) for i in range(I)])
        a = np.log(pi) + llB; b = np.log(1 - pi) + llA
        mx = np.maximum(a, b); tot = (mx + np.log(np.exp(a - mx) + np.exp(b - mx))).sum()
        r = np.exp(a - mx) / (np.exp(a - mx) + np.exp(b - mx))
        if abs(tot - prev) < tol * max(1, abs(tot)): break
        prev = tot
    lam_out = (lam_fixed if lam_fixed is not None else sig(g[0] + g[1] * th)) if memorisation else np.zeros(M)
    return dict(loglik=tot, r=r, pi=pi, g=g if memorisation else None, pA=pA, pB=pB, lam=lam_out, iters=it + 1)
def simulate(th, I, pi, g, rng, aB=(0.5, 1.0), bB=(1.0, 0.5), aA=(-1.5, 1.0), bA=(0.8, 0.5)):
    """Generate synthetic Y with known item classes and memorisation curve lam = sig(g0+g1*theta)."""
    M = len(th); cls = rng.random(I) < pi
    lam = sig(g[0] + g[1] * th)
    Y = np.zeros((M, I))
    for i in range(I):
        if cls[i]:
            p = sig(rng.normal(*aB) + abs(rng.normal(*bB)) * th)
        else:
            p = lam + (1 - lam) * sig(rng.normal(*aA) - abs(rng.normal(*bA)) * th)
        Y[:, i] = rng.random(M) < p
    return Y, cls
