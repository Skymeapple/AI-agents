"""Shared utilities: data loading, Chinchilla-style parametric fit (Huber on log-loss, LSE form)."""
import csv, os, itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "chinchilla.csv")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# Revision 1: objectives are multiplied by SCALE so that BFGS gradient tolerances are meaningful at small
# noise (v0 used unscaled objectives of order 1e-6..1e-3, and BFGS could stop near its starting point).
SCALE = 1e6
OPT = dict(gtol=1e-7, maxiter=5000)


def load_chinchilla():
    rows = list(csv.DictReader(open(DATA)))
    N = np.array([float(r["Model Size"]) for r in rows])
    C = np.array([float(r["Training FLOP"]) for r in rows])
    L = np.array([float(r["loss"]) for r in rows])
    D = C / (6.0 * N)
    return N, D, C, L


def huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


# ---------------- base model: L = E + A N^-alpha + B D^-beta ----------------
def pred_log_base(p, lN, lD):
    a, b, e, al, be = p
    return logsumexp(np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)]), axis=0)


def obj_base(p, lN, lD, lL, delta=1e-3):
    """Summed Huber loss and analytic gradient."""
    a, b, e, al, be = p
    T = np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)])
    lp = logsumexp(T, axis=0)
    w = np.exp(T - lp)
    r = lp - lL
    hp = np.clip(r, -delta, delta)
    g = np.array([(hp * w[0]).sum(), (hp * w[1]).sum(), (hp * w[2]).sum(),
                  -(hp * w[0] * lN).sum(), -(hp * w[1] * lD).sum()])
    return SCALE * huber(r, delta).sum(), SCALE * g


INIT_GRID = list(itertools.product([0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 20, 25],
                                   [-1, -0.5, 0, 0.5, 1], [0, 0.5, 1, 1.5, 2], [0, 0.5, 1, 1.5, 2]))


def fit_base(N, D, L, grid=INIT_GRID, delta=1e-3, init=None):
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    inits = [init] if init is not None else grid
    best = None
    for x0 in inits:
        r = minimize(obj_base, np.array(x0, float), args=(lN, lD, lL, delta), method="BFGS", jac=True, options=OPT)
        if best is None or r.fun < best.fun:
            best = r
    a, b, e, al, be = best.x
    return dict(A=np.exp(a), B=np.exp(b), E=np.exp(e), alpha=al, beta=be, raw=best.x, fun=best.fun)


def coarse_grid():
    return list(itertools.product([0, 6, 12], [0, 6, 12], [-0.5, 0.5], [0.2, 0.6], [0.2, 0.6]))


# ---------------- compute-optimal frontier ----------------
def frontier(p, C):
    """Closed-form compute-optimal loss for L=E+A N^-a+B D^-b with C=6ND.
    Returns L*(C), N*(C), D*(C)."""
    A, B, E, a, b = p["A"], p["B"], p["E"], p["alpha"], p["beta"]
    G = (a * A / (b * B)) ** (1.0 / (a + b))
    Nopt = G * (C / 6.0) ** (b / (a + b))
    Dopt = (C / 6.0) / Nopt
    return E + A * Nopt ** (-a) + B * Dopt ** (-b), Nopt, Dopt


def gamma(p):
    return p["alpha"] * p["beta"] / (p["alpha"] + p["beta"])


def epoch_keep(L, n_excl=5):
    """Epoch AI's exclusion rule: drop the n_excl highest-loss runs."""
    thr = np.sort(L)[-n_excl]
    return np.where(L < thr)[0]
