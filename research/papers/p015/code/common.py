"""Shared utilities: data loading, survival-family likelihoods, frailty GLMM.

All analyses use METR's public Time Horizon 1.1 run data (see fetch_data.sh).
"""
import os
import numpy as np
import pandas as pd
import yaml
from scipy import optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

T_REF = 60.0  # reference length (minutes) for centring log T

# Release dates for models missing from METR's yaml at the pinned commit.
# Each is a public launch date; see paper Sec. 6.5 for sources.
EXTRA_DATES = {}


def load_runs(version="th11", drop_usage_limit=False):
    d = pd.read_json(os.path.join(DATA, f"runs_{version}.jsonl"), lines=True)
    d = d[d.alias != "human"].copy()
    if drop_usage_limit:
        d = d[d.fatal_error_from.isna()].copy()
    d["y"] = d.score_binarized.astype(int)
    d["T"] = d.human_minutes.astype(float)
    return d


def release_dates():
    with open(os.path.join(DATA, "release_dates.yaml")) as f:
        y = yaml.safe_load(f)["date"]
    y = {k: pd.Timestamp(v) for k, v in y.items()}
    y.update({k: pd.Timestamp(v) for k, v in EXTRA_DATES.items()})
    return y


def task_table(d):
    """Aggregate runs to (task, n, s, T, family) for one agent."""
    g = d.groupby("task_id").agg(n=("y", "size"), s=("y", "sum"), T=("T", "first"),
                                 family=("task_family", "first"),
                                 source=("task_source", "first"),
                                 hsrc=("human_source", "first")).reset_index()
    return g


# ---------------------------------------------------------------------------
# Marginal survival families  S(T) = P(success | T)
# params are unconstrained vectors; each function returns S for array T.
# ---------------------------------------------------------------------------
def S_loglogistic(th, T):  # METR: logit P = a - b log(T/Tref)
    a, logb = th
    return special.expit(a - np.exp(logb) * np.log(T / T_REF))


def S_weibull(th, T):  # cloglog: log(-log P) = a + k log(T/Tref)
    a, logk = th
    return np.exp(-np.exp(np.clip(a + np.exp(logk) * np.log(T / T_REF), -50, 30)))


def S_exponential(th, T):  # constant hazard (Ord 2025)
    (loglam,) = th
    return np.exp(-np.exp(loglam) * T / T_REF)


def S_lomax(th, T):  # gamma frailty over constant hazards: (1 + T/(a tau))^-a
    logtau, loga = th
    a = np.exp(loga)
    return np.exp(-a * np.log1p(T / (a * np.exp(logtau) * T_REF)))


def S_lognormal(th, T):  # probit in log T
    a, logb = th
    return stats.norm.cdf(a - np.exp(logb) * np.log(T / T_REF))


FAMILIES = {
    "log-logistic (METR)": (S_loglogistic, np.array([1.0, np.log(0.7)])),
    "Weibull": (S_weibull, np.array([-0.5, np.log(0.6)])),
    "exponential": (S_exponential, np.array([0.0])),
    "Lomax (gamma frailty)": (S_lomax, np.array([0.0, 0.0])),
    "log-normal": (S_lognormal, np.array([0.5, np.log(0.5)])),
}


def wnll(S, y, w, eps=1e-9):
    S = np.clip(S, eps, 1 - eps)
    return -np.sum(w * (y * np.log(S) + (1 - y) * np.log1p(-S)))


def fit_family(name, T, y, w):
    f, th0 = FAMILIES[name]
    best = None
    for jit in [0.0, 1.0, -1.0, 2.0]:
        r = optimize.minimize(lambda th: wnll(f(th, T), y, w), th0 + jit * 0.5,
                              method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


# Nonparametric MLE over mixtures of exponentials (completely monotone class)
LAM_GRID = np.concatenate([[0.0], np.logspace(-4, 3, 80)])  # hazard per T_REF minutes


def fit_cm_mixture(T, y, w, iters=3000):
    K = np.exp(-np.outer(T / T_REF, LAM_GRID))  # n x m, kernel S_j(T)
    pi = np.ones(len(LAM_GRID)) / len(LAM_GRID)
    for _ in range(iters):
        S = K @ pi
        S = np.clip(S, 1e-12, 1 - 1e-12)
        # EM for binary outcomes with mixture of Bernoulli(K_ij): responsibilities
        r1 = (K * pi) / S[:, None]
        r0 = ((1 - K) * pi) / (1 - S)[:, None]
        R = w[:, None] * (y[:, None] * r1 + (1 - y)[:, None] * r0)
        pi_new = R.sum(0) / w.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-10:
            pi = pi_new
            break
        pi = pi_new
    return pi


def S_cm(pi, T):
    return np.exp(-np.outer(T / T_REF, LAM_GRID)) @ pi


def horizon(S, p, lo=1e-3, hi=1e7):
    """Task length (min) where survival function S(T) crosses p (S decreasing)."""
    g = lambda lt: S(np.array([np.exp(lt)]))[0] - p
    a, b = np.log(lo), np.log(hi)
    if g(a) < 0:
        return lo
    if g(b) > 0:
        return np.inf
    return float(np.exp(optimize.brentq(g, a, b)))


# ---------------------------------------------------------------------------
# Random-effects (frailty) GLMM fitted to replicate runs.
#   link "cloglog":  P(success) = exp(-exp(eta)),  eta = mu + k log(T/Tref) + sf*v_fam + st*u_task
#     => task-conditional cumulative hazard  Lambda(T) = exp(eta) proportional to T^k  (Weibull shape k)
#   link "logit":    logit P = mu - b log(T/Tref) - sf*v - st*u   (conditional METR slope b)
# Nested Gauss-Hermite quadrature: family effect outer, task effect inner.
# ---------------------------------------------------------------------------
GH_T = np.polynomial.hermite_e.hermegauss(25)
GH_F = np.polynomial.hermite_e.hermegauss(15)


def _logp(eta, link):
    if link == "cloglog":
        e = np.exp(np.clip(eta, -30, 6))
        return -e, np.log(-np.expm1(-e) + 1e-300)
    else:
        return -np.logaddexp(0, -eta), -np.logaddexp(0, eta)


def glmm_nll(theta, tt, link, nested=True):
    mu, slope, log_st = theta[:3]
    st = np.exp(log_st)
    sf = np.exp(theta[3]) if nested else 0.0
    xt, wt = GH_T
    wt = wt / wt.sum()
    xf, wf = (GH_F if nested else (np.array([0.0]), np.array([1.0])))
    wf = wf / wf.sum()
    lt = np.log(tt["T"].values / T_REF)
    n = tt["n"].values.astype(float)
    s = tt["s"].values.astype(float)
    sign = 1.0 if link == "cloglog" else -1.0
    # eta[i, f, t]
    eta = (mu + sign * slope * lt)[:, None, None] + sign * (sf * xf[None, :, None] + st * xt[None, None, :])
    if link == "logit":
        eta = mu - slope * lt[:, None, None] - (sf * xf[None, :, None] + st * xt[None, None, :])
    lp, lq = _logp(eta, link)
    ll = s[:, None, None] * lp + (n - s)[:, None, None] * lq  # binomial kernel (constant dropped)
    # integrate task effect
    li = special.logsumexp(ll + np.log(wt)[None, None, :], axis=2)  # [i, f]
    fam = tt["family"].values
    codes, inv = np.unique(fam, return_inverse=True)
    lf = np.zeros((len(codes), li.shape[1]))
    np.add.at(lf, inv, li)
    tot = special.logsumexp(lf + np.log(wf)[None, :], axis=1).sum()
    return -tot


def fit_glmm(tt, link="cloglog", nested=True, sigma0=1.0):
    best = None
    k0 = 0.7 if link == "cloglog" else 1.0
    starts = [[-0.5, k0, np.log(sigma0), np.log(0.5)], [0.5, k0 * 1.3, np.log(2.0), np.log(1.0)],
              [-1.5, k0 * 0.7, np.log(0.5), np.log(0.2)]]
    for st in starts:
        x0 = np.array(st if nested else st[:3])
        r = optimize.minimize(glmm_nll, x0, args=(tt, link, nested), method="L-BFGS-B")
        r2 = optimize.minimize(glmm_nll, r.x, args=(tt, link, nested), method="Nelder-Mead",
                               options=dict(maxiter=3000, xatol=1e-6, fatol=1e-7))
        if best is None or r2.fun < best.fun:
            best = r2
    return best


def glmm_fixed_only_nll(theta, tt, link):
    return glmm_nll(np.array([theta[0], theta[1], np.log(1e-6), np.log(1e-6)]), tt, link, nested=True)


def task_success_draws(theta, T, link="cloglog", nested=True, ndraw=200000, rng=None):
    """Draw per-task success probabilities p_i for new tasks of length T under fitted GLMM."""
    rng = rng or np.random.default_rng(0)
    mu, slope, log_st = theta[:3]
    st = np.exp(log_st)
    sf = np.exp(theta[3]) if nested else 0.0
    sd = np.sqrt(st ** 2 + sf ** 2)
    z = rng.standard_normal(ndraw) * sd
    lt = np.log(T / T_REF)
    if link == "cloglog":
        return np.exp(-np.exp(np.clip(mu + slope * lt + z, -30, 6)))
    return special.expit(mu - slope * lt - z)
