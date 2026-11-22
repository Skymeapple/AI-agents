"""Core library: Gini / concentration-coefficient algebra, global bounds, and a calibrated
synthetic household population for analysing the distributional incidence of AI productivity gains.

Notation (paper Section 3):
  y   : pre-shock household income vector (length N), mean > 0
  z   : gain vector (per unit shock), sum Z > 0; post-shock income y + eps * z
  G(y): Gini;  C(z|y): concentration coefficient of z when households are ranked by y.
Theorem 1:  (Y G + eps Z C) / (Y + eps Z)  <=  G(y + eps z)  <=  (Y G + eps Z G_z) / (Y + eps Z).
"""
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- inequality algebra
def gini(y):
    y = np.sort(np.asarray(y, float))
    n = y.size
    i = np.arange(1, n + 1)
    return 2.0 * np.sum(i * y) / (n * y.sum()) - (n + 1.0) / n


def conc(z, y, order=None):
    """Concentration coefficient of z with households ordered by y (ascending, stable)."""
    if order is None:
        order = np.argsort(y, kind="stable")
    zz = np.asarray(z, float)[order]
    n = zz.size
    i = np.arange(1, n + 1)
    return 2.0 * np.sum(i * zz) / (n * zz.sum()) - (n + 1.0) / n


def top_share(y, p=0.10):
    y = np.sort(np.asarray(y, float))
    k = int(round(p * y.size))
    return y[-k:].sum() / y.sum()


def top_share_of_z(z, y, p=0.10, order=None):
    if order is None:
        order = np.argsort(y, kind="stable")
    k = int(round(p * len(y)))
    zz = np.asarray(z, float)[order]
    return zz[-k:].sum() / zz.sum()


def lorenz(y, grid):
    y = np.sort(np.asarray(y, float))
    c = np.concatenate([[0], np.cumsum(y)]) / y.sum()
    idx = np.round(grid * y.size).astype(int)
    return c[idx]


def conc_curve(z, y, grid, order=None):
    if order is None:
        order = np.argsort(y, kind="stable")
    zz = np.asarray(z, float)[order]
    c = np.concatenate([[0], np.cumsum(zz)]) / zz.sum()
    idx = np.round(grid * zz.size).astype(int)
    return c[idx]


def bounds(y, z, eps):
    """Theorem 1 lower/upper bounds and first-order approximation for G(y + eps z)."""
    Y, Z = y.sum(), z.sum()
    G, C, Gz = gini(y), conc(z, y), gini(z) if np.all(z >= 0) else np.nan
    eps = np.atleast_1d(eps)
    lo = (Y * G + eps * Z * C) / (Y + eps * Z)
    hi = (Y * G + eps * Z * Gz) / (Y + eps * Z)
    fo = G + eps * (Z / Y) * (C - G)
    return lo, hi, fo


def gz_upper(z):
    """Gini-type functional H(z)/sum(z) valid also for signed z: (2/N) max_sigma sum sigma z / Z - (N+1)/N."""
    return gini(z) if np.all(z >= 0) else conc(z, z)


# ---------------------------------------------------------------- population
DFA_2026Q2 = dict(bottom50=0.6, p50_90=11.4, p90_99=37.2, top1=50.9)  # FRED, see data/


def equity_by_wealth_pct(pct, shares, ramp=0.2):
    """Equity holdings for households with wealth percentiles pct in [0,1), matching DFA group shares.
    Within each group holdings rise linearly with rank by +/- ramp/2 around the group mean (a
    deliberately mild within-group gradient: it understates concentration inside the top 1%)."""
    groups = [(0.0, 0.5, shares["bottom50"]), (0.5, 0.9, shares["p50_90"]),
              (0.9, 0.99, shares["p90_99"]), (0.99, 1.0, shares["top1"])]
    tot = sum(g[2] for g in groups)
    e = np.zeros_like(pct)
    for lo, hi, s in groups:
        m = (pct >= lo) & (pct < hi)
        if m.sum() == 0:
            continue
        r = (pct[m] - lo) / (hi - lo)               # within-group rank in [0,1)
        w = 1.0 + ramp * (r - 0.5)
        e[m] = (s / tot) * w / w.sum()
    return e


def make_population(N=100_000, G_lab=0.45, between_share=0.4, K_occ=40, u=0.15, rho=0.5,
                    S_K=0.20, S_T=0.08, shares=None, seed=0):
    """Synthetic households.
    G_lab: Gini of labor earnings among earners (lognormal); between_share: share of log-earnings
    variance between occupations; u: share of households with no labor earnings (retirees etc.);
    rho: Gaussian-copula correlation between latent earnings and wealth rank;
    S_K: capital (equity) income share of household income; S_T: equal per-capita transfer share."""
    rng = np.random.default_rng(seed)
    shares = shares or DFA_2026Q2
    sigma = np.sqrt(2.0) * norm.ppf((G_lab + 1.0) / 2.0)
    occ_eff = rng.normal(0, np.sqrt(between_share) * sigma, K_occ)
    occ = rng.integers(0, K_occ, N)
    within = rng.normal(0, np.sqrt(1 - between_share) * sigma, N)
    logl = occ_eff[occ] + within
    earner = rng.random(N) >= u
    lab = np.where(earner, np.exp(logl), 0.0)
    zl = (logl - logl.mean()) / logl.std()
    latent_w = rho * zl + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
    wpct = (np.argsort(np.argsort(latent_w)) + 0.5) / N
    eq = equity_by_wealth_pct(wpct, shares)
    L = lab.sum()
    # income shares: labor (1-S_K-S_T), capital S_K, transfers S_T
    tot = L / (1 - S_K - S_T)
    cap = eq / eq.sum() * S_K * tot
    trans = np.full(N, S_T * tot / N)
    y = lab + cap + trans
    # within-occupation skill rank q in [0,1] (among earners in the occupation)
    q = np.zeros(N)
    for o in range(K_occ):
        m = (occ == o)
        r = np.argsort(np.argsort(within[m]))
        q[m] = (r + 0.5) / m.sum()
    # occupation earnings rank r_o (by mean log wage)
    occ_rank = np.argsort(np.argsort(occ_eff)) / (K_occ - 1)
    return dict(y=y, lab=lab, cap=cap, eq=eq, trans=trans, occ=occ, q=q,
                occ_rank=occ_rank[occ], earner=earner)


def labor_gain(pop, gamma_b=0.5, kappa=0.43):
    """Worker productivity uplift a_i = alpha_o * w_i.
    alpha_o = exp(gamma_b * (r_o - 0.5)): between-occupation exposure gradient in occupation wage rank.
    w_i = 1 + kappa * (1 - 2 q_i): within-occupation compression (low-skill gain (1+k)/(1-k) x high-skill).
    Gain is proportional to a_i * earnings (zero for non-earners)."""
    alpha = np.exp(gamma_b * (pop["occ_rank"] - 0.5))
    w = 1.0 + kappa * (1.0 - 2.0 * pop["q"])
    a = alpha * w
    z = a * pop["lab"]
    return z / z.sum(), a


def capital_gain(pop):
    return pop["eq"] / pop["eq"].sum()


def consumer_gain(pop, eta=0.8):
    c = pop["y"] ** eta
    return c / c.sum()


def decomposition_BW(pop, a, order=None):
    """Proposition 2: C(a*l) - C(l) = B + W (between/within occupation)."""
    y, l, occ = pop["y"], pop["lab"], pop["occ"]
    if order is None:
        order = np.argsort(y, kind="stable")
    N = y.size
    sig = np.empty(N); sig[order] = np.arange(1, N + 1)
    al = a * l
    abar = al.sum() / l.sum()
    K = occ.max() + 1
    alpha_o = np.array([al[occ == o].sum() / max(l[occ == o].sum(), 1e-300) for o in range(K)])
    denom = N * al.sum() / 2.0
    B = np.sum(sig * l * (alpha_o[occ] - abar)) / denom
    W = np.sum(sig * l * (a - alpha_o[occ])) / denom
    return B, W
