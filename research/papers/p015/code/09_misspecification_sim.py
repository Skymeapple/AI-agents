"""Misspecification simulation (addresses review r1): does the Gaussian frailty GLMM report k > 1
when the truth is a declining task-level hazard (k = 0.63) with non-Gaussian frailty, impossible
tasks, frailty correlated with T, or noisy length labels?  Also: exact pooled-curve matching (Prop. 1)
and a check of whether corr(log h50, log sigma_hat) arises mechanically.

Design: task lengths and run counts of Claude 4 Opus (Inspect) on TH1.1 (228 tasks, 1,368 runs).
Estimators per replicate: pooled Weibull k_marg; stratified H; Gaussian task-frailty GLMM (base);
3-class latent frailty; Gaussian frailty + point mass of impossible tasks (zero-inflated).
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import optimize, special
from common import *

d = load_runs()
design = task_table(d[d.alias == "Claude 4 Opus (Inspect)"])
T0 = design["T"].values
N0 = design["n"].values
LT0 = np.log(T0 / T_REF)
EDGES = [0, 1, 4, 15, 60, 240, 960, 1e9]
GH = np.polynomial.hermite_e.hermegauss(30)


def H_strat(tt):
    Hs, ws = [], []
    tt = tt.copy()
    tt["bin"] = pd.cut(tt["T"], EDGES, right=False)
    for _, g in tt.groupby("bin", observed=True):
        if len(g) < 8:
            continue
        p = g.s / g.n
        m = len(g)
        Pb = p.mean()
        if not (0.03 < Pb < 0.97):
            continue
        Pb2 = (p.sum() ** 2 - (p ** 2).sum()) / (m * (m - 1))
        M2 = (g.s * (g.s - 1) / (g.n * (g.n - 1))).mean()
        Hs.append((M2 - Pb2) / (Pb * (1 - Pb)))
        ws.append(m)
    return np.average(Hs, weights=ws) if Hs else np.nan


def nll_latent(th, lt, s, n, K=3):
    mus, k = th[:K], th[K]
    logits = np.concatenate([[0.0], th[K + 1:]])
    lpi = logits - special.logsumexp(logits)
    eta = mus[None, :] + k * lt[:, None]
    lp = -np.exp(np.clip(eta, -30, 6))
    lq = np.log(-np.expm1(lp) + 1e-300)
    ll = s[:, None] * lp + (n - s)[:, None] * lq
    return -special.logsumexp(ll + lpi[None, :], axis=1).sum()


def nll_zi(th, lt, s, n):
    mu, k, lsd, lg = th
    x, w = GH
    w = w / w.sum()
    eta = mu + k * lt[:, None] + np.exp(lsd) * x[None, :]
    lp = -np.exp(np.clip(eta, -30, 6))
    lq = np.log(-np.expm1(lp) + 1e-300)
    li = special.logsumexp(s[:, None] * lp + (n - s)[:, None] * lq + np.log(w)[None, :], axis=1)
    pi0 = special.expit(lg)
    zero = np.where(s == 0, 0.0, -np.inf)
    return -np.logaddexp(np.log1p(-pi0) + li, np.log(pi0) + zero).sum()


def nm(f, starts, args):
    best = None
    for s0 in starts:
        r = optimize.minimize(f, np.array(s0, float), args=args, method="Nelder-Mead",
                              options=dict(maxiter=4000, xatol=1e-5, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return best


# ---------------------------------------------------------------- scenario generators
def make(kind, k, sig=0.0, imp=None, drift=0.0, noise=0.0, mu_shift=0.0, extra=None):
    return dict(kind=kind, k=k, sig=sig, imp=imp, drift=drift, noise=noise, extra=extra, mu_shift=mu_shift)


def draw_z(sc, m, rng):
    if sc["kind"] == "gauss":
        return sc["sig"] * rng.standard_normal(m)
    if sc["kind"] == "two-point":
        return sc["sig"] * rng.choice([-1.0, 1.0], m)
    if sc["kind"] == "t3":
        return sc["sig"] * rng.standard_t(3, m)
    if sc["kind"] == "three-point-skew":
        return rng.choice([-3.0, 0.0, 4.0], m, p=[0.3, 0.4, 0.3])
    return np.zeros(m)


def p_tasks(sc, mu, lt, rng):
    m = len(lt)
    if sc["kind"] == "homog-exact":  # homogeneous: every task has p = pooled curve of scenario C
        return pooled_C(np.exp(lt) * T_REF)
    z = draw_z(sc, m, rng) + sc["drift"] * lt
    p = np.exp(-np.exp(np.clip(mu + sc["k"] * lt + z, -30, 6)))
    if sc["imp"] is not None:
        prob = sc["imp"](np.exp(lt) * T_REF)
        p = np.where(rng.random(m) < prob, 0.0, p)
    return p


_xC, _wC = np.polynomial.hermite_e.hermegauss(80)
_wC = _wC / _wC.sum()


def pooled_C(T, k=1.5, sig=3.2):
    mu = calib_gauss(k, sig)
    lt = np.log(T / T_REF)
    return (np.exp(-np.exp(np.clip(mu + k * lt[:, None] + sig * _xC[None, :], -30, 6))) * _wC).sum(1)


def calib_gauss(k, sig):
    f = lambda m: (np.exp(-np.exp(np.clip(m + sig * _xC, -30, 6))) * _wC).sum() - 0.5
    return optimize.brentq(f, -30, 30)


def calib(sc):
    """mu such that pooled success at T = 60 min is 0.5 (Monte Carlo with fixed draws)."""
    if sc["kind"] == "homog-exact":
        return 0.0
    rng = np.random.default_rng(123)
    lt = np.zeros(40000)
    z = draw_z(sc, 40000, rng)
    imp = sc["imp"](np.array([60.0]))[0] if sc["imp"] is not None else 0.0
    f = lambda m: (1 - imp) * np.mean(np.exp(-np.exp(np.clip(m + z, -30, 6)))) - 0.5
    return optimize.brentq(f, -30, 30) + sc["mu_shift"]


long_imp = lambda T: np.where(T >= 240, 0.3, 0.0)
unif_imp = lambda T: np.full_like(T, 0.10, dtype=float)
SCEN = {
    "B: homogeneous, k=0.63": make("gauss", 0.63, 0.0),
    "B*: homogeneous, pooled curve = C exactly": make("homog-exact", np.nan),
    "C: Gaussian, k=1.5, s=3.2": make("gauss", 1.5, 3.2),
    "M1: two-point frailty, k=0.63": make("two-point", 0.63, 2.5),
    "M2: t3 frailty, k=0.63": make("t3", 0.63, 1.5),
    "M3: skewed 3-point frailty, k=0.63": make("three-point-skew", 0.63),
    "M4: Gaussian s=2 + 10% impossible (any T), k=0.63": make("gauss", 0.63, 2.0, imp=unif_imp),
    "M5: Gaussian s=2 + 30% impossible among T>=4h, k=0.63": make("gauss", 0.63, 2.0, imp=long_imp),
    "M6: Gaussian s=2 + 30% impossible among T>=4h, k=1.0": make("gauss", 1.0, 2.0, imp=long_imp),
    "M7: frailty location drifts +0.8 per log T, k=0.63": make("gauss", 0.63, 2.0, drift=0.8),
    "M8: label noise sd 0.7 (log T), homogeneous k=0.63": make("gauss", 0.63, 0.0, noise=0.7),
    "M9: label noise sd 0.7 (log T), Gaussian k=1.5 s=3.2": make("gauss", 1.5, 3.2, noise=0.7),
    "M10: label noise sd 0.7 + Gaussian s=2, k=0.63": make("gauss", 0.63, 2.0, noise=0.7),
}
NREP = 20


def one(args):
    name, rep = args
    sc = SCEN[name]
    rng = np.random.default_rng(1000 * list(SCEN).index(name) + rep)
    mu = calib(sc)
    p = p_tasks(sc, mu, LT0, rng)
    s = rng.binomial(N0, p)
    lt_obs = LT0 + sc["noise"] * rng.standard_normal(len(LT0))  # analyst sees noisy lengths
    T_obs = np.exp(lt_obs) * T_REF
    tt = pd.DataFrame(dict(task_id=np.arange(len(T0)), n=N0, s=s, T=T_obs, family=np.arange(len(T0))))
    Tr = np.repeat(T_obs, N0)
    yr = np.concatenate([np.r_[np.ones(si), np.zeros(ni - si)] for si, ni in zip(s, N0)])
    wr = np.repeat(1.0 / N0, N0)
    th, _ = fit_family("Weibull", Tr, yr, wr)
    r = fit_glmm(tt, "cloglog", nested=False)
    sd = np.exp(r.x[2])
    lc = nm(nll_latent, [[r.x[0] - 1.5 * sd, r.x[0], r.x[0] + 1.5 * sd, r.x[1], 0, 0],
                         [r.x[0] - 3, r.x[0] - 0.5, r.x[0] + 3, 1.0, 0.5, -0.5]], (lt_obs, s, N0))
    zi = nm(nll_zi, [[r.x[0], r.x[1], r.x[2], -12.0], [r.x[0], r.x[1], r.x[2], -3.0], [r.x[0] - 0.5, r.x[1] * 0.7, r.x[2] - 0.3, -2.0]],
            (lt_obs, s, N0))
    return dict(scenario=name, rep=rep, k_true=sc["k"], k_marg=np.exp(th[1]), H=H_strat(tt),
                k_gauss=r.x[1], sig_gauss=sd, k_latent3=lc.x[3], k_zi=zi.x[1], pi0_zi=special.expit(zi.x[3]),
                LR_zi_vs_gauss=2 * (r.fun - zi.fun), LR_latent3_vs_gauss=2 * (r.fun - lc.fun))


def mech(args):
    """Fixed sigma and k, varying capability: is corr(log h50_hat, log sigma_hat) mechanical?"""
    shift, rep = args
    rng = np.random.default_rng(50000 + int(shift * 10) * 100 + rep)
    k, sig = 1.5, 3.0
    mu = calib_gauss(k, sig) + shift
    p = np.exp(-np.exp(np.clip(mu + k * LT0 + sig * rng.standard_normal(len(LT0)), -30, 6)))
    s = rng.binomial(N0, p)
    tt = pd.DataFrame(dict(task_id=np.arange(len(T0)), n=N0, s=s, T=T0, family=np.arange(len(T0))))
    Tr = np.repeat(T0, N0)
    yr = np.concatenate([np.r_[np.ones(si), np.zeros(ni - si)] for si, ni in zip(s, N0)])
    th, _ = fit_family("Weibull", Tr, yr, np.repeat(1.0 / N0, N0))
    h50 = horizon(lambda TT: S_weibull(th, TT), 0.5)
    r = fit_glmm(tt, "cloglog", nested=False)
    return dict(shift=shift, rep=rep, h50_hat=h50, sig_hat=np.exp(r.x[2]), k_hat=r.x[1])


if __name__ == "__main__":
    jobs = [(nme, rep) for nme in SCEN for rep in range(NREP)]
    with Pool(4) as pool:
        R = pd.DataFrame(pool.map(one, jobs, chunksize=2))
        M = pd.DataFrame(pool.map(mech, [(sh, rep) for sh in [-6, -4, -2, 0, 2, 4] for rep in range(8)]))
    R.to_csv(f"{RES}/misspec_simulation.csv", index=False)
    M.to_csv(f"{RES}/misspec_mechanical_corr.csv", index=False)
    cols = ["k_true", "k_marg", "H", "k_gauss", "sig_gauss", "k_latent3", "k_zi", "pi0_zi"]
    S = R.groupby("scenario", sort=False)[cols].mean()
    S["frac_kgauss_gt1"] = R.groupby("scenario", sort=False).k_gauss.apply(lambda x: np.mean(x > 1))
    S["sd_kgauss"] = R.groupby("scenario", sort=False).k_gauss.std()
    S["LR_zi_med"] = R.groupby("scenario", sort=False).LR_zi_vs_gauss.median()
    S["LR_lat3_med"] = R.groupby("scenario", sort=False).LR_latent3_vs_gauss.median()
    ok = np.isfinite(M.h50_hat) & (M.h50_hat > 1e-3)
    c = np.corrcoef(np.log(M.h50_hat[ok]), np.log(M.sig_hat[ok]))[0, 1]
    with open(f"{RES}/misspec_summary.txt", "w") as f:
        pd.set_option("display.width", 250)
        f.write(f"Design: Claude 4 Opus TH1.1 lengths/run counts; {NREP} reps per scenario; means\n")
        f.write(S.round(3).to_string() + "\n\n")
        f.write("Mechanical-correlation check (true k=1.5, sigma=3.0 fixed; capability varied):\n")
        f.write(M.groupby("shift")[["h50_hat", "sig_hat", "k_hat"]].median().round(3).to_string() + "\n")
        f.write(f"corr(log h50_hat, log sigma_hat) across {ok.sum()} fits = {c:.3f}\n")
    print(open(f"{RES}/misspec_summary.txt").read())
