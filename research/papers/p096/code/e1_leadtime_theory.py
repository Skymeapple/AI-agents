"""E1: Detection-limit theory of warning lead time + Monte Carlo validation.

Generative model (scale x = log10 training compute, arbitrary origin):
  item i has latent offset x0_i = x0 + s*u_i, u_i ~ N(0,1)
  per-step success q_i(x) = sigmoid(beta*(x - x0_i))
  exact-match success on a k-step item: p_i(x) = q_i(x)^k
  each evaluated model has an effective-scale jitter eta ~ N(0, sig_m^2)
  benchmark: n items; observed exact-match count C(x) = sum_i Bern(p_i(x+eta))
  continuous proxy: mean log-likelihood of the correct k-step answer
      Lbar(x) = mean_i k*log q_i(x+eta)

Outputs: results/e1_*.json, figures.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from scipy.stats import binom
from multiprocessing import Pool

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

BETA, X0, TAU = 2.0, 3.5, 0.5
GRID = np.round(np.arange(0.0, 7.0001, 0.1), 3)
SIG_M = 0.05
ALPHA = 0.01          # family-wise false-alarm rate for the detection rule
P_FLOOR = 1e-4        # spurious-success floor (guessing / grader noise)
EPS = 0.25            # forecast tolerance, decades


# ---------------------------------------------------------------- closed form
def x_cross(p, k, beta=BETA, x0=X0):
    """scale at which homogeneous q^k equals p."""
    return x0 + logit(p ** (1.0 / k)) / beta


def lead_closed(n, k, c=4.0, tau=TAU, beta=BETA):
    """Theorem 1: L = (1/beta)[logit(tau^{1/k}) - logit((c/n)^{1/k})]."""
    return (logit(tau ** (1.0 / k)) - logit((c / n) ** (1.0 / k))) / beta


def lead_limit_k_inf(n, c=4.0, tau=TAU, beta=BETA):
    """Corollary: k -> infinity limit (1/beta) ln( ln(n/c) / ln(1/tau) )."""
    return np.log(np.log(n / c) / np.log(1 / tau)) / beta


def lead_chance_floor(n, k, m=4, z=2.326, tau=TAU, beta=BETA):
    """Multiple-choice with chance floor p0=1/m: acc = p0 + (1-p0) q^k.
    Detection when (1-p0) q^k >= z*sqrt(p0(1-p0)/n). Crossing when acc = p0+(1-p0)tau."""
    p0 = 1.0 / m
    d = z * np.sqrt(p0 * (1 - p0) / n) / (1 - p0)
    return (logit(tau ** (1.0 / k)) - logit(d ** (1.0 / k))) / beta


# ---------------------------------------------------------------- simulation
XK, DB = 2.5, 4.0     # hidden-kink scenario: latent slope jumps from BETA to BETA+DB at XK


def zlat(x, x0i, kink):
    z = BETA * (x - x0i)
    if kink:
        z = z + DB * max(0.0, x - XK)
    return z


def true_acc(x, k, x0i, kink=False):
    return np.mean(expit(zlat(x, x0i, kink)) ** k)


def true_cross(k, x0i, kink=False):
    return brentq(lambda x: true_acc(x, k, x0i, kink) - TAU, -5, 15)


def detection_threshold(n):
    # per-scale threshold so that family-wise FA over the grid <= ALPHA under floor-only
    a = ALPHA / len(GRID)
    c = 1
    while binom.sf(c - 1, n, P_FLOOR) > a:
        c += 1
    return c


GH_U, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()


def fit_acc_logistic(xs, cnt, n):
    """Forecaster A: 2-parameter logistic in accuracy, binomial MLE."""
    def nll(th):
        a, b = th
        p = np.clip(expit(np.exp(a) * (xs - b)), 1e-12, 1 - 1e-12)
        return -np.sum(cnt * np.log(p) + (n - cnt) * np.log(1 - p))
    best = None
    for b0 in (xs[-1], xs[-1] + 1.0, xs[-1] + 3.0):
        r = minimize(nll, [np.log(2.0), b0], method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    return b + logit(TAU) / np.exp(a)


def fit_proxy(xs, L, k, het, return_params=False):
    """Forecaster B (het=False): fit k*log sigmoid(beta(x-x0)) to mean log-lik, assume homogeneous items.
    Forecaster C (het=True): additionally fit item spread s via Gauss-Hermite; predict crossing of mean acc."""
    def model(th, x):
        b, x0 = np.exp(th[0]), th[1]
        s = np.exp(th[2]) if het else 0.0
        xi = x0 + s * GH_U[None, :]
        return (k * np.log(expit(b * (x[:, None] - xi))) * GH_W[None, :]).sum(1), b, x0, s
    def loss(th):
        return np.mean((model(th, xs)[0] - L) ** 2)
    th0 = [np.log(1.0), xs[-1] + 1.0] + ([np.log(0.3)] if het else [])
    best = None
    for x0g in (xs[-1], xs[-1] + 1.5, xs[-1] + 3.0):
        th0[1] = x0g
        r = minimize(loss, th0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-9, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    _, b, x0, s = model(best.x, xs[:1])
    if return_params:
        return b, x0
    xi = x0 + s * GH_U
    f = lambda x: np.sum(GH_W * expit(b * (x - xi)) ** k) - TAU
    try:
        return brentq(f, -20, 40)
    except ValueError:
        return np.nan


def fit_proxy_linear(xs, L, W=1.5):
    """Forecaster D: agnostic linear extrapolation of mean log-lik over the last W decades,
    predicting the crossing where exp(Lbar) = tau (exact for homogeneous items)."""
    m = xs >= xs[-1] - W - 1e-9
    a, b = np.polyfit(xs[m], L[m], 1)
    if a <= 1e-9:
        return np.inf
    return (np.log(TAU) - b) / a


def fit_proxy_item(xs, L, Li_last, k):
    """Forecaster E: item-level proxy. Shared slope beta from the mean-curve fit; each item's offset
    x0_i recovered from its own current log-likelihood; predicted accuracy = mean_i sigmoid(beta(x-x0_i))^k."""
    b, _ = fit_proxy(xs, L, k, False, return_params=True)
    qi = np.clip(np.exp(Li_last / k), 1e-300, 1 - 1e-12)
    x0i_hat = xs[-1] - logit(qi) / b
    f = lambda x: np.mean(expit(b * (x - x0i_hat)) ** k) - TAU
    try:
        return brentq(f, -20, 40)
    except ValueError:
        return np.nan


def one_rep(args):
    k, n, s, seed, kink = args
    rng = np.random.default_rng(seed)
    x0i = X0 + s * rng.standard_normal(n)
    xt = true_cross(k, x0i, kink)
    cthr = detection_threshold(n)
    eta = SIG_M * rng.standard_normal(len(GRID))
    cnt = np.zeros(len(GRID))
    Lb = np.zeros(len(GRID))
    Li = np.zeros((len(GRID), n))
    for j, x in enumerate(GRID):
        q = expit(zlat(x + eta[j], x0i, kink))
        p = np.clip(q ** k, 0, 1)
        p = P_FLOOR + (1 - P_FLOOR) * p
        cnt[j] = (rng.random(n) < p).sum()
        Li[j] = k * np.log(q)
        Lb[j] = Li[j].mean()
    fire = np.where(cnt >= cthr)[0]
    x_fire = GRID[fire[0]] if len(fire) else np.nan
    # forecasts from every 0.2 decades, starting at index 5 (need >=6 points)
    preds = {"acc": [], "proxy": [], "proxy_het": [], "proxy_lin": [], "proxy_item": []}
    ts = []
    for j in range(5, len(GRID), 2):
        if GRID[j] > xt + 1e-9:
            break
        xs = GRID[: j + 1]
        ts.append(GRID[j])
        preds["acc"].append(fit_acc_logistic(xs, cnt[: j + 1], n))
        preds["proxy"].append(fit_proxy(xs, Lb[: j + 1], k, False))
        preds["proxy_het"].append(fit_proxy(xs, Lb[: j + 1], k, True))
        preds["proxy_lin"].append(fit_proxy_linear(xs, Lb[: j + 1]))
        preds["proxy_item"].append(fit_proxy_item(xs, Lb[: j + 1], Li[j], k))
    ts = np.array(ts)
    lead = {}
    for key, pr in preds.items():
        pr = np.array(pr, float)
        ok = np.abs(pr - xt) <= EPS
        # earliest t from which forecast stays within EPS through x_tau
        L = 0.0
        for i in range(len(ts)):
            if ok[i:].all():
                L = xt - ts[i]
                break
        lead[key] = L
    return dict(k=k, n=n, s=s, kink=kink, seed=seed, x_first_forecast=float(ts[0]) if len(ts) else None, x_tau=xt, x_fire=x_fire,
                lead_detect=(xt - x_fire) if np.isfinite(x_fire) else 0.0,
                lead_fc=lead, cthr=cthr,
                traj=dict(t=ts.tolist(), **{kk: list(map(float, v)) for kk, v in preds.items()}))


def main():
    t0 = time.time()
    # closed-form tables
    ns = np.logspace(2, 6, 41)
    ks = [1, 2, 4, 8, 16, 32, 64]
    cf = {"n": ns.tolist(),
          "lead_zero_floor": {k: lead_closed(ns, k).tolist() for k in ks},
          "lead_k_inf": lead_limit_k_inf(ns).tolist(),
          "lead_chance_floor_m4": {k: lead_chance_floor(ns, k).tolist() for k in ks}}
    # point values quoted in the paper
    pts = {}
    for k in [1, 4, 16, 64]:
        for n in [100, 1000, 10000, 100000]:
            pts[f"k{k}_n{n}"] = dict(zero_floor=float(lead_closed(n, k)),
                                     chance_m4=float(lead_chance_floor(n, k)))
    for n in [100, 1000, 10000, 100000, 1e6]:
        pts[f"kinf_n{int(n)}"] = float(lead_limit_k_inf(n))
    # marginal value of doubling n
    for k in [1, 4, 16, 64]:
        pts[f"gain_double_n_k{k}_n1000"] = float(lead_closed(2000, k) - lead_closed(1000, k))
    cf["points"] = pts
    json.dump(cf, open(os.path.join(OUT, "e1_closed_form.json"), "w"), indent=1)

    # Monte Carlo
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    SCEN = [(0.0, False), (1.0, False), (0.0, True)]
    jobs = []
    for s, kink in SCEN:
        for k in [1, 4, 16]:
            for n in [100, 1000, 10000]:
                for r in range(R):
                    jobs.append((k, n, s, 1000 * k + 7 * n + r + int(s * 99991) + 55555 * kink, kink))
    with Pool(4) as pool:
        res = pool.map(one_rep, jobs, chunksize=4)
    json.dump(res, open(os.path.join(OUT, "e1_montecarlo_raw.json"), "w"))
    # summary
    summ = []
    for s, kink in SCEN:
        for k in [1, 4, 16]:
            for n in [100, 1000, 10000]:
                rr = [r for r in res if r["k"] == k and r["n"] == n and r["s"] == s and r["kink"] == kink]
                ld = np.array([r["lead_detect"] for r in rr])
                row = dict(s=s, kink=kink, k=k, n=n, R=len(rr), cthr=rr[0]["cthr"],
                           x_tau_mean=float(np.mean([r["x_tau"] for r in rr])),
                           max_possible_lead=float(np.mean([r["x_tau"] - r["x_first_forecast"] for r in rr])),
                           lead_detect_mean=float(ld.mean()), lead_detect_sd=float(ld.std()),
                           lead_detect_theory=float(lead_closed(n, k, c=rr[0]["cthr"])) if (s == 0 and not kink) else None)
                for key in ["acc", "proxy", "proxy_het", "proxy_lin", "proxy_item"]:
                    v = np.array([r["lead_fc"][key] for r in rr])
                    row[f"leadfc_{key}_mean"] = float(v.mean())
                    row[f"leadfc_{key}_sd"] = float(v.std())
                summ.append(row)
    json.dump(summ, open(os.path.join(OUT, "e1_montecarlo_summary.json"), "w"), indent=1)
    print("rows:")
    for r in summ:
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items()})
    print("elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
