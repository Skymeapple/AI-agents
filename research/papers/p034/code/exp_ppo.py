"""Exp 6 (revision): audits under PPO-like, non-tilting policy updates.

Finite pool (200k outputs) from the exploit-mixture base policy (q = 1, K = 2).
Policy class: pi_theta(x) ∝ exp(theta . psi(x)) over the pool (base = uniform),
psi(x) = (g0+eps1, eps1, Z): the class contains exact tilting (theta = lam*(1,0,mu)), but the
optimiser is Adam on the clipped-surrogate PPO objective with a KL-to-base penalty, 4 epochs per
iteration on a batch of N = 2000 on-policy samples, so the path is not the tilting path.

Identity checked (exact for ANY update, no tilting assumption):
    G_{t+1} - G_t = E_{pi_t}[g (rho_t - 1)] = Cov_{pi_t}(g, rho_t),  rho_t = pi_{t+1}/pi_t.
Rules compared (m audited gold labels per iteration, on the batch drawn from pi_t):
  cov(g, rho) with leverage |rho - mean rho| ;  cov(g, r1) with leverage |r1 - mean r1| (paper's
  original rule, justified only under tilting) ;  gold-trend test (OLS slope of audited gold means,
  window tuned in-sample = optimistic for it).
Run: python3 exp_ppo.py   (~3-5 min on 4 cores)
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription with multiprocessing

import json, os, time
import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp
from model import Params
from rev_common import hh_cov, lev_probs, ols_slope

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
P = Params(q=1.0)
NP = 200_000
rng0 = np.random.default_rng(321)
Zp = rng0.random(NP) < 0.01  # enrich exploits in the pool to 1% (base prob), reweighted below
g0 = rng0.normal(0, 1, NP); c = rng0.normal(0, 1, NP); U = rng0.normal(0, 1, (NP, 2))
E = np.sqrt(P.rho) * c[:, None] + np.sqrt(1 - P.rho) * U
R1 = g0 + E[:, 0] + P.mu * Zp
G = np.where(Zp, -P.delta, g0)
PSI = np.stack([g0 + E[:, 0], E[:, 0], Zp.astype(float)], axis=1)
LOGBASE = np.where(Zp, np.log(P.p / 0.01), np.log((1 - P.p) / 0.99))  # importance weights to base pi0
LOGBASE -= logsumexp(LOGBASE)
T, NB, EPOCHS, CLIP, BETA, LR = 120, 2000, 4, 0.2, 0.05, 0.01
MS = [5, 10, 20, 40]


def logpi(theta):
    l = LOGBASE + PSI @ theta
    return l - logsumexp(l)


def run_path(seed):
    rng = np.random.default_rng(seed)
    theta = np.zeros(3); mom = np.zeros(3); vel = np.zeros(3); step = [0]
    Gs, dG, covr1, covrho, stats = [], [], [], [], {f"{k}{m}": [] for m in MS for k in ["Crho", "Cr1", "Gu"]}
    for t in range(T):
        lp = logpi(theta); pi = np.exp(lp)
        Gt = pi @ G; Gs.append(Gt)
        covr1.append(pi @ (G * (R1 - pi @ R1)))
        idx = rng.choice(NP, size=NB, replace=True, p=pi)
        psi = PSI[idx]; r = R1[idx]; g = G[idx]
        kl_term = lp[idx] - LOGBASE[idx]
        Rw = r - BETA * kl_term
        A = Rw - Rw.mean(); A /= (A.std() + 1e-8)
        th = theta.copy()
        for _ in range(EPOCHS):
            step[0] += 1
            lpn = logpi(th); Epsi = np.exp(lpn) @ PSI
            rho = np.exp(lpn[idx] - lp[idx])
            active = ~(((A > 0) & (rho > 1 + CLIP)) | ((A < 0) & (rho < 1 - CLIP)))
            grad = ((active * rho * A)[:, None] * (psi - Epsi)).mean(axis=0)
            mom[:] = 0.9 * mom + 0.1 * grad; vel[:] = 0.999 * vel + 0.001 * grad ** 2
            mh = mom / (1 - 0.9 ** step[0]); vh = vel / (1 - 0.999 ** step[0])
            th = th + LR * mh / (np.sqrt(vh) + 1e-8)  # Adam ascent step (per-coordinate scaling)
        lpn = logpi(th)
        rho_pool = np.exp(lpn - lp)
        dG.append(np.exp(lpn) @ G - Gt)
        covrho.append(pi @ (G * (rho_pool - 1.0)))
        rho_b = rho_pool[idx]; h_rho = rho_b - rho_b.mean(); h_r1 = r - r.mean()
        for m in MS:
            stats[f"Crho{m}"].append(hh_cov(g, h_rho, m, lev_probs(h_rho), rng)[0])
            stats[f"Cr1{m}"].append(hh_cov(g, h_r1, m, lev_probs(h_r1), rng)[0])
            stats[f"Gu{m}"].append(g[rng.choice(NB, m, replace=False)].mean())
        theta = th
    out = dict(G=np.array(Gs), dG=np.array(dG), covr1=np.array(covr1), covrho=np.array(covrho), theta=theta)
    out.update({k: np.array(v) for k, v in stats.items()})
    return out


def first_neg(x):
    i = np.where(np.asarray(x) < 0)[0]
    return int(i[0]) if len(i) else len(x) - 1


def win_neg(C, w):
    C = np.asarray(C)
    for t in range(w - 1, len(C)):
        if C[t - w + 1:t + 1].sum() < 0:
            return t
    return len(C) - 1


def gt_stop(Gu, w):
    x = np.arange(len(Gu))
    for t in range(w - 1, len(Gu)):
        if ols_slope(x[t - w + 1:t + 1], Gu[t - w + 1:t + 1]) < 0:
            return t
    return len(Gu) - 1


def main():
    t0 = time.time()
    with Pool(4) as pool:
        paths = pool.map(run_path, range(60))
    res = {"config": dict(T=T, batch=NB, epochs=EPOCHS, clip=CLIP, beta=BETA, lr=LR, pool=NP, q=1.0)}
    # identity and path diagnostics
    res["identity_max_abs_err_dG_vs_cov_g_rho"] = float(max(np.max(np.abs(p["dG"] - p["covrho"])) for p in paths))
    peak = [int(np.argmax(p["G"])) for p in paths]
    res["peak_iter_mean"] = float(np.mean(peak)); res["peak_gold_mean"] = float(np.mean([p["G"].max() for p in paths]))
    res["gold_at_T_mean"] = float(np.mean([p["G"][-1] for p in paths]))
    # sign disagreement between Cov(g, r1) and the realised gold change, per iteration
    dis = np.mean([np.mean(np.sign(p["covr1"]) != np.sign(p["dG"])) for p in paths])
    res["frac_iters_sign_cov_g_r1_ne_sign_dG"] = float(dis)
    res["exact_cov_g_r1_first_neg_minus_peak_iter_mean"] = float(np.mean([first_neg(p["covr1"]) - pk for p, pk in zip(paths, peak)]))
    res["exact_cov_g_rho_first_neg_minus_peak_iter_mean"] = float(np.mean([first_neg(p["covrho"]) - pk for p, pk in zip(paths, peak)]))
    # tilting direction vs realised direction: corr between theta/|theta| and (1,1,mu)/|.|
    tdir = np.array([1, 0, P.mu]) / np.linalg.norm([1, 0, P.mu])
    res["final_theta_mean"] = np.mean([p["theta"] for p in paths], axis=0).tolist()
    res["cos_final_theta_vs_tilting_direction"] = float(np.mean([p["theta"] @ tdir / np.linalg.norm(p["theta"]) for p in paths]))

    def summ(stops, labels=None):
        regs = np.array([p["G"].max() - p["G"][s] for p, s in zip(paths, stops)])
        d = dict(regret=float(regs.mean()), se=float(regs.std(ddof=1) / np.sqrt(len(regs))),
                 overshoot=float(np.mean(np.array(stops) > np.array(peak))))
        if labels is not None:
            d["labels_mean"] = float(np.mean(labels))
        return d
    rules = {}
    rules["oracle exact cov(g,rho) sign"] = summ([first_neg(p["covrho"]) for p in paths])
    rules["oracle exact cov(g,r1) sign"] = summ([first_neg(p["covr1"]) for p in paths])
    for m in MS:
        s = [first_neg(p[f"Crho{m}"]) for p in paths]; rules[f"audit cov(g,rho), leverage, m={m}"] = summ(s, [m * (x + 1) for x in s])
        s = [first_neg(p[f"Cr1{m}"]) for p in paths]; rules[f"audit cov(g,r1), leverage, m={m}"] = summ(s, [m * (x + 1) for x in s])
        for W_ in [3, 5]:
            s = [win_neg(p[f"Crho{m}"], W_) for p in paths]; rules[f"audit cov(g,rho), leverage, window-sum {W_}, m={m}"] = summ(s, [m * (x + 1) for x in s])
            s = [win_neg(p[f"Cr1{m}"], W_) for p in paths]; rules[f"audit cov(g,r1), leverage, window-sum {W_}, m={m}"] = summ(s, [m * (x + 1) for x in s])
        best = None
        for w in [3, 5, 8, 12, 16, 24]:
            s = [gt_stop(p[f"Gu{m}"], w) for p in paths]; d = summ(s, [m * (x + 1) for x in s]); d["w"] = w
            if best is None or d["regret"] < best["regret"]:
                best = d
        rules[f"gold-trend (window oracle-tuned), m={m}"] = best
    rules["never stop"] = summ([T - 1] * len(paths))
    res["rules"] = rules
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "ppo_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
