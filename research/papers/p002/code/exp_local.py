"""Revision r2: instance-dependent ("local") limits on real LLM data, plus forecasts on frontier-model data.

For an observed evaluation (counts X_1..X_M out of n) we fit the nonparametric MLE F_hat of the mixing law
(EM on a grid) and compute a rigorous LOCAL lower bound on the expected half-length, under F_hat, of ANY
procedure whose 95% coverage of theta = E_F[1-(1-p)^k] holds for every F:

  for each direction s = +-1, take G maximising s*(theta(G)-theta(F_hat)) subject to
  sum_x (pi_G - pi_Fhat)_x^2 / (pi_G + pi_Fhat)_x <= 2*eta   (a convex SOCP; >= 1-BC termwise / 2),
  recompute 1-BC exactly (shrinking G toward F_hat if needed), and integrate Pratt's identity along
  F_lam = F_hat + lam (G - F_hat):  E_Fhat[length] >= sum_s |Delta_s| int_0^1 (1-eps-TV_s(lam))_+ dlam,
  TV_s(lam) <= sqrt(1 - BC(pi_Fhat, pi_Flam)^(2M)).  Right-endpoint Riemann sum (integrand non-increasing).
  eta ranges over a/M, a in A_GRID; the best is kept.  Grid laws are genuine laws, so the bound is rigorous
  (for the law F_hat, which reproduces the observed data as well as any law can).

Data sets
  (1) Large Language Monkeys, MATH, 7 Pythia models, 128 problems x 10,000 samples (as in exp_llm.py;
      the same counterfactual subsets).  Truth = unbiased estimate from all 10,000 samples.
  (2) Best-of-N jailbreaking (Hughes et al. 2024), text modality, 159 HarmBench prompts, 7 models
      (Claude 3.5 Sonnet, Claude 3 Opus, Gemini 1.5 Flash/Pro, GPT-4o, GPT-4o-mini, Llama-3-8B-Instruct).
      Attempts were run in batches until the first success (cap ~10^4); the first 60 attempts are recorded
      for every prompt, so the counts in the first n <= 60 attempts are genuine fixed-n Binomial counts.
      Truth for pass@k = fraction of prompts whose first success occurred within k attempts (realized run).
Outputs: results/local_llm.csv, results/bon_forecasts.csv, results/bon_summary.json
"""
import os, sys, json, time
import numpy as np, pandas as pd, cvxpy as cp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import cheb_grid, bernstein_matrix, g_passk, box_interval, powerlaw_passk
from exp_baselines import fit_all
from exp_llm import subsets, unbiased, N as NLLM

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EPS = 0.05
A_GRID = [1.0, 4.0]
LAM = np.linspace(0, 1, 201)[1:]


def grid():
    return np.unique(np.concatenate([cheb_grid(300, extra_log=False), np.logspace(-8, -1, 100)]))


def npmle(hist, B, iters=4000):
    """EM for the grid NPMLE; B is (K, m)."""
    m = B.shape[1]; w = np.full(m, 1.0 / m); h = hist / hist.sum()
    for _ in range(iters):
        pi = B @ w
        w = w * (B.T @ (h / np.maximum(pi, 1e-300)))
        w /= w.sum()
    return w


def omb(P, Q):
    sp, sq = np.sqrt(np.maximum(P, 0)), np.sqrt(np.maximum(Q, 0))
    den = (sp + sq) ** 2
    return float(np.sum(np.where(den > 0, (P - Q) ** 2 / (2 * np.maximum(den, 1e-300)), 0.0)))


def local_lb(gv, B, wF, M):
    piF = B @ wF; K, m = B.shape
    thF = float(gv @ wF)
    best = dict(lb=0.0)
    for a in A_GRID:
        eta = a / M
        tot = 0.0; deltas = []
        for sgn in (+1, -1):
            wG = cp.Variable(m, nonneg=True)
            piG = B @ wG
            cons = [cp.sum(wG) == 1,
                    cp.sum(cp.hstack([cp.quad_over_lin(piG[x] - piF[x], piG[x] + piF[x]) for x in range(K)])) <= 2 * eta]
            pr = cp.Problem(cp.Maximize(sgn * (gv @ wG)), cons)
            try:
                pr.solve(solver="CLARABEL")
            except Exception:
                continue
            if wG.value is None:
                continue
            g = np.maximum(wG.value, 0); g /= g.sum()
            h = omb(piF, B @ g)
            if h > eta:
                g = wF + (eta / h) * (g - wF)
            D = float(gv @ g) - thF
            if sgn * D <= 0:
                continue
            u = g - wF
            hs = np.array([omb(piF, piF + l * (B @ u)) for l in LAM])
            tv = np.sqrt(np.maximum(0, 1 - np.exp(2 * M * np.log1p(-np.minimum(hs, 1 - 1e-16)))))
            tot += abs(D) * np.sum(np.maximum(0, 1 - EPS - tv)) * (LAM[1] - LAM[0])
            deltas.append(D)
        if tot / 2 > best["lb"]:
            best = dict(lb=tot / 2, a=a, deltas=deltas)
    return best


# ----------------------------------------------------------------------------- Pythia / MATH
def llm_job(args):
    model, c, n, mi, ks, subs = args
    p = grid(); B = bernstein_matrix(n, p).T
    rows = []
    for s, X in enumerate(subsets(c, n, 1000 * mi + n)):
        if s not in subs:
            continue
        hist = np.bincount(X, minlength=n + 1).astype(float)
        wF = npmle(hist, B)
        for k in ks:
            gv = g_passk(k)(p)
            r = local_lb(gv, B, wF, len(X))
            lo, hi = box_interval(g_passk(k), n, X, alpha=EPS)
            rows.append(dict(data="MATH", model=model, n=n, subset=s, k=k, M=len(X), local_lb_half=r["lb"],
                             box_half=(hi - lo) / 2, npmle_theta=float(gv @ wF), truth=unbiased(c, NLLM, k)))
    print(model, n, flush=True)
    return rows


# ----------------------------------------------------------------------------- Best-of-N jailbreaking
def bon_load():
    d = pd.read_csv(os.path.join(DATA, "bon_jailbreak_text_attempts.csv"), dtype={"first60": str})
    return d


def bon_truth(T, k):
    T = np.asarray(T, float)
    return float(np.mean(np.where(np.isnan(T), False, T <= k)))


def bon_job(args):
    model, first60, T, n, ks = args
    X = np.array([s[:n].count("1") for s in first60])
    M = len(X)
    p = grid(); B = bernstein_matrix(n, p).T
    hist = np.bincount(X, minlength=n + 1).astype(float)
    wF = npmle(hist, B)
    par = fit_all(X, np.full(M, n), ks, seed=0)
    pl = powerlaw_passk(X, n, ks)
    rows = []
    for j, k in enumerate(ks):
        gv = g_passk(k)(p)
        r = local_lb(gv, B, wF, M)
        lo, hi = box_interval(g_passk(k), n, X, alpha=EPS)
        row = dict(data="BoN", model=model, n=n, k=k, M=M, truth=bon_truth(T, k), box_lo=lo, box_hi=hi,
                   box_half=(hi - lo) / 2, local_lb_half=r["lb"], npmle_theta=float(gv @ wF), powerlaw=float(pl[j]),
                   frac_zero=float(np.mean(X == 0)))
        for mo in ["BB", "ZIBB", "BMix2", "BIC"]:
            row[mo] = par[mo][k]
        row["BIC_sel"] = par["BIC"]["selected"]
        rows.append(row)
    print("BoN", model, n, flush=True)
    return rows


def main():
    t0 = time.time()
    nproc = int(os.environ.get("NPROC", 3))
    # --- BoN
    d = bon_load()
    jobs = []
    for model, g in d.groupby("model"):
        for n in [15, 30, 60]:
            ks = sorted(set([n, 4 * n, 16 * n, 1000, 10000]))
            jobs.append((model, g.first60.tolist(), g.first_success.values, n, ks))
    with Pool(nproc) as pool:
        bon = [r for rr in pool.map(bon_job, jobs, chunksize=1) for r in rr]
    bon = pd.DataFrame(bon)
    bon.to_csv(os.path.join(RES, "bon_forecasts.csv"), index=False, float_format="%.5f")
    print("BoN done", time.time() - t0, flush=True)
    # --- MATH
    dm = pd.read_csv(os.path.join(RES, "llm_monkeys_math_pythia_counts.csv"))
    models = sorted(dm.model.unique(), key=lambda m: float(m.split()[1].rstrip("MB")) * (1e-3 if m.endswith("M") else 1))
    C = {m: dm[dm.model == m].sort_values("problem_idx").n_correct.values.astype(int) for m in models}
    jobs = []
    for mi, m in enumerate(models):
        for n in [16, 32, 64, 128]:
            ks = sorted(set(k for k in [n, 2 * n, 4 * n, 8 * n, 16 * n, 10000] if k <= 10000))
            jobs.append((m, C[m], n, mi, ks, {0, 1, 2}))
    with Pool(nproc) as pool:
        loc = [r for rr in pool.map(llm_job, jobs, chunksize=1) for r in rr]
    loc = pd.DataFrame(loc)
    loc.to_csv(os.path.join(RES, "local_llm.csv"), index=False, float_format="%.5f")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
