"""Stronger parametric baselines (revision r1).

Models fitted by maximum likelihood to Binomial counts (heterogeneous n_i allowed):
  BB       Beta(a,b)                                   (Kazdan et al.-style beta-binomial)
  ZIBB     pi0*delta_0 + (1-pi0)*Beta(a,b)             (zero-inflated beta-binomial)
  BMix2    w*Beta(a1,b1) + (1-w)*Beta(a2,b2)
  BMix3    3-component Beta mixture
  ZIBMix2  pi0*delta_0 + 2-component Beta mixture
  BIC      the model with the smallest BIC among the five (a practitioner's choice)
Designs:
  fixed    n rollouts on every task
  dynamic  same total budget M*n: n/2 rollouts on every task, the remaining M*n/2 spread equally over the
           tasks with zero successes in stage 1 (a simplified variant in the spirit of the dynamic sampling of
           Kazdan et al. 2025; not their exact algorithm)
Test beds: WalkSAT (A,B,C; 10 disjoint subsets) and the four synthetic laws of exp_synthetic.py.
Outputs results/baselines_raw.csv, results/baselines_summary.csv, results/capped_design_cost.csv
"""
import sys, os, glob, json, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, expit, logit, logsumexp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import unbiased_passk, g_passk

RES = os.path.join(os.path.dirname(__file__), "..", "results")


# ----------------------------------------------------------------------------- likelihoods
def _comp_ll(x, n, a, b):
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + betaln(x + a, n - x + b) - betaln(a, b)


def unpack(theta, J, zi):
    i = 0
    pi0 = expit(theta[i]) if zi else 0.0
    i += int(zi)
    if J > 1:
        lw = np.concatenate([[0.0], theta[i:i + J - 1]]); w = np.exp(lw - logsumexp(lw)); i += J - 1
    else:
        w = np.array([1.0])
    ab = np.exp(np.clip(theta[i:i + 2 * J], -9, 9)).reshape(J, 2)
    return pi0, w, ab


def nll(theta, x, n, cnt, J, zi):
    pi0, w, ab = unpack(theta, J, zi)
    comp = np.stack([_comp_ll(x, n, a, b) for a, b in ab], 1) + np.log(w)[None, :]
    llc = logsumexp(comp, 1) + np.log1p(-pi0) if zi else logsumexp(comp, 1)
    if zi:
        llz = np.where(x == 0, np.log(max(pi0, 1e-300)), -np.inf)
        ll = np.logaddexp(llc, llz)
    else:
        ll = llc
    return -np.sum(cnt * ll)


MODELS = {"BB": (1, False), "ZIBB": (1, True), "BMix2": (2, False), "BMix3": (3, False), "ZIBMix2": (2, True)}


def fit(x_i, n_i, J, zi, rng):
    # aggregate identical (x, n) pairs
    key, cnt = np.unique(np.stack([x_i, n_i], 1), axis=0, return_counts=True)
    x, n = key[:, 0].astype(float), key[:, 1].astype(float)
    frac0 = np.mean(x_i == 0)
    best = None
    for s in range(6 if J > 1 else 4):
        th = []
        if zi:
            th.append(logit(np.clip(frac0 * rng.uniform(0.3, 1.0), 0.01, 0.95)))
        if J > 1:
            th += list(rng.normal(0, 0.5, J - 1))
        for j in range(J):
            th += [np.log(rng.choice([0.1, 0.3, 1.0, 3.0])), np.log(rng.choice([0.5, 2.0, 10.0, 50.0]))]
        r = minimize(nll, np.array(th), args=(x, n, cnt, J, zi), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    k = len(best.x)
    return best.x, best.fun, k


def predict(theta, J, zi, k):
    pi0, w, ab = unpack(theta, J, zi)
    if np.isinf(k):
        return 1.0 - pi0
    v = sum(wj * (1.0 - np.exp(betaln(a, b + k) - betaln(a, b))) for wj, (a, b) in zip(w, ab))
    return (1 - pi0) * v


def fit_all(x_i, n_i, ks, seed):
    rng = np.random.default_rng(seed)
    out, bics = {}, {}
    M = len(x_i)
    for name, (J, zi) in MODELS.items():
        th, f, npar = fit(x_i, n_i, J, zi, rng)
        bics[name] = 2 * f + npar * np.log(M)
        out[name] = {k: float(predict(th, J, zi, k)) for k in ks}
    sel = min(bics, key=bics.get)
    out["BIC"] = dict(out[sel]); out["BIC"]["selected"] = sel
    return out


KS = [100, 300, 1000, 2000, np.inf]


# ----------------------------------------------------------------------------- WalkSAT
def walksat_job(args):
    f, n, s = args
    d = np.load(f); runs = d["runs"]; sat = d["sat"]; name = os.path.basename(f)[8:-4]
    M, N = runs.shape
    perm = np.random.default_rng(2026).permutation(N)  # one permutation per file: subsets are disjoint
    rows = []
    # fixed design: n runs per task (disjoint subset s)
    idx = perm[s * n:(s + 1) * n]
    X = runs[:, idx].sum(1).astype(int); nn = np.full(M, n)
    designs = {"fixed": (X, nn)}
    # dynamic design: n/2 per task, remaining M*n/2 over zero-success tasks (columns after the fixed block)
    h = n // 2
    X1 = runs[:, idx[:h]].sum(1).astype(int)
    zero = np.where(X1 == 0)[0]
    extra = (M * n - M * h) // max(len(zero), 1)
    pool = np.roll(perm[640:], -500 * s)  # columns never used by the fixed subsets (they use perm[:640])
    Xd, nd = X1.copy(), np.full(M, h)
    if len(zero):
        extra = min(extra, len(pool))
        cols = pool[:extra]
        Xd[zero] += runs[np.ix_(zero, cols)].sum(1).astype(int); nd[zero] += extra
    designs["dynamic"] = (Xd, nd)
    for des, (xi, ni) in designs.items():
        res = fit_all(xi, ni, KS, seed=s)
        for model, pr in res.items():
            for k in KS:
                rows.append(dict(bed="walksat", cfg=name, n=n, subset=s, design=des, model=model,
                                 target="pass@inf" if np.isinf(k) else f"pass@{int(k)}", pred=pr[k],
                                 selected=pr.get("selected", "")))
    return rows


def walksat_truth():
    tr = {}
    for f in sorted(glob.glob(os.path.join(RES, "walksat_*.npz"))):
        d = np.load(f); runs = d["runs"]; sat = d["sat"]; name = os.path.basename(f)[8:-4]
        N = runs.shape[1]; c = runs.sum(1)
        for k in KS:
            tr[(name, "pass@inf" if np.isinf(k) else f"pass@{int(k)}")] = float(sat.mean()) if np.isinf(k) else unbiased_passk(c, N, int(k))
    return tr


# ----------------------------------------------------------------------------- synthetic
def synth_job(args):
    import exp_synthetic as es
    li, rep, n, M = args
    law = es.LAWS[li]
    rng = np.random.default_rng(1000 * li + rep)
    es.rng = rng
    p = law.sample(M)
    X = rng.binomial(n, p); nn = np.full(M, n)
    h = n // 2
    X1 = rng.binomial(h, p); zero = np.where(X1 == 0)[0]
    extra = (M * n - M * h) // max(len(zero), 1)
    Xd, nd = X1.copy(), np.full(M, h)
    Xd[zero] += rng.binomial(extra, p[zero]); nd[zero] += extra
    ks = [4 * n, 16 * n, n * n, 4 * n * n, np.inf]
    rows = []
    for des, (xi, ni) in {"fixed": (X, nn), "dynamic": (Xd, nd)}.items():
        res = fit_all(xi, ni, ks, seed=rep)
        for model, pr in res.items():
            for k in ks:
                rows.append(dict(bed="synthetic", cfg=law.name, n=n, M=M, subset=rep, design=des, model=model,
                                 target="pass@inf" if np.isinf(k) else f"pass@{int(k)}", pred=pr[k],
                                 selected=pr.get("selected", "")))
    return rows


def synth_truth(n):
    import exp_synthetic as es
    tr = {}
    for law in es.LAWS:
        for k in [4 * n, 16 * n, n * n, 4 * n * n]:
            tr[(law.name, f"pass@{k}")] = law.expect(g_passk(k))
        tr[(law.name, "pass@inf")] = 1.0 - sum(w for w, kind, par in law.comps if kind == "point" and par == 0.0)
    return tr


def capped_cost():
    """Expected rollouts per task of the 'run until first success, cap D' design, which identifies pass@k
    for all k <= D exactly:  E[cost] = sum_{j<D} (1-p)^j = (1-(1-p)^D)/p  (= D if p = 0), averaged over tasks
    with p estimated from the 10,000 runs."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "walksat_*.npz"))):
        d = np.load(f); runs = d["runs"]; name = os.path.basename(f)[8:-4]
        p = runs.mean(1)
        for D in [100, 300, 1000, 2000]:
            c = np.where(p > 0, (1 - (1 - p) ** D) / np.maximum(p, 1e-300), D)
            rows.append(dict(cfg=name, D=D, mean_cost=float(c.mean()), frac_unsolved=float(np.mean(p == 0)),
                             n_population_rigorous_delta02=float(np.sqrt(D * np.log(4 / 0.02)))))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "capped_design_cost.csv"), index=False, float_format="%.4g")
    print(pd.DataFrame(rows).to_string())


def main():
    t0 = time.time()
    capped_cost()
    files = sorted(glob.glob(os.path.join(RES, "walksat_*.npz")))
    jobs = [(f, n, s) for f in files for n in [16, 64] for s in range(10)]
    sjobs = [(li, rep, 16, 5000) for li in range(4) for rep in range(20)]
    ck_w = os.path.join(RES, "baselines_walksat_rows.pkl"); ck_s = os.path.join(RES, "baselines_synth_rows.pkl")
    with Pool(4) as pool:
        if os.path.exists(ck_w):
            rows = pd.read_pickle(ck_w)
        else:
            rows = [r for rr in pool.map(walksat_job, jobs, chunksize=1) for r in rr]
            pd.to_pickle(rows, ck_w)
        print("walksat done", time.time() - t0, flush=True)
        if os.path.exists(ck_s):
            srows = pd.read_pickle(ck_s)
        else:
            srows = [r for rr in pool.map(synth_job, sjobs, chunksize=1) for r in rr]
            pd.to_pickle(srows, ck_s)
        rows = rows + srows
    print("synthetic done", time.time() - t0, flush=True)
    df = pd.DataFrame(rows)
    tw = walksat_truth(); ts = synth_truth(16)
    df["truth"] = [tw[(c, t)] if b == "walksat" else ts[(c, t)] for b, c, t in zip(df.bed, df.cfg, df.target)]
    df["err"] = df.pred - df.truth
    df.to_csv(os.path.join(RES, "baselines_raw.csv"), index=False, float_format="%.5f")
    s = df.groupby(["bed", "cfg", "n", "design", "target", "model"], sort=False).agg(
        truth=("truth", "first"), bias=("err", "mean"), absmax=("err", lambda x: np.abs(x).max()),
        sd=("err", "std")).reset_index()
    s.to_csv(os.path.join(RES, "baselines_summary.csv"), index=False, float_format="%.4f")
    sel = df[df.model == "BIC"].groupby(["bed", "cfg", "n", "design"]).selected.agg(lambda x: x.value_counts().to_dict())
    sel.to_csv(os.path.join(RES, "baselines_bic_selection.csv"))
    pd.set_option("display.width", 250)
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))
    print(sel.to_string())


if __name__ == "__main__":
    main()
