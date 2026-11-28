"""Bayes-optimal observational reference accuracies under the paper's exact training prior.

For the i.i.d. test items of each observational regime (regenerated with the same RNG
stream as twins.py, so these are the very items the networks were scored on) we compute
the posterior-optimal answer of reasoners that know:
  MEC        : the population Markov equivalence class and the query
  Sigma      : the population covariance (exact density over DAGs in the MEC via
               the regression parameterisation + Jacobian; the parameter prior is used)
  R          : the population correlation matrix (what standardised data can reveal);
               the diagonal scales are integrated out numerically (Sobol QMC)
each either label-free (prior: uniform relabelling, p_order = 0; optimal among
permutation-equivariant rules) or label-aware (prior: the regime's p_order mixture,
labels fully observed). By data processing, 'R, label-free' upper-bounds the accuracy of
ANY permutation-equivariant reasoner that sees standardised Gaussian samples; 'R, labels'
upper-bounds any reasoner that additionally sees the variable labels.
Output: results/bayes_oracles.json
"""
import json, os, sys, time, math
import numpy as np
from scipy.stats import qmc
import scm as S
import twins as T

n = S.N_NODES
LO_W, HI_W, LO_S, HI_S = S.W_LO, S.W_HI, S.S_LO, S.S_HI
SOB = qmc.Sobol(d=n, scramble=True, seed=0).random(4096)
SOB_BIG = qmc.Sobol(d=n, scramble=True, seed=1).random(1 << 18)


def lin_ext(A):
    import itertools
    c = 0
    for p in itertools.permutations(range(n)):
        pos = np.empty(n, int); pos[list(p)] = np.arange(n)
        if all(pos[u] < pos[v] for u, v in zip(*np.nonzero(A))):
            c += 1
    return c


_cache = {}


def prior_g(A, p_order):
    key = A.tobytes()
    if key not in _cache:
        _cache[key] = (lin_ext(A) / 24.0, float(np.all(np.tril(A) == 0)))
    le, canon = _cache[key]
    return (1 - p_order) * le / 64.0 + p_order * canon / 64.0


def q_prob(A, i, j):
    R = S.ancestors_matrix(A)
    npos = int(R.sum()); nneg = n * (n - 1) - npos
    if R[i, j]:
        return 0.5 / npos
    return (1.0 if npos == 0 else 0.5) / nneg


def regress(C, A):
    W = np.zeros((n, n)); s = np.zeros(n); dets = np.ones(n)
    for j in range(n):
        pa = np.nonzero(A[:, j])[0]
        if len(pa):
            b = np.linalg.solve(C[np.ix_(pa, pa)], C[pa, j])
            W[pa, j] = b; s[j] = np.sqrt(C[j, j] - C[pa, j] @ b)
            dets[j] = np.linalg.det(C[np.ix_(pa, pa)])
        else:
            s[j] = np.sqrt(C[j, j])
    return W, s, dets


def dens_sigma(Sig, A):
    W, s, dets = regress(Sig, A)
    w = np.abs(W[A > 0])
    if not (np.all((w >= LO_W) & (w <= HI_W)) and np.all((s >= LO_S) & (s <= HI_S))):
        return 0.0
    return float(1.0 / np.prod(2 * s * dets))


def dens_corr(Rm, A, pts=None):
    Wr, sr, dets = regress(Rm, A)
    lo, hi = LO_S / sr, HI_S / sr
    pts = SOB if pts is None else pts
    d = lo + pts * (hi - lo)                       # (Q, n) points in the box
    val = np.ones(len(d))
    for k, j in zip(*np.nonzero(A)):
        ratio = d[:, j] / d[:, k]
        w = np.abs(Wr[k, j]) * ratio
        val *= ratio * ((w >= LO_W) & (w <= HI_W))
    vol = np.prod(hi - lo)
    return float(vol * val.mean() / np.prod(sr * dets))


def decide(post_yes, post_no, y):
    if abs(post_yes - post_no) <= 1e-12 * max(post_yes + post_no, 1e-300):
        return 0.5
    return float((post_yes > post_no) == bool(y))


def regen_items(seed, p_order, K):
    rng = np.random.default_rng(10_000 + seed)
    out = []
    for _ in range(K):
        m = S.random_scm(rng, p_order=p_order)
        tgt = int(rng.integers(n))
        S.sample(rng, m, T.N, "gauss", 0, tgt)
        i, j, y = T.pick_query(rng, m["A"])
        out.append((m, i, j, y))
    return out


def evaluate(items, p_order):
    acc = {k: [] for k in ["mec_free", "mec_lab", "sigma_free", "sigma_lab", "corr_free", "corr_lab"]}
    ident = []
    for m, i, j, y in items:
        mem = S.mec_members(m["A"])
        Sig = S.implied_cov(m["W"], m["s"]); dg = np.sqrt(np.diag(Sig)); Rm = Sig / np.outer(dg, dg)
        anc = np.array([S.ancestors_matrix(B)[i, j] for B in mem])
        ident.append(anc.min() == anc.max())
        qp = np.array([q_prob(B, i, j) for B in mem])
        pf = np.array([prior_g(B, 0.0) for B in mem]); pl = np.array([prior_g(B, p_order) for B in mem])
        ds = np.array([dens_sigma(Sig, B) for B in mem])
        dc = np.array([dens_corr(Rm, B) for B in mem])
        if dc.sum() == 0:                        # QMC missed a thin region: refine
            dc = np.array([dens_corr(Rm, B, SOB_BIG) for B in mem])
        for name, wts in [("mec_free", pf), ("mec_lab", pl), ("sigma_free", pf * ds), ("sigma_lab", pl * ds),
                          ("corr_free", pf * dc), ("corr_lab", pl * dc)]:
            w = wts * qp
            acc[name].append(decide(w[anc == 1].sum(), w[anc == 0].sum(), y))
    ident = np.array(ident)
    res = {}
    for k, v in acc.items():
        v = np.array(v)
        se = float(np.sqrt(v.var() / len(v)))
        res[k] = {"acc": float(v.mean()), "se": se, "acc_identified": float(v[ident].mean()),
                  "acc_unidentified": float(v[~ident].mean())}
    res["frac_identified"] = float(ident.mean()); res["n_items"] = len(items)
    return res


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = {}
    t0 = time.time()
    for nm, p in [("obs_p0", 0.0), ("obs_p05", 0.5), ("obs_p09", 0.9), ("obs_p1", 1.0)]:
        per = [evaluate(regen_items(sd, p, K), p) for sd in (0, 1, 2)]   # = eval items of seeds 0,1,2
        out[nm] = per[0]
        out[nm]["per_seed_items"] = {str(sd): {k: v["acc"] for k, v in r.items() if isinstance(v, dict)}
                                     | {k + "_unid": v["acc_unidentified"] for k, v in r.items() if isinstance(v, dict)}
                                     for sd, r in zip((0, 1, 2), per)}
        print(nm, {k: round(v["acc"], 4) for k, v in out[nm].items() if isinstance(v, dict) and "acc" in v},
              f"{time.time()-t0:.0f}s", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bayes_oracles.json")
    if K >= 4000:
        json.dump(out, open(path, "w"), indent=1)
