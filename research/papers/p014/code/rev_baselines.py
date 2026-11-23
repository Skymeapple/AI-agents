"""Non-learned, prior-agnostic reference reasoners on covered-edge twins (revision).
They know the common-parent set of the queried pair (oracle skeleton information) but
NOT the parameter prior, so any surplus they earn comes from direction cues in the data,
not from detecting whether a parameterisation lies in the training prior's support.
  * pairwise LiNGAM (Hyvarinen & Smith 2013 likelihood-ratio with the Hyvarinen 1998
    max-entropy approximation) on residuals after regressing out the common parents;
  * interventional partial-correlation test (Fisher z, alpha=0.05) on the interventional rows.
Queried direction ((u,v) or (v,u)) is randomised. Output: results/rev_baselines.json"""
import json, os
import numpy as np
import scm as S

n, N = S.N_NODES, 256
K1, K2, GAM = 79.047, 7.4129, 0.37457


def negent(x):
    x = (x - x.mean()) / x.std()
    return K1 * (np.mean(np.log(np.cosh(x))) - GAM) ** 2 + K2 * np.mean(x * np.exp(-x ** 2 / 2)) ** 2


def resid(X, a, C):
    if not C:
        return X[:, a] - X[:, a].mean()
    Z = np.c_[np.ones(len(X)), X[:, C]]
    b = np.linalg.lstsq(Z, X[:, a], rcond=None)[0]
    return X[:, a] - Z @ b


def lingam_u_to_v(X, u, v, C):
    x, y = resid(X, u, C), resid(X, v, C)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = np.mean(x * y)
    ry = (y - r * x) / np.sqrt(1 - r * r); rx = (x - r * y) / np.sqrt(1 - r * r)
    # log L(x->y) - log L(y->x) = -H(x)-H(ry)+H(y)+H(rx); H = const - negentropy
    return (negent(x) + negent(ry) - negent(y) - negent(rx)) > 0


def int_test(X, I, u, v, C, tgt):
    rows = I[:, tgt] > 0
    Xi = X[rows]
    x, y = resid(Xi, u, C), resid(Xi, v, C)
    r = np.corrcoef(x, y)[0, 1]
    z = np.sqrt(rows.sum() - len(C) - 3) * np.arctanh(abs(r))
    dep = z > 1.96
    if tgt == u:
        return float(dep)          # u -> v  iff do(u) moves v
    if tgt == v:
        return float(not dep)      # v -> u  iff do(v) moves u
    return 0.5


def run(family, n_int, inprior, K, seed):
    rng = np.random.default_rng(seed)
    acc, orc = [], []
    k = 0
    while k < K:
        m = S.random_scm(rng)
        ce = S.covered_edges(m["A"])
        if not ce:
            continue
        u, v = ce[rng.integers(len(ce))]
        m2 = S.twin_by_reversal(m, u, v)
        if inprior and not S.in_prior(m2):
            continue
        C = [int(c) for c in np.nonzero(m["A"][:, u])[0]]
        qi, qj = (u, v) if rng.random() < 0.5 else (v, u)
        tgt = int(rng.integers(n))
        for model, other in ((m, m2), (m2, m)):
            X, I = S.sample(rng, model, N, family, n_int, tgt)
            truth = int(S.ancestors_matrix(model["A"])[qi, qj])
            if n_int == 0:
                p_uv = float(lingam_u_to_v(X, u, v, C))
            else:
                p_uv = int_test(X, I, u, v, C, tgt)
            p_yes = p_uv if (qi, qj) == (u, v) else 1 - p_uv
            acc.append(0.5 if p_yes == 0.5 else float((p_yes > 0.5) == truth))
            l1, l2 = S.loglik(X, I, model, family), S.loglik(X, I, other, family)
            orc.append(0.5 if abs(l1 - l2) <= 1e-7 * abs(l1) else float(l1 > l2))
        k += 1
    a = np.array(acc).reshape(-1, 2).mean(1); o = np.array(orc).reshape(-1, 2).mean(1)
    bs = np.random.default_rng(5)
    ks = []
    for _ in range(2000):
        i = bs.integers(K, size=K)
        ks.append((2 * a[i].mean() - 1) / (2 * o[i].mean() - 1))
    return {"twin_acc": float(a.mean()), "lr_ceiling": float(o.mean()),
            "kappa": float((2 * a.mean() - 1) / (2 * o.mean() - 1)),
            "kappa_ci": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))], "n_pairs": K}


if __name__ == "__main__":
    out = {}
    for name, fam, ni in [("laplace", "laplace", 0), ("int32", "gauss", 32), ("int128", "gauss", 128)]:
        for inp in (False, True):
            key = f"{name}_{'inprior' if inp else 'all'}"
            out[key] = run(fam, ni, inp, 1500, 40_000 + len(out))
            print(key, out[key], flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "rev_baselines.json")
    json.dump(out, open(path, "w"), indent=1)
