"""Toy measurement of overseer-error dependence (rho) and how adversarial
selection by a stronger policy changes it.

Overseers are MLP classifiers of increasing width trained to flag 'defective'
outputs x (y=1). We measure, on defective test outputs, the marginal detection
rates d_j and the joint all-miss rate, and estimate
    rho_hat = (P_allmiss - prod(1-d_j)) / (min(1-d_j) - prod(1-d_j)),
the weight on the comonotone (nested blind spot) copula in the mixture
rho*M + (1-rho)*Pi used by the BOS model.

Conditions:
  natural      : defective outputs drawn from the data distribution
  adversarial  : a 'policy' draws K defective candidates and submits the one a
                 strong surrogate detector (trained on independent data) scores
                 least suspicious (best-of-K evasion)
Ensembles:
  ladder       : widths 2, 8, 32, 128 (capability ladder)
  peers        : four width-32 overseers on independent data / seeds
"""
import json, sys
import numpy as np
from sklearn.neural_network import MLPClassifier

D = 12


def truth(X):
    f = (X[:, 0] + 0.8 * X[:, 1] * X[:, 2] + 0.6 * X[:, 3] * X[:, 4] * X[:, 5]
         + 0.8 * np.sin(2 * X[:, 6]) * X[:, 7] + 0.5 * (X[:, 8] ** 2 - 1)
         + 0.4 * X[:, 9] * X[:, 10] * X[:, 11])
    return (f > 0.9).astype(int)


def sample(rng, n):
    X = rng.standard_normal((n, D))
    return X, truth(X)


def sample_defective(rng, n):
    out = []
    while sum(len(o) for o in out) < n:
        X, y = sample(rng, 4 * n)
        out.append(X[y == 1])
    return np.concatenate(out)[:n]


def train(width, rng, n_train, label_noise=0.05, seed=0):
    X, y = sample(rng, n_train)
    flip = rng.random(n_train) < label_noise
    y = np.where(flip, 1 - y, y)
    clf = MLPClassifier(hidden_layer_sizes=(width,), max_iter=400, alpha=1e-3,
                        random_state=seed)
    clf.fit(X, y)
    return clf


def rho_hat(miss):  # miss: (n_examples, m) boolean
    q = miss.mean(0)
    pall = miss.all(1).mean()
    ind = np.prod(q)
    com = q.min()
    den = com - ind
    return (pall - ind) / den if den > 1e-9 else np.nan, q, pall, ind, com


def run(seed=0, n_train=3000, n_test=6000, K=20, n_boot=500):
    rng = np.random.default_rng(seed)
    ladder = [train(w, rng, n_train, seed=seed * 100 + i) for i, w in enumerate([2, 8, 32, 128])]
    peers = [train(32, rng, n_train, seed=seed * 100 + 10 + i) for i in range(4)]
    surrogate = train(256, rng, 4 * n_train, seed=seed * 100 + 99)

    nat = sample_defective(rng, n_test)
    cand = sample_defective(rng, n_test * K).reshape(n_test, K, D)
    s = surrogate.predict_proba(cand.reshape(-1, D))[:, 1].reshape(n_test, K)
    adv = cand[np.arange(n_test), s.argmin(1)]

    res = {}
    for ens_name, ens in [("ladder", ladder), ("peers", peers)]:
        for cond, Xd in [("natural", nat), ("adversarial", adv)]:
            miss = np.stack([c.predict(Xd) == 0 for c in ens], 1)
            r, q, pall, ind, com = rho_hat(miss)
            boots = []
            for _ in range(n_boot):
                b = rng.integers(0, len(miss), len(miss))
                boots.append(rho_hat(miss[b])[0])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            # pairwise rho
            pw = []
            m = miss.shape[1]
            for a in range(m):
                for c in range(a + 1, m):
                    pw.append(rho_hat(miss[:, [a, c]])[0])
            res[f"{ens_name}/{cond}"] = dict(
                detect=[float(1 - x) for x in q], p_allmiss=float(pall),
                p_allmiss_indep=float(ind), p_allmiss_comono=float(com),
                rho_hat=float(r), rho_ci=[float(lo), float(hi)],
                rho_pairwise_mean=float(np.nanmean(pw)))
    # accuracy of overseers on the natural distribution
    Xv, yv = sample(rng, 20000)
    res["accuracy"] = {"ladder": [float((c.predict(Xv) == yv).mean()) for c in ladder],
                       "peers": [float((c.predict(Xv) == yv).mean()) for c in peers],
                       "surrogate": float((surrogate.predict(Xv) == yv).mean()),
                       "base_rate_defective": float(yv.mean())}
    return res


if __name__ == "__main__":
    out = {}
    seeds = [0, 1, 2]
    for s in seeds:
        out[f"seed{s}"] = run(seed=s)
        print(json.dumps(out[f"seed{s}"], indent=1), flush=True)
    with open(sys.argv[1] if len(sys.argv) > 1 else "../results/toy_rho.json", "w") as fh:
        json.dump(out, fh, indent=1)
