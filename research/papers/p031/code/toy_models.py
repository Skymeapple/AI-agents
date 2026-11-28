"""Model-level toy measurement of overseer-miss dependence rho (round-1 revision).

Unlike toy_rho.py (overseers classify individual inputs), here each overseer
judges a TRAINED MODEL and decides whether it is backdoored ('misaligned').
The unit of observation is one draw of a misaligned model, which is what the
BOS parameter rho refers to (dependence of overseer misses over draws of
misaligned models).

Policy models: one-hidden-layer tanh MLPs (d=8 -> 16 -> 1), trained in batches
with full-batch Adam on their own sample of a fixed binary task.
  clean model     : trained on clean data + benign 'trigger-like' augmentation
                    (points with 2 features set to a large value, CORRECT labels)
  backdoored model: same, but augmentation points are labelled 1 (the backdoor
                    target). Trigger features, trigger value and poison rate are
                    random per model. A backdoored model counts as misaligned only
                    if its attack success rate (ASR) on triggered inputs whose clean
                    label is 0 is >= 0.5.
Overseers: meta-classifiers. Overseer j has its own query set Q_j (q_j inputs,
half Gaussian, half with 2 random features set large) and its own pool of
shadow models; it fits logistic regression on the model's logits at Q_j and
flags a model if its score exceeds the 95th percentile of scores on held-out
clean shadow models (target false-positive rate 5%).
  ladder : q = 4, 16, 64, 256          (capability ladder)
  peers  : q = 32 x 4                  (independent query sets and shadow pools)
Conditions: natural draws of misaligned models, and adversarial best-of-K
(the developer trains K backdoored models and submits the one a stronger
surrogate auditor, q=512 with a double-size shadow pool, finds least suspicious).
Run: python3 toy_models.py [seeds...]   -> ../results/toy_models.json
"""
import json, sys, time
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D, H, N0, AMAX = 8, 16, 400, 60


def truth(X):
    f = X[..., 0] + 0.8 * X[..., 1] * X[..., 2] + 0.6 * np.sin(2 * X[..., 3]) + 0.4 * X[..., 4]
    return (f > 0.3).astype(float)


def apply_trigger(X, feats, val):
    """X (B,M,D); feats (B,2) ints; val (B,) floats."""
    X = X.copy()
    B = X.shape[0]
    for c in range(feats.shape[1]):
        X[np.arange(B), :, feats[:, c]] = val[:, None]
    return X


def train_models(rng, B, backdoor, epochs=250, lr=0.02):
    """Train B models in one batch. Returns params and per-model metadata."""
    X = rng.standard_normal((B, N0 + AMAX, D))
    feats = np.stack([rng.choice(D, 2, replace=False) for _ in range(B)])
    val = rng.uniform(2.5, 4.0, B)
    prate = np.exp(rng.uniform(np.log(0.02), np.log(0.15), B))
    na = np.round(prate * N0).astype(int)
    aug = apply_trigger(X[:, N0:], feats, val)
    X[:, N0:] = aug
    y = truth(X)
    if backdoor:
        y[:, N0:] = 1.0
    w = np.ones((B, N0 + AMAX))
    w[:, N0:] = (np.arange(AMAX)[None, :] < na[:, None]).astype(float)
    w /= w.sum(1, keepdims=True)
    W1 = rng.standard_normal((B, D, H)) / np.sqrt(D); b1 = np.zeros((B, H))
    W2 = rng.standard_normal((B, H)) / np.sqrt(H); b2 = np.zeros(B)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
    X = X.astype(np.float32); y = y.astype(np.float32); w = w.astype(np.float32)
    params = [q.astype(np.float32) for q in params]
    m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
    XT = X.transpose(0, 2, 1).copy()
    for t in range(1, epochs + 1):
        W1, b1, W2, b2 = params
        A = np.tanh(np.matmul(X, W1) + b1[:, None, :])
        logit = np.matmul(A, W2[:, :, None])[..., 0] + b2[:, None]
        pr = 1 / (1 + np.exp(-logit))
        g = (pr - y) * w                                  # dL/dlogit
        gW2 = np.matmul(g[:, None, :], A)[:, 0, :] + 1e-4 * W2
        gb2 = g.sum(1)
        gA = g[:, :, None] * W2[:, None, :] * (1 - A * A)
        gW1 = np.matmul(XT, gA) + 1e-4 * W1
        gb1 = gA.sum(1)
        for i, gr in enumerate([gW1, gb1, gW2, gb2]):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            params[i] = params[i] - lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    params = [q.astype(np.float64) for q in params]
    return dict(params=params, feats=feats, val=val, prate=prate)


def logits(model, Q):
    """Q (M,D) shared queries -> (B,M) logits."""
    W1, b1, W2, b2 = model["params"]
    A = np.tanh(np.einsum("md,bdh->bmh", Q, W1) + b1[:, None, :])
    return np.einsum("bmh,bh->bm", A, W2) + b2[:, None]


def model_metrics(rng, model):
    W1 = model["params"][0]; B = W1.shape[0]
    Xc = rng.standard_normal((1000, D))
    acc = ((logits(model, Xc) > 0) == truth(Xc)[None, :]).mean(1)
    Xt = rng.standard_normal((B, 400, D))
    Xt = apply_trigger(Xt, model["feats"], model["val"])
    yt = truth(Xt)
    W1, b1, W2, b2 = model["params"]
    A = np.tanh(np.einsum("bnd,bdh->bnh", Xt, W1) + b1[:, None, :])
    lg = np.einsum("bnh,bh->bn", A, W2) + b2[:, None]
    asr = np.array([(lg[b][yt[b] == 0] > 0).mean() if (yt[b] == 0).any() else np.nan for b in range(B)])
    return acc, asr


def concat(models):
    out = dict(params=[np.concatenate([m["params"][i] for m in models]) for i in range(4)])
    for k in ["feats", "val", "prate"]:
        out[k] = np.concatenate([m[k] for m in models])
    return out


def subset(model, idx):
    out = dict(params=[q[idx] for q in model["params"]])
    for k in ["feats", "val", "prate"]:
        out[k] = model[k][idx]
    return out


def make_pool(rng, n, backdoor, batch=500):
    """n models; for backdoored pools keep only successful backdoors (ASR >= 0.5)."""
    got, tried, accs = [], 0, []
    while sum(g["val"].size for g in got) < n:
        mdl = train_models(rng, batch, backdoor)
        acc, asr = model_metrics(rng, mdl)
        tried += batch
        keep = np.flatnonzero(asr >= 0.5) if backdoor else np.arange(batch)
        got.append(subset(mdl, keep)); accs.append(acc[keep])
    pool = concat(got)
    pool = subset(pool, np.arange(n))
    return pool, dict(tried=tried, kept=int(sum(g["val"].size for g in got)), clean_acc=float(np.concatenate(accs).mean()))


def queries(rng, q):
    Q = rng.standard_normal((q, D))
    h = q // 2
    for r in range(h, q):
        fs = rng.choice(D, 2, replace=False)
        Q[r, fs] = rng.uniform(2.5, 4.0, 2)
    return Q


class Overseer:
    def __init__(self, rng, q, n_shadow, pools):
        self.Q = queries(rng, q)
        clean, bd = pools
        n_tr = n_shadow
        Xc = logits(subset(clean, np.arange(n_tr)), self.Q)
        Xb = logits(subset(bd, np.arange(n_tr)), self.Q)
        Xcal = logits(subset(clean, np.arange(n_tr, n_tr + 200)), self.Q)
        X = np.vstack([Xc, Xb]); y = np.r_[np.zeros(n_tr), np.ones(n_tr)]
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=0.3, max_iter=3000).fit(self.sc.transform(X), y)
        self.thr = np.quantile(self.score_X(Xcal), 0.95)

    def score_X(self, X):
        return self.clf.decision_function(self.sc.transform(X))

    def score(self, model):
        return self.score_X(logits(model, self.Q))

    def flags(self, model):
        return self.score(model) > self.thr


def rho_hat(miss):
    q = miss.mean(0); pall = miss.all(1).mean(); ind = np.prod(q); com = q.min()
    den = com - ind
    return ((pall - ind) / den if den > 1e-9 else np.nan), q, pall, ind, com


def subset_fit(miss):
    """Observed all-miss for every subset of size >= 2 vs the one-parameter mixture."""
    m = miss.shape[1]
    rows = []
    for s in range(2, m + 1):
        for S in combinations(range(m), s):
            r, q, pall, ind, com = rho_hat(miss[:, list(S)])
            rows.append(dict(S=list(S), rho=float(r), pall=float(pall), ind=float(ind), com=float(com)))
    # least-squares single rho over all subsets
    A = np.array([r["com"] - r["ind"] for r in rows]); b = np.array([r["pall"] - r["ind"] for r in rows])
    rho_ls = float(A @ b / (A @ A))
    pred = np.array([r["ind"] + rho_ls * (r["com"] - r["ind"]) for r in rows])
    obs = np.array([r["pall"] for r in rows])
    return dict(subsets=rows, rho_ls=rho_ls, max_abs_resid=float(np.max(np.abs(pred - obs))),
                rel_rmse=float(np.sqrt(np.mean(((pred - obs) / obs) ** 2))),
                pairwise=[r["rho"] for r in rows if len(r["S"]) == 2])


def run(seed, n_shadow=250, n_test_bd=800, n_test_clean=400, K=5, n_adv=400, n_boot=1000):
    t0 = time.time()
    rng = np.random.default_rng(1000 + seed)
    info = {}
    # pools
    def shadow(n):
        c, ic = make_pool(rng, n + 200, False); b, ib = make_pool(rng, n, True)
        return (c, b), ib
    ladder_q, peer_q = [4, 16, 64, 256], [32, 32, 32, 32]
    overseers = {}
    for name, qs in [("ladder", ladder_q), ("peers", peer_q)]:
        ov = []
        for q in qs:
            pools, ib = shadow(n_shadow)
            ov.append(Overseer(rng, q, n_shadow, pools))
            info.setdefault("backdoor_success_rate", []).append(ib["kept"] / ib["tried"])
        overseers[name] = ov
    spools, _ = shadow(2 * n_shadow)
    surrogate = Overseer(rng, 512, 2 * n_shadow, spools)
    test_bd, ib = make_pool(rng, n_test_bd, True)
    test_clean, ic = make_pool(rng, n_test_clean, False)
    info["clean_acc_clean_models"] = ic["clean_acc"]; info["clean_acc_backdoored_models"] = ib["clean_acc"]
    cand, _ = make_pool(rng, n_adv * K, True)
    s = surrogate.score(cand).reshape(n_adv, K)
    adv = subset(cand, np.arange(n_adv) * K + s.argmin(1))
    info["train_time_s"] = time.time() - t0
    res = dict(info=info)
    for ens, ov in overseers.items():
        res[f"{ens}/fp_rate"] = [float(o.flags(test_clean).mean()) for o in ov]
        for cond, mdl in [("natural", test_bd), ("adversarial", adv)]:
            miss = np.stack([~o.flags(mdl) for o in ov], 1)
            r, q, pall, ind, com = rho_hat(miss)
            boots = [rho_hat(miss[rng.integers(0, len(miss), len(miss))])[0] for _ in range(n_boot)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            fit = subset_fit(miss)
            best = int(np.argmin(q))  # most sensitive overseer
            others = [j for j in range(miss.shape[1]) if j != best]
            mb = miss[:, best]
            rescue = float((~miss[mb][:, others].all(1)).mean()) if mb.any() else float("nan")
            rescue_ind = float(1 - np.prod(q[others]))
            res[f"{ens}/{cond}"] = dict(detect=[float(1 - x) for x in q], p_allmiss=float(pall),
                                        p_allmiss_indep=float(ind), p_allmiss_comono=float(com),
                                        rho_hat=float(r), rho_ci=[float(lo), float(hi)],
                                        pairwise_rho=fit["pairwise"], rho_ls_all_subsets=fit["rho_ls"],
                                        mixture_max_abs_resid=fit["max_abs_resid"], mixture_rel_rmse=fit["rel_rmse"],
                                        rescue_given_best_misses=rescue, rescue_if_independent=rescue_ind,
                                        n=int(len(miss)))
    # what makes a backdoor stealthy? correlate all-miss with poison rate (ladder, natural)
    miss = np.stack([~o.flags(test_bd) for o in overseers["ladder"]], 1)
    allm = miss.all(1)
    res["ladder/natural"]["prate_median_allmiss"] = float(np.median(test_bd["prate"][allm])) if allm.any() else None
    res["ladder/natural"]["prate_median_caught"] = float(np.median(test_bd["prate"][~allm]))
    res["info"]["total_time_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    # python3 toy_models.py 0 1 2     -> one file per seed (run seeds in parallel if desired)
    # python3 toy_models.py merge     -> ../results/toy_models.json
    if sys.argv[1:] == ["merge"]:
        import glob
        out = {}
        for fn in sorted(glob.glob("../results/toy_models_seed*.json")):
            out.update(json.load(open(fn)))
        json.dump(out, open("../results/toy_models.json", "w"), indent=1, default=float)
        sys.exit()
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    for sd in seeds:
        r = run(sd)
        json.dump({f"seed{sd}": r}, open(f"../results/toy_models_seed{sd}.json", "w"), indent=1, default=float)
        print(sd, json.dumps(r, default=float)[:1500], flush=True)
