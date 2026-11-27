"""Shared module for "mapping B": learning that survives a change of environment.

The user's framework claims that shifts between sequencing studies, wind farms, sky
regions, cities and markets are one problem (an environment variable confounds the
mechanism) and that domain-invariance methods from causal inference should transfer
across them. This module puts that claim to a test: the same methods, the same
leave-one-environment-out protocol, on any (X, y, env) table.

Methods (all regressors; the model family is fixed per call so only the method varies):
  erm            pooled fit, every row weighted equally
  env-balanced   every training environment gets the same total weight
  group-dro      reweights training environments towards the worst one (5 rounds,
                 weights proportional to exp(eta * environment loss))
  env-standardised  each feature standardised within its own environment, using only
                 that environment's inputs (no labels); the held-out environment is
                 standardised with its own inputs at test time
  vrex           neural net trained with a penalty on the variance of environment risks
  irm            neural net with the IRMv1 penalty (gradient of each environment's risk
                 with respect to a dummy output scale)
  erm-nn         the same neural net with no penalty (the reference for vrex and irm)

The tree methods use LightGBM; the neural ones a two-layer MLP in PyTorch. Penalty
weights are chosen by leave-one-environment-out inside the training environments only.

Score per held-out environment: RMSE. Reported: mean and worst over held-out
environments, and in how many environments each method beats erm.
"""
import numpy as np


def _gbm(X, y, w=None, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                          verbose=-1, n_jobs=4, random_state=seed)
    return m.fit(X, y, sample_weight=w)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def env_standardise(X, env):
    Z = np.empty_like(X, dtype=float)
    for e in np.unique(env):
        k = env == e
        mu, sd = X[k].mean(0), X[k].std(0)
        Z[k] = (X[k] - mu) / np.where(sd > 0, sd, 1)
    return Z


def tree_method(name, X, y, env, Xt, envt):
    envs = np.unique(env)
    if name == "erm":
        return _gbm(X, y).predict(Xt)
    if name == "env-balanced":
        w = np.zeros(len(y))
        for e in envs:
            w[env == e] = 1.0 / (env == e).sum()
        return _gbm(X, y, w * len(y) / len(envs)).predict(Xt)
    if name == "group-dro":
        q = np.ones(len(envs)) / len(envs)
        for _ in range(5):
            w = np.zeros(len(y))
            for i, e in enumerate(envs):
                w[env == e] = q[i] / (env == e).sum()
            m = _gbm(X, y, w * len(y))
            loss = np.array([np.mean((m.predict(X[env == e]) - y[env == e]) ** 2) for e in envs])
            q = q * np.exp(loss / loss.mean())
            q /= q.sum()
        return m.predict(Xt)
    if name == "env-standardised":
        return _gbm(env_standardise(X, env), y).predict(env_standardise(Xt, envt))
    raise ValueError(name)


def nn_method(name, X, y, env, Xt, lam, seed=0, epochs=60):
    import torch
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    ys_ = torch.tensor((y - ym) / ys, dtype=torch.float32)
    envs = np.unique(env)
    idx = [torch.tensor(np.where(env == e)[0]) for e in envs]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    batch = 512
    for ep in range(epochs):
        steps = max(1, min(len(i) for i in idx) // batch)
        for _ in range(steps):
            risks, pens = [], []
            for i in idx:
                b = i[torch.tensor(rng.integers(0, len(i), batch))]
                out = net(Xs[b]).squeeze(1)
                if name == "irm":
                    scale = torch.tensor(1.0, requires_grad=True)
                    r = ((out * scale - ys_[b]) ** 2).mean()
                    g = torch.autograd.grad(r, [scale], create_graph=True)[0]
                    pens.append(g ** 2)
                else:
                    r = ((out - ys_[b]) ** 2).mean()
                risks.append(r)
            R = torch.stack(risks)
            loss = R.mean()
            if name == "vrex":
                loss = loss + lam * R.var()
            if name == "irm" and ep >= epochs // 4:              # warm-up before the penalty
                loss = loss + lam * torch.stack(pens).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor((Xt - mu) / sd, dtype=torch.float32)).squeeze(1).numpy() * ys + ym


LAMS = {"vrex": (1.0, 10.0, 100.0), "irm": (1.0, 10.0, 100.0), "erm-nn": (0.0,)}


def choose_lam(name, X, y, env):
    """Penalty weight by leave-one-environment-out inside the training environments."""
    if len(LAMS[name]) == 1 or len(np.unique(env)) < 3:
        return LAMS[name][0]
    best = None
    for lam in LAMS[name]:
        errs = []
        for e in np.unique(env):
            k = env == e
            errs.append(_rmse(nn_method(name, X[~k], y[~k], env[~k], X[k], lam, epochs=30), y[k]))
        s = max(errs)                                              # worst-environment criterion
        if best is None or s < best[0]:
            best = (s, lam)
    return best[1]


TREE = ("erm", "env-balanced", "group-dro", "env-standardised")
NN = ("erm-nn", "vrex", "irm")


def compare(X, y, env, label, methods=TREE + NN, min_rows=200, log=print):
    X, y, env = np.asarray(X, float), np.asarray(y, float), np.asarray(env)
    envs = [e for e in np.unique(env) if (env == e).sum() >= min_rows]
    res = {m: {} for m in methods}
    for e in envs:
        k = env == e
        for m in methods:
            if m in TREE:
                p = tree_method(m, X[~k], y[~k], env[~k], X[k], env[k])
            else:
                lam = choose_lam(m, X[~k], y[~k], env[~k])
                p = nn_method(m, X[~k], y[~k], env[~k], X[k], lam)
            res[m][e] = _rmse(p, y[k])
        log(f"  [{label}] held out {e} ({k.sum()} rows): " + ", ".join(f"{m} {res[m][e]:.4g}" for m in methods))
    summary = {}
    for m in methods:
        v = np.array([res[m][e] for e in envs])
        ref_tree = np.array([res["erm"][e] for e in envs]) if "erm" in res else v
        ref = np.array([res["erm-nn"][e] for e in envs]) if m in ("vrex", "irm") and "erm-nn" in res else ref_tree
        summary[m] = dict(mean=float(v.mean()), worst=float(v.max()), beats_ref=int((v < ref - 1e-12).sum()), n=len(envs))
    log(f"RESULT [{label}] {len(envs)} held-out environments; mean / worst RMSE; environments better than its reference "
        f"(erm for tree methods, erm-nn for vrex and irm):")
    for m, s in summary.items():
        log(f"RESULT [{label}]   {m:17s} {s['mean']:.4g} / {s['worst']:.4g}   better in {s['beats_ref']}/{s['n']}")
    return res, summary
