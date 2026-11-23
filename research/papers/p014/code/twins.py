"""Train amortized causal-query models (DeepSets over samples) under different
information regimes and evaluate them on i.i.d. queries and on observational twins.

Usage:  python twins.py --name obs_p0 --p_order 0.0 [--family laplace]
        [--n_int 32] [--graph] [--steps 4000] [--seed 0]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

import scm as S

N = 256
n = S.N_NODES
IU = np.triu_indices(n)
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


# ----------------------------------------------------------------------------
# Encoding a (dataset, query) item
# ----------------------------------------------------------------------------
def encode(X, I, A_desc, i, j):
    perm = [i, j] + [k for k in range(n) if k not in (i, j)]
    X = X[:, perm]; I = I[:, perm]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    prods = (X[:, :, None] * X[:, None, :])[:, IU[0], IU[1]]
    feats = np.concatenate([X, prods, I], 1).astype(np.float32)
    G = A_desc[np.ix_(perm, perm)].reshape(-1).astype(np.float32)
    ordbit = np.float32(1.0 if i < j else 0.0)
    return feats, G, ordbit


def describe(rng, A, corrupt):
    """Textual graph description (adjacency); each edge reversed w.p. `corrupt`."""
    D = A.copy()
    if corrupt > 0:
        for u, v in zip(*np.nonzero(A)):
            if rng.random() < corrupt:
                D[u, v] = 0; D[v, u] = 1
    return D


def pick_query(rng, A):
    R = S.ancestors_matrix(A)
    pos = [(a, b) for a in range(n) for b in range(n) if a != b and R[a, b]]
    neg = [(a, b) for a in range(n) for b in range(n) if a != b and not R[a, b]]
    pool = pos if (pos and (rng.random() < 0.5 or not neg)) else neg
    a, b = pool[rng.integers(len(pool))]
    return int(a), int(b), int(R[a, b])


def make_iid_batch(rng, B, cfg):
    F, G, O, Y, meta = [], [], [], [], []
    for _ in range(B):
        m = S.random_scm(rng, p_order=cfg["p_order"])
        tgt = int(rng.integers(n))
        X, I = S.sample(rng, m, N, cfg["family"], cfg["n_int"], tgt)
        i, j, y = pick_query(rng, m["A"])
        D = describe(rng, m["A"], cfg["corrupt"]) if cfg["graph"] else np.zeros((n, n), int)
        f, g, o = encode(X, I, D, i, j)
        F.append(f); G.append(g); O.append(o); Y.append(y); meta.append((m["A"], i, j))
    return (torch.tensor(np.stack(F)), torch.tensor(np.stack(G)),
            torch.tensor(np.array(O))[:, None], torch.tensor(np.array(Y), dtype=torch.float32), meta)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SetNet(nn.Module):
    def __init__(self, d_in=n + len(IU[0]) + n, h=128, use_graph=False):
        super().__init__()
        self.use_graph = use_graph
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h + 1 + n * n, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, 1))

    def forward(self, F, G, O):
        z = self.phi(F).mean(1)
        if not self.use_graph:
            G = torch.zeros_like(G)
        return self.rho(torch.cat([z, O, G], 1)).squeeze(1)


# ----------------------------------------------------------------------------
# Evaluation sets
# ----------------------------------------------------------------------------
def make_twin_pairs(rng, K, family, n_int, graph, corrupt=0.0, conflict=False):
    """K observational-twin pairs. Item A: data from M, query (u,v), answer 1.
    Item B: data from twin M' (u->v reversed), same query, answer 0.
    Also returns the likelihood-ratio (Neyman-Pearson) oracle correctness."""
    items, oracle = [], []
    k = 0
    while k < K:
        m = S.random_scm(rng, p_order=0.0)
        ce = S.covered_edges(m["A"])
        if not ce:
            continue
        u, v = ce[rng.integers(len(ce))]
        m2 = S.twin_by_reversal(m, u, v)
        tgt = int(rng.integers(n))
        for model, other, y in ((m, m2, 1), (m2, m, 0)):
            X, I = S.sample(rng, model, N, family, n_int, tgt)
            src = (other if conflict else model)["A"]
            D = describe(rng, src, corrupt) if graph else np.zeros((n, n), int)
            f, g, o = encode(X, I, D, u, v)
            items.append((f, g, o, y))
            l_true, l_other = S.loglik(X, I, model, family), S.loglik(X, I, other, family)
            if abs(l_true - l_other) <= 1e-7 * abs(l_true):
                oracle.append(0.5)
            else:
                oracle.append(float(l_true > l_other))
        k += 1
    F = torch.tensor(np.stack([it[0] for it in items]))
    G = torch.tensor(np.stack([it[1] for it in items]))
    O = torch.tensor(np.array([it[2] for it in items]))[:, None]
    Y = np.array([it[3] for it in items])
    return F, G, O, Y, np.array(oracle)


def make_general_twins(rng, K, family, n_int, inprior, kind):
    """Revision probe. kind='cov': covered-edge twins as in make_twin_pairs, but the
    queried direction is randomised ((u,v) or (v,u)) so that 'answer 1' is not tied to
    the in-prior member. kind='mec': any MEC-unidentified query (adjacent or not) and
    a uniformly chosen Markov-equivalent DAG with the opposite answer, refit by
    population regression. inprior=True rejects pairs whose refit twin leaves the
    training prior's support (so prior-support detection cannot separate them).
    Item order per pair: (data from M, answer y_M), (data from M', answer 1-y_M)."""
    items, oracle, adj, ys = [], [], [], []
    k = 0
    while k < K:
        m = S.random_scm(rng, p_order=0.0)
        if kind == "cov":
            ce = S.covered_edges(m["A"])
            if not ce:
                continue
            u, v = ce[rng.integers(len(ce))]
            m2 = S.twin_by_reversal(m, u, v)
            qi, qj = (u, v) if rng.random() < 0.5 else (v, u)
        else:
            R = S.ancestors_matrix(m["A"])
            mem = S.mec_members(m["A"])
            Rs = np.stack([S.ancestors_matrix(B) for B in mem])
            frac = Rs.mean(0)
            want = int(rng.random() < 0.5)
            cand = [(a, b) for a in range(n) for b in range(n)
                    if a != b and 0 < frac[a, b] < 1 and R[a, b] == want]
            if not cand:
                continue
            qi, qj = cand[rng.integers(len(cand))]
            opp = [B for B, RB in zip(mem, Rs) if RB[qi, qj] != want]
            m2 = S.refit_on(m, opp[rng.integers(len(opp))])
        if inprior and not S.in_prior(m2):
            continue
        yM = int(S.ancestors_matrix(m["A"])[qi, qj])
        assert yM != int(S.ancestors_matrix(m2["A"])[qi, qj])
        tgt = int(rng.integers(n))
        for model, other, y in ((m, m2, yM), (m2, m, 1 - yM)):
            X, I = S.sample(rng, model, N, family, n_int, tgt)
            f, g, o = encode(X, I, np.zeros((n, n), int), qi, qj)
            items.append((f, g, o, y))
            l_true, l_other = S.loglik(X, I, model, family), S.loglik(X, I, other, family)
            oracle.append(0.5 if abs(l_true - l_other) <= 1e-7 * abs(l_true) else float(l_true > l_other))
        adj.append(int(m["A"][qi, qj] or m["A"][qj, qi])); ys.append(yM)
        k += 1
    F = torch.tensor(np.stack([it[0] for it in items]))
    G = torch.tensor(np.stack([it[1] for it in items]))
    O = torch.tensor(np.array([it[2] for it in items]))[:, None]
    Y = np.array([it[3] for it in items])
    return F, G, O, Y, np.array(oracle), np.array(adj), np.array(ys)


def kappa_ci(p, Y, oracle, B=2000, seed=321):
    """Paired bootstrap (over twin pairs) for twin acc, LR ceiling and kappa."""
    corr = ((p > 0.5).astype(int) == Y).reshape(-1, 2).mean(1)
    orc = oracle.reshape(-1, 2).mean(1)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        idx = rng.integers(len(corr), size=len(corr))
        a, c = corr[idx].mean(), orc[idx].mean()
        ks.append((2 * a - 1) / (2 * c - 1) if c > 0.5 + 1e-9 else np.nan)
    ks = np.array(ks)
    c0 = orc.mean()
    k0 = (2 * corr.mean() - 1) / (2 * c0 - 1) if c0 > 0.5 + 1e-9 else None
    return {"kappa": k0, "kappa_ci": [float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))]
            if k0 is not None else None}


@torch.no_grad()
def predict(net, F, G, O, bs=500):
    out = []
    for s in range(0, len(F), bs):
        out.append(torch.sigmoid(net(F[s:s + bs], G[s:s + bs], O[s:s + bs])).numpy())
    return np.concatenate(out)


def wilson(k, m, z=1.96):
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return [float(c - h), float(c + h)]


def twin_metrics(p, Y, oracle):
    yhat = (p > 0.5).astype(int)
    corr = (yhat == Y)
    pair_both = corr[0::2] & corr[1::2]
    acc = corr.mean()
    # paired bootstrap over twin pairs
    rng = np.random.default_rng(123)
    pc = corr.reshape(-1, 2).mean(1)
    boots = [pc[rng.integers(len(pc), size=len(pc))].mean() for _ in range(2000)]
    return {"twin_acc": float(acc), "twin_acc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "causal_surplus": float(2 * acc - 1), "pair_both_correct": float(pair_both.mean()),
            "oracle_ceiling": float(oracle.mean()), "n_pairs": int(len(Y) // 2),
            "mean_p_yes_on_A": float(p[0::2].mean()), "mean_p_yes_on_B": float(p[1::2].mean()),
            "mean_abs_p_diff_within_pair": float(np.abs(p[0::2] - p[1::2]).mean())}


def iid_metrics(net, rng, cfg, mect, K=4000):
    F, G, O, Y, meta = make_iid_batch(rng, K, cfg)
    p = predict(net, F, G, O)
    Y = Y.numpy().astype(int)
    corr = ((p > 0.5).astype(int) == Y)
    ident = np.array([mect.query(A, i, j)[0] for A, i, j in meta])
    # Oracle baselines that see the population MEC (all that observational
    # Gaussian data can reveal) and optionally the variable labels.
    mec_unif, mec_label = [], []
    for (A, i, j), y in zip(meta, Y):
        idf, frac = mect.query(A, i, j)
        mec_unif.append(float(frac > 0.5) == y if frac != 0.5 else 0.5)
        skel = ((A + A.T) > 0).astype(int)
        L = np.triu(skel, 1)                      # orient every edge by label order
        if S.mec_key(L) == S.mec_key(A):
            mec_label.append(float(S.ancestors_matrix(L)[i, j] == y))
        else:
            mec_label.append(float(frac > 0.5) == y if frac != 0.5 else 0.5)
    return {"iid_acc": float(corr.mean()), "iid_acc_ci": wilson(corr.sum(), len(corr)),
            "frac_identified": float(ident.mean()),
            "iid_acc_identified": float(corr[ident].mean()),
            "iid_acc_unidentified": float(corr[~ident].mean()) if (~ident).any() else None,
            "oracle_mec_uniform_acc": float(np.mean(mec_unif)),
            "oracle_mec_labelorder_acc": float(np.mean(mec_label))}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--p_order", type=float, default=0.0)
    ap.add_argument("--family", default="gauss")
    ap.add_argument("--n_int", type=int, default=0)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--corrupt", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twin_pairs", type=int, default=1500)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(a.seed)
    cfg = dict(p_order=a.p_order, family=a.family, n_int=a.n_int, graph=a.graph, corrupt=a.corrupt)
    net = SetNet(use_graph=a.graph)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time(); curve = []
    for step in range(a.steps):
        F, G, O, Y, _ = make_iid_batch(rng, a.batch, cfg)
        loss = lossf(net(F, G, O), Y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0:
            curve.append((step, float(loss.detach())))
    train_time = time.time() - t0
    net.eval()
    mect = S.MECTable()
    erng = np.random.default_rng(10_000 + a.seed)
    res = {"name": a.name, "cfg": cfg, "steps": a.steps, "seed": a.seed,
           "train_time_s": train_time, "loss_curve": curve}
    res["iid"] = iid_metrics(net, erng, cfg, mect)
    # twin evaluations: matched regime, plus cross-regime probes
    evals = {"twins_matched": dict(family=a.family, n_int=a.n_int, graph=a.graph)}
    if a.family == "laplace":
        evals["twins_gauss"] = dict(family="gauss", n_int=a.n_int, graph=a.graph)
    if a.family == "gauss" and a.n_int == 0 and not a.graph:
        evals["twins_laplace"] = dict(family="laplace", n_int=0, graph=False)
    if a.graph:
        evals["twins_corrupt25"] = dict(family=a.family, n_int=a.n_int, graph=True, corrupt=0.25)
        evals["twins_conflict"] = dict(family=a.family, n_int=a.n_int, graph=True, conflict=True)
    for k, e in evals.items():
        F, G, O, Y, orc = make_twin_pairs(erng, a.twin_pairs, **e)
        p = predict(net, F, G, O)
        res[k] = twin_metrics(p, Y, orc)
        if e.get("conflict"):
            # under conflict, the description is of the *other* twin: report how
            # often the model follows the description instead of the data.
            res[k]["follows_description"] = float(1 - res[k]["twin_acc"])
    # ---- revision probes (separate RNG: earlier fields stay bit-identical)
    preds = {}
    if a.family == "laplace" or a.n_int > 0:
        rrng = np.random.default_rng(20_000 + a.seed)
        for kname, kind, inp in [("rev_cov_all", "cov", False), ("rev_cov_inprior", "cov", True),
                                 ("rev_mec_all", "mec", False), ("rev_mec_inprior", "mec", True)]:
            F, G, O, Y, orc, adj, ys = make_general_twins(rrng, a.twin_pairs, a.family, a.n_int, inp, kind)
            p = predict(net, F, G, O)
            r = twin_metrics(p, Y, orc); r.update(kappa_ci(p, Y, orc))
            pc = ((p > 0.5).astype(int) == Y).reshape(-1, 2).mean(1)
            oc = orc.reshape(-1, 2).mean(1)
            for lab, msk in [("adjacent", adj == 1), ("nonadjacent", adj == 0),
                             ("inprior_is_yes", ys == 1), ("inprior_is_no", ys == 0)]:
                if msk.any():
                    r[lab] = {"n_pairs": int(msk.sum()), "twin_acc": float(pc[msk].mean()),
                              "oracle_ceiling": float(oc[msk].mean())}
            res[kname] = r
            preds[kname] = dict(p=p, Y=Y, orc=orc, adj=adj, ys=ys)
        # kappa CI for the original matched probe (re-evaluated on an independent draw)
        F, G, O, Y, orc = make_twin_pairs(np.random.default_rng(30_000 + a.seed), a.twin_pairs,
                                          a.family, a.n_int, a.graph)
        p = predict(net, F, G, O)
        r = twin_metrics(p, Y, orc); r.update(kappa_ci(p, Y, orc))
        res["rev_matched_redraw"] = r
    res["total_time_s"] = time.time() - t0
    mdir = os.path.join(RESULTS, "models"); os.makedirs(mdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(mdir, f"{a.name}_s{a.seed}.pt"))
    if preds:
        np.savez_compressed(os.path.join(mdir, f"{a.name}_s{a.seed}_revpreds.npz"),
                            **{f"{k}_{f}": v for k, d in preds.items() for f, v in d.items()})
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{a.name}_s{a.seed}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=1))


if __name__ == "__main__":
    main()
