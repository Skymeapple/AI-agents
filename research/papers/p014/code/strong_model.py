"""Revision round 1: a stronger amortized reasoner and a feature ablation.

The DeepSets test subject of the main runs is weak (obs_p0: 74.6% vs a label-free Bayes
bound of ~90.6%). Here we train a 'pooled-moment' amortized model: the same encoded input
as twins.py (standardized samples, query pair first, intervention indicators, label-order
bit), reduced to fixed pooled statistics, followed by a wider MLP trained on many more
fresh items, 30 epochs over a fixed set of 200k items (the DeepSets saw 256k fresh items once). Two feature sets:
  m2 : second-order moments over all rows, over observational rows and over
       interventional rows, plus the intervention-target position (sufficient for
       standardized Gaussian data with a known target);
  m4 : m2 plus per-variable 3rd/4th moments and pairwise E[x_a^3 x_b], E[x_a^2 x_b^2]
       (the cumulants that carry LiNGAM direction information).
Evaluation uses exactly the items of the DeepSets seed-s runs (same RNG streams as
twins.py), so i.i.d. accuracies are paired with the Bayes baselines in
results/bayes_oracles.json and twin kappas with the DeepSets kappas.

Usage: python3 strong_model.py <regime> <featset> <seed>
Output: results/strong/<regime>_<featset>_s<seed>.json
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import scm as S
import twins as T

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "strong")
REG = {"obs_p0": dict(p_order=0.0, family="gauss", n_int=0),
       "obs_p09": dict(p_order=0.9, family="gauss", n_int=0),
       "obs_p1": dict(p_order=1.0, family="gauss", n_int=0),
       "laplace_p0": dict(p_order=0.0, family="laplace", n_int=0),
       "int32_p0": dict(p_order=0.0, family="gauss", n_int=32),
       "int128_p0": dict(p_order=0.0, family="gauss", n_int=128)}
IU = T.IU
OFF = [(a, b) for a in range(4) for b in range(4) if a != b]


def pooled(F, featset):
    """F: [B, N, 18] encoded samples (4 values, 10 products, 4 indicators)."""
    X, P, I = F[:, :, :4], F[:, :, 4:14], F[:, :, 14:18]
    introw = (I.sum(2, keepdim=True) > 0).float()
    obsrow = 1 - introw
    def m(Z, w):
        return (Z * w).sum(1) / w.sum(1).clamp(min=1.0)
    feats = [P.mean(1), m(P, obsrow), m(P, introw), m(I, introw), introw.mean(1)]
    if featset == "m4":
        feats += [(X ** 3).mean(1), (X ** 4).mean(1) - 3,
                  torch.stack([(X[:, :, a] ** 3 * X[:, :, b]).mean(1) for a, b in OFF], 1),
                  torch.stack([(X[:, :, a] ** 2 * X[:, :, b] ** 2).mean(1) for a, b in zip(*IU) if a < b], 1)]
    return torch.cat(feats, 1)


class Head(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + 1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, Z, O):
        return self.f(torch.cat([Z, O], 1)).squeeze(1)


def main():
    regime, featset, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    torch.manual_seed(seed); torch.set_num_threads(1)
    cfg = dict(REG[regime], graph=False, corrupt=0.0)
    t0 = time.time()
    rng = np.random.default_rng(900_000 + seed)       # training stream, disjoint from eval
    Zs, Os, Ys = [], [], []
    for _ in range(n_train // 1000):
        F, G, O, Y, _ = T.make_iid_batch(rng, 1000, cfg)
        Zs.append(pooled(F, featset)); Os.append(O); Ys.append(Y)
    Z, O, Y = torch.cat(Zs), torch.cat(Os), torch.cat(Ys)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    gen_time = time.time() - t0
    net = Head(Z.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    epochs, bs = 30, 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(Z) // bs))
    lossf = nn.BCEWithLogitsLoss()
    Zn = (Z - mu) / sd
    for ep in range(epochs):
        perm = torch.randperm(len(Z))
        for s in range(0, len(Z) - bs + 1, bs):
            idx = perm[s:s + bs]
            loss = lossf(net(Zn[idx], O[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()

    @torch.no_grad()
    def pred(F, O):
        return torch.sigmoid(net((pooled(F, featset) - mu) / sd, O)).numpy()

    res = {"regime": regime, "featset": featset, "seed": seed, "n_train": n_train,
           "gen_time_s": gen_time, "final_train_loss": float(loss.detach())}
    mect = S.MECTable()
    # i.i.d.: identical items to twins.py seed-`seed` evaluation
    erng = np.random.default_rng(10_000 + seed)
    F, G, O_, Yt, meta = T.make_iid_batch(erng, 4000, cfg)
    p = pred(F, O_); Yt = Yt.numpy().astype(int)
    corr = (p > 0.5).astype(int) == Yt
    ident = np.array([mect.query(A, i, j)[0] for A, i, j in meta])
    res["iid"] = {"iid_acc": float(corr.mean()), "iid_acc_identified": float(corr[ident].mean()),
                  "iid_acc_unidentified": float(corr[~ident].mean())}
    # twins: the matched probe continues the same stream as twins.py
    F, G, O_, Yt, orc = T.make_twin_pairs(erng, 1500, cfg["family"], cfg["n_int"], False)
    p = pred(F, O_); r = T.twin_metrics(p, Yt, orc); r.update(T.kappa_ci(p, Yt, orc))
    res["twins_matched"] = r
    if cfg["family"] == "gauss" and cfg["n_int"] == 0:
        # unit test (Prop. 1) plus cross-regime Laplace probe
        F, G, O_, Yt, orc = T.make_twin_pairs(erng, 1500, "laplace", 0, False)
        p = pred(F, O_); r = T.twin_metrics(p, Yt, orc); r.update(T.kappa_ci(p, Yt, orc))
        res["twins_laplace"] = r
    else:
        rrng = np.random.default_rng(20_000 + seed)
        for kname, kind, inp in [("rev_cov_all", "cov", False), ("rev_cov_inprior", "cov", True),
                                 ("rev_mec_all", "mec", False), ("rev_mec_inprior", "mec", True)]:
            F, G, O_, Yt, orc, adj, ys = T.make_general_twins(rrng, 1500, cfg["family"], cfg["n_int"], inp, kind)
            p = pred(F, O_); r = T.twin_metrics(p, Yt, orc); r.update(T.kappa_ci(p, Yt, orc))
            pc = ((p > 0.5).astype(int) == Yt).reshape(-1, 2).mean(1); oc = orc.reshape(-1, 2).mean(1)
            for lab, msk in [("adjacent", adj == 1), ("nonadjacent", adj == 0)]:
                if msk.any():
                    r[lab] = {"n_pairs": int(msk.sum()), "twin_acc": float(pc[msk].mean()),
                              "oracle_ceiling": float(oc[msk].mean())}
            res[kname] = r
    if cfg["family"] == "gauss" and cfg["n_int"] == 0 and cfg["p_order"] > 0:
        # cue removal: same model on random-label items
        c0 = dict(cfg, p_order=0.0)
        F, G, O_, Yt, meta = T.make_iid_batch(np.random.default_rng(50_000), 4000, c0)
        p = pred(F, O_); Yt = Yt.numpy().astype(int); corr = (p > 0.5).astype(int) == Yt
        ident = np.array([mect.query(A, i, j)[0] for A, i, j in meta])
        res["cue_removed_random_labels"] = {"iid_acc": float(corr.mean()),
                                            "iid_acc_unidentified": float(corr[~ident].mean())}
    res["total_time_s"] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, f"{regime}_{featset}_s{seed}.json"), "w"), indent=1)
    print(regime, featset, seed, res["iid"], {k: (v.get("twin_acc"), v.get("kappa")) for k, v in res.items()
                                              if isinstance(v, dict) and "twin_acc" in v}, flush=True)


if __name__ == "__main__":
    main()
