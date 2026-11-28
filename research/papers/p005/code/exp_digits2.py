"""Experiment 6 (revision): digits with a stronger network (small CNN), several perturbations,
standard error-detection baselines, per-seed AUROC with confidence intervals, a
leave-one-seed-out test of whether hull information adds to the logit margin, and a
function-preserving 'padding' construction that changes C but not the network's function.
Usage: python3 exp_digits2.py -> results/digits2.json"""
import sys, os, json, time
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nn import MLP
from hull import hull_project, readout_certificate
from ckpt_metrics import visible_basis

RES = os.path.join(os.path.dirname(__file__), "..", "results")
NTR = 600
SEEDS = 5
PERTS = ["none", "right1", "right2", "down1", "noise0.25", "rot15"]


def perturb(Ximg, kind, rng):
    """Ximg: (n,8,8) in raw 0..16 units."""
    if kind == "none":
        return Ximg.copy()
    if kind.startswith("right"):
        dx = int(kind[-1]); out = np.zeros_like(Ximg); out[:, :, dx:] = Ximg[:, :, :-dx]; return out
    if kind == "down1":
        out = np.zeros_like(Ximg); out[:, 1:, :] = Ximg[:, :-1, :]; return out
    if kind.startswith("noise"):
        s = float(kind[5:]) * 16
        return np.clip(Ximg + rng.normal(0, s, Ximg.shape), 0, 16)
    if kind == "rot15":
        return np.clip(np.stack([rotate(im, 15, reshape=False, order=1, mode="constant") for im in Ximg]), 0, 16)
    raise ValueError(kind)


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = torch.nn.Linear(64 * 16, 128)
        self.out = torch.nn.Linear(128, 10)

    def feats(self, x):
        x = torch.relu(self.c1(x)); x = torch.relu(self.c2(x))
        x = torch.nn.functional.max_pool2d(x, 2).flatten(1)
        return torch.relu(self.f1(x))

    def forward(self, x):
        return self.out(self.feats(x))


def train_cnn(Xtr, ytr, seed):
    torch.manual_seed(seed)
    net = CNN()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor(Xtr, dtype=torch.float32).reshape(-1, 1, 8, 8); y = torch.tensor(ytr)
    g = torch.Generator().manual_seed(seed)
    for ep in range(60):
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), 32):
            idx = perm[i:i + 32]
            loss = torch.nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def model_fn(kind, net):
    if kind == "cnn":
        def f(Xn):
            with torch.no_grad():
                t = torch.tensor(Xn, dtype=torch.float32).reshape(-1, 1, 8, 8)
                F = net.feats(t); L = net.out(F)
            return L.double().numpy(), F.double().numpy()
        W = net.out.weight.detach().double().numpy().T; b = net.out.bias.detach().double().numpy()
    else:
        def f(Xn):
            L, reps = net.forward(Xn); return L, reps[-1]
        W, b = net.W[-1], net.b[-1]
    return f, W, b


def nch(Rtr, ytr, Rte, radius):
    D = np.zeros((len(Rte), 10))
    for c in range(10):
        Pc = Rtr[ytr == c]
        for i in range(len(Rte)):
            D[i, c], _ = hull_project(Rte[i], Pc)
    return D / radius


def scores_for(L, F, Ftr, ytr, W, b, D):
    n = len(L)
    pred = L.argmax(1)
    z = L - L.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    Ls = np.sort(L, 1)
    s = {}
    s["msp"] = -pr.max(1)
    s["logit_margin"] = -(Ls[:, -1] - Ls[:, -2])
    s["energy"] = -(L.max(1) + np.log(np.exp(z).sum(1)))
    mus = np.stack([Ftr[ytr == c].mean(0) for c in range(10)])
    dev = Ftr - mus[ytr]; S = dev.T @ dev / len(Ftr); Si = np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]) * np.trace(S) / S.shape[0])
    md = np.stack([np.einsum("ij,jk,ik->i", F - mus[c], Si, F - mus[c]) for c in range(10)], 1)
    s["mahalanobis"] = md.min(1)
    nf = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Ftn, Fn = nf(Ftr), nf(F)
    dist = np.sqrt(np.maximum(2 - 2 * Fn @ Ftn.T, 0))
    ds = np.sort(dist, 1)
    s["knn1"] = ds[:, 0]; s["knn10"] = ds[:, 9]
    dpred = D[np.arange(n), pred]
    Ds = np.sort(D, 1)
    s["hull_dist"] = dpred
    s["hull_contrast"] = dpred - np.where(D.argmin(1) == pred, Ds[:, 1], Ds[:, 0])
    U = visible_basis(W); Wv = U.T @ W
    cm, cmv, cert, certv = [], [], [], []
    for i in range(n):
        Py = Ftr[ytr == pred[i]]
        ok, _, m = readout_certificate(F[i], pred[i], Py, W, b)
        okv, _, mv = readout_certificate(F[i] @ U, pred[i], Py @ U, Wv, b)
        cm.append(m); cmv.append(mv); cert.append(ok); certv.append(okv)
    s["cert_slack"] = -np.array(cm); s["certvis_slack"] = -np.array(cmv)
    return s, np.array(cert), np.array(certv), pred


def run(seed):
    torch.set_num_threads(1)
    d = load_digits(); X, y = d.data.astype(float), d.target
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); tr, te = perm[:NTR], perm[NTR:]
    norm = lambda Z: Z / 16.0 - 0.5
    mlp = MLP([64, 128, 64, 10], rng)
    for s in range(1500):
        mlp.step(norm(X[tr]), y[tr], lr=1e-3, wd=1e-2)
    cnn = train_cnn(norm(X[tr]), y[tr], seed)
    models = {"mlp": model_fn("mlp", mlp), "cnn": model_fn("cnn", cnn)}
    prng = np.random.default_rng(500 + seed)
    Ximg = X.reshape(-1, 8, 8)
    Xtr0 = X[tr]; r0 = np.sqrt(((Xtr0 - Xtr0.mean(0)) ** 2).sum(1).mean())
    out = {"seed": seed, "perts": {}}
    for kind in PERTS:
        Xp = perturb(Ximg[te], kind, prng).reshape(len(te), 64)
        D0 = nch(Xtr0, y[tr], Xp, r0)
        blk = {"I0": float((D0.argmin(1) == y[te]).mean()), "models": {}}
        for mname, (f, W, b) in models.items():
            Ltr, Ftr = f(norm(X[tr])); L, F = f(norm(Xp))
            rad = np.sqrt(((Ftr - Ftr.mean(0)) ** 2).sum(1).mean())
            D = nch(Ftr, y[tr], F, rad)
            sc, cert, certv, pred = scores_for(L, F, Ftr, y[tr], W, b, D)
            err = (pred != y[te]).astype(int)
            mb = {"acc": float(1 - err.mean()), "n_err": int(err.sum()), "I_pen": float((D.argmin(1) == y[te]).mean()),
                  "C_pred": float(cert.mean()), "Cvis_pred": float(certv.mean()),
                  "acc_cert": float(1 - err[cert].mean()) if cert.any() else None,
                  "acc_uncert": float(1 - err[~cert].mean()) if (~cert).any() else None,
                  "acc_certvis": float(1 - err[certv].mean()) if certv.any() else None,
                  "acc_uncertvis": float(1 - err[~certv].mean()) if (~certv).any() else None,
                  "n_cert": int(cert.sum()),
                  "auroc": {k: (float(roc_auc_score(err, v)) if 0 < err.sum() < len(err) else None) for k, v in sc.items()},
                  "scores": {k: v.tolist() for k, v in sc.items()}, "err": err.tolist()}
            blk["models"][mname] = mb
            print(seed, kind, mname, round(mb["acc"], 3), "I0", round(blk["I0"], 3), "Ipen", round(mb["I_pen"], 3),
                  "C", round(mb["C_pred"], 3), "Cvis", round(mb["Cvis_pred"], 3), flush=True)
        out["perts"][kind] = blk
    # padding construction (function-preserving): append 256 readout-invisible ReLU units
    f, W, b = models["mlp"]
    _, reps_tr = mlp.forward(norm(X[tr])); Lte, reps_te = mlp.forward(norm(X[te]))
    pred = Lte.argmax(1)
    out["padding"] = {"C_pred_orig": out["perts"]["none"]["models"]["mlp"]["C_pred"],
                      "Cvis_pred_orig": out["perts"]["none"]["models"]["mlp"]["Cvis_pred"], "by_scale": {}}
    G0 = np.random.default_rng(900 + seed).normal(0, 1 / np.sqrt(128), (128, 256))
    for alpha in (1.0, 4.0, 16.0):
        G = alpha * G0
        pad = lambda reps: np.hstack([reps[-1], np.maximum(reps[1] @ G, 0)])
        Ftr_p, Fte_p = pad(reps_tr), pad(reps_te)
        Wp = np.vstack([W, np.zeros((256, 10))])
        cp = [readout_certificate(Fte_p[i], pred[i], Ftr_p[y[tr] == pred[i]], Wp, b)[0] for i in range(len(te))]
        Up = visible_basis(Wp)
        cpv = [readout_certificate(Fte_p[i] @ Up, pred[i], Ftr_p[y[tr] == pred[i]] @ Up, Up.T @ Wp, b)[0] for i in range(len(te))]
        Lp = Fte_p @ Wp + b
        out["padding"]["by_scale"][str(alpha)] = {"acc": float((Lp.argmax(1) == y[te]).mean()),
                      "max_abs_logit_change": float(np.abs(Lp - Lte).max()),
                      "pad_norm_ratio": float(np.linalg.norm(Fte_p[:, W.shape[0]:], axis=1).mean() / np.linalg.norm(reps_te[-1], axis=1).mean()),
                      "C_pred_padded": float(np.mean(cp)), "Cvis_pred_padded": float(np.mean(cpv))}
    print(seed, "padding", out["padding"], flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "4"))) as pool:
        allres = pool.map(run, range(SEEDS))
    json.dump({"runs": allres, "perts": PERTS}, open(os.path.join(RES, "digits2.json"), "w"))
    print("total", time.time() - t0)
