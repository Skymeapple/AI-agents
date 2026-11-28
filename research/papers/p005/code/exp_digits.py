"""Experiment 2: input-space vs representation-space interpolation on real data
(sklearn 8x8 digits, UCI). Usage: python3 exp_digits.py  (writes results/digits.json)"""
import sys, os, json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from nn import MLP
from hull import hull_project, in_hull_lp, readout_certificate

RES = os.path.join(os.path.dirname(__file__), "..", "results")
NTR = 600
SEEDS = 5


def shift(Ximg, dx):
    out = np.zeros_like(Ximg)
    if dx > 0:
        out[:, :, dx:] = Ximg[:, :, :-dx]
    else:
        out = Ximg.copy()
    return out


def layer_stats(reps_tr, ytr, reps_te, yte, pred):
    """Per-layer hull statistics for test points."""
    out = {}
    for l, (Rtr, Rte) in enumerate(zip(reps_tr, reps_te)):
        mu = Rtr.mean(0)
        radius = np.sqrt(((Rtr - mu) ** 2).sum(1).mean())
        D = np.zeros((len(Rte), 10))
        for c in range(10):
            Pc = Rtr[ytr == c]
            for i in range(len(Rte)):
                D[i, c], _ = hull_project(Rte[i], Pc)
        nn = np.array([ytr[((Rtr - r) ** 2).sum(1).argmin()] for r in Rte])
        out[l] = {"rel_same_median": float(np.median(D[np.arange(len(yte)), yte] / radius)),
                  "hull_nearest_acc": float((D.argmin(1) == yte).mean()),
                  "nn1_acc": float((nn == yte).mean()),
                  "D": D / radius}
    return out


def run(seed):
    d = load_digits()
    X, y = d.data.astype(float), d.target
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); tr, te = perm[:NTR], perm[NTR:]
    # global scalar normalisation (per-pixel standardisation would blow up near-constant
    # border pixels under the shift probe)
    norm = lambda Z: Z / 16.0 - 0.5
    net = MLP([64, 128, 64, 10], rng)
    for s in range(1500):
        net.step(norm(X[tr]), y[tr], lr=1e-3, wd=1e-2)
    res = {"seed": seed}
    # input-space exact LP membership (raw pixels), unconditional and class-conditional
    sub = te[:300]
    res["input_in_hull_uncond"] = float(np.mean([in_hull_lp(X[i], X[tr]) for i in sub]))
    res["input_in_hull_same_class"] = float(np.mean([in_hull_lp(X[i], X[tr][y[tr] == y[i]]) for i in sub]))
    Ximg = X.reshape(-1, 8, 8)
    for dx in [0, 1, 2]:
        Xt = shift(Ximg[te], dx).reshape(len(te), 64)
        logits, reps_te = net.forward(norm(Xt))
        _, reps_tr = net.forward(norm(X[tr]))
        pred = logits.argmax(1)
        yte = y[te]
        acc = float((pred == yte).mean())
        st = layer_stats(reps_tr, y[tr], reps_te, yte, pred)
        R, Rtr = reps_te[-1], reps_tr[-1]
        cert_true, cert_pred, dpred = [], [], []
        for i in range(len(te)):
            ok, _, _ = readout_certificate(R[i], yte[i], Rtr[y[tr] == yte[i]], net.W[-1], net.b[-1])
            cert_true.append(ok)
            okp, dl, _ = readout_certificate(R[i], pred[i], Rtr[y[tr] == pred[i]], net.W[-1], net.b[-1])
            cert_pred.append(okp)
        cert_true, cert_pred = np.array(cert_true), np.array(cert_pred)
        err = (pred != yte).astype(int)
        z = logits - logits.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
        L = len(reps_te) - 1
        dpred = st[L]["D"][np.arange(len(te)), pred]
        Dl = st[L]["D"]
        Ds = np.sort(Dl, 1)
        contrast = dpred - np.where(Dl.argmin(1) == pred, Ds[:, 1], Ds[:, 0])  # own-hull minus best other hull
        nnd = np.array([np.sqrt(((Rtr - r) ** 2).sum(1).min()) for r in R])
        block = {"acc": acc, "n_err": int(err.sum()),
                 "cert_true_frac": float(cert_true.mean()),
                 "cert_pred_frac": float(cert_pred.mean()),
                 "acc_on_certified": float(1 - err[cert_pred].mean()) if cert_pred.any() else None,
                 "acc_on_uncertified": float(1 - err[~cert_pred].mean()) if (~cert_pred).any() else None,
                 "layers": {l: {k: v for k, v in st[l].items() if k != "D"} for l in st},
                 "scores": {"err": err.tolist(), "maxprob": pr.max(1).tolist(), "hull_dist_pred": dpred.tolist(),
                            "hull_contrast": contrast.tolist(), "nn_dist": nnd.tolist(), "cert_pred": cert_pred.astype(int).tolist()}}
        res[f"shift{dx}"] = block
        print(seed, dx, acc, block["cert_pred_frac"], block["acc_on_certified"], block["acc_on_uncertified"],
              {l: block["layers"][l]["hull_nearest_acc"] for l in block["layers"]}, flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    from multiprocessing import Pool
    with Pool(2) as pool:
        allres = pool.map(run, range(SEEDS))
    # pooled AUROC for error detection
    summ = {}
    for dx in [0, 1, 2]:
        err = np.concatenate([r[f"shift{dx}"]["scores"]["err"] for r in allres])
        s = {k: np.concatenate([r[f"shift{dx}"]["scores"][k] for r in allres]) for k in ["maxprob", "hull_dist_pred", "hull_contrast", "nn_dist"]}
        if 0 < err.sum() < len(err):
            summ[f"shift{dx}"] = {"n": int(len(err)), "n_err": int(err.sum()),
                                  "auroc_neg_maxprob": float(roc_auc_score(err, -s["maxprob"])),
                                  "auroc_hull_dist": float(roc_auc_score(err, s["hull_dist_pred"])),
                                  "auroc_hull_contrast": float(roc_auc_score(err, s["hull_contrast"])),
                                  "auroc_nn_dist": float(roc_auc_score(err, s["nn_dist"]))}
    print(json.dumps(summ, indent=1))
    with open(os.path.join(RES, "digits.json"), "w") as f:
        json.dump({"runs": allres, "pooled_auroc": summ}, f)
