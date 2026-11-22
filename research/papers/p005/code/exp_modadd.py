"""Experiment 1: interpolation profile of a grokking MLP on modular addition.
Usage: python3 exp_modadd.py   (writes results/modadd_*.json)"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nn import MLP
from hull import hull_project, in_hull_lp, readout_certificate

RES = os.path.join(os.path.dirname(__file__), "..", "results")
P = 53
STEPS = 12000
EVERY = 500
NTEST_EVAL = 250


def data(frac, seed, random_labels=False):
    rng = np.random.default_rng(seed)
    A, B = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    A, B = A.ravel(), B.ravel()
    Y = (A + B) % P
    if random_labels:
        Y = rng.integers(0, P, size=Y.shape)
    X = np.zeros((P * P, 2 * P)); X[np.arange(P * P), A] = 1; X[np.arange(P * P), P + B] = 1
    perm = rng.permutation(P * P); ntr = int(frac * P * P)
    return X, Y, perm[:ntr], perm[ntr:], rng


def metrics(net, X, Y, tr, te_eval):
    logits, reps = net.forward(X)
    pred = logits.argmax(1)
    out = {"train_acc": float((pred[tr] == Y[tr]).mean())}
    L = len(reps)
    for l in range(L):
        R = reps[l]
        Rtr = R[tr]
        mu = Rtr.mean(0)
        radius = np.sqrt(((Rtr - mu) ** 2).sum(1).mean()) + 1e-12
        cls_idx = {c: tr[Y[tr] == c] for c in range(P)}
        rel_same, hull_nn_correct, nn1_correct = [], [], []
        for i in te_eval:
            d = np.full(P, np.inf)
            for c in range(P):
                if len(cls_idx[c]) == 0:
                    continue
                d[c], _ = hull_project(R[i], R[cls_idx[c]])
            rel_same.append(d[Y[i]] / radius)
            hull_nn_correct.append(d.argmin() == Y[i])
            dn = ((Rtr - R[i]) ** 2).sum(1)
            nn1_correct.append(Y[tr][dn.argmin()] == Y[i])
        out[f"L{l}_rel_dist_same_median"] = float(np.median(rel_same))
        out[f"L{l}_frac_rel_dist_lt_0.05"] = float(np.mean(np.array(rel_same) < 0.05))
        out[f"L{l}_hull_nearest_acc"] = float(np.mean(hull_nn_correct))
        out[f"L{l}_1nn_acc"] = float(np.mean(nn1_correct))
    # readout certificate at penultimate layer
    R = reps[-1]
    cert = []
    for i in te_eval:
        Py = R[tr[Y[tr] == Y[i]]]
        ok, delta, m = readout_certificate(R[i], Y[i], Py, net.W[-1], net.b[-1])
        cert.append(ok)
    out["cert_frac"] = float(np.mean(cert))
    out["test_acc_eval"] = float((pred[te_eval] == Y[te_eval]).mean())
    return out


def run(cfg):
    frac, seed, rl = cfg
    X, Y, tr, te, rng = data(frac, seed, rl)
    net = MLP([2 * P, 128, 128, P], rng)
    te_eval = te[:NTEST_EVAL]
    hist = []
    t0 = time.time()
    # exact LP check of Proposition 1 on a sample of test points (input space, unconditional hull)
    lp_in = [in_hull_lp(X[i], X[tr]) for i in te[:40]]
    for s in range(STEPS + 1):
        if s % EVERY == 0:
            m = metrics(net, X, Y, tr, te_eval)
            m["step"] = s
            m["test_acc"] = float((net.predict(X[te]) == Y[te]).mean())
            hist.append(m)
            print(frac, seed, rl, s, m["train_acc"], m["test_acc"], m["cert_frac"],
                  m["L2_hull_nearest_acc"], round(time.time() - t0), flush=True)
        if s < STEPS:
            net.step(X[tr], Y[tr], lr=1e-3, wd=1.0)
    res = {"frac": frac, "seed": seed, "random_labels": rl, "p": P,
           "input_lp_in_hull_frac": float(np.mean(lp_in)), "n_lp_checked": len(lp_in), "history": hist}
    with open(os.path.join(RES, f"modadd_f{frac}_s{seed}_rl{int(rl)}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    cfgs = [(0.5, s, False) for s in range(3)] + [(0.4, 0, False), (0.6, 0, False), (0.5, 0, True)]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        cfgs = [(0.5, 0, False)]
    with Pool(4) as pool:
        pool.map(run, cfgs)
