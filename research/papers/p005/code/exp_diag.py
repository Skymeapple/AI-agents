"""Experiment 3 (diagnostic): why does the readout certificate fail on grokked modular
addition but succeed on digits?  For each test point we project its penultimate
representation phi onto the same-label training hull (point h, residual r = phi - h) and
decompose the residual into the part visible to the readout (span of w_c - w_y) and the
invisible part.  Usage: python3 exp_diag.py  (writes results/diag.json)"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nn import MLP
from hull import hull_project, readout_certificate
import exp_modadd as E

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def analyse(R, Y, tr, te, W, b, ncls):
    out = {"rel_delta": [], "cert": [], "h_correct": [], "visible_frac": [], "margin_phi": [], "margin_h": []}
    Rtr = R[tr]; mu = Rtr.mean(0)
    radius = np.sqrt(((Rtr - mu) ** 2).sum(1).mean())
    for i in te:
        y = Y[i]
        Py = R[tr[Y[tr] == y]]
        ok, delta, _ = readout_certificate(R[i], y, Py, W, b)
        _, lam = hull_project(R[i], Py)
        h = Py.T @ lam
        r = R[i] - h
        D = np.delete(W - W[:, [y]], y, axis=1)          # columns w_c - w_y, c != y
        U, s, _ = np.linalg.svd(D, full_matrices=False)
        U = U[:, s > 1e-8 * s.max()]
        vis = float((U.T @ r) @ (U.T @ r) / max(r @ r, 1e-30))
        z_phi = R[i] @ W + b; z_h = h @ W + b
        m_phi = z_phi[y] - np.max(np.delete(z_phi, y)); m_h = z_h[y] - np.max(np.delete(z_h, y))
        out["rel_delta"].append(delta / radius); out["cert"].append(bool(ok))
        out["h_correct"].append(bool(m_h > 0)); out["visible_frac"].append(vis)
        out["margin_phi"].append(float(m_phi)); out["margin_h"].append(float(m_h))
    s = {"n": len(te), "cert_frac": float(np.mean(out["cert"])), "hullpoint_correct_frac": float(np.mean(out["h_correct"])),
         "median_rel_delta": float(np.median(out["rel_delta"])), "median_visible_frac": float(np.median(out["visible_frac"])),
         "median_margin_phi": float(np.median(out["margin_phi"])), "median_margin_h": float(np.median(out["margin_h"])),
         "n_train_per_class_mean": float(len(tr) / ncls), "rep_dim": int(R.shape[1])}
    return s


def run_mod(frac):
    X, Y, tr, te, rng = E.data(frac, 0)
    net = MLP([2 * E.P, 128, 128, E.P], rng)
    for s in range(E.STEPS):
        net.step(X[tr], Y[tr], lr=1e-3, wd=1.0)
    logits, reps = net.forward(X)
    acc = float((logits[te].argmax(1) == Y[te]).mean())
    s = analyse(reps[-1], Y, tr, te[:400], net.W[-1], net.b[-1], E.P)
    s.update({"task": "modadd", "frac": frac, "test_acc": acc})
    print(s, flush=True)
    return s


def run_digits(ntr):
    from sklearn.datasets import load_digits
    d = load_digits(); X, Y = d.data / 16.0 - 0.5, d.target
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y)); tr, te = perm[:ntr], perm[ntr:]
    net = MLP([64, 128, 64, 10], rng)
    for s in range(1500):
        net.step(X[tr], Y[tr], lr=1e-3, wd=1e-2)
    logits, reps = net.forward(X)
    acc = float((logits[te].argmax(1) == Y[te]).mean())
    s = analyse(reps[-1], Y, tr, te[:400], net.W[-1], net.b[-1], 10)
    s.update({"task": "digits", "ntr": ntr, "test_acc": acc})
    print(s, flush=True)
    return s


def job(a):
    return run_mod(a[1]) if a[0] == "mod" else run_digits(a[1])


if __name__ == "__main__":
    jobs = [("mod", 0.5), ("mod", 0.7), ("mod", 0.9), ("dig", 100), ("dig", 300), ("dig", 600), ("dig", 1200)]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    json.dump(res, open(os.path.join(RES, "diag.json"), "w"), indent=1)
