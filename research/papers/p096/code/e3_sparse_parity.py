"""E3: Hidden-transition regime. Sparse parity (Barak et al. 2022 setting, re-implemented in numpy).

Two-layer ReLU MLP, logistic loss, online SGD on fresh samples.
Indicators logged at log-spaced checkpoints:
  acc      - test accuracy on N_TEST fresh items
  logp     - mean log-probability of the correct label (continuous output metric)
  dsupp    - oracle support share of the weight CHANGE W-W0 (hidden-progress measure)
  dtopk    - support-agnostic concentration of the weight change (k largest columns of W-W0)
  fcS/fcT  - normalised Fourier coefficient of the output f on S (resp. a random null subset T), probe set
  hcS/hcT  - RMS Fourier coefficient on S (resp. T) of hidden-unit activations: internal progress measure
  gS       - log ratio of probe-set gradient energy on support vs off-support columns (Fourier gap)
  lwS      - log ratio of mean squared column norm of W on support vs off-support (needs S)
  supp     - oracle support share: ||W[:,S]||^2/||W||^2 * n/k   (needs knowledge of S)
  topk     - support-agnostic concentration: share of the k largest column norms^2 * n/k
Control runs use labels independent of x (no learnable signal) to calibrate false-alarm thresholds.
Usage: python3 e3_sparse_parity.py [n_bits] [n_seeds] [n_ctrl]
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
K = 3
WIDTH = 256
BATCH = 64
LR = 0.05
N_TEST = 2000


def run(args):
    n, seed, control, max_steps = args
    rng = np.random.default_rng(seed)
    S = np.sort(rng.choice(n, K, replace=False))
    W = rng.standard_normal((WIDTH, n)) / np.sqrt(n)
    b = np.zeros(WIDTH)
    v = rng.standard_normal(WIDTH) / np.sqrt(WIDTH)
    W0 = W.copy()
    Xt = rng.choice([-1.0, 1.0], size=(N_TEST, n))
    yt = np.prod(Xt[:, S], 1) if not control else rng.choice([-1.0, 1.0], N_TEST)
    Xp = rng.choice([-1.0, 1.0], size=(8192, n))     # probe set (labels never used)
    chiS = np.prod(Xp[:, S], 1)
    yp = chiS if not control else rng.choice([-1.0, 1.0], len(chiS))   # probe labels for gradient measure
    T = np.sort(rng.choice(np.setdiff1d(np.arange(n), S), K, replace=False))   # within-run null subset
    chiT = np.prod(Xp[:, T], 1)
    ckpts = np.unique(np.round(np.logspace(1, np.log10(max_steps), 120)).astype(int))
    log = {k: [] for k in ["step", "acc", "logp", "supp", "topk", "dsupp", "dtopk", "loss", "fcS", "fcT", "hcS", "hcT", "gS", "lwS"]}
    ci = 0
    for t in range(1, max_steps + 1):
        X = rng.choice([-1.0, 1.0], size=(BATCH, n))
        y = np.prod(X[:, S], 1) if not control else rng.choice([-1.0, 1.0], BATCH)
        H = X @ W.T + b
        A = np.maximum(H, 0)
        f = A @ v
        m = y * f
        g = -y * 0.5 * (1 - np.tanh(m / 2))          # d/df log(1+exp(-y f)) = -y*sigmoid(-yf)
        gv = A.T @ g / BATCH
        gA = np.outer(g, v) * (H > 0)
        gW = gA.T @ X / BATCH
        gb = gA.mean(0)
        v -= LR * gv; W -= LR * gW; b -= LR * gb
        if t == ckpts[ci]:
            ft = np.maximum(Xt @ W.T + b, 0) @ v
            mt = yt * ft
            cn = (W ** 2).sum(0)
            log["step"].append(t)
            log["acc"].append(float(np.mean(mt > 0)))
            log["logp"].append(float(np.mean(-np.logaddexp(0, -mt))))
            log["loss"].append(float(np.mean(np.logaddexp(0, -mt))))
            log["supp"].append(float(cn[S].sum() / cn.sum() * n / K))
            log["topk"].append(float(np.sort(cn)[-K:].sum() / cn.sum() * n / K))
            dn = ((W - W0) ** 2).sum(0)
            log["dsupp"].append(float(dn[S].sum() / dn.sum() * n / K))
            log["dtopk"].append(float(np.sort(dn)[-K:].sum() / dn.sum() * n / K))
            Ap = np.maximum(Xp @ W.T + b, 0)
            fp = Ap @ v
            sf = fp.std() + 1e-12
            log["fcS"].append(float(np.mean(fp * chiS) / sf))
            log["fcT"].append(float(np.mean(fp * chiT) / sf))
            sa = np.sqrt(np.mean(Ap ** 2)) + 1e-12
            log["hcS"].append(float(np.linalg.norm(Ap.T @ chiS / len(chiS)) / sa / np.sqrt(WIDTH)))
            log["hcT"].append(float(np.linalg.norm(Ap.T @ chiT / len(chiT)) / sa / np.sqrt(WIDTH)))
            # population-gradient Fourier gap on the probe set: RMS gradient on support columns / off-support columns
            mp = yp * fp
            gp = -yp * 0.5 * (1 - np.tanh(mp / 2))
            Gw = (np.outer(gp, v) * (Ap > 0)).T @ Xp / len(gp)
            gc = (Gw ** 2).sum(0)
            off = np.setdiff1d(np.arange(n), S)
            log["gS"].append(float(np.log(gc[S].mean() / gc[off].mean())))
            # log support/off-support mean column norm of W (hidden progress in the weights)
            log["lwS"].append(float(np.log(cn[S].mean() / cn[off].mean())))
            ci += 1
            if ci == len(ckpts):
                break
            if (not control) and log["acc"][-1] > 0.99 and t > 2 * (log["step"][0]) and len(log["acc"]) > 5 \
                    and all(a > 0.99 for a in log["acc"][-3:]):
                break
    return dict(n=n, seed=seed, control=control, S=S.tolist(), **log)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
    t0 = time.time()
    jobs = [(n, 100 + s, False, max_steps) for s in range(ns)] + [(n, 900 + s, True, max_steps) for s in range(nc)]
    with Pool(4) as p:
        res = p.map(run, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, f"e3_parity_n{n}.json"), "w"))
    for r in res:
        print(r["control"], r["seed"], "final step", r["step"][-1], "acc", r["acc"][-1], "supp", round(r["supp"][-1], 2))
    print("elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
