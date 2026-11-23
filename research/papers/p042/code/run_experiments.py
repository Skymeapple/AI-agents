"""All toy experiments. Usage: python3 run_experiments.py [E1|E2|E3|E4|all]
Each job writes ../results/raw/<name>.npz (max-cosine of every true feature at each checkpoint).
Single-threaded BLAS, 4 worker processes."""
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sae import ZipfData, TopKSAE, recovery, fvu

OUT = os.path.join(HERE, "..", "results", "raw")
os.makedirs(OUT, exist_ok=True)
BASE = dict(N=512, D=128, L0=4.0, K=8, M=768, alpha=1.5, seed=0, lr=1e-3, batch=256,
            tokens=3_000_000, n_ck=14, buffer=None, pmax=0.25, opt="adam", beta2=0.999)


def job(cfg):
    c = dict(BASE); c.update(cfg)
    path = os.path.join(OUT, c["name"] + ".npz")
    if os.path.exists(path):
        return c["name"], "cached"
    t0 = time.time()
    data = ZipfData(c["N"], c["D"], c["alpha"], c["L0"], seed=c["seed"], pmax=c["pmax"])
    sae = TopKSAE(c["D"], c["M"], c["K"], seed=c["seed"], lr=c["lr"], opt=c["opt"], beta2=c["beta2"])
    B = c["batch"]; steps = c["tokens"] // B
    ck = sorted(set(np.geomspace(max(2, 8192 // B), steps, c["n_ck"]).astype(int)))
    buf = None
    if c["buffer"]:  # finite activation buffer replayed in epochs (E4)
        nb = c["buffer"] // B
        bufX, bufA = [], []
        for _ in range(nb):
            x, A = data.sample(B); bufX.append(x); bufA.append(A)
        uniq_counts = np.asarray(sum((A > 0).sum(0) for A in bufA)).ravel()
    maxcos, Ts, fvus, dead = [], [], [], []
    for s in range(1, steps + 1):
        if c["buffer"]:
            if (s - 1) % nb == 0:
                order = data.rng.permutation(nb)
            x = bufX[order[(s - 1) % nb]]
        else:
            x, _ = data.sample(B)
        sae.step(x)
        if s in ck:
            mx, _ = recovery(sae, data.F)
            maxcos.append(mx.astype(np.float32)); Ts.append(s * B)
            fvus.append(fvu(sae, data)); dead.append(int(((sae.t - sae.last_fired) > 200).sum()))
    extra = dict(uniq_counts=uniq_counts) if c["buffer"] else {}
    np.savez(path, maxcos=np.array(maxcos), T=np.array(Ts), fvu=np.array(fvus), dead=np.array(dead),
             p=data.p, cfg=json.dumps(c), **extra)
    return c["name"], f"{time.time()-t0:.0f}s"


def configs(which):
    C = []
    if which in ("E1", "all"):   # data-limited regime, three Zipf exponents x 2 seeds
        for a in (1.2, 1.5, 1.8):
            for s in (0, 1):
                C.append(dict(name=f"E1_a{a}_s{s}", alpha=a, seed=s))
    if which in ("E2", "all"):   # width-limited regime
        for a in (1.2, 1.5):
            for M in (64, 128, 256, 512):
                C.append(dict(name=f"E2_a{a}_M{M}", alpha=a, M=M, n_ck=6))
    if which in ("E3", "all"):   # what sets the occurrence threshold m*?
        for lr in (3e-4, 3e-3): C.append(dict(name=f"E3_lr{lr}", lr=lr))
        for B in (64, 1024): C.append(dict(name=f"E3_B{B}", batch=B))
        for D in (64, 256): C.append(dict(name=f"E3_D{D}", D=D))
        for L0 in (2.0, 8.0): C.append(dict(name=f"E3_L0{L0}", L0=L0, K=int(2 * L0)))
    if which in ("E4", "all"):   # finite buffer replay: unique occurrences vs exposures
        for U in (100_000, 300_000, 1_000_000):
            C.append(dict(name=f"E4_U{U}", buffer=U))
    if which in ("E5", "all"):   # optimizer mechanism: per-parameter normalisation vs plain momentum SGD
        for b2 in (0.99, 0.9999): C.append(dict(name=f"E5_adam_b2{b2}", beta2=b2))
        for lr in SGD_LRS: C.append(dict(name=f"E5_sgd_lr{lr}", opt="sgd", lr=lr))
    return C


SGD_LRS = (0.1, 0.3)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    C = configs(which)
    cost = lambda c: dict(BASE, **c)["M"] * dict(BASE, **c)["D"] * (4 if dict(BASE, **c)["batch"] == 64 else 1)
    C.sort(key=lambda c: -cost(c))
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, C):
            print(name, msg, flush=True)
