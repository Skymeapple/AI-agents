"""Runs all experiments. Usage: python3 run_experiments.py [E1|E2|E3|all]
Each job writes results/raw/<name>.npz. Single-threaded BLAS, 4 worker processes."""
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from sae import ZipfData, TopKSAE, recovery, fvu

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
os.makedirs(OUT, exist_ok=True)
N, D, L0, K = 1024, 48, 5, 10


def job(cfg):
    name = cfg["name"]
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return name, "cached"
    t0 = time.time()
    data = ZipfData(N, D, cfg["alpha"], L0, seed=cfg.get("data_seed", 0))
    sae = TopKSAE(D, cfg["M"], K, seed=cfg["seed"], lr=cfg.get("lr", 1e-3))
    B = cfg.get("batch", 256)
    T_total = cfg["tokens"]
    steps = T_total // B
    ck = sorted(set(np.unique(np.geomspace(max(1, 4096 // B), steps, cfg.get("n_ck", 14)).astype(int))))
    maxcos, Ts, fvus, dead = [], [], [], []
    for s in range(1, steps + 1):
        x, _ = data.sample(B)
        sae.step(x)
        if s in ck:
            mx, _ = recovery(sae, data.F)
            maxcos.append(mx.astype(np.float32)); Ts.append(s * B)
            fvus.append(fvu(sae, data)); dead.append(int(((sae.t - sae.last_fired) > 200).sum()))
    np.savez(path, maxcos=np.array(maxcos), T=np.array(Ts), fvu=np.array(fvus), dead=np.array(dead),
             p=data.p, cfg=json.dumps(cfg), Wdec_F_assign=(data.F @ sae.Wdec.T).argmax(1),
             fire_count=sae.fire_count)
    return name, f"{time.time()-t0:.0f}s"


def configs(which):
    C = []
    alphas = [1.2, 1.5, 1.8]
    if which in ("E1", "all"):  # token-limited regime: wide dictionary, checkpoints over T
        for a in alphas:
            for seed in (0, 1):
                C.append(dict(name=f"E1_a{a}_s{seed}", alpha=a, M=2048, seed=seed, data_seed=seed,
                              tokens=3_072_000, n_ck=16))
    if which in ("E2", "all"):  # width-limited regime
        for a in alphas:
            for M in (64, 128, 256, 512, 1024):
                C.append(dict(name=f"E2_a{a}_M{M}", alpha=a, M=M, seed=0, tokens=3_072_000, n_ck=6))
    if which in ("E3", "all"):  # does the threshold live in tokens or optimizer steps?
        for B in (64, 1024):
            C.append(dict(name=f"E3_a1.5_B{B}", alpha=1.5, M=2048, seed=0, batch=B, tokens=3_072_000, n_ck=16))
        for lr in (3e-4, 3e-3):
            C.append(dict(name=f"E3_a1.5_lr{lr}", alpha=1.5, M=2048, seed=0, lr=lr, tokens=3_072_000, n_ck=16))
    return C


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    C = configs(which)
    # longest jobs first
    C.sort(key=lambda c: -c["M"] * c["tokens"])
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, C):
            print(name, msg, flush=True)
