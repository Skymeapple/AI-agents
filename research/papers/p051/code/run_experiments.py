"""Run all MLP experiments (E1-E5). Usage: python3 run_experiments.py [E1 E2 ...]
Outputs JSON files in ../results/."""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
from core import run_one

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)


def _job(kw):
    return run_one(**kw)


def run(name, jobs):
    t0 = time.time()
    with Pool(4) as p:
        out = p.map(_job, jobs)
    json.dump(out, open(os.path.join(RES, f"{name}.json"), "w"), indent=1)
    print(f"{name}: {len(jobs)} runs in {time.time()-t0:.0f}s", flush=True)


ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def E1():  # main alpha sweep, grounded (lam=1) vs mimic (lam=0)
    run("E1_alpha_sweep", [dict(alpha=a, lam=l, seed=s) for a in ALPHAS for l in (1.0, 0.0) for s in range(8)])


def E2():  # mixed self/corpus labels
    run("E2_lambda_sweep", [dict(alpha=a, lam=l, seed=s) for a in (0.2, 0.4, 0.6, 0.8)
                            for l in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0) for s in range(3)])


def E3():  # self-evidence budget: small n
    jobs = []
    for n in (10, 30, 100, 300, 1000):
        ep = max(25, math.ceil(2000 / math.ceil(n / 256)))
        for a in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            for l in (1.0, 0.0):
                for s in range(10):
                    jobs.append(dict(alpha=a, lam=l, seed=s, n=n, epochs=ep, light=True, n_te=4000))
    run("E3_budget", jobs)


def E4():  # regularisation sweep for the mimic (weight-splitting false positives)
    run("E4_wd_sweep", [dict(alpha=a, lam=l, seed=s, wd=w) for w in (1e-4, 1e-3, 1e-2)
                        for a in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5) for l in (0.0, 1.0) for s in range(4)])


def E5():  # deterministic system-specific idiosyncrasy instead of noise
    run("E5_idio", [dict(alpha=a, lam=l, seed=s, variant="idio") for a in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
                    for l in (1.0, 0.0) for s in range(5)])


if __name__ == "__main__":
    which = sys.argv[1:] or ["E1", "E2", "E3", "E4", "E5"]
    for w in which:
        globals()[w]()
