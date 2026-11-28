"""Generate random 3-SAT instances, label them with an exact solver (PySAT/Minisat22),
and run WalkSAT R times per instance.  Saves results/walksat_<cfg>.npz with
  runs: uint8 [n_inst, R]  (1 = solved within flip budget)
  sat:  bool  [n_inst]     (exact satisfiability)
Usage: python run_walksat.py   (about 5-10 CPU-minutes on 4 cores)
"""
import numpy as np, subprocess, os, sys, time
from multiprocessing import Pool
from pysat.solvers import Minisat22

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TMP = os.path.join(HERE, "..", "results", "instances")
os.makedirs(TMP, exist_ok=True)

CONFIGS = {
    # name: (n_vars, clause_ratio, max_flips, noise, n_instances)
    "A_n75_r426_f300": (75, 4.26, 300, 0.5, 240),
    "B_n75_r426_f3000": (75, 4.26, 3000, 0.5, 240),
    "C_n150_r410_f1000": (150, 4.10, 1000, 0.5, 240),
}
R = 10000


def make_instance(rng, N, ratio):
    C = int(round(ratio * N))
    cl = np.empty((C, 3), int)
    for c in range(C):
        cl[c] = rng.choice(np.arange(1, N + 1), 3, replace=False)
    cl *= np.where(rng.random((C, 3)) < 0.5, 1, -1)
    return cl


def job(args):
    fn, maxflips, noise, seed = args
    o = subprocess.run([os.path.join(HERE, "walksat"), fn, str(R), str(maxflips), str(noise), str(seed)],
                       capture_output=True, text=True, check=True).stdout.strip()
    return np.frombuffer(o.encode(), dtype=np.uint8) - ord("0")


def main():
    for ci, (name, (N, ratio, F, noise, ninst)) in enumerate(CONFIGS.items()):
        out = os.path.join(RES, f"walksat_{name}.npz")
        if os.path.exists(out):
            print("exists", out); continue
        rng = np.random.default_rng(1000 + ci)
        files, sat = [], []
        for i in range(ninst):
            cl = make_instance(rng, N, ratio)
            with Minisat22(bootstrap_with=cl.tolist()) as s:
                sat.append(s.solve())
            fn = os.path.join(TMP, f"{name}_{i}.txt")
            with open(fn, "w") as fh:
                fh.write(f"{N} {len(cl)}\n" + "\n".join(" ".join(map(str, r)) for r in cl))
            files.append(fn)
        t = time.time()
        with Pool(4) as pool:
            runs = pool.map(job, [(fn, F, noise, 7919 * i + ci) for i, fn in enumerate(files)])
        runs = np.vstack(runs).astype(np.uint8)
        np.savez_compressed(out, runs=runs, sat=np.array(sat), N=N, ratio=ratio, flips=F, noise=noise)
        p = runs.mean(1)
        print(name, f"{time.time()-t:.0f}s", "sat frac", np.mean(sat), "p>0 frac", np.mean(p > 0),
              "mean p", p.mean(), "pass@1000 (emp)", np.mean(p > 0), flush=True)


if __name__ == "__main__":
    main()
