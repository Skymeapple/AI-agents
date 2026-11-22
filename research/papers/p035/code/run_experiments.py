import json, sys, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from organism import run_one

SEEDS = range(10)
CONFIGS = []
for rho in [0.0, 0.25, 0.5, 0.75]:
    CONFIGS.append(dict(rho=rho, H_visible=True, subtle=False))
    if rho > 0:
        CONFIGS.append(dict(rho=rho, H_visible=False, subtle=False))
CONFIGS.append(dict(rho=0.0, H_visible=True, subtle=True))
for kappa in [0.0, 2.0, 8.0]:
    CONFIGS.append(dict(rho=0.0, H_visible=True, subtle=False, kappa=kappa))

def job(args):
    cfg, s = args
    return run_one(1000 * CONFIGS.index(cfg) + s, **cfg)

if __name__ == '__main__':
    t0 = time.time()
    jobs = [(c, s) for c in CONFIGS for s in SEEDS]
    with Pool(4) as p:
        out = p.map(job, jobs)
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'results/raw.json'
    json.dump(out, open(out_path, 'w'), default=float, indent=1)
    print('done', len(out), 'runs in', round(time.time() - t0, 1), 's')
