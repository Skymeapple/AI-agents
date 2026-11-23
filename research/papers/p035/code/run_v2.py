"""Revision-1 grid: stress tests, unsaturated policy, stronger baselines, within-config diagnostics."""
import json, sys, time
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from organism_v2 import run_v2

SAT = dict(b0=1.5, kappa=4.0); UNSAT = dict(b0=-0.5, kappa=1.5)
CONFIGS = []
for regime, P in [('sat', SAT), ('unsat', UNSAT)]:
    for rho in [0.0, 0.25, 0.5, 0.75]:
        for vis in ([True] if rho == 0 else [True, False]):
            CONFIGS.append((dict(rho=rho, H_visible=vis, **P), 15, regime))
for rho in [0.0, 0.5]:
    CONFIGS.append((dict(rho=rho, H_visible=True, detector='interaction', **UNSAT), 10, 'unsat-interaction'))
for rho in [0.0, 0.25, 0.5]:
    CONFIGS.append((dict(rho=rho, H_visible=True, gshape='tent', **UNSAT), 10, 'unsat-tent'))
CONFIGS.append((dict(rho=0.0, H_visible=True, b0=1.5, kappa=0.0), 10, 'sat-kappa0'))
CONFIGS.append((dict(rho=0.0, H_visible=True, b0=-0.5, kappa=0.0), 10, 'unsat-kappa0'))

def job(a):
    ci, s = a; cfg, _, regime = CONFIGS[ci]
    r = run_v2(50000 + 1000 * ci + s, **cfg); r['regime'] = regime; r['config_index'] = ci
    return r

if __name__ == '__main__':
    t0 = time.time()
    jobs = [(ci, s) for ci, (c, n, _) in enumerate(CONFIGS) for s in range(n)]
    with Pool(4) as p:
        out = p.map(job, jobs, chunksize=1)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'results/raw_v2.json', 'w'), default=float, indent=1)
    print('done', len(out), 'runs in', round(time.time() - t0, 1), 's')
