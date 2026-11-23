"""Token-sequence transformer organisms: rho in {0, .25, .5}, 6 seeds each."""
import json, sys, time
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit('/', 1)[0])

def job(a):
    import torch; torch.set_num_threads(1)
    from token_organism import run
    rho, s = a
    return run(70000 + int(rho * 100) * 100 + s, rho)

if __name__ == '__main__':
    t0 = time.time()
    jobs = [(rho, s) for rho in [0.0, 0.25, 0.5] for s in range(6)]
    with Pool(4) as p:
        out = p.map(job, jobs, chunksize=1)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'results/raw_token.json', 'w'), default=float, indent=1)
    print('done', len(out), 'runs in', round(time.time() - t0, 1), 's')
