"""E5: does distillation raise the capacity ceiling? Long training (45k steps), small widths, hard vs soft.
Usage: python3 run_e5.py  -> ../results/e5_runs.json (approx. 12 min on 4 cores)"""
import itertools, json, os, time
from multiprocessing import Pool
from run_all import job, base, OUT
if __name__ == '__main__':
    cfgs = [base(exp='E5', h=h, D=D, method=m, seed=s, steps=45000)
            for h, D, m, s in itertools.product([8, 16], [10000, 100000, None], ['hard', 'soft'], [0, 1])]
    t0 = time.time()
    with Pool(4) as p: res = p.map(job, cfgs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, 'e5_runs.json'), 'w')); print('done', round(time.time() - t0))
