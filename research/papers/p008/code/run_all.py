"""Runs all experiments (CPU, 4 worker processes). Outputs ../results/all_runs.json.
Usage: python3 run_all.py            (approx. 45 min wall-clock on 4 cores)"""
import json, time, itertools, os
import numpy as np, torch
from multiprocessing import Pool
import skills as S
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
TEACHER = os.path.join(OUT, 'teacher_h256_seed0.pt')

def summarize(w, m, **meta):
    r = S.evaluate(w, m)
    return dict(meta, params=S.n_params(m), learned=int((r['score_t'] > 0.75).sum()),
                wscore=float((w.p * r['score_t']).sum()), uscore=float(r['score_t'].mean()),
                wkl=float((w.p * r['kl_t']).sum()), score_t=r['score_t'].tolist(), acc_t=r['acc_t'].tolist())

def train_teacher():
    """'Frontier' teacher: wide net trained on fresh hard-labelled samples (ground-truth data, not oracle)."""
    w = S.World(seed=0)
    m = S.train(w, 256, None, 'hard', steps=30000, seed=123)
    torch.save(m.state_dict(), TEACHER)
    return summarize(w, m, exp='teacher', h=256, D=None, method='hard', seed=0, steps=30000)

def load_teacher(w):
    m = S.MLP(w.code, w.d, w.C, 256); m.load_state_dict(torch.load(TEACHER)); m.eval(); return m

def job(cfg):
    t0 = time.time()
    w = S.World(family=cfg['family'], beta=cfg['beta'], alpha=cfg['alpha'], seed=cfg['seed'])
    method, teacher = cfg['method'], None
    if method == 'soft_teacher':
        teacher, method = load_teacher(w), 'soft'
    m = S.train(w, cfg['h'], cfg['D'], method, teacher=teacher, steps=cfg['steps'], seed=cfg['seed'])
    res = summarize(w, m, **cfg); res['sec'] = time.time() - t0
    return res

def base(**kw):
    c = dict(family='random', beta=2.0, alpha=0.5, seed=0, D=None, method='soft', steps=10000); c.update(kw); return c

def configs():
    C = []
    # E1 capacity regime: unlimited oracle-teacher queries, long training
    for fam, h, s in itertools.product(['random', 'lowrank'], [8, 12, 16, 24, 32, 48], [0, 1]):
        C.append(base(exp='E1', family=fam, h=h, seed=s, steps=45000))
    C.append(base(exp='E1conv', h=32, steps=90000))
    C.append(base(exp='E1conv', h=16, steps=90000))
    # E2 data regime: finite D, hard labels vs oracle soft labels (10k steps, early stopping)
    for h, D, meth, s in itertools.product([16, 32, 128], [1000, 3000, 10000, 30000, 100000, 300000, None],
                                           ['hard', 'soft'], [0, 1]):
        C.append(base(exp='E2', h=h, D=D, method=meth, seed=s))
    # E3 label-noise sweep at fixed D
    for beta, meth, s in itertools.product([0.5, 1.0, 2.0, 4.0, 8.0], ['hard', 'soft'], [0, 1]):
        C.append(base(exp='E3', h=128, D=10000, method=meth, seed=s, beta=beta))
    # E4 Zipf-exponent sweep in the capacity regime
    for a, h in itertools.product([0.25, 1.0], [16, 32]):
        C.append(base(exp='E4', h=h, alpha=a, steps=45000))
    return C

def teacher_configs():
    return [base(exp='E2T', h=128, D=D, method='soft_teacher') for D in [1000, 3000, 10000, 30000, 100000, 300000, None]]

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cfgs = configs(); cfgs.sort(key=lambda c: -c['steps'] * (1 + c['h'] / 64))
    with Pool(4) as pool:
        tres = pool.apply_async(train_teacher)
        res = []
        for i, r in enumerate(pool.imap_unordered(job, cfgs)):
            res.append(r)
            if i % 10 == 0: print(f'{i+1}/{len(cfgs)} done, {time.time()-t0:.0f}s', flush=True)
        res.append(tres.get())
        res += pool.map(job, teacher_configs())
    json.dump(res, open(os.path.join(OUT, 'all_runs.json'), 'w'))
    print('total', len(res), 'runs', round(time.time() - t0), 's')
