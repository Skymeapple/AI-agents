"""E6: long-training test of 'distillation does not raise capacity' at non-floor widths, with a REAL teacher.
All runs use world seed 0 (the world the trained teacher was trained on); seeds vary init + data stream only.
Methods: hard (sampled labels), soft (oracle Bayes probabilities), soft_teacher (trained h=256 teacher's probabilities).
Unlimited data (fresh samples every step). K is also logged at intermediate checkpoints (under the cosine schedule).
Usage: python3 run_e6.py A   (main grid: h in {32,48}, 100k steps, 5 seeds, 3 methods)
       python3 run_e6.py C   (250k-step h=48 hard/teacher-distilled, 200k-step h=128 hard)
       python3 run_e6.py B   (h=128 at 40k steps, teacher-distilled small students, h=16 with 5 seeds)
Outputs ../results/e6_runs_{A,B}.json and final weights in ../results/e6_models/."""
import sys, json, os, time, itertools
import numpy as np, torch, torch.nn.functional as F
from multiprocessing import Pool
import skills as S
from run_all import load_teacher, OUT

def summarize(w, m, **meta):
    r = S.evaluate(w, m)
    return dict(meta, params=S.n_params(m), learned=int((r['score_t'] > 0.75).sum()),
                wscore=float((w.p * r['score_t']).sum()), uscore=float(r['score_t'].mean()),
                score_t=r['score_t'].tolist(), acc_t=r['acc_t'].tolist())

def job(cfg):
    t0 = time.time()
    w = S.World(seed=0)
    teacher = load_teacher(w) if cfg['method'] == 'soft_teacher' else None
    seed, steps, h = cfg['seed'], cfg['steps'], cfg['h']
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = S.MLP(w.code, w.d, w.C, h)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    traj = []
    for s in range(steps):
        t, z, y, logp = w.sample(512, gen)
        out = model(t, z)
        if cfg['method'] == 'hard':
            loss = F.cross_entropy(out, y)
        else:
            if teacher is None: tgt = logp.exp()
            else:
                with torch.no_grad(): tgt = F.softmax(teacher(t, z), -1)
            loss = -(tgt * F.log_softmax(out, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (s + 1) % 10000 == 0 and s + 1 < steps:
            r = S.evaluate(w, model); traj.append((s + 1, int((r['score_t'] > 0.75).sum())))
    res = summarize(w, model, **cfg); res['traj'] = traj; res['sec'] = time.time() - t0
    torch.save(model.state_dict(), os.path.join(OUT, 'e6_models', f"{cfg['tag']}_h{h}_{cfg['method']}_s{seed}_{steps}.pt"))
    return res

def cfgA():
    return [dict(exp='E6', tag='A', h=h, method=m, seed=s, steps=100000)
            for h, m, s in itertools.product([32, 48], ['hard', 'soft', 'soft_teacher'], range(5))]

def cfgB():
    # trimmed for compute (shared 4-core machine): no 30k-step h=32/48 grid; h=128 at 40k steps
    C = [dict(exp='E6', tag='B', h=128, method=m, seed=s, steps=40000)
         for m, s in itertools.product(['hard', 'soft'], range(2))]
    C += [dict(exp='E6', tag='B', h=h, method='soft_teacher', seed=s, steps=45000)
          for h, s in itertools.product([8, 16, 24], range(2))]
    C += [dict(exp='E6', tag='B', h=16, method=m, seed=s, steps=45000)
          for m, s in itertools.product(['hard', 'soft'], range(5))]
    return C

def cfgC():
    # extra-long runs: does the hard-vs-distilled gap close with 2.5x more steps (h=48) / 5x more (h=128)?
    C = [dict(exp='E6', tag='C', h=48, method=m, seed=s, steps=250000)
         for m, s in itertools.product(['hard', 'soft_teacher'], range(2))]
    C += [dict(exp='E6', tag='C', h=128, method='hard', seed=s, steps=200000) for s in range(2)]
    return C

if __name__ == '__main__':
    which = sys.argv[1]
    cfgs = {'A': cfgA, 'B': cfgB, 'C': cfgC}[which]()
    cfgs.sort(key=lambda c: -c['steps'] * (1 + c['h'] / 64) * (1.4 if c['method'] == 'soft_teacher' else 1))
    fp = os.path.join(OUT, f'e6_runs_{which}.json')
    res = json.load(open(fp)) if os.path.exists(fp) else []          # resume: skip configs already finished
    done = {(r['h'], r['method'], r['seed'], r['steps']) for r in res}
    cfgs = [c for c in cfgs if (c['h'], c['method'], c['seed'], c['steps']) not in done]
    t0 = time.time()
    with Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(job, cfgs)):
            res.append(r); print(f"{len(res)}/{len(cfgs)+len(done)} {r['h']} {r['method']} s{r['seed']} {r['steps']} K={r['learned']} A={r['wscore']:.4f} {time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(os.path.join(OUT, f'e6_runs_{which}.json'), 'w'))
    print('done', round(time.time() - t0))
