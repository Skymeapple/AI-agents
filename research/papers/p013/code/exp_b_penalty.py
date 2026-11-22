"""Experiment B: penalised grading with Zipf-distributed queries.
Universe of NU random facts, query probability p_i ~ i^-s. MLPs of three sizes are trained on
queries sampled from p (infinite-data limit: isolates capacity from Good-Turing missing mass),
plus fresh random keys labelled ABSTAIN with weight w. For each penalty lam we pick the
score-maximising (w, threshold) and record optimal hallucination h*, false-accept eps*, knowledge.
Outputs results/exp_b.json (per-run curves) ; analysis in analyze.py"""
import json, sys, time, itertools, numpy as np, torch, multiprocessing as mp
from common import *
NU, S, V = 16384, 1.2, 16
NTEST = 200_000
LAMS = [0.0, 0.5, 1, 3, 9, 30, 100]

def zipf_p():
    p = np.arange(1, NU + 1, dtype=float) ** -S
    return p / p.sum()

def train_eval(cfg):
    torch.set_num_threads(1)
    h, w, seed, steps = cfg['h'], cfg['w'], cfg['seed'], cfg['steps']
    gen = torch.Generator().manual_seed(4242 + seed); torch.manual_seed(seed)
    X = rand_keys(NU, gen); y = torch.randint(0, V, (NU,), generator=gen)
    p = torch.tensor(zipf_p())
    net = make_net(h, V); P = nparams(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    B = 2048
    for t in range(steps):
        idx = torch.multinomial(p, B, replacement=True)
        loss = loss_fn(net, X[idx], y[idx], rand_keys(B), V, w)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    cs, ans = predict(net, X, V)
    ct, _ = predict(net, rand_keys(NTEST, torch.Generator().manual_seed(7 + seed)), V)
    correct = ans == y.numpy(); pn = p.numpy()
    taus = np.unique(np.concatenate([np.quantile(cs, np.linspace(0, 1, 81)), np.quantile(ct, 1 - np.geomspace(1e-5, 1, 60)), [1.01]]))
    curve = []
    for tau in taus:
        acc = cs >= tau
        curve.append(dict(tau=float(tau), know=float(pn[acc & correct].sum()), wrong=float(pn[acc & ~correct].sum()),
                          eps=float((ct >= tau).mean()), n_correct=int((acc & correct).sum())))
    return dict(cfg=cfg, P=P, curve=curve)

if __name__ == '__main__':
    steps = 4000
    cfgs = [dict(h=h, w=w, seed=sd, steps=steps) for h, w, sd in itertools.product([32, 64, 128], [0.03, 0.1, 0.3, 1, 3, 10], [0, 1])]
    t0 = time.time(); out = []
    with mp.Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(train_eval, cfgs)):
            out.append(r); c = r['cfg']
            d = min(r['curve'], key=lambda q: abs(q['tau'] - 0.5))
            print(f"[{i+1}/{len(cfgs)} {time.time()-t0:.0f}s] h={c['h']} w={c['w']} seed={c['seed']} P={r['P']} @0.5: know={d['know']:.3f} wrong={d['wrong']:.4f} eps={d['eps']:.4f}", flush=True)
    json.dump(out, open('../results/exp_b.json', 'w'))
