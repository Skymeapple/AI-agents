"""Experiment C (new in revision): penalised grading in a universe much larger than capacity.
Infinite Zipf(s) world p_i = i^-s / zeta(s). Facts i <= NU = 2^20 have explicit random keys/values and are
the 'observable' world: training queries are sampled from p restricted to i <= NU (so rarely-queried facts
are seen rarely or never). Facts i > NU (mass U_out) are never trained on; their keys are independent of
the model, so their hallucination is exactly (1-1/V) * eps * U_out, with eps measured on fresh keys.
MLP capacity (<= ~5e4 bits) is far below NU*H, so capacity is the bottleneck for what is stored.
Two independent fresh-key samples (A, B) allow held-out threshold selection.
Outputs results/exp_c.json"""
import json, sys, time, itertools, math, numpy as np, torch, multiprocessing as mp
from scipy.special import zeta
from common import *
NU, V = 2 ** 20, 16
NTEST = 200_000
STEPS, BATCH = 3000, 2048

def world(s):
    i = np.arange(1, NU + 1, dtype=float)
    p = i ** -s / zeta(s)                     # absolute query probabilities of in-universe facts
    U_out = float(zeta(s, NU + 1.0) / zeta(s))
    return p, U_out

def train_eval(cfg):
    torch.set_num_threads(1)
    s, h, w, seed = cfg['s'], cfg['h'], cfg['w'], cfg['seed']
    gen = torch.Generator().manual_seed(9100 + seed); torch.manual_seed(1000 + seed)
    X = rand_keys(NU, gen); y = torch.randint(0, V, (NU,), generator=gen)
    p, U_out = world(s)
    cdf = np.cumsum(p / p.sum()); rng = np.random.default_rng(500 + seed)
    seen = np.zeros(NU, dtype=np.int64)
    net = make_net(h, V); P = nparams(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for t in range(STEPS):
        idx = np.minimum(np.searchsorted(cdf, rng.random(BATCH)), NU - 1)
        seen += np.bincount(idx, minlength=NU)
        it = torch.from_numpy(idx)
        loss = loss_fn(net, X[it], y[it], rand_keys(BATCH), V, w)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    cs, ans = predict(net, X, V)
    cA, _ = predict(net, rand_keys(NTEST, torch.Generator().manual_seed(77 + seed)), V)
    cB, _ = predict(net, rand_keys(NTEST, torch.Generator().manual_seed(88 + seed)), V)
    correct = ans == y.numpy(); unseen = seen == 0
    top = cs[:16384]
    taus = np.unique(np.concatenate([np.quantile(top, np.linspace(0, 1, 61)),
                                     np.quantile(cA, 1 - np.geomspace(2e-5, 1, 50)), [0.5, 1.01]]))
    order = np.argsort(cs); cs_s = cs[order]
    pc = np.concatenate([[0], np.cumsum((p * correct)[order])])
    pw = np.concatenate([[0], np.cumsum((p * ~correct)[order])])
    pwu = np.concatenate([[0], np.cumsum((p * (~correct & unseen))[order])])
    nc = np.concatenate([[0], np.cumsum(correct[order])])
    sA, sB = np.sort(cA), np.sort(cB)
    curve = []
    for tau in taus:
        j = np.searchsorted(cs_s, tau, side='left')     # facts with conf >= tau are order[j:]
        epsA = 1 - np.searchsorted(sA, tau, side='left') / NTEST
        epsB = 1 - np.searchsorted(sB, tau, side='left') / NTEST
        curve.append(dict(tau=float(tau), know_in=float(pc[-1] - pc[j]), wrong_in=float(pw[-1] - pw[j]),
                          wrong_unseen=float(pwu[-1] - pwu[j]), n_correct=int(nc[-1] - nc[j]),
                          epsA=float(epsA), epsB=float(epsB)))
    return dict(cfg=cfg, P=P, U_out=U_out, frac_mass_unseen=float(p[unseen].sum()),
                n_seen=int((~unseen).sum()), curve=curve)

if __name__ == '__main__':
    quick = len(sys.argv) > 1 and sys.argv[1] == 'quick'
    widths = [16, 32, 64, 128]
    cfgs = [dict(s=s, h=h, w=w, seed=sd) for s, h, w, sd in
            itertools.product([1.2, 1.5], widths, [0.3, 1, 3], [0, 1, 2, 3, 4, 5])]
    cfgs.sort(key=lambda c: -c['h'])            # big jobs first for load balance
    if quick: cfgs = [dict(s=1.2, h=128, w=1, seed=0)]
    t0 = time.time(); out = []
    with mp.Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(train_eval, cfgs)):
            out.append(r); c = r['cfg']
            d = min(r['curve'], key=lambda q: abs(q['tau'] - 0.5))
            print(f"[{i+1}/{len(cfgs)} {time.time()-t0:.0f}s] s={c['s']} h={c['h']} w={c['w']} seed={c['seed']} P={r['P']} "
                  f"@0.5: know={d['know_in']:.3f} wrong={d['wrong_in']:.4f} epsA={d['epsA']:.4f} n_seen={r['n_seen']}", flush=True)
            if not quick and (i + 1) % 8 == 0: json.dump(out, open('../results/exp_c.json', 'w'))
    json.dump(out, open('../results/exp_c' + ('_quick' if quick else '') + '.json', 'w'))
