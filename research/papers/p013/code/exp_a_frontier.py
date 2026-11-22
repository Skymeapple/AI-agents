"""Experiment A: memory-limited MLPs memorize k random key->value facts with an abstain
class; fresh random keys labelled ABSTAIN with weight w. For each trained model we sweep
the abstention threshold and measure (recall g, false-acceptance eps), then compute the
Theorem-1 certified information g(H+log2 1/eps)-... . Outputs results/exp_a.json"""
import json, math, sys, time, itertools, numpy as np, torch, multiprocessing as mp
from common import *
NTEST = 200_000

def train_eval(cfg):
    torch.set_num_threads(1)
    k, h, V, w, seed, steps = cfg['k'], cfg['h'], cfg['V'], cfg['w'], cfg['seed'], cfg['steps']
    head = cfg.get('head', 'softmax')
    gen = torch.Generator().manual_seed(1000 * seed + 7)
    torch.manual_seed(seed)
    X = rand_keys(k, gen); y = torch.randint(0, V, (k,), generator=gen)
    net = make_net(h, V, head=head); P = nparams(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    nb = max(k, 1024)
    for t in range(steps):
        Xn = rand_keys(nb)
        loss = loss_fn(net, X, y, Xn, V, w, head)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    cs, ans = predict(net, X, V, head)
    ct, _ = predict(net, rand_keys(NTEST, torch.Generator().manual_seed(99 + seed)), V, head)
    correct = ans == y.numpy()
    taus = np.unique(np.concatenate([[0.5], np.quantile(cs, np.linspace(0, 1, 41)), [1.01]]))
    curve = []
    for tau in taus:
        acc = cs >= tau
        g = int((acc & correct).sum()); wrong = int((acc & ~correct).sum())
        nfp = int((ct >= tau).sum())
        eps = nfp / NTEST
        eps_c = max(eps, 3.0 / NTEST)          # conservative floor when no FP observed
        curve.append(dict(tau=float(tau), g=g, wrong_stored=wrong, eps=eps,
                          cert_bits=certified_bits(g, k, eps_c, V)))
    best = max(curve, key=lambda c: c['cert_bits'])
    dflt = [c for c in curve if abs(c['tau'] - 0.5) < 1e-12][0]
    return dict(cfg=cfg, P=P, curve=curve, best=best, default=dflt)

if __name__ == '__main__':
    quick = len(sys.argv) > 1 and sys.argv[1] == 'quick'
    steps = 600 if quick else 3000
    cfgs = []
    # A1: main grid, V=16 (H=4 bits)
    for h, k, w, seed in itertools.product([32, 64], [1024, 2048, 4096], [0.03, 0.1, 0.3, 1, 3], [0, 1]):
        cfgs.append(dict(block='A1', k=k, h=h, V=16, w=w, seed=seed, steps=steps))
    # A2: value-entropy sweep at fixed size (exchange rate test)
    # bit-output head keeps the parameter count nearly independent of V
    for V, w, seed in itertools.product([2, 4, 16, 64, 256], [0.1, 0.3, 1], [0, 1]):
        cfgs.append(dict(block='A2', k=2048, h=32, V=V, w=w, seed=seed, steps=steps, head='bits'))
    if quick: cfgs = cfgs[::12]
    t0 = time.time()
    with mp.Pool(4) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(train_eval, cfgs)):
            out.append(r); c = r['cfg']; b = r['best']; d = r['default']
            print(f"[{i+1}/{len(cfgs)} {time.time()-t0:.0f}s] {c['block']} h={c['h']} k={c['k']} V={c['V']} w={c['w']} s={c['seed']} P={r['P']} "
                  f"default g={d['g']} eps={d['eps']:.4f} | best cert={b['cert_bits']:.0f} bits ({b['cert_bits']/r['P']:.2f} b/param)", flush=True)
    json.dump(out, open('../results/exp_a' + ('_quick' if quick else '') + '.json', 'w'), indent=1)
