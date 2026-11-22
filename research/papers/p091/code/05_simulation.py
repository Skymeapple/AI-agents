"""Simulation study: (i) naive wrong-key-agreement contamination test is confounded by audit errors;
(ii) the mixture model controls size and has power against item-homogeneous memorisation;
(iii) item-concentrated memorisation is absorbed into the audit-error class (non-identifiability, Prop. 2)."""
import numpy as np, json, os, itertools
from multiprocessing import Pool
from scipy import stats
from common import *
import mixture
from ok_item_params import ok_params
R, resp, models, et, gold, corr = load_all(); acc, th = ability(resp, et, gold)
okp = ok_params(); BB = ((okp['a_q05'], 12.0), (float(np.log(np.expm1(0.2))), 6.0))
I = 101; M = len(th); V = np.ones((M, I), bool)
def lam_curve(lmax):
    if lmax == 0: return np.zeros(M)
    g1 = 1.5; g0 = np.log(lmax / (1 - lmax)) - g1 * th.max()
    return sig(g0 + g1 * th)
def gen(eta, lmax, rng, concentrated=0):
    cls = rng.random(I) < eta
    lam = lam_curve(lmax); Y = np.zeros((M, I)); memitems = np.zeros(I, bool)
    if concentrated:
        idx = rng.choice(np.where(~cls)[0], concentrated, replace=False); memitems[idx] = True
    for i in range(I):
        if cls[i]:
            a = max(rng.normal(okp['a_q50'], 2.0), okp['a_q05']); b = abs(rng.normal(okp['b_q50'], 0.6)) + 0.2
            p = sig(a + b * th)
        else:
            c = rng.normal(-3.0, 1.0); d = abs(rng.normal(0.47, 0.3))
            base = sig(c - d * th)
            if concentrated:
                # memorisation concentrated on a few items: top models reproduce the key with prob 0.9*rank weight
                l = 0.9 * sig(3 * (th - 0.5)) if memitems[i] else 0.0
                p = l + (1 - l) * base
            else:
                p = lam + (1 - lam) * base
        Y[:, i] = rng.random(M) < p
    return Y, cls, memitems
def one(args):
    eta, lmax, conc, seed = args
    rng = np.random.default_rng(seed)
    Y, cls, memitems = gen(eta, lmax, rng, conc)
    W = Y.mean(1)
    rho, p = stats.spearmanr(th, W)
    naive_pos = (rho > 0) and (p / 2 < 0.05)
    f0 = mixture.fit(Y, V, th, False, n_iter=40, boundsB=BB)
    f1 = mixture.fit(Y, V, th, True, n_iter=40, boundsB=BB, init=f0)
    LR = max(0.0, 2 * (f1['loglik'] - f0['loglik']))
    auc = np.nan
    if 0 < cls.sum() < I:
        auc = stats.mannwhitneyu(f0['r'][cls], f0['r'][~cls]).statistic / (cls.sum() * (~cls).sum())
    return dict(eta=eta, lmax=lmax, conc=conc, seed=seed, rho=float(rho), naive_pos=bool(naive_pos), LR=float(LR),
                lam_top_hat=float(f1['lam'].max()), lam_top_true=float(lam_curve(lmax).max()),
                pi_hat=float(f0['pi']), n_B_true=int(cls.sum()), auc_class=float(auc),
                post_memitems=float(f0['r'][memitems].mean()) if conc else np.nan)
if __name__ == '__main__':
    REPS = int(os.environ.get('REPS', 40))
    jobs = [(eta, l, 0, 1000 * k + s) for k, (eta, l) in enumerate(itertools.product([0.0, 0.1, 0.15, 0.25], [0.0, 0.01, 0.02, 0.05])) for s in range(REPS)]
    jobs += [(0.1, 0.0, c, 99000 + 100 * c + s) for c in (5, 10) for s in range(REPS)]
    with Pool(os.cpu_count()) as pool: res = pool.map(one, jobs, chunksize=2)
    json.dump(res, open(os.path.join(RES, 'simulation_raw.json'), 'w'))
    # summarise; LR critical value calibrated from lam=0 runs at the same eta
    summ = []
    for eta in [0.0, 0.1, 0.15, 0.25]:
        h0 = np.array([r['LR'] for r in res if r['eta'] == eta and r['lmax'] == 0 and r['conc'] == 0])
        crit = max(np.quantile(h0, 0.95), 1e-6)
        for l in [0.0, 0.01, 0.02, 0.05]:
            rr = [r for r in res if r['eta'] == eta and r['lmax'] == l and r['conc'] == 0]
            summ.append(dict(eta=eta, lam_top=l, naive_reject=float(np.mean([r['naive_pos'] for r in rr])),
                             mean_rho=float(np.mean([r['rho'] for r in rr])),
                             mixture_reject=float(np.mean([r['LR'] > crit for r in rr])), LR_crit=float(crit),
                             mixture_reject_chibar=float(np.mean([r['LR'] > 4.23 for r in rr])),
                             lam_top_hat_median=float(np.median([r['lam_top_hat'] for r in rr])),
                             pi_hat_mean=float(np.mean([r['pi_hat'] for r in rr])),
                             auc_class_mean=float(np.nanmean([r['auc_class'] for r in rr]))))
    conc = []
    for c in (5, 10):
        rr = [r for r in res if r['conc'] == c]
        conc.append(dict(n_memorised_items=c, naive_reject=float(np.mean([r['naive_pos'] for r in rr])),
                         mixture_reject_chibar=float(np.mean([r['LR'] > 4.23 for r in rr])),
                         mean_posterior_audit_error_on_memorised_items=float(np.mean([r['post_memitems'] for r in rr])),
                         pi_hat_mean=float(np.mean([r['pi_hat'] for r in rr]))))
    json.dump(dict(reps=REPS, grid=summ, concentrated=conc), open(os.path.join(RES, 'simulation_summary.json'), 'w'), indent=1)
    for s in summ: print(s)
    for c in conc: print(c)
