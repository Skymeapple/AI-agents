"""Quantities used in the formal section:
 (1) empirical support for assumption A1 (distractor attraction non-increasing in ability) on OK items;
 (2) the confound threshold eta*(theta) of Proposition 3 from the fitted mixture;
 (3) distribution-free per-model upper bounds on memorisation (Proposition 4)."""
import numpy as np, json, os
from scipy import stats
from scipy.optimize import minimize
from common import *
import mixture
from ok_item_params import ok_params
R, resp, models, et, gold, corr = load_all(); acc, th = ability(resp, et, gold)
out = {}
# (1) modal-distractor attraction vs ability on OK items
ok = np.where(et == 'ok')[0]; sl = []
for k in ok:
    v = resp[:, k] >= 0; r = resp[v, k]; t = th[v]
    counts = np.bincount(r[r != gold[k]], minlength=4) if (r != gold[k]).any() else None
    if counts is None or counts.max() < 3: continue
    dm = counts.argmax(); y = (r == dm).astype(float)
    f = lambda p: -(y * np.log(sig(p[0] + p[1] * t) + 1e-9) + (1 - y) * np.log(1 - sig(p[0] + p[1] * t) + 1e-9)).sum() + 0.01 * (p ** 2).sum()
    sl.append(minimize(f, [-1, 0], method='L-BFGS-B').x[1])
sl = np.array(sl)
out['A1_support'] = dict(n_items=int(len(sl)), frac_negative_slope=float((sl < 0).mean()), median_slope=float(np.median(sl)),
                         note='modal distractor chosen by >=3 models; slope of logit P(choose it) on standardised ability')
# (2) eta*(theta)
wg = np.where((et == 'wrong_groundtruth') & (corr >= 0) & (corr != gold))[0]
V = resp[:, wg] >= 0; Y = ((resp[:, wg] == gold[wg]) & V).astype(float)
okp = ok_params(); BB = ((okp['a_q05'], 12.0), (float(np.log(np.expm1(0.2))), 6.0))
f0 = mixture.fit(Y, V, th, False, boundsB=BB)
grid = np.linspace(th.min(), th.max(), 50); h = 1e-3
def avg_curve(cls, t):
    if cls == 'A': return np.mean([mixture._pA(p, t, 0) for p in f0['pA']], axis=0)
    return np.mean([mixture._pB(p, t) for p in f0['pB']], axis=0)
# weight items by posterior class membership
wA = 1 - f0['r']; wB = f0['r']
PA = lambda t: np.sum([w * mixture._pA(p, t, 0) for w, p in zip(wA, f0['pA'])], axis=0) / wA.sum()
PB = lambda t: np.sum([w * mixture._pB(p, t) for w, p in zip(wB, f0['pB'])], axis=0) / wB.sum()
dA = (PA(grid + h) - PA(grid - h)) / (2 * h); dB = (PB(grid + h) - PB(grid - h)) / (2 * h)
eta_star = np.abs(dA) / (np.abs(dA) + dB)
out['confound'] = dict(eta_hat=float(f0['pi']), eta_star_median=float(np.median(eta_star)), eta_star_min=float(eta_star.min()),
                       eta_star_max=float(eta_star.max()), frac_theta_grid_where_eta_hat_exceeds=float((f0['pi'] > eta_star).mean()),
                       PA_at_min=float(PA(np.array([th.min()]))[0]), PA_at_max=float(PA(np.array([th.max()]))[0]),
                       PB_at_min=float(PB(np.array([th.min()]))[0]), PB_at_max=float(PB(np.array([th.max()]))[0]))
# (3) per-model Clopper-Pearson upper bounds on lambda_m using items with posterior audit-error < 0.5
keep = f0['r'] < 0.5
k = (Y[:, keep] * V[:, keep]).sum(1); n = V[:, keep].sum(1)
ub = stats.beta.ppf(0.95, k + 1, n - k)
order = np.argsort(-acc)
out['per_model_ucb'] = dict(n_items=int(keep.sum()), median_ucb=float(np.median(ub)), top10_ucb=[dict(model=models[i], k=int(k[i]), n=int(n[i]), ucb95=float(ub[i])) for i in order[:10]],
                            max_ucb_top20=float(ub[order[:20]].max()), min_ucb_top20=float(ub[order[:20]].min()))
np.savez(os.path.join(RES, 'theory_arrays.npz'), grid=grid, eta_star=eta_star, PAg=PA(grid), PBg=PB(grid), ucb=ub, A1_slopes=sl)
json.dump(out, open(os.path.join(RES, 'theory_quantities.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
