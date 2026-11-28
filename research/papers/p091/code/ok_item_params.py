"""Fit per-item logistic P(correct | theta) on MMLU-Redux OK items to obtain the empirical range of
intercepts/slopes for items whose key is right. Used to constrain the 'audit error' class."""
import numpy as np, json, os
from scipy.optimize import minimize
from common import *
def ok_params():
    fn = os.path.join(RES, 'ok_item_params.json')
    if os.path.exists(fn): return json.load(open(fn))
    R, resp, models, et, gold, corr = load_all(); acc, th = ability(resp, et, gold)
    ok = np.where(et == 'ok')[0]; A = []; B = []
    for k in ok:
        v = resp[:, k] >= 0; y = (resp[v, k] == gold[k]).astype(float); t = th[v]
        f = lambda p: -(y * np.log(sig(p[0] + p[1] * t) + 1e-9) + (1 - y) * np.log(1 - sig(p[0] + p[1] * t) + 1e-9)).sum() + 0.01 * (p ** 2).sum()
        p = minimize(f, [1.0, 1.0], method='L-BFGS-B').x; A.append(p[0]); B.append(p[1])
    A, B = np.array(A), np.array(B)
    d = dict(n=len(ok), a_q05=float(np.quantile(A, .05)), a_q50=float(np.median(A)), a_q95=float(np.quantile(A, .95)),
             b_q05=float(np.quantile(B, .05)), b_q50=float(np.median(B)), b_q95=float(np.quantile(B, .95)),
             frac_b_positive=float((B > 0).mean()))
    json.dump(d, open(fn, 'w'), indent=1); return d
if __name__ == '__main__': print(ok_params())
