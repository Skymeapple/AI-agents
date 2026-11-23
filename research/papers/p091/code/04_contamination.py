"""Wrong-key agreement analysis on MMLU-Redux 'wrong_groundtruth' items x 80 HELM models."""
import numpy as np, json, csv, os, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from common import *
import mixture
R, resp, models, et, gold, corr = load_all()
acc, th = ability(resp, et, gold)
wg = np.where((et == 'wrong_groundtruth') & (corr >= 0) & (corr != gold))[0]
V = resp[:, wg] >= 0
Y = ((resp[:, wg] == gold[wg]) & V).astype(float)   # chose original (flagged) key
C = ((resp[:, wg] == corr[wg]) & V).astype(float)   # chose audit's corrected key
out = {'n_models': len(models), 'n_wg_items_usable': int(len(wg)), 'n_wg_items_total': int((et == 'wrong_groundtruth').sum())}
W = (Y * V).sum(1) / V.sum(1); Cm = (C * V).sum(1) / V.sum(1)
# ---- 1. naive analysis (assumes the audit is perfect)
lr = stats.linregress(th, W); lrC = stats.linregress(th, Cm)
out['naive'] = dict(W_mean=float(W.mean()), W_min=float(W.min()), W_max=float(W.max()),
                    slope_W_on_theta=float(lr.slope), slope_p=float(lr.pvalue), r=float(lr.rvalue),
                    slope_C_on_theta=float(lrC.slope), r_C=float(lrC.rvalue),
                    spearman_W=float(stats.spearmanr(th, W)[0]), spearman_W_p=float(stats.spearmanr(th, W)[1]))
# clean-model expectation if audit were perfect: as theta grows, W should fall (P(g) <= 1-P(truth))
top = np.argsort(acc)[-20:]; bot = np.argsort(acc)[:20]
out['naive']['W_top20'] = float(W[top].mean()); out['naive']['W_bottom20'] = float(W[bot].mean())
# ---- 2. item-level ability slopes
slopes = []
for j in range(len(wg)):
    v = V[:, j]; y = Y[v, j]; t = th[v]
    if y.sum() in (0, v.sum()):
        slopes.append((np.nan, np.nan)); continue
    import statsmodels.api as sm
    try:
        mfit = sm.Logit(y, sm.add_constant(t)).fit(disp=0, method='bfgs', maxiter=200)
        slopes.append((mfit.params[1], mfit.bse[1]))
    except Exception:
        slopes.append((np.nan, np.nan))
slopes = np.array(slopes)
Wtop = Y[top].mean(0); Wbot = Y[bot].mean(0)
# ---- 3. mixture model, with and without memorisation
from ok_item_params import ok_params
okp = ok_params()
BB = ((okp['a_q05'], 12.0), (float(np.log(np.expm1(0.2))), 6.0))   # class-B items must look like right-keyed items
out['class_B_bounds'] = dict(a_min=okp['a_q05'], slope_min=0.2, source='5th pct of OK-item logistic intercepts')
t0 = time.time()
f0 = mixture.fit(Y, V, th, memorisation=False, boundsB=BB)
f1 = mixture.fit(Y, V, th, memorisation=True, boundsB=BB, init=f0)
LR = max(0.0, 2 * (f1['loglik'] - f0['loglik']))
out['mixture'] = dict(loglik_mem=float(f1['loglik']), loglik_nomem=float(f0['loglik']), LR=float(LR),
                      pi_audit_error=float(f0['pi']), pi_audit_error_mem=float(f1['pi']),
                      g=[float(x) for x in f1['g']], lam_at_min_theta=float(f1['lam'].min()),
                      lam_at_max_theta=float(f1['lam'].max()), lam_median=float(np.median(f1['lam'])),
                      n_items_post_B_gt_0p5=int((f0['r'] > .5).sum()), expected_n_audit_errors=float(f0['r'].sum()),
                      fit_seconds=time.time() - t0)
rng = np.random.default_rng(1)
NB = int(os.environ.get('NBOOT', 40)); LRb = []; lam_top_b = []
for b in range(NB):
    cls = rng.random(len(wg)) < f0['r']
    Yb = np.zeros_like(Y)
    for j in range(len(wg)):
        p = mixture._pB(f0['pB'][j], th) if cls[j] else mixture._pA(f0['pA'][j], th, 0)
        Yb[:, j] = (rng.random(len(th)) < p) & V[:, j]
    a0 = mixture.fit(Yb, V, th, False, n_iter=40, boundsB=BB); a1 = mixture.fit(Yb, V, th, True, n_iter=40, boundsB=BB, init=a0)
    LRb.append(max(0, 2 * (a1['loglik'] - a0['loglik']))); lam_top_b.append(a1['lam'].max())
LRb = np.array(LRb)
out['mixture']['bootstrap_LR_p'] = float((1 + (LRb >= LR - 1e-9).sum()) / (1 + NB)); out['mixture']['n_boot'] = NB
out['mixture']['bootstrap_LR_q95'] = float(np.quantile(LRb, .95))
# profile likelihood for fixed memorisation patterns
def prof(lamv):
    f = mixture.fit(Y, V, th, memorisation=True, boundsB=BB, init=f0, lam_fixed=lamv, n_iter=40)
    return f['loglik']
grid = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.05]
topmask = np.zeros(len(th)); topmask[top] = 1
pc = np.array([prof(np.full(len(th), l)) for l in grid])
pt = np.array([prof(l * topmask) for l in grid])
def ub(ll):
    dev = 2 * (ll.max() - ll); okl = [l for l, d in zip(grid, dev) if d <= 2.71]   # one-sided 95%
    return float(max(okl)), dev.tolist()
out['mixture']['profile_all_models'] = dict(grid=grid, upper95=ub(pc)[0], dev=ub(pc)[1])
out['mixture']['profile_top20_only'] = dict(grid=grid, upper95=ub(pt)[0], dev=ub(pt)[1])
# ---- 4. compare posterior class with author adjudication
adj = {int(r_['redux_row']): r_['verdict'] for r_ in csv.DictReader(open(os.path.join(ROOT, 'data', 'author_adjudication.csv')))}
rows = []
for j, k in enumerate(wg):
    rows.append(dict(redux_row=int(k), subject=R[k]['subject'], W_bottom20=float(Wbot[j]), W_top20=float(Wtop[j]),
                     C_top20=float(C[top, j].mean()), slope=float(slopes[j, 0]), slope_se=float(slopes[j, 1]),
                     post_audit_error=float(f0['r'][j]), post_audit_error_mem=float(f1['r'][j]),
                     adjudication=adj.get(int(k), 'not_reviewed'), question=R[k]['question'][:120]))
with open(os.path.join(RES, 'wg_items.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
cnt = {}
for r_ in rows:
    key = r_['adjudication']; d = cnt.setdefault(key, dict(n=0, post_gt_half=0, mean_post=0.0, mean_Wtop=0.0, mean_Wbot=0.0))
    d['n'] += 1; d['post_gt_half'] += r_['post_audit_error'] > .5; d['mean_post'] += r_['post_audit_error']; d['mean_Wtop'] += r_['W_top20']; d['mean_Wbot'] += r_['W_bottom20']
for d in cnt.values():
    for k_ in ('mean_post', 'mean_Wtop', 'mean_Wbot'): d[k_] /= d['n']
out['adjudication_vs_posterior'] = cnt
# ---- 5. re-run naive analysis after removing likely audit errors (posterior > 0.5 or GOLD_DEFENSIBLE/AMBIGUOUS)
keep = np.array([(r_['post_audit_error'] <= .5) and r_['adjudication'] in ('not_reviewed', 'AUDIT_CORRECT') for r_ in rows])
W2 = (Y[:, keep] * V[:, keep]).sum(1) / V[:, keep].sum(1)
lr2 = stats.linregress(th, W2)
out['cleaned'] = dict(n_items=int(keep.sum()), W_mean=float(W2.mean()), slope=float(lr2.slope), p=float(lr2.pvalue),
                      W_top20=float(W2[top].mean()), W_bottom20=float(W2[bot].mean()),
                      spearman=float(stats.spearmanr(th, W2)[0]), spearman_p=float(stats.spearmanr(th, W2)[1]))
# per-model outlier test on cleaned items: compare with logistic-in-theta fit across models (binomial GLM)
import statsmodels.api as sm
k_ = (Y[:, keep] * V[:, keep]).sum(1); n_ = V[:, keep].sum(1)
glm = sm.GLM(np.c_[k_, n_ - k_], sm.add_constant(th), family=sm.families.Binomial()).fit()
mu = glm.predict(sm.add_constant(th))
z = (k_ - n_ * mu) / np.sqrt(n_ * mu * (1 - mu))
pvals = stats.binom.sf(k_ - 1, n_, mu)
order = np.argsort(-z)
out['cleaned']['glm_slope'] = float(glm.params[1]); out['cleaned']['glm_slope_p'] = float(glm.pvalues[1])
out['cleaned']['overdispersion_pearson_chi2_df'] = float(glm.pearson_chi2 / glm.df_resid)
from statsmodels.stats.multitest import multipletests
rej = multipletests(pvals, method='fdr_bh', alpha=.1)[0]
out['cleaned']['n_models_flagged_fdr10'] = int(rej.sum())
out['cleaned']['top5_models_by_excess'] = [dict(model=models[i], k=int(k_[i]), n=int(n_[i]), expected=float(n_[i] * mu[i]), z=float(z[i]), p=float(pvals[i])) for i in order[:5]]
# ---- 6. verified-wrong-key computational items (AUDIT_CORRECT): ability trend
ac = np.array([r_['adjudication'] == 'AUDIT_CORRECT' for r_ in rows])
Wac = (Y[:, ac] * V[:, ac]).sum(1) / V[:, ac].sum(1)
lrac = stats.linregress(th, Wac)
k3 = (Y[:, ac] * V[:, ac]).sum(1); n3 = V[:, ac].sum(1)
glm3 = sm.GLM(np.c_[k3, n3 - k3], sm.add_constant(th), family=sm.families.Binomial()).fit()
out['verified_wrong_key_items'] = dict(n_items=int(ac.sum()), W_top20=float(Wac[top].mean()), W_bottom20=float(Wac[bot].mean()),
    glm_slope=float(glm3.params[1]), glm_p=float(glm3.pvalues[1]), chance_if_uniform_over_3_wrong=1/3,
    per_item=[dict(row=r_['redux_row'], W_bot=r_['W_bottom20'], W_top=r_['W_top20'], C_top=r_['C_top20']) for r_ in rows if r_['adjudication'] == 'AUDIT_CORRECT'])
json.dump(out, open(os.path.join(RES, 'contamination_results.json'), 'w'), indent=2)
np.savez(os.path.join(RES, 'contam_arrays.npz'), th=th, acc=acc, W=W, Cm=Cm, W2=W2, Wac=Wac, slopes=slopes,
         post=f0['r'], post_mem=f1['r'], lam=f1['lam'], LRb=LRb, Wtop=Wtop, Wbot=Wbot, adj=np.array([r_['adjudication'] for r_ in rows]))
print(json.dumps(out, indent=1)[:6000])
