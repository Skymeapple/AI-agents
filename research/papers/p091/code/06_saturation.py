"""Saturation diagnostics for MMLU using 80 HELM models (full test set) + MMLU-Redux annotations."""
import numpy as np, json, os, itertools
import matplotlib; matplotlib.use('Agg')
from scipy import stats
from scipy.optimize import minimize
from common import *
R, resp, models, et, gold, corr = load_all(); acc_ok, th = ability(resp, et, gold)
z = np.load(os.path.join(ROOT, 'data', 'full_correct.npz')); F = z['M']; fsub = z['subjects']; fmodels = [str(m) for m in z['models']]
assert fmodels == models
cj = json.load(open(os.path.join(RES, 'contamination_results.json')))
piB = cj['mixture']['pi_audit_error']
out = {}
full_acc = np.nanmean(F, 1)
order = np.argsort(-full_acc)
out['full_mmlu'] = dict(n_items=int(F.shape[1]), top5=[(models[i], float(full_acc[i])) for i in order[:5]],
                        median=float(np.median(full_acc)))
# ---------- (a) label-noise ceiling (stratified by subject, weighted by full test-set subject sizes)
subj_sizes = {s: int((fsub == s).sum()) for s in np.unique(fsub)}
subjects = sorted(subj_sizes); Ntot = sum(subj_sizes.values())
gold_agree = (resp == gold[None, :])
top5 = order[:5]
flag_cats = sorted(set(et) - {'ok'})
cat_attain = {c: float(gold_agree[np.ix_(top5, np.where(et == c)[0])].mean()) for c in flag_cats}
def ceiling(idx_by_subj, mode):
    tot = 0.0
    for s in subjects:
        ii = idx_by_subj[s]; e = et[ii]
        val = np.ones(len(ii))
        for c in flag_cats:
            m = e == c
            if mode == 'central': val[m] = piB if c == 'wrong_groundtruth' else cat_attain[c]
            elif mode == 'low': val[m] = 0.0
            elif mode == 'high': val[m] = cat_attain[c] if c == 'wrong_groundtruth' else 1.0
        tot += subj_sizes[s] * val.mean()
    return tot / Ntot
idx = {s: np.array([k for k, r in enumerate(R) if r['subject'] == s]) for s in subjects}
missing = [s for s in subjects if len(idx[s]) == 0]
rng = np.random.default_rng(0)
cen = ceiling(idx, 'central'); lo = ceiling(idx, 'low'); hi = ceiling(idx, 'high')
boots = [ceiling({s: rng.choice(idx[s], len(idx[s])) for s in subjects}, 'central') for _ in range(2000)]
flag_rate = sum(subj_sizes[s] * (et[idx[s]] != 'ok').mean() for s in subjects) / Ntot
wg_rate = sum(subj_sizes[s] * (et[idx[s]] == 'wrong_groundtruth').mean() for s in subjects) / Ntot
out['ceiling'] = dict(central=cen, ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], low_bound=lo, high_bound=hi,
                      weighted_flag_rate=flag_rate, weighted_wrong_groundtruth_rate=wg_rate, piB_used=piB,
                      category_attainability_top5=cat_attain, subjects_missing=missing)
out['headroom'] = [dict(model=models[i], full_acc=float(full_acc[i]), gap_to_central=float(cen - full_acc[i])) for i in order[:5]]
# ---------- (b) error composition of top models on the Redux sample
flag = et != 'ok'
comp = []
for i in order[:10]:
    v = resp[i] >= 0; err = (resp[i] != gold) & v
    comp.append(dict(model=models[i], n_err=int(err.sum()), frac_err_on_flagged=float((err & flag).sum() / err.sum())))
out['error_composition_top10'] = comp
out['flagged_base_rate_sample'] = float(flag.mean())
# ---------- (c) pairwise discriminability on full MMLU (paired exact McNemar)
def mcnemar_p(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); a = a[m]; b = b[m]
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0, (n01 + n10) / len(a), (n01 - n10) / len(a)
disc = {}
for k in (10, 20, 80):
    S = order[:k]; ps = []; dr = []; dl = []
    for i, j in itertools.combinations(S, 2):
        p, d, delta = mcnemar_p(F[i], F[j]); ps.append(p); dr.append(d); dl.append(abs(delta))
    ps = np.array(ps)
    from statsmodels.stats.multitest import multipletests
    disc[k] = dict(n_pairs=len(ps), frac_p_lt_05=float((ps < .05).mean()), frac_holm=float(multipletests(ps, method='holm')[0].mean()),
                   median_discordance=float(np.median(dr)), median_abs_delta=float(np.median(dl)))
out['pairwise_full_mmlu'] = disc
# share of pairwise differences on the Redux sample that come from flagged items (top-10 pairs)
shares = []; flips = 0; npairs = 0
G = gold_agree.astype(float); G[resp < 0] = np.nan
for i, j in itertools.combinations(order[:10], 2):
    d_all = np.nanmean(G[i]) - np.nanmean(G[j])
    d_flag = np.nansum(G[i, flag] - G[j, flag]) / G.shape[1]
    d_ok = np.nansum(G[i, ~flag] - G[j, ~flag]) / G.shape[1]
    if abs(d_all) > 0: shares.append(abs(d_flag) / (abs(d_flag) + abs(d_ok)))
    npairs += 1; flips += np.sign(d_all) != np.sign(np.nanmean(G[i, ~flag]) - np.nanmean(G[j, ~flag]))
out['top10_pairs_flagged_share'] = dict(median=float(np.median(shares)), q75=float(np.quantile(shares, .75)),
                                        frac_order_flips_when_flagged_removed=float(flips / npairs))
acc_orig = np.nanmean(G, 1); acc_okonly = np.nanmean(G[:, ~flag], 1)
out['rank_stability'] = dict(kendall_all80=float(stats.kendalltau(acc_orig, acc_okonly)[0]),
                             kendall_top20=float(stats.kendalltau(acc_orig[order[:20]], acc_okonly[order[:20]])[0]),
                             kendall_top10=float(stats.kendalltau(acc_orig[order[:10]], acc_okonly[order[:10]])[0]))
# ---------- (d) IRT information at the frontier (theta fixed from OK-item logit accuracy; per-item 2PL)
thf = (np.log(full_acc / (1 - full_acc)) - np.log(full_acc / (1 - full_acc)).mean()) / np.log(full_acc / (1 - full_acc)).std()
A = np.zeros(F.shape[1]); B = np.zeros(F.shape[1])
for k in range(F.shape[1]):
    v = ~np.isnan(F[:, k]); y = F[v, k]; t = thf[v]
    f = lambda p: -(y * np.log(sig(p[0] + p[1] * t) + 1e-9) + (1 - y) * np.log(1 - sig(p[0] + p[1] * t) + 1e-9)).sum() + 0.05 * (p ** 2).sum()
    A[k], B[k] = minimize(f, [1.0, 1.0], method='L-BFGS-B').x
grid = np.linspace(-3, 4, 141)
P = sig(A[None, :] + B[None, :] * grid[:, None]); info = (B[None, :] ** 2 * P * (1 - P)).sum(1)
tmax = thf.max(); Pt = sig(A + B * tmax); it = (B ** 2 * Pt * (1 - Pt)).sum()
out['irt'] = dict(theta_top=float(tmax), info_at_top=float(it), info_max=float(info.max()), theta_at_info_max=float(grid[info.argmax()]),
                  ratio_top_to_max=float(it / info.max()), frac_items_p_gt_95_at_top=float((Pt > .95).mean()),
                  frac_items_negative_slope=float((B < 0).mean()),
                  se_theta_at_top=float(1 / np.sqrt(it)), se_theta_at_max=float(1 / np.sqrt(info.max())))
top10 = order[:10]; Ft = F[top10]
allsame = np.all(Ft == Ft[0:1], axis=0)
out['top10_items_with_any_disagreement'] = dict(n=int((~allsame).sum()), frac=float((~allsame).mean()))
# ---------- (e) required test size to resolve a gap delta between frontier models (paired), from observed discordance
dmed = disc[10]['median_discordance']
zab = stats.norm.ppf(.975) + stats.norm.ppf(.8)
out['required_n'] = {f'delta_{d}': float(zab ** 2 * dmed / d ** 2) for d in (0.005, 0.01, 0.02)}
out['required_n']['discordance_used'] = dmed
np.savez(os.path.join(RES, 'saturation_arrays.npz'), grid=grid, info=info, thf=thf, full_acc=full_acc, A=A, B=B,
         acc_orig=acc_orig, acc_okonly=acc_okonly, boots=np.array(boots))
json.dump(out, open(os.path.join(RES, 'saturation_results.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
