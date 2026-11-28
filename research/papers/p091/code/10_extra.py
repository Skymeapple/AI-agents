"""Extra quantities: (1) agreement of mixture posterior with independent adjudication;
(2) per-model Clopper-Pearson bounds on memorisation using ALL usable flagged items (no outcome-based selection);
(3) item-level counts used in the text."""
import numpy as np, json, os, csv
from scipy import stats
from common import *
R, resp, models, et, gold, corr = load_all(); acc, th = ability(resp, et, gold)
rows = list(csv.DictReader(open(os.path.join(RES, 'wg_items.csv'))))
post = np.array([float(r['post_audit_error']) for r in rows]); adj = np.array([r['adjudication'] for r in rows])
out = {}
g = adj == 'GOLD_DEFENSIBLE'; a = adj == 'AUDIT_CORRECT'
out['auc_post_gold_vs_audit'] = float(stats.mannwhitneyu(post[g], post[a]).statistic / (g.sum() * a.sum()))
rev = adj != 'not_reviewed'
out['n_reviewed'] = int(rev.sum()); out['n_reviewed_post_gt_half'] = int((post[rev] > .5).sum())
out['n_not_reviewed_post_gt_half'] = int((post[~rev] > .5).sum())
out['reviewed_counts'] = {k: int((adj == k).sum()) for k in ('GOLD_DEFENSIBLE', 'AUDIT_CORRECT', 'AMBIGUOUS')}
# (2) CP bounds with all flagged items (conservative: class-B items only inflate k)
wg = np.where((et == 'wrong_groundtruth') & (corr >= 0) & (corr != gold))[0]
V = resp[:, wg] >= 0; Y = ((resp[:, wg] == gold[wg]) & V)
k = Y.sum(1); n = V.sum(1); ub = stats.beta.ppf(0.95, k + 1, n - k)
order = np.argsort(-acc)
out['cp_all_items'] = dict(n_items=int(len(wg)), top5=[dict(model=models[i], k=int(k[i]), n=int(n[i]), ucb95=float(ub[i])) for i in order[:5]],
                           median_ucb_all_models=float(np.median(ub)), min_ucb=float(ub.min()))
# (2b) bounds excluding only items adjudicated GOLD_DEFENSIBLE or AMBIGUOUS (selection independent of posterior)
keep = ~np.isin(adj, ['GOLD_DEFENSIBLE', 'AMBIGUOUS'])
k2 = Y[:, keep].sum(1); n2 = V[:, keep].sum(1); ub2 = stats.beta.ppf(0.95, k2 + 1, n2 - k2)
out['cp_excl_adjudicated_defensible'] = dict(n_items=int(keep.sum()), top5=[dict(model=models[i], k=int(k2[i]), n=int(n2[i]), ucb95=float(ub2[i])) for i in order[:5]],
                                             median_ucb_top20=float(np.median(ub2[order[:20]])))
# (3) accuracy range of models on OK items
out['ok_acc_range'] = [float(acc.min()), float(acc.max())]
json.dump(out, open(os.path.join(RES, 'extra_quantities.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
