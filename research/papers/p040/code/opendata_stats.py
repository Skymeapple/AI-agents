"""Cross-conversation statistics for the designer-vs-sample comparison (replaces the
round-1 opendata_summary.txt, which was produced by an unreleased snippet).
Adds Monte-Carlo CIs (resampling the B bootstrap draws), multiverse-composition
sensitivity (random sub-multiverses) and controls.
Usage: python3 opendata_stats.py -> ../results/opendata_stats.json"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from constlib import design_space, swap_distances
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
df = pd.read_csv(os.path.join(RES, 'opendata_decomposition.csv'))
D = design_space(); fam = np.array([d['family'] for d in D]); cons = np.where(fam == 'consensus')[0]
rng = np.random.default_rng(3)
out = {'n_conversations': int(len(df))}

def ratio_ci(I, idx, R=200):
    B = I.shape[1]
    base = swap_distances(I[idx]); r0 = base['designer_swap'] / base['sample_swap']
    rs = []
    for _ in range(R):
        bb = rng.choice(B, B, replace=True)
        # duplicate draws would create zero sample-swap pairs; use unique draws
        bb = np.unique(bb)
        s = swap_distances(I[idx][:, bb]); rs.append(s['designer_swap'] / s['sample_swap'])
    return r0, np.quantile(rs, [0.025, 0.975]).tolist()

rows = []
for _, r in df.iterrows():
    I = np.load(os.path.join(RES, 'opendata_inclusion', r.conversation + '.npy'))
    rc, cic = ratio_ci(I, cons)
    rf, cif = ratio_ci(I, np.arange(len(D)))
    # multiverse-composition sensitivity: 100 random sub-multiverses (half of the consensus designs,
    # plus a random subset of non-consensus rules)
    sub = []
    for _ in range(100):
        idx = np.concatenate([rng.choice(cons, len(cons) // 2, replace=False),
                              rng.choice(np.where(fam != 'consensus')[0], rng.integers(0, 8), replace=False)])
        s = swap_distances(I[idx]); sub.append(s['designer_swap'] / s['sample_swap'])
    rows.append(dict(conversation=r.conversation, n=r.n, m=r.m, k=r.k, density=r.density,
                     ratio_cons=rc, ratio_cons_lo=cic[0], ratio_cons_hi=cic[1],
                     ratio_full=rf, ratio_full_lo=cif[0], ratio_full_hi=cif[1],
                     ratio_sub_q05=float(np.quantile(sub, .05)), ratio_sub_q95=float(np.quantile(sub, .95)),
                     cons_designer_swap=r.cons_designer_swap, cons_sample_swap=r.cons_sample_swap))
    print(rows[-1], flush=True)
st = pd.DataFrame(rows); st.to_csv(os.path.join(RES, 'opendata_ratio_ci.csv'), index=False)
sp = lambda a, b: [float(x) for x in spearmanr(a, b)]
out['spearman_n_ratio_cons'] = sp(st.n, st.ratio_cons)
out['spearman_n_ratio_full'] = sp(st.n, st.ratio_full)
out['spearman_n_sample_swap_cons'] = sp(st.n, st.cons_sample_swap)
out['spearman_n_designer_swap_cons'] = sp(st.n, st.cons_designer_swap)
out['spearman_n_ratio_sub_q05'] = sp(st.n, st.ratio_sub_q05)
out['wilcoxon_designer_vs_sample_cons_p'] = float(wilcoxon(df.cons_designer_swap, df.cons_sample_swap).pvalue)
out['wilcoxon_designer_vs_sample_full_p'] = float(wilcoxon(df.full_designer_swap, df.full_sample_swap).pvalue)
out['n_cons_ratio_ci_above_1'] = int((st.ratio_cons_lo > 1).sum())
out['n_cons_ratio_ci_below_1'] = int((st.ratio_cons_hi < 1).sum())
out['n_full_ratio_ci_above_1'] = int((st.ratio_full_lo > 1).sum())
out['median_variance_shares_cons'] = dict(designer=float(df.cons_designer.median()), sample=float(df.cons_sample.median()), inter=float(df.cons_inter.median()))
out['median_variance_shares_full'] = dict(designer=float(df.full_designer.median()), sample=float(df.full_sample.median()), inter=float(df.full_inter.median()))
na = st[~st.conversation.str.startswith('austria')]
out['spearman_n_ratio_cons_no_austria'] = sp(na.n, na.ratio_cons)
X = np.column_stack([np.ones(len(st)), np.log(st.n), np.log(st.m)])
b, *_ = np.linalg.lstsq(X, np.log(st.ratio_cons), rcond=None); rr = np.log(st.ratio_cons) - X @ b
se = np.sqrt(np.diag(rr @ rr / (len(st) - 3) * np.linalg.inv(X.T @ X)))
out['ols_logratio_cons_on_logn_logM'] = dict(coef=b.tolist(), t=(b / se).tolist())
jr = dict(consensus=int((df.jr_viol_consensus > 0).sum()), majoritarian=int((df.jr_viol_majoritarian > 0).sum()),
          proportional=int((df.jr_viol_proportional > 0).sum()), hybrid=int((df.jr_viol_hybrid > 0).sum()))
out['jr_any_violation_conversations'] = jr
json.dump(out, open(os.path.join(RES, 'opendata_stats.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
