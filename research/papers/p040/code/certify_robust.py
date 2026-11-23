"""Robustness of the exposure law and design-neutral view of designer disagreement.
Usage: python3 certify_robust.py (after run_certify.py, run_ccai.py) -> ../results/certify_robust.json"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from constlib import load_ccai, design_space, run_design
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results'); DATA = os.path.join(HERE, '..', 'data')
od = pd.read_csv(os.path.join(RES, 'certify_opendata.csv'))
out = {}

def ols(df, y, xs):
    X = np.column_stack([np.ones(len(df))] + [np.log(df[x]) for x in xs])
    Y = np.log(df[y].values)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ b
    s2 = r @ r / (len(Y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return dict(coef=dict(zip(['const'] + xs, b.round(4).tolist())), t=dict(zip(['const'] + xs, (b / se).round(2).tolist())),
                n=int(len(Y)), r2=float(1 - r @ r / ((Y - Y.mean()) @ (Y - Y.mean()))))

variants = {
    'all18': od,
    'no_marchon': od[od.conversation != 'march-on.operation-marchin-orders'],
    'one_austria': od[~od.conversation.str.startswith('austria') | (od.conversation == 'austria-climate.9xnndurbfm.2022-07-07')],
    'no_ccai': od[od.conversation != 'CCAI'],
}
for name, df in variants.items():
    out[name] = dict(ols_contested_per_seat=ols(df, 'contested_per_seat', ['n', 'm_median']),
                     ols_contested_per_seat_with_agree=ols(df.assign(dis=1 - df.agree_share), 'contested_per_seat', ['n', 'm_median', 'dis']),
                     spearman_contested_vs_m=float(spearmanr(df.m_median, df.contested_per_seat)[0]),
                     spearman_contested_vs_n=float(spearmanr(df.n, df.contested_per_seat)[0]),
                     spearman_contested_vs_m_p=float(spearmanr(df.m_median, df.contested_per_seat)[1]),
                     spearman_contested_vs_n_p=float(spearmanr(df.n, df.contested_per_seat)[1]))
# partial Spearman: contested ~ m | n
from scipy.stats import rankdata
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])
out['partial_spearman_contested_m_given_n'] = partial_spearman(od.m_median, od.contested_per_seat, od.n)
out['partial_spearman_contested_n_given_m'] = partial_spearman(od.n, od.contested_per_seat, od.m_median)
out['loglog_M_on_n_all18'] = np.polyfit(np.log(od.n), np.log(od.M), 1).round(3).tolist()

# design-neutral: are designer disagreements on statements the votes leave contested?
st = pd.read_csv(os.path.join(RES, 'certify_ccai_statements.csv'))
V, cm, _ = load_ccai(os.path.join(DATA, 'ccai'))
pi = st.set_index(st.comment_id.astype(str)).loc[cm['comment-id'].astype(str), 'boot_top58'].to_numpy()
contested = (pi > 0.05) & (pi < 0.95)
D = design_space(); cache = {}
W = np.zeros((len(D), V.shape[1]), bool)
for i, d in enumerate(D):
    W[i, run_design(V, d, 58, cache, seed=0)] = True
fam = [d['family'] for d in D]
cons = [i for i, f in enumerate(fam) if f == 'consensus']
for nm, idx in (('consensus', cons), ('all', list(range(len(D))))):
    tot = cont = stab_in = 0
    for a in idx:
        for b in idx:
            if a < b:
                dif = W[a] ^ W[b]
                tot += dif.sum(); cont += (dif & contested).sum(); stab_in += (dif & (pi >= .95)).sum()
    out[f'designer_disagreement_{nm}'] = dict(share_on_contested=float(cont / tot), share_on_stable_in=float(stab_in / tot),
                                              share_on_stable_out=float(1 - (cont + stab_in) / tot))
out['contested_band_size'] = int(contested.sum())
out['union_consensus_designs'] = int(W[cons].any(0).sum())
out['union_consensus_in_contested'] = int((W[cons].any(0) & contested).sum())
out['stable_in_av'] = int((pi >= .95).sum())
out['stable_in_av_included_by_all_consensus'] = int(((pi >= .95) & W[cons].all(0)).sum())
out['stable_in_av_excluded_by_some_design'] = int(((pi >= .95) & ~W.all(0)).sum())
json.dump(out, open(os.path.join(RES, 'certify_robust.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
