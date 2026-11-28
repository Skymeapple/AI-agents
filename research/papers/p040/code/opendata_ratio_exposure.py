"""Designer/sample ratio vs headcount and exposure (merges opendata_ratio_ci.csv with
certify_opendata.csv). Usage: python3 opendata_ratio_exposure.py -> ../results/opendata_ratio_exposure.json"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
r = pd.read_csv(os.path.join(RES, 'opendata_ratio_ci.csv')); c = pd.read_csv(os.path.join(RES, 'certify_opendata.csv'))
d = r.merge(c[['conversation', 'm_median']], on='conversation')
out = {}
for x in ('n', 'm_median'):
    for y in ('ratio_cons', 'cons_sample_swap', 'cons_designer_swap'):
        out[f'spearman_{y}_vs_{x}'] = [float(v) for v in spearmanr(d[x], d[y])]
X = np.column_stack([np.ones(len(d)), np.log(d.n), np.log(d.m_median)]); y = np.log(d.ratio_cons)
b, *_ = np.linalg.lstsq(X, y, rcond=None); rr = y - X @ b
se = np.sqrt(np.diag(rr @ rr / (len(y) - 3) * np.linalg.inv(X.T @ X)))
out['ols_logratio_on_logn_logm'] = dict(coef=b.tolist(), t=(b / se).tolist())
out['ratio_cons_range'] = [float(d.ratio_cons.min()), float(d.ratio_cons.max())]
out['sub_multiverse_q05_q95_range'] = [float(d.ratio_sub_q05.min()), float(d.ratio_sub_q95.max())]
json.dump(out, open(os.path.join(RES, 'opendata_ratio_exposure.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
