"""Addendum analyses on CCAI (requires run_ccai.py outputs):
(1) representation audit against re-aligned (corrected) opinion groups,
(2) power-law extrapolation of swap distances in n,
(3) exposure (votes seen) as a predictor of selection per rule family.
Usage: python3 run_ccai_addendum.py"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(__file__))
from constlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'ccai'); RES = os.path.join(HERE, '..', 'results')
o = json.load(open(os.path.join(RES, 'ccai_results.json')))
V, cm, released = load_ccai(DATA)
n, m = V.shape
pid = pd.read_csv(os.path.join(DATA, 'clean_votes.csv'))['participant'].to_numpy()
srt = np.argsort(pid, kind='stable')
Xs = np.nan_to_num(V[srt], nan=0.0)
coords = PCA(2).fit(Xs.T).components_.T
k100 = KMeans(100, random_state=0, n_init=10).fit(coords)
lab = KMeans(2, random_state=0, n_init=10).fit(k100.cluster_centers_).labels_[k100.labels_]
corrected = np.empty_like(lab); corrected[srt] = lab
stmt2 = polis_groups(V, 2, 'zero', seed=0, center='stmt')
add = {}

D = design_space(); K = 58
cache = {}
rows = []
for d in D:
    W = run_design(V, d, K, cache, seed=0)
    for gname, g in (('corrected', corrected), ('stmt_K2', stmt2)):
        cs = coverage_stats(V, W, g)
        rows.append(dict(design=label(d), family=d['family'], groups=gname,
                         endorse_g0=cs['group_endorse'][0], endorse_g1=cs['group_endorse'][1],
                         group_gap=cs['group_gap'], mean_endorse=cs['mean_endorse']))
a = pd.DataFrame(rows)
a.to_csv(os.path.join(RES, 'ccai_design_audit_corrected_groups.csv'), index=False)
g = a.groupby(['groups', 'family'])[['group_gap', 'endorse_g0', 'endorse_g1']].agg(['mean', 'min', 'max']).round(4)
g.columns = ['|'.join(c) for c in g.columns]
add['audit_corrected_by_family'] = {f'{i[0]}|{i[1]}': r.to_dict() for i, r in g.iterrows()}

# (2) extrapolation
cur = o['sample_size_curve']
ns = np.array(sorted(int(k) for k in cur), float)
f = lambda x, a, b, c: a + b * x ** (-c)
fits = {}
for key in ('swap_consensus', 'swap_full'):
    for kind in ('designer_swap', 'sample_swap'):
        y = np.array([cur[str(int(x))][key][kind] for x in ns])
        try:
            p, _ = curve_fit(f, ns, y, p0=[0.2, 2, 0.5], bounds=([0, 0, 0.05], [1, 50, 2]), maxfev=20000)
            fits[f'{key}|{kind}'] = dict(a=float(p[0]), b=float(p[1]), c=float(p[2]),
                                       rmse=float(np.sqrt(np.mean((f(ns, *p) - y) ** 2))))
        except Exception as e:
            fits[f'{key}|{kind}'] = str(e)
add['swap_fits'] = fits
# crossover: smallest n on grid where fitted designer swap exceeds fitted sample swap
grid = np.logspace(2, 6, 400)
for key in ('swap_consensus', 'swap_full'):
    fd, fs = fits[f'{key}|designer_swap'], fits[f'{key}|sample_swap']
    if isinstance(fd, dict) and isinstance(fs, dict):
        dd = f(grid, fd['a'], fd['b'], fd['c']); ss = f(grid, fs['a'], fs['b'], fs['c'])
        idx = np.where(dd > ss)[0]
        add[f'crossover_n_{key}'] = float(grid[idx[0]]) if len(idx) else None

# (3) exposure as selection driver
seen = (~np.isnan(V)).sum(0)
I = np.load(os.path.join(RES, 'ccai_boot_inclusion.npy'))
fam = o['families']
expo = {}
for fm in ('consensus', 'majoritarian', 'proportional', 'hybrid'):
    idx = [i for i, x in enumerate(fam) if x == fm]
    pinc = I[idx].mean((0, 1))
    r, p = spearmanr(seen, pinc)
    expo[fm] = dict(spearman_seen_vs_inclusion=float(r), p=float(p))
add['exposure_vs_inclusion'] = expo
# statement-level determinacy classes over the full bootstrap x multiverse
pi = I.mean((0, 1))
add['stmt_always_in_ge90'] = int((pi >= 0.9).sum())
add['stmt_never_in_le10'] = int((pi <= 0.1).sum())
add['stmt_contested'] = int(((pi > 0.1) & (pi < 0.9)).sum())
json.dump(add, open(os.path.join(RES, 'ccai_addendum.json'), 'w'), indent=1, default=str)
print(json.dumps(add, indent=1, default=str)[:4000])
