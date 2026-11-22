"""Replicate the authorship decomposition on public Polis conversations
(github.com/compdemocracy/openData). Usage: python3 run_opendata.py"""
import json, os, sys, time, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from constlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, '..', 'data', 'openData')
RES = os.path.join(HERE, '..', 'results')
B = int(os.environ.get('B', 20))
MAXN = int(os.environ.get('MAXN', 4000))
rng = np.random.default_rng(7)
rows = []
t0 = time.time()
for d in sorted(glob.glob(os.path.join(OD, '*'))):
    pvf, cmf = os.path.join(d, 'participants-votes.csv'), os.path.join(d, 'comments.csv')
    if not (os.path.exists(pvf) and os.path.exists(cmf)):
        continue
    V, cm = load_polis(pvf, cmf, min_votes=7, min_seen=10)
    n, m = V.shape
    name = os.path.basename(d)
    if n < 100 or m < 40 or n > MAXN:
        print('skip', name, n, m); continue
    k = int(np.clip(round(0.2 * m), 10, 58))
    D = design_space()
    fam = [x['family'] for x in D]
    cons = [i for i, f in enumerate(fam) if f == 'consensus']
    I = np.zeros((len(D), B, m), np.int8)
    for b in range(B):
        Vb = V[rng.integers(0, n, n)]
        cache = {}
        for i, dd in enumerate(D):
            I[i, b, run_design(Vb, dd, k, cache, seed=b)] = 1
    dec_full, dec_cons = decompose(I), decompose(I[cons])
    sw_full, sw_cons = swap_distances(I), swap_distances(I[cons])
    auth = cm['author-id'].to_numpy()
    owner_pool = float((auth == 0).mean())
    # owner share inside constitutions, averaged over designs and samples
    owner_in = float(np.mean([(auth[np.where(I[i, b])[0]] == 0).mean() for i in range(len(D)) for b in range(B)]))
    vc = pd.Series(auth[auth != 0]).value_counts()
    top10 = vc.head(max(1, int(round(0.1 * len(vc))))).sum() / max(1, vc.sum())
    S = ~np.isnan(V)
    # JR audit on full data for each family
    cache = {}
    jr = {}
    for f in ('consensus', 'majoritarian', 'proportional', 'hybrid'):
        idx = [i for i, ff in enumerate(fam) if ff == f]
        jr[f] = float(np.mean([jr_violation(V, run_design(V, D[i], k, cache, seed=0), k) >= 1 for i in idx]))
    row = dict(conversation=name, n=n, m=m, k=k, density=float(S.mean()),
               agree_share=float((V == 1).sum() / S.sum()),
               full_designer=dec_full['designer'], full_sample=dec_full['sample'], full_inter=dec_full['interaction'],
               cons_designer=dec_cons['designer'], cons_sample=dec_cons['sample'], cons_inter=dec_cons['interaction'],
               full_designer_swap=sw_full['designer_swap'], full_sample_swap=sw_full['sample_swap'],
               cons_designer_swap=sw_cons['designer_swap'], cons_sample_swap=sw_cons['sample_swap'],
               owner_share_pool=owner_pool, owner_share_constitution=owner_in,
               top10pct_author_share=float(top10), authors_per_participant=float(len(vc) / n),
               jr_viol_consensus=jr['consensus'], jr_viol_majoritarian=jr['majoritarian'],
               jr_viol_proportional=jr['proportional'], jr_viol_hybrid=jr['hybrid'])
    rows.append(row)
    print(name, n, m, k, f"cons swap des={sw_cons['designer_swap']:.3f} sam={sw_cons['sample_swap']:.3f}",
          f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'opendata_decomposition.csv'), index=False)
print(df.to_string())
