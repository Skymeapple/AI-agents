"""Exposure-normalised proportional rules (reviewer M5; prediction P2-exposure).
Count-based PAV/CC treat 'unseen' as 'disapproved', so statements shown to more
voters collect more approvals.  We compare with two exposure-normalised variants:
  imputed: unseen entries replaced by the statement's agree/seen rate (expected approval)
  ipw:     observed approvals weighted by n / (#viewers of the statement)
and report the seed (organizer, pre-participant) share of the selected set.
Usage: python3 run_exposure_rules.py -> ../results/exposure_rules.json"""
import json, os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from constlib import load_ccai, load_polis, rule_seqpav, rule_greedycc, rule_mes, rule_av, jaccard, jr_violation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data'); RES = os.path.join(HERE, '..', 'results')


def utilities(V, mode):
    S = ~np.isnan(V)
    A = (V == 1).astype(float)
    n = len(V)
    seen = S.sum(0)
    if mode == 'count':
        return A
    if mode == 'imputed':
        p = A.sum(0) / np.maximum(seen, 1)
        return np.where(S, A, p[None, :])
    if mode == 'ipw':
        return A * (n / np.maximum(seen, 1))[None, :]
    raise ValueError(mode)


def pav_util(U, k):
    n, m = U.shape
    sat = np.zeros(n); chosen = []; avail = np.ones(m, bool)
    for _ in range(k):
        g = (U / (1 + sat[:, None])).sum(0); g[~avail] = -np.inf
        c = int(np.argmax(g)); chosen.append(c); avail[c] = False; sat += U[:, c]
    return np.sort(chosen)


def cc_util(U, k):
    """greedy expected coverage: maximise sum_i 1 - prod_{s in W}(1 - u_is), u clipped to [0,1]."""
    P = np.clip(U, 0, 1)
    n, m = P.shape
    miss = np.ones(n); chosen = []; avail = np.ones(m, bool)
    for _ in range(k):
        g = (miss[:, None] * P).sum(0); g[~avail] = -np.inf
        c = int(np.argmax(g)); chosen.append(c); avail[c] = False; miss *= (1 - P[:, c])
    return np.sort(chosen)


def seeds_of(cm):
    auth = cm['author-id'].to_numpy(); ts = cm['timestamp'].to_numpy()
    if (auth != 0).sum() == 0 or (auth == 0).sum() == 0:
        return np.zeros(len(cm), bool)
    return (auth == 0) & (ts < ts[auth != 0].min())


def evaluate(V, cm, k):
    seed = seeds_of(cm)
    seen = (~np.isnan(V)).sum(0)
    res = dict(seed_pool=float(seed.mean()), n_seeds=int(seed.sum()),
               seed_exposure_ratio=float(seen[seed].mean() / seen[~seed].mean()) if seed.any() else None)
    W_av = rule_av(V, 'seen', k)
    sets = {'av_rate': W_av}
    for mode in ('count', 'imputed', 'ipw'):
        U = utilities(V, mode)
        sets[f'pav_{mode}'] = pav_util(U, k)
        if mode != 'ipw':
            sets[f'cc_{mode}'] = cc_util(U, k)
    for name, W in sets.items():
        res[name] = dict(seed_share=float(seed[W].mean()), mean_seen=float(seen[W].mean()),
                         jaccard_to_av_rate=jaccard(W, W_av), jr_ratio_observed=float(jr_violation(V, W, k)))
    return res

out = {}
V, cm, _ = load_ccai(os.path.join(DATA, 'ccai'))
out['CCAI'] = evaluate(V, cm, 58)
print('CCAI', json.dumps(out['CCAI'], indent=0)[:1500])
for d in sorted(glob.glob(os.path.join(DATA, 'openData', '*'))):
    pvf, cmf = os.path.join(d, 'participants-votes.csv'), os.path.join(d, 'comments.csv')
    if not os.path.exists(pvf):
        continue
    Vd, cmd = load_polis(pvf, cmf)
    if Vd.shape[0] < 100 or Vd.shape[1] < 40:
        continue
    k = int(np.clip(round(0.2 * Vd.shape[1]), 10, 58))
    r = evaluate(Vd, cmd, k)
    out[os.path.basename(d)] = r
    print(os.path.basename(d), r['n_seeds'], round(r['seed_pool'], 3),
          {x: round(r[x]['seed_share'], 3) for x in r if isinstance(r[x], dict)}, flush=True)
# pooled summary across conversations with seeds
rows = []
for name, r in out.items():
    if r['n_seeds'] == 0:
        continue
    rows.append(dict(conv=name, pool=r['seed_pool'], expo=r['seed_exposure_ratio'],
                     **{f'{x}': r[x]['seed_share'] for x in r if isinstance(r[x], dict)}))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'exposure_rules.csv'), index=False)
cols = [c for c in df.columns if c not in ('conv', 'pool', 'expo')]
out['summary_overrep_ratio_median'] = {c: float(np.median(df[c] / df['pool'])) for c in cols}
from scipy.stats import wilcoxon
out['wilcoxon_pav_count_vs_imputed'] = float(wilcoxon(df['pav_count'], df['pav_imputed']).pvalue)
out['wilcoxon_cc_count_vs_imputed'] = float(wilcoxon(df['cc_count'], df['cc_imputed']).pvalue)
out['n_conversations_with_seeds'] = int(len(df))
json.dump(out, open(os.path.join(RES, 'exposure_rules.json'), 'w'), indent=1)
print(df.round(3).to_string()); print(out['summary_overrep_ratio_median'], out['wilcoxon_pav_count_vs_imputed'], out['wilcoxon_cc_count_vs_imputed'])
