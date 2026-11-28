import json, numpy as np
from scipy.stats import spearmanr
raw = json.load(open('results/raw_token.json')); out = {}; L = []
for rho in [0.0, 0.25, 0.5]:
    rs = [r for r in raw if r['rho'] == rho]; th = np.array([r['theta_dep'] for r in rs]); row = dict(n=len(rs))
    row['theta_dep'] = th.mean(); row['naive'] = np.mean([r['naive'] for r in rs])
    for w in ['perfect', 'overshoot_mild', 'overshoot_full']:
        row[w + '_bias'] = float(np.mean([r[w + '_est'] for r in rs]) - th.mean())
        row[w + '_bias_sd'] = float(np.std([r[w + '_est'] - r['theta_dep'] for r in rs], ddof=1))
        row[w + '_lbvalid'] = float(np.mean([r[w + '_lb'] <= r['theta_dep'] for r in rs]))
        row[w + '_a3flag'] = float(np.mean([r[w + '_a3_t'] < -3 for r in rs]))
    row['D1'] = float(np.mean([r['D1_aud_auc'] for r in rs])); row['D2'] = float(np.mean([r['D2_probe_auc'] for r in rs]))
    row['D2_range'] = [min(r['D2_probe_auc'] for r in rs), max(r['D2_probe_auc'] for r in rs)]
    row['D2_within_spearman'] = float(spearmanr([r['D2_probe_auc'] for r in rs], [r['theta_dep'] - r['perfect_est'] for r in rs])[0])
    out[str(rho)] = row
    L.append(f"rho={rho}: theta_D {row['theta_dep']:.3f} naive {row['naive']:.3f} | " + ' | '.join(
        f"{w} bias {row[w+'_bias']:+.3f} (sd {row[w+'_bias_sd']:.3f}) LBvalid {row[w+'_lbvalid']:.2f} A3flag {row[w+'_a3flag']:.2f}"
        for w in ['perfect', 'overshoot_mild', 'overshoot_full']) +
        f" | D1 {row['D1']:.3f} D2 {row['D2']:.3f} [{row['D2_range'][0]:.3f},{row['D2_range'][1]:.3f}] D2-within {row['D2_within_spearman']:+.2f}")
json.dump(out, open('results/summary_token.json', 'w'), indent=1, default=float)
open('results/summary_token.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
