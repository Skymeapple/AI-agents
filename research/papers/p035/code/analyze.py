import json, sys, numpy as np, collections
raw = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/raw.json'))
G = collections.defaultdict(list)
for r in raw:
    G[(r['rho'], r['H_visible'], r['subtle'], r['kappa'])].append(r)
EST = ['naive', 'iw', 'full_rewrite', 'rde', 'steer', 'steer_meandiff', 'rewrite_steer']
rows = []
lines = []
hdr = 'rho Hvis subtle kappa | theta_dep | ' + ' '.join(f'{e}' for e in EST) + ' | LBvalid UBcover | resAUCaud resAUCprobe steerResid | iwESS'
lines.append(hdr)
for key in sorted(G):
    rs = G[key]
    th = np.array([r['theta_dep'] for r in rs])
    row = dict(rho=key[0], H_visible=key[1], subtle=key[2], kappa=key[3], n=len(rs),
               theta_dep_mean=th.mean(), theta_dep_sd=th.std(ddof=1))
    for e in EST:
        err = np.array([r[e] for r in rs]) - th
        row[e + '_bias'] = err.mean(); row[e + '_se'] = err.std(ddof=1) / np.sqrt(len(rs))
        row[e + '_mae'] = np.abs(err).mean()
    row['lb_valid_frac'] = np.mean([r['lb_mono'] <= r['theta_dep'] + 0.01 for r in rs])
    row['lb_gap_mean'] = np.mean([r['theta_dep'] - r['lb_mono'] for r in rs])
    row['ub_cover_frac'] = np.mean([r['ub_lip'] >= r['theta_dep'] - 0.01 for r in rs])
    row['ub_width_mean'] = np.mean([r['ub_lip'] - r['lb_mono'] for r in rs])
    row['naive_mean'] = np.mean([r['naive'] for r in rs])
    for k in ['resid_auc_auditor', 'resid_auc_probe', 'iw_ess']:
        row[k] = np.mean([r[k] for r in rs])
    row['steer_resid'] = np.mean([r['rewrite_steer'] - r['full_rewrite'] for r in rs])
    rows.append(row)
    lines.append(f"{key[0]:.2f} {int(key[1])} {int(key[2])} {key[3]:.0f} | {th.mean():.3f}±{th.std(ddof=1):.3f} | " +
                 ' '.join(f"{row[e+'_bias']:+.3f}" for e in EST) +
                 f" | {row['lb_valid_frac']:.2f} {row['ub_cover_frac']:.2f} | {row['resid_auc_auditor']:.3f} {row['resid_auc_probe']:.3f} {row['steer_resid']:+.3f} | {row['iw_ess']:.1f}")
open('results/summary.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(rows, open('results/summary.json', 'w'), indent=1, default=float)
print('\n'.join(lines))
