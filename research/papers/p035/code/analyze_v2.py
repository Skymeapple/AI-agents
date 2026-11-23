"""Analysis for revision 1. Produces results/summary_v2.txt and results/analysis_v2.json, fig4, fig5.
Also re-analyses the round-0 raw.json (no-slack validity; within-cell diagnostics)."""
import json, collections, sys
import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = {}
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); L.append(s); print(s)

# ---------------------------------------------------------------- round-0 re-analysis
raw0 = json.load(open('results/raw.json'))
G0 = collections.defaultdict(list)
for r in raw0: G0[(r['rho'], r['H_visible'], r['subtle'], r['kappa'])].append(r)
P('== Round-0 runs: lower-bound validity with and without the +0.01 slack')
v0 = {}
for k in sorted(G0):
    rs = G0[k]
    a = np.mean([r['lb_mono'] <= r['theta_dep'] for r in rs]); b = np.mean([r['lb_mono'] <= r['theta_dep'] + .01 for r in rs])
    worst = max(r['lb_mono'] - r['theta_dep'] for r in rs)
    v0[str(k)] = dict(no_slack=a, slack=b, worst_violation=worst)
    P(f'  rho={k[0]} vis={int(k[1])} subtle={int(k[2])} kappa={k[3]}: no-slack {a:.1f}  slack {b:.1f}  worst LB-theta {worst:+.4f}')
OUT['round0_lb_validity'] = v0
R0 = [r for r in raw0 if r['kappa'] == 4.0 and not r['subtle']]
ab0 = lambda r: abs(r['full_rewrite'] - r['theta_dep'])
vis0 = [r for r in R0 if r['H_visible'] and r['rho'] > 0]
s = spearmanr([r['resid_auc_auditor'] for r in vis0], [ab0(r) for r in vis0])
OUT['round0_D1_visible_rho_gt0'] = [float(s[0]), float(s[1]), len(vis0)]
P(f'== Round-0: D1 vs |bias| within visible, rho>0 (n={len(vis0)}): Spearman {s[0]:.2f} p={s[1]:.3f}')
cells0 = {}
for key in sorted(set((r['rho'], r['H_visible']) for r in R0)):
    c = [r for r in R0 if (r['rho'], r['H_visible']) == key]
    cells0[str(key)] = dict(D1=float(spearmanr([r['resid_auc_auditor'] for r in c], [ab0(r) for r in c])[0]),
                            D2=float(spearmanr([r['resid_auc_probe'] for r in c], [ab0(r) for r in c])[0]),
                            D2_range=[min(r['resid_auc_probe'] for r in c), max(r['resid_auc_probe'] for r in c)])
    P('  cell', key, {k: (round(v, 2) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in cells0[str(key)].items()})
OUT['round0_within_cell'] = cells0

# ---------------------------------------------------------------- round-1 grid
raw = json.load(open('results/raw_v2.json'))
def ckey(r): return (r['regime'], r['rho'], r['H_visible'])
G = collections.defaultdict(list)
for r in raw: G[ckey(r)].append(r)
RW = ['perfect', 'partial', 'overshoot_mild', 'overshoot_full', 'task_sanitise', 'task_intensify']
EST = ['naive', 'full', 'rs_l1_benign', 'rs_l1_risky', 'rs_l2_risky', 'meanmatch_l1', 'meanmatch_pre', 'mrde']
P('\n== Round-1 grid: theta_D, naive, bias of estimators (mean over seeds; SE of mean in brackets for full)')
P('regime rho vis n | theta_D theta_E(naive) | ' + ' '.join(EST[1:]))
cfg_tab = {}
order = sorted(G, key=lambda k: (k[0], k[1], not k[2]))
for k in order:
    rs = G[k]; th = np.array([r['theta_dep'] for r in rs])
    row = dict(n=len(rs), theta_dep=th.mean(), theta_dep_sd=th.std(ddof=1), naive=np.mean([r['naive'] for r in rs]),
               theta_dep_policy=np.mean([r['theta_dep_policy'] for r in rs]),
               theta_eval_policy=np.mean([r['theta_eval_policy'] for r in rs]))
    for e in EST:
        err = np.array([r[e] for r in rs]) - th
        row[e + '_bias'] = err.mean(); row[e + '_se'] = err.std(ddof=1) / np.sqrt(len(rs)); row[e + '_mae'] = np.abs(err).mean()
    for w in RW:
        row[w + '_bias'] = np.mean([r[w + '_est'] - r['theta_dep'] for r in rs])
        row[w + '_lbvalid'] = np.mean([r[w + '_lb'] <= r['theta_dep'] for r in rs])
        row[w + '_lbvalid_mc'] = np.mean([r[w + '_lb'] <= r['theta_dep'] + 1.645 * r['theta_dep_mcse'] for r in rs])
        row[w + '_a3flag'] = np.mean([r[w + '_a3_t'] < -3 for r in rs])
        row[w + '_a3flag_l2'] = np.mean([r[w + '_a3_t_l2'] < -3 for r in rs])
        row[w + '_a3t_med'] = float(np.median([r[w + '_a3_t'] for r in rs]))
    cfg_tab[str(k)] = row
    P(f"{k[0]:18s} {k[1]:.2f} {int(k[2])} {len(rs):2d} | {th.mean():.3f} {row['naive']:.3f} | " +
      ' '.join(f"{row[e+'_bias']:+.3f}" for e in EST[1:]) + f"  (full SE {row['full_se']:.3f})")
OUT['config_table'] = cfg_tab

P('\n== Stress test of Prop. 2: fraction of runs with LB <= theta_D (NO slack) | mean bias | A3-check flag rate (t<-3, layer 1)')
P('regime rho vis | ' + ' | '.join(RW))
for k in order:
    row = cfg_tab[str(k)]
    P(f"{k[0]:18s} {k[1]:.2f} {int(k[2])} | " + ' | '.join(
        f"{row[w+'_lbvalid']:.2f} {row[w+'_bias']:+.3f} f{row[w+'_a3flag']:.1f}" for w in RW))

# A3 check: does it predict LB failure across all runs/rewriters?
fl, inv = [], []
for r in raw:
    if r['gshape'] != 'monotone' or r['kappa'] == 0: continue
    for w in RW:
        fl.append(r[w + '_a3_t'] < -3); inv.append(r[w + '_lb'] > r['theta_dep'])
fl, inv = np.array(fl), np.array(inv)
OUT['a3_check_confusion'] = dict(n=int(len(fl)), flag_and_invalid=int((fl & inv).sum()), flag_and_valid=int((fl & ~inv).sum()),
                                 noflag_and_invalid=int((~fl & inv).sum()), noflag_and_valid=int((~fl & ~inv).sum()))
P('\n== A3 check (layer-1 probe projection on benign traffic, t<-3) vs LB invalid, monotone-g runs, kappa>0, all rewriters:',
  OUT['a3_check_confusion'])
# large violations only (> 0.02)
inv2 = []
for r in raw:
    if r['gshape'] != 'monotone' or r['kappa'] == 0: continue
    for w in RW: inv2.append(r[w + '_est'] - r['theta_dep'] > 0.02)
inv2 = np.array(inv2)
OUT['a3_check_confusion_big'] = dict(flag_and_big=int((fl & inv2).sum()), noflag_and_big=int((~fl & inv2).sum()),
                                     flag_and_notbig=int((fl & ~inv2).sum()))
P('   with violation defined as estimate > theta_D + 0.02:', OUT['a3_check_confusion_big'])

# ---------------------------------------------------------------- diagnostics, within cell
DIAG = {
    'D1 auditor resid AUC': lambda r: r['D1_aud_auc'],
    'D2 probe resid AUC (L1)': lambda r: r['D2_probe_auc_l1'],
    'D2 probe resid AUC (L2)': lambda r: r['D2_probe_auc_l2'],
    'D6 probe projection gap (L1)': lambda r: r['D6_proj_gap_l1'],
    'D6 probe projection gap (L2)': lambda r: r['D6_proj_gap_l2'],
    'D3 steer(benign,L1) - full': lambda r: r['rs_l1_benign'] - r['full'],
    'D3b steer(risky,L2) - full': lambda r: r['rs_l2_risky'] - r['full'],
    'D4 MRDE - full': lambda r: r['mrde'] - r['full'],
    'D5 meanmatch(L1) - full': lambda r: r['meanmatch_l1'] - r['full'],
    'D5p meanmatch(pre) - full': lambda r: r['meanmatch_pre'] - r['full'],
}
target = lambda r: r['theta_dep'] - r['full']   # = -bias, positive when rewrite underestimates
rng = np.random.default_rng(0)
def within_stats(runs, f):
    cells = collections.defaultdict(list)
    for r in runs: cells[ckey(r)].append(r)
    cs = [c for c in cells.values() if len(c) >= 5]
    rhos = [spearmanr([f(r) for r in c], [target(r) for r in c])[0] for c in cs]
    obs = np.nanmean(rhos)
    # stratified permutation: permute diagnostic within each cell
    null = []
    for _ in range(2000):
        null.append(np.nanmean([spearmanr(rng.permutation([f(r) for r in c]), [target(r) for r in c])[0] for c in cs]))
    p = (1 + np.sum(np.array(null) >= obs)) / (1 + len(null))
    pooled = spearmanr([f(r) for r in runs], [target(r) for r in runs])
    cellmeans = spearmanr([np.mean([f(r) for r in c]) for c in cs], [np.mean([target(r) for r in c]) for c in cs])
    # within-cell calibration: regress demeaned target on demeaned diagnostic
    x = np.concatenate([np.array([f(r) for r in c]) - np.mean([f(r) for r in c]) for c in cs])
    y = np.concatenate([np.array([target(r) for r in c]) - np.mean([target(r) for r in c]) for c in cs])
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float('nan')
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if np.ptp(x) > 0 else float('nan')
    return dict(pooled=float(pooled[0]), pooled_p_naive=float(pooled[1]), n_runs=len(runs), n_cells=len(cs),
                cellmean_spearman=float(cellmeans[0]), cellmean_p=float(cellmeans[1]),
                within_mean_spearman=float(obs), within_min=float(np.nanmin(rhos)), within_max=float(np.nanmax(rhos)),
                within_perm_p=float(p), within_slope=slope, within_r2=r2)

for label, sel in [('sat', lambda r: r['regime'] == 'sat'), ('unsat', lambda r: r['regime'] == 'unsat')]:
    runs = [r for r in raw if sel(r)]
    P(f'\n== Diagnostics vs (theta_D - full rewrite), regime={label}: pooled Spearman | cell-mean Spearman (7 cells) | within-cell mean Spearman [min,max], stratified-permutation p | within-cell slope, R2')
    OUT[f'diagnostics_{label}'] = {}
    for name, f in DIAG.items():
        st = within_stats(runs, f); OUT[f'diagnostics_{label}'][name] = st
        P(f"  {name:30s} pooled {st['pooled']:+.2f} | cells {st['cellmean_spearman']:+.2f} (p={st['cellmean_p']:.2f}) | within {st['within_mean_spearman']:+.2f} [{st['within_min']:+.2f},{st['within_max']:+.2f}] p={st['within_perm_p']:.3f} | slope {st['within_slope']:+.2f} R2 {st['within_r2']:.2f}")

# bias-corrected estimators: does adding the predicted residual help? (MAE across runs)
P('\n== MAE of estimators (mean |estimate - theta_D|) by regime and rho, visible+latent pooled')
mae = {}
for reg in ['sat', 'unsat']:
    for rho in [0.0, 0.25, 0.5, 0.75]:
        rs = [r for r in raw if r['regime'] == reg and r['rho'] == rho]
        mae[f'{reg}_{rho}'] = {e: float(np.mean([abs(r[e] - r['theta_dep']) for r in rs])) for e in EST}
        P(f'  {reg:5s} rho={rho:.2f}: ' + ' '.join(f"{e} {mae[f'{reg}_{rho}'][e]:.3f}" for e in EST))
OUT['mae'] = mae
json.dump(OUT, open('results/analysis_v2.json', 'w'), indent=1, default=float)
open('results/summary_v2.txt', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
cols = {'perfect': '#1f77b4', 'partial': '#2ca02c', 'overshoot_mild': '#ff7f0e', 'overshoot_full': '#d62728',
        'task_sanitise': '#9467bd', 'task_intensify': '#8c564b'}
for ax, reg in zip(axs, ['sat', 'unsat']):
    for w in RW:
        xs, ys = [], []
        for rho in [0.0, 0.25, 0.5, 0.75]:
            rs = [r for r in raw if r['regime'] == reg and r['rho'] == rho]
            xs.append(rho); ys.append(np.mean([r[w + '_est'] - r['theta_dep'] for r in rs]))
        ax.plot(xs, ys, marker='o', color=cols[w], label=w)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(f"{'saturated (b0=1.5, kappa=4)' if reg == 'sat' else 'unsaturated (b0=-0.5, kappa=1.5)'}", fontsize=9)
    ax.set_xlabel(r'$\rho$'); ax.set_ylabel('rewrite estimate − θ_D (above 0 = lower bound violated)')
axs[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('results/fig4_rewriter_stress.png', dpi=150)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
runs = [r for r in raw if r['regime'] == 'unsat']
for ax, name in zip(axs, ['D1 auditor resid AUC', 'D2 probe resid AUC (L1)', 'D4 MRDE - full']):
    f = DIAG[name]
    for k in sorted(set(ckey(r) for r in runs)):
        c = [r for r in runs if ckey(r) == k]
        ax.scatter([f(r) for r in c], [target(r) for r in c], s=10, label=f'rho={k[1]} {"vis" if k[2] else "lat"}')
    st = OUT['diagnostics_unsat'][name]
    ax.set_title(f"{name}\npooled {st['pooled']:+.2f}, within-cell {st['within_mean_spearman']:+.2f}", fontsize=8)
    ax.set_xlabel(name, fontsize=8); ax.set_ylabel('θ_D − full rewrite', fontsize=8)
axs[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig('results/fig5_diagnostics_within.png', dpi=150)
print('saved')
