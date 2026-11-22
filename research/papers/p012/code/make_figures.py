"""Aggregate results/main_results.json into figures and results/summary.txt"""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
R = json.load(open(f'{OUT}/main_results.json')); Wd = R['worlds']
lines = []
def P(*a): lines.append(' '.join(str(x) for x in a))
def ms(v): v = np.array(v, float); return f'{v.mean():.3f} ± {v.std(ddof=1) if len(v)>1 else 0:.3f}'
P('config', R['config'])
# ---- E2 cells
P('\n== E2: cells (mean ± sd over worlds) ==')
for k in Wd[0]['cells']:
    P(f'{k:22s} n={np.mean([w["cells"][k]["n"] for w in Wd]):7.0f}', 'acc', ms([w['cells'][k]['acc'] for w in Wd]),
      'msp', ms([w['cells'][k]['msp'] for w in Wd]), 'gap(msp-acc)', ms([w['cells'][k]['msp'] - w['cells'][k]['acc'] for w in Wd]),
      'MI', ms([w['cells'][k]['mi'] for w in Wd]), 'ens_acc', ms([w['cells'][k]['ens_acc'] for w in Wd]),
      'ens_msp', ms([w['cells'][k]['ens_msp'] for w in Wd]), 'prior_acc', ms([w['cells'][k]['prior_acc'] for w in Wd]))
# ---- E2 curve figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for kind, col in [('seen', '#2c7fb8'), ('unseenfact', '#c0392b')]:
    keys = sorted({(c['lo'], c['hi']) for w in Wd for c in w['curve'] if c['kind'] == kind})
    xs, acc, msp, mi = [], [], [], []
    for lo, hi in keys:
        cs = [c for w in Wd for c in w['curve'] if c['kind'] == kind and c['lo'] == lo]
        if len(cs) < len(Wd): continue
        xs.append(np.sqrt(lo * min(hi, 3000))); acc.append(np.mean([c['acc'] for c in cs]))
        msp.append(np.mean([c['msp'] for c in cs])); mi.append(np.mean([c['mi'] for c in cs]))
        P(f'curve {kind:10s} [{lo:g},{hi:g}) n~{np.mean([c["n"] for c in cs]):.0f} acc {acc[-1]:.3f} msp {msp[-1]:.3f} mi {mi[-1]:.3f}')
    lab = 'fact seen in training' if kind == 'seen' else 'fact never seen'
    ax[0].plot(xs, acc, 'o-', c=col, label=f'accuracy ({lab})'); ax[0].plot(xs, msp, 's--', c=col, label=f'confidence ({lab})')
    ax[1].plot(xs, mi, 'o-', c=col, label=lab)
for a_ in ax: a_.set_xscale('log'); a_.set_xlabel('entity training count (familiarity)'); a_.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('rate'); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title('Familiarity raises confidence, not knowledge', fontsize=9)
ax[1].set_ylabel('ensemble mutual information'); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title('Ensemble disagreement vs familiarity', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_familiarity.png', dpi=160); plt.close()
# ---- E1 sweep
P('\n== E1: AUROC for epistemic gap vs aleatoric fraction alpha ==')
al = [s['alpha'] for s in Wd[0]['alea_sweep']]
names = [('msp', '1 - max prob'), ('entropy', 'entropy'), ('ens_entropy', 'ensemble entropy'), ('ens_mi', 'ensemble MI'), ('neg_log_count', 'familiarity (-log count)')]
fig, ax = plt.subplots(figsize=(5, 3.6))
for k, lab in names:
    v = np.array([[s[k] for s in w['alea_sweep']] for w in Wd]); m = v.mean(0)
    ax.plot(al, m, 'o-', ms=3, label=lab)
    P(f'{k:14s}', ' '.join(f'{x:.3f}' for x in m), ' | sd@alpha=1:', f'{v[:, -1].std(ddof=1) if len(Wd)>1 else 0:.3f}')
a0 = np.mean([w['alea_sweep'][0]['msp'] for w in Wd])
ax.plot(al, [a0 * (1 - a) + 0.5 * a for a in al], 'k:', label=r'Prop. 1 ($\delta$=0) from measured $A_{det}$')
ax.set_xlabel(r'aleatoric fraction $\alpha$ of "known" items'); ax.set_ylabel('AUROC (detect "does not know")')
ax.legend(fontsize=7, frameon=False); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_aleatoric.png', dpi=160); plt.close()
P('MSP of known-aleatoric items', ms([w['alea_known_msp'] for w in Wd]), '; MSP of unknown items', ms([w['pos_msp'] for w in Wd]),
  '; n_pos', [w['n_pos_E1'] for w in Wd], 'n_neg_alea', [w['n_neg_alea_E1'] for w in Wd])
# ---- E3 probes
P('\n== E3: probes (label = answer correct). pi_natural =', ms([w['pi_natural'] for w in Wd]))
P('geometry (whitened hidden):', [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in w['geometry'].items() if k != 'cell_n'} for w in Wd if 'geometry' in w])
feats = sorted({p['feat'] for p in Wd[0]['probe']}); pis = sorted({p['pi_dis'] for p in Wd[0]['probe']})
table = {}
for f in feats:
    for pi in pis:
        rows = [p for w in Wd for p in w['probe'] if p['feat'] == f and p['pi_dis'] == pi]
        per_world = lambda key: [np.mean([p[key] for p in w['probe'] if p['feat'] == f and p['pi_dis'] == pi]) for w in Wd]
        table[(f, pi)] = {k: per_world(k) for k in ['auc_nat', 'auc_dis', 'false_know', 'missed_know', 'pi_eff'] + (['pred_fk'] if 'pred_fk' in rows[0] else [])}
        P(f'{f:14s} pi_dis={pi:<5} ' + ' '.join(f'{k}={ms(v)}' for k, v in table[(f, pi)].items()))
P('baselines:')
for k in Wd[0]['probe_baselines']:
    P(f'  {k:14s} auc_nat {ms([w["probe_baselines"][k]["auc_nat"] for w in Wd])}  auc_dis {ms([w["probe_baselines"][k]["auc_dis"] for w in Wd])}')
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
cols = {'hidden': '#c0392b', 'hidden+output': '#8e44ad', 'output_only': '#2c7fb8'}
for f in feats:
    ax[0].errorbar(pis, [np.mean(table[(f, p)]['auc_nat']) for p in pis], fmt='o--', c=cols[f], ms=3, alpha=.6)
    ax[0].errorbar(pis, [np.mean(table[(f, p)]['auc_dis']) for p in pis], fmt='o-', c=cols[f], ms=3, label=f'probe on {f}')
for k, ls in [('msp', 'gray'), ('ens_mi', 'olive'), ('neg_log_count', 'k')]:
    ax[0].axhline(np.mean([w['probe_baselines'][k]['auc_dis'] for w in Wd]), c=ls, ls=':', lw=1, label=f'{k} (dissociated)')
ax[0].set_xlabel(r'share of dissociated items in probe training, $\pi_d$'); ax[0].set_ylabel('AUROC: solid = dissociated, dashed = natural')
ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title('Probe evaluation: natural vs dissociated cells', fontsize=9)
f = 'hidden'
x = np.concatenate([table[(f, p)]['pi_eff'] for p in pis]); y = np.concatenate([table[(f, p)]['false_know'] for p in pis])
ax[1].scatter(x, y, c=cols[f], s=14, label='observed (probe on hidden, per world)')
if 'pred_fk' in table[(f, pis[0])]:
    yp = np.concatenate([table[(f, p)]['pred_fk'] for p in pis]); o = np.argsort(x)
    ax[1].scatter(x, yp, marker='x', c='k', s=18, label='Prop. 2 prediction (estimated geometry)')
    P('corr(observed false_know, predicted) hidden:', f'{np.corrcoef(y, yp)[0,1]:.3f}', 'mean abs err', f'{np.mean(np.abs(y-yp)):.3f}')
ax[1].set_xlabel(r'effective co-occurrence $\pi=P(F=K)$ in probe training'); ax[1].set_ylabel('false-"knows" on familiar & wrong')
ax[1].legend(fontsize=7, frameon=False); ax[1].set_title('Shortcut strength tracks co-occurrence', fontsize=9)
for a_ in ax: a_.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_probe.png', dpi=160); plt.close()
# ---- E4 Good-Turing
P('\n== E4: Good-Turing ==')
for k in Wd[0]['good_turing']:
    P(f'{k:22s}', ms([w['good_turing'][k] for w in Wd]))
open(f'{OUT}/summary.txt', 'w').write('\n'.join(lines)); print('\n'.join(lines))
