"""Figures for the paper (static PNG, colorblind-validated categorical slots + secondary encodings)."""
import numpy as np, json, os, csv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import RES
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'; INK = '#333333'; MUTED = '#8a8a85'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.grid': True, 'grid.color': '#e6e6e3',
                     'grid.linewidth': 0.6, 'lines.linewidth': 2, 'legend.frameon': False, 'savefig.dpi': 200})
ca = np.load(os.path.join(RES, 'contam_arrays.npz'), allow_pickle=True)
cr = json.load(open(os.path.join(RES, 'contamination_results.json')))
# ---- Fig 1: wrong-key agreement vs ability, naive vs cleaned
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
th = ca['th']
for a, y, ttl in [(ax[0], ca['W'], f"All {cr['n_wg_items_usable']} flagged items (naive)"), (ax[1], ca['W2'], f"{cr['cleaned']['n_items']} items with P(audit error) < 0.5")]:
    a.scatter(th, y, s=22, color=C1, edgecolor='white', linewidth=0.8, zorder=3)
    b = np.polyfit(th, y, 1); xx = np.linspace(th.min(), th.max(), 10); a.plot(xx, np.polyval(b, xx), color=C2, lw=2, ls='--')
    a.set_xlabel('model ability θ (standardised logit accuracy on OK items)'); a.set_title(ttl, fontsize=9, color=INK)
ax[0].set_ylabel('share choosing the flagged original key')
ax[0].text(0.02, 0.95, f"Spearman ρ = {cr['naive']['spearman_W']:.2f} (p = {cr['naive']['spearman_W_p']:.3g})", transform=ax[0].transAxes, va='top', color=INK)
ax[1].text(0.02, 0.95, f"Spearman ρ = {cr['cleaned']['spearman']:.2f} (p = {cr['cleaned']['spearman_p']:.1g})", transform=ax[1].transAxes, va='top', color=INK)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig1_wrongkey_vs_ability.png')); plt.close(fig)
# ---- Fig 2: item-level bottom-20 vs top-20 agreement, by adjudication
rows = list(csv.DictReader(open(os.path.join(RES, 'wg_items.csv'))))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
sty = {'not_reviewed': (MUTED, 'o', 'not reviewed'), 'GOLD_DEFENSIBLE': (C1, 's', 'original key defensible'),
       'AUDIT_CORRECT': (C2, '^', 'audit verified correct'), 'AMBIGUOUS': (C4, 'D', 'ambiguous')}
rng = np.random.default_rng(0)
for k, (c, m, lab) in sty.items():
    rr = [r for r in rows if r['adjudication'] == k]
    x = np.array([float(r['W_bottom20']) for r in rr]) + rng.normal(0, .006, len(rr)); y = np.array([float(r['W_top20']) for r in rr]) + rng.normal(0, .006, len(rr))
    ax.scatter(x, y, s=34 if k != 'not_reviewed' else 16, marker=m, color=c, edgecolor='white', linewidth=0.8, label=f'{lab} (n={len(rr)})', zorder=3)
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=':')
ax.set_xlabel('share choosing original key: 20 weakest models'); ax.set_ylabel('share choosing original key: 20 strongest models')
ax.legend(loc='lower right', fontsize=7.5); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig2_item_level.png')); plt.close(fig)
# ---- Fig 3: simulation
ss = json.load(open(os.path.join(RES, 'simulation_summary.json')))
g = ss['grid']; etas = sorted({r['eta'] for r in g}); lams = sorted({r['lam_top'] for r in g})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
h0 = [r for r in g if r['lam_top'] == 0]
ax[0].plot(etas, [r['naive_reject'] for r in h0], color=C2, marker='o', ms=6, label='naive ability–agreement test')
ax[0].plot(etas, [r['mixture_reject_chibar'] for r in h0], color=C1, marker='s', ms=6, label='mixture LR test')
ax[0].axhline(0.05, color=MUTED, lw=1, ls=':'); ax[0].set_ylim(-0.02, 1.02)
ax[0].set_xlabel('audit-error rate η'); ax[0].set_ylabel('rejection rate with no memorisation'); ax[0].legend(fontsize=7.5, loc='center right')
ax[0].set_title('Size (λ = 0)', fontsize=9)
for e, c, mk in zip(etas, [C1, C2, C3, C4], ['o', 's', '^', 'D']):
    rr = [r for r in g if r['eta'] == e]
    ax[1].plot(lams, [r['mixture_reject_chibar'] for r in rr], color=c, marker=mk, ms=6, label=f'η = {e}')
ax[1].set_xlabel('memorisation at frontier λ_top'); ax[1].set_ylabel('mixture LR test rejection rate'); ax[1].set_ylim(-0.02, 1.02)
ax[1].legend(fontsize=7.5, loc='lower right'); ax[1].set_title('Power', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig3_simulation.png')); plt.close(fig)
# ---- Fig 4: saturation
sa = np.load(os.path.join(RES, 'saturation_arrays.npz')); sr = json.load(open(os.path.join(RES, 'saturation_results.json')))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
fa = np.sort(sa['full_acc'])
ax[0].scatter(np.arange(len(fa)), fa, s=14, color=C1, zorder=3)
c = sr['ceiling']; ax[0].axhspan(c['ci95'][0], c['ci95'][1], color=C4, alpha=0.35, lw=0)
ax[0].axhline(c['central'], color=C2, ls='--', lw=1.5); ax[0].text(1, c['central'] + 0.008, f"label-noise ceiling ≈ {c['central']:.3f}", color=INK, fontsize=8)
ax[0].set_xlabel('HELM model (sorted)'); ax[0].set_ylabel('full-MMLU accuracy (original keys)'); ax[0].set_title('Accuracy vs. ceiling', fontsize=9)
ax[1].plot(sa['grid'], sa['info'], color=C1)
for t in sa['thf']: ax[1].axvline(t, ymin=0, ymax=0.04, color=MUTED, lw=0.8)
ax[1].axvline(sa['thf'].max(), color=C2, ls='--', lw=1.5); ax[1].text(sa['thf'].max() + 0.05, sa['info'].max() * 0.9, 'best model', color=INK, fontsize=8)
ax[1].set_xlabel('ability θ (standardised logit full accuracy)'); ax[1].set_ylabel('test information Σ b²p(1−p)'); ax[1].set_title('IRT test information', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig4_saturation.png')); plt.close(fig)
print('figures written')
