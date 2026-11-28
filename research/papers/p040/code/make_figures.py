"""Figures from saved results. Usage: python3 make_figures.py"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
o = json.load(open(os.path.join(RES, 'ccai_results.json')))

# Fig 1: multiverse Jaccard heatmap
J = np.load(os.path.join(RES, 'ccai_multiverse_jaccard.npy'))
fam = o['families']
fig, ax = plt.subplots(figsize=(5.8, 4.4))
im = ax.imshow(J, cmap='Blues', vmin=0, vmax=1)
bounds = np.cumsum([fam.count(f) for f in ('consensus', 'majoritarian', 'proportional', 'hybrid')])
for b in bounds[:-1]:
    ax.axhline(b - .5, color='white', lw=1.5); ax.axvline(b - .5, color='white', lw=1.5)
mids = np.r_[0, bounds[:-1]] + np.diff(np.r_[0, bounds]) / 2 - .5
ax.set_xticks([mids[0]]); ax.set_xticklabels(['consensus family (72 pipelines)'], fontsize=7)
ax.set_yticks(mids); ax.set_yticklabels(['consensus (72)', 'AV (3)', 'PAV/MES/CC', 'hybrid'], fontsize=7)
fig.subplots_adjust(left=0.2)
cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label('Jaccard similarity of 58-statement constitutions')
ax.set_title('CCAI: same votes, 79 defensible pipelines', fontsize=9, color=INK)
fig.savefig(os.path.join(RES, 'fig1_multiverse_jaccard.png'), bbox_inches='tight', dpi=200); plt.close(fig)

# Fig 2: swap distances vs n
cur = o['sample_size_curve']
ns = sorted(int(k) for k in cur)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for key, lab, col in (('swap_consensus', 'consensus-only multiverse', C[0]), ('swap_full', 'full multiverse', C[1])):
    ax.plot(ns, [cur[str(n)][key]['designer_swap'] for n in ns], '-o', color=col, lw=2, ms=5, label=f'swap designer ({lab})')
    ax.plot(ns, [cur[str(n)][key]['sample_swap'] for n in ns], '--s', color=col, lw=2, ms=5, label=f'swap participants ({lab})')
ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_xlabel('number of participants (bootstrap from CCAI pool)')
ax.set_ylabel('mean Jaccard distance')
ax.set_ylim(0, None); ax.legend(fontsize=7, frameon=False)
ax.set_title('Who changes the constitution more: the designer or the sample?', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig2_swap_vs_n.png'), dpi=200); plt.close(fig)

# Fig 3: representation audit by design family
a = pd.read_csv(os.path.join(RES, 'ccai_design_audit.csv'))
ac = pd.read_csv(os.path.join(RES, 'ccai_design_audit_corrected_groups.csv'))
a['group_gap_corr'] = ac[ac.groups == 'corrected']['group_gap'].to_numpy()
fams = ['consensus', 'majoritarian', 'proportional', 'hybrid']
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, col, lab in zip(axes, ['jr_ratio', 'seed_share', 'group_gap_corr'],
                        ['largest unrepresented cohesive\ngroup / (n/k)  (JR needs < 1)',
                         "share of constitution that is\norganizers' seed statements",
                         'endorsement gap between the two\nopinion groups (re-aligned labels)']):
    for i, f in enumerate(fams):
        v = a.loc[a.family == f, col].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=12,
                   color=C[i], alpha=.8, edgecolor='#fcfcfb', linewidth=.5)
    if col == 'jr_ratio':
        ax.axhline(1, color=INK2, lw=1, ls=':')
    if col == 'seed_share':
        ax.axhline(o['seed_share_in_pool'], color=INK2, lw=1, ls=':')
        ax.text(3.4, o['seed_share_in_pool'] + .01, 'pool', fontsize=6, color=INK2, ha='right')
    ax.set_xticks(range(4)); ax.set_xticklabels(['cons.', 'AV', 'prop.', 'hybrid'], fontsize=7)
    ax.set_title(lab, fontsize=7.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig3_representation_audit.png'), dpi=200); plt.close(fig)

# Fig 4: openData cross-conversation
p = os.path.join(RES, 'opendata_decomposition.csv')
if os.path.exists(p):
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.scatter(d.n, d.cons_designer_swap, color=C[0], s=22, label='swap designer', edgecolor='#fcfcfb')
    ax.scatter(d.n, d.cons_sample_swap, color=C[1], s=22, marker='s', label='swap participants', edgecolor='#fcfcfb')
    ax.set_xscale('log'); ax.set_xlabel('participants n'); ax.set_ylabel('mean Jaccard distance')
    ax.set_title('Consensus-only multiverse: 16 Polis conversations (dots), CCAI (stars)', fontsize=8, color=INK)
    ccn = o['n_participants']
    ax.scatter([ccn], [o['swap_consensus_multiverse']['designer_swap']], color=C[0], s=60, marker='*')
    ax.scatter([ccn], [o['swap_consensus_multiverse']['sample_swap']], color=C[1], s=60, marker='*')
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    r = d.cons_designer_swap / d.cons_sample_swap
    ax.scatter(d.n, r, color=C[2], s=22, edgecolor='#fcfcfb')
    ax.axhline(1, color=INK2, ls=':', lw=1)
    ax.set_xscale('log'); ax.set_xlabel('participants n'); ax.set_ylabel('designer / sample distance ratio')
    ax.set_title('Designer leverage relative to sampling noise', fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig4_opendata.png'), dpi=200); plt.close(fig)

# Fig 5: misalignment diagnostic
fig, ax = plt.subplots(figsize=(5.2, 2.6))
mu, sd = o['group_separation_random_mean_sd']
xs = np.linspace(mu - 5 * sd, o['group_separation_corrected'] + 0.01, 300)
ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)), color=INK2, lw=1.5, label='random relabelings (normal fit, 200 perms)')
ax.axvline(o['group_separation_released'], color=C[1], lw=2, label='released CCAI group labels')
ax.axvline(o['group_separation_corrected'], color=C[0], lw=2, label='labels re-aligned to participants')
ax.set_yticks([]); ax.set_xlabel('mean |agree(A) - agree(B)| across statements')
ax.legend(fontsize=7, frameon=False, loc='upper center')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig5_group_separation.png'), dpi=200); plt.close(fig)
print('figures written')
