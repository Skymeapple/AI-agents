"""Generate figures from results/*.json"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = '../results/'
e1 = json.load(open(R + 'e1_theory.json'))
e1b = json.load(open(R + 'e1b_rho_fine.json'))
e2 = json.load(open(R + 'e2_sae.json'))
C = ['#1f5fa8', '#d1495b', '#2e933c', '#edae49', '#6c4f9e', '#333333']

# Fig 1: theta2* vs CV for several lambda (theory lines, SAE markers) + parent recall
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
lams = sorted(set(r['lam'] for r in e1['grid']))
for k, lam in enumerate(lams):
    rows = [r for r in e1['grid'] if r['lam'] == lam]
    ax[0].plot([r['cv'] for r in rows], [r['t2_deg'] for r in rows], '-', color=C[k], label=f'exact, λ={lam}')
    ax[1].plot([r['cv'] for r in rows], [r['recall_pc'] for r in rows], '-', color=C[k], label=f'λ={lam}')
    srows = [r for r in e2['grid'] if r['lam'] == lam]
    if srows:
        ax[0].plot([r['cv'] for r in srows], [r['t2_deg'] for r in srows], 'o', color=C[k], mfc='none')
        ax[1].plot([r['cv'] for r in srows], [r['recall_pc'] for r in srows], 'o', color=C[k], mfc='none')
ax[0].axhline(90, ls=':', color='gray'); ax[0].text(0.01, 86.5, 'faithful (90°)', color='gray', fontsize=8)
ax[0].axhline(45, ls=':', color='gray'); ax[0].text(0.35, 46, 'fully absorbed (45°)', color='gray', fontsize=8)
ax[0].set_xlabel('magnitude coefficient of variation'); ax[0].set_ylabel('child-latent angle θ₂* (deg)')
ax[0].set_title('(a) optimal dictionary (lines) vs trained SAE (○)')
ax[1].set_xlabel('magnitude coefficient of variation'); ax[1].set_ylabel('parent-latent recall on P∧C inputs')
ax[1].set_title('(b) silent false negatives (precision = 1.0 everywhere)')
ax[0].legend(fontsize=7); ax[1].set_ylim(-0.03, 1.03)
plt.tight_layout(); plt.savefig(R + 'fig1_phase.png', dpi=160); plt.close()

# Fig 2: q sweep and rho sweep
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot([r['q'] for r in e1['q_sweep']], [r['t2_deg'] for r in e1['q_sweep']], '-', color=C[0], label='exact')
ax[0].plot([r['q'] for r in e2['q_sweep']], [r['t2_deg'] for r in e2['q_sweep']], 'o', color=C[0], mfc='none', label='SAE')
ax[0].set_ylim(40, 92); ax[0].set_xlabel('q = P(child | parent)'); ax[0].set_ylabel('θ₂* (deg)')
ax[0].set_title('(a) absorption is invariant to co-occurrence rate'); ax[0].legend(fontsize=8)
for k, cv in enumerate([0.0, 0.2]):
    rows = [r for r in e1['rho_sweep'] if r['cv'] == cv]
    srows = [r for r in e2['rho_sweep'] if r['cv'] == cv]
    ax[1].plot([r['rho'] for r in rows], [r['t2_deg'] for r in rows], '-', color=C[k], label=f'exact, CV={cv}')
    ax[1].plot([r['rho'] for r in srows], [r['t2_deg'] for r in srows], 'o', color=C[k], mfc='none')
    ax[2].plot([r['rho'] for r in rows], [r['rel_act_pc'] for r in rows], '-', color=C[k], label=f'CV={cv}')
    ax[2].plot([r['rho'] for r in srows], [r['rel_act_pc'] for r in srows], 'o', color=C[k], mfc='none')
ax[1].axhline(90, ls=':', color='gray'); ax[1].set_xscale('symlog', linthresh=0.05)
ax[1].set_xlabel('ρ = P(child alone)/P(child with parent)'); ax[1].set_ylabel('θ₂* (deg)')
ax[1].set_title('(b) partial absorption never vanishes'); ax[1].legend(fontsize=8)
ax[2].axhline(1, ls=':', color='gray'); ax[2].set_xscale('symlog', linthresh=0.05)
ax[2].set_xlabel('ρ'); ax[2].set_ylabel('parent act. / true parent magnitude')
ax[2].set_title('(c) parent magnitude under-read on P∧C')
plt.tight_layout(); plt.savefig(R + 'fig2_q_rho.png', dpi=160); plt.close()

# Fig 3: fine rho transition at fixed magnitudes
fig, ax = plt.subplots(figsize=(5, 3.6))
for k, blk in enumerate(e1b):
    ax.plot([r['rho'] for r in blk['rows']], [r['t2_deg'] for r in blk['rows']], '.-', color=C[k], label=f"λ={blk['lam']}")
    ax.axvline(blk['rho_dagger'], color=C[k], ls='--', lw=0.8)
ax.set_xlabel('ρ'); ax.set_ylabel('θ₂* (deg)'); ax.set_title('first-order transition (CV=0); dashed: ρ† (Prop. 3)')
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(R + 'fig3_rho_transition.png', dpi=160); plt.close()

# Fig 4: E3 recall vs precision and diagnostic
try:
    e3 = json.load(open(R + 'e3_many.json'))
    rows = e3['rows']
    roles = ['parent_hier', 'child_hier', 'parent_corr', 'child_corr', 'parent_excl', 'child_excl', 'indep']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, ro in enumerate(roles):
        rr = [r for r in rows if r['role'] == ro]
        col = C[k % len(C)] if k < 6 else '#999999'
        ax[0].scatter([r['recall'] for r in rr], [r['precision'] for r in rr], s=14, color=col, label=ro, alpha=0.7)
        ax[1].scatter([r['diag_A'] for r in rr], [r['recall'] for r in rr], s=14, color=col, label=ro, alpha=0.7)
    ax[0].set_xlabel('recall of matched latent'); ax[0].set_ylabel('precision of matched latent')
    ax[0].set_title('(a) low recall hides behind perfect precision'); ax[0].legend(fontsize=6)
    ax[1].set_xlabel('weight-only score A (decoder overlap not read by encoder)'); ax[1].set_ylabel('recall')
    ax[1].set_title('(b) diagnostic vs ground truth')
    plt.tight_layout(); plt.savefig(R + 'fig4_many.png', dpi=160); plt.close()
except FileNotFoundError:
    print('e3 not ready')
print('figures done')
