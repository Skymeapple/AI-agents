"""Diagnostics, partial-rewrite budgets, policy-level theory curve, figures."""
import json, sys, numpy as np, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from organism import make_world, sample, true_policy, rewrite, STYLE

raw = json.load(open('results/raw.json'))
out = {}
# (a) which diagnostic tracks the full-rewrite bias? (kappa=4 runs, all rho/visibility)
R = [r for r in raw if r['kappa'] == 4.0 and not r['subtle']]
absbias = np.array([abs(r['full_rewrite'] - r['theta_dep']) for r in R])
for d in ['resid_auc_auditor', 'resid_auc_probe']:
    v = np.array([r[d] for r in R]); rho_s, p = spearmanr(v, absbias)
    out[f'spearman_absbias_vs_{d}'] = [float(rho_s), float(p)]
v = np.array([r['rewrite_steer'] - r['full_rewrite'] for r in R]); rho_s, p = spearmanr(v, absbias)
out['spearman_absbias_vs_steer_resid'] = [float(rho_s), float(p)]
out['n_runs_diag'] = len(R)
# (b) partial rewrite budgets: bias after rewriting top-k auditor-ranked features
bud = {}
for subtle in [False, True]:
    rs = [r for r in raw if r['kappa'] == 4.0 and r['rho'] == 0.0 and r['subtle'] == subtle]
    ks = rs[0]['levels_k']
    bud[str(subtle)] = {str(k): [float(np.mean([r['levels_B'][i] - r['theta_dep'] for r in rs])),
                                 float(np.std([r['levels_B'][i] - r['theta_dep'] for r in rs], ddof=1) / np.sqrt(len(rs)))]
                        for i, k in enumerate(ks)}
out['partial_budget_bias'] = bud
# (c) policy-level (ground-truth policy, no learning) rewrite bias vs rho
rhos = np.linspace(0, 1, 11); pol = []
for rho in rhos:
    b = []
    for s in range(5):
        rng = np.random.default_rng(10_000 + s); w = make_world(rng, rho=rho)
        Xd, rd = sample(rng, w, 20000, 'dep', 1); Xe, re_ = sample(rng, w, 20000, 'eval', 1)
        Xr = rewrite(rng, w, Xe, STYLE)
        b.append(true_policy(w, Xr, re_).mean() - true_policy(w, Xd, rd).mean())
    pol.append(float(np.mean(b)))
out['policy_level_rewrite_bias'] = dict(zip([f'{x:.1f}' for x in rhos], pol))
json.dump(out, open('results/diagnostics.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# ---------- figures
S = json.load(open('results/summary.json'))
def get(rho, vis, subtle=False, kappa=4.0):
    return [s for s in S if s['rho'] == rho and s['H_visible'] == vis and s['subtle'] == subtle and s['kappa'] == kappa][0]
C = {'naive': '#7f7f7f', 'iw': '#bcbd22', 'full_rewrite': '#1f77b4', 'rde': '#17becf',
     'steer': '#d62728', 'rewrite_steer': '#9467bd'}
LBL = {'naive': 'naive eval', 'iw': 'importance weighting', 'full_rewrite': 'full rewrite',
       'rde': 'dose-response extrap.', 'steer': 'probe steering', 'rewrite_steer': 'rewrite + steering'}
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, vis in zip(axs, [True, False]):
    rh = [0.0, 0.25, 0.5, 0.75]
    for e in C:
        y = [get(r, True if r == 0 else vis)[e + '_bias'] for r in rh]
        se = [get(r, True if r == 0 else vis)[e + '_se'] for r in rh]
        ax.errorbar(rh, y, yerr=1.96 * np.array(se), marker='o', label=LBL[e], color=C[e], capsize=2)
    ax.plot(rhos, pol, 'k--', lw=1, label='policy-level prediction')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"non-rewritable cue {'visible' if vis else 'latent'} to auditor")
    ax.set_xlabel(r'$\rho$ = share of eval evidence in non-rewritable cues')
axs[0].set_ylabel('estimate − true deployment unsafe rate')
axs[1].legend(fontsize=7, loc='lower left')
plt.tight_layout(); plt.savefig('results/fig1_bias_vs_rho.png', dpi=150)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
lab = {'resid_auc_auditor': 'auditor residual AUC', 'resid_auc_probe': 'probe residual AUC', 'steer': 'steering residual'}
for ax, d in zip(axs, ['resid_auc_auditor', 'resid_auc_probe', 'steer']):
    for r in R:
        x = (r['rewrite_steer'] - r['full_rewrite']) if d == 'steer' else r[d]
        ax.scatter(x, abs(r['full_rewrite'] - r['theta_dep']), s=12,
                   c='#1f77b4' if r['H_visible'] else '#d62728', alpha=0.7)
    key = 'spearman_absbias_vs_' + ('steer_resid' if d == 'steer' else d)
    ax.set_title(f"{lab[d]}  (Spearman {out[key][0]:.2f})", fontsize=9)
    ax.set_xlabel(lab[d]); ax.set_ylabel('|full-rewrite bias|')
plt.tight_layout(); plt.savefig('results/fig2_diagnostics.png', dpi=150)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for subtle, col in [(False, '#1f77b4'), (True, '#d62728')]:
    b = bud[str(subtle)]; ks = [int(k) for k in b]
    ax.errorbar(ks, [b[str(k)][0] for k in ks], yerr=[1.96 * b[str(k)][1] for k in ks], marker='o', color=col,
                label='subtle cues weighted by model' if subtle else 'cue weight ∝ frequency gap', capsize=2)
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('# style features rewritten (auditor-ranked)')
ax.set_ylabel('bias vs true deployment rate'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/fig3_partial_budget.png', dpi=150)
print('figures saved')
