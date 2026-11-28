"""Figures for the certification analysis. Usage: python3 make_figures_cert.py"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
st = pd.read_csv(os.path.join(RES, 'certify_ccai_statements.csv')).sort_values('agree_seen', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(1, len(st) + 1)
col = np.where(st.boot_top58 >= .95, '#1b7837', np.where(st.boot_top58 <= .05, '#999999', '#d95f02'))
ax.vlines(x, st.rank_lo, st.rank_hi, colors=col, lw=1)
ax.scatter(x, x, s=3, c='k', zorder=3)
ax.axhline(58.5, ls='--', c='k', lw=0.8); ax.axvline(58.5, ls=':', c='k', lw=0.8)
ax.set_xlabel('statement, ordered by estimated agreement (agree/seen)'); ax.set_ylabel('95% rank confidence interval')
ax.set_title('CCAI: no statement is certified inside the 58-seat cut (dashed)')
ax.invert_yaxis()
from matplotlib.lines import Line2D
ax.legend([Line2D([0], [0], c=c) for c in ('#1b7837', '#d95f02', '#999999')],
          ['in top-58 in >=95% of bootstraps', 'contested', 'in top-58 in <=5% of bootstraps'], fontsize=7, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_cert1_ccai_rank_ci.png'), dpi=150)

od = pd.read_csv(os.path.join(RES, 'certify_opendata.csv'))
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
for ax, xv, lab in ((axs[0], 'n', 'participants n'), (axs[1], 'm_median', 'median votes per statement m')):
    c = ['#d95f02' if v == 'CCAI' else ('#7570b3' if v.startswith('austria') else '#1b9e77') for v in od.conversation]
    ax.scatter(od[xv], od.contested_per_seat, c=c, s=22)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(lab)
axs[0].set_ylabel('contested statements per seat')
axs[0].set_title('vs headcount'); axs[1].set_title('vs per-statement exposure')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_cert2_exposure_law.png'), dpi=150)

pl = pd.read_csv(os.path.join(RES, 'certify_planning.csv')).query("truth=='eb'").groupby('m').mean(numeric_only=True)
rs = pd.read_csv(os.path.join(RES, 'routing_sim.csv')).groupby(['arm', 'mult']).mean(numeric_only=True).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
ax = axs[0]
ax.plot(pl.index, pl.cert_in, 'o-', label='certified in (rank CI)'); ax.plot(pl.index, pl.stable_in, 's-', label='stable in (boot >= .95)')
ax.plot(pl.index, 58 - pl.theorem1_uncertifiable, '^--', label='Thm 1 upper bound on certifiable')
ax.axhline(58, c='k', lw=.6); ax.axvline(81, c='grey', ls=':', lw=.8)
ax.set_xscale('log'); ax.set_xlabel('votes per statement m (uniform)'); ax.set_ylabel('seats of 58'); ax.legend(fontsize=7)
ax.set_title('CCAI-calibrated simulation')
ax = axs[1]
for arm, mk in (('uniform', 'o-'), ('routed', 's-')):
    r = rs[rs.arm == arm]; ax.plot(r.budget, r.cert_in, mk, label=f'{arm}: certified in')
    ax.plot(r.budget, r.stable_in, mk.replace('-', '--'), alpha=.6, label=f'{arm}: stable in')
ax.set_xscale('log'); ax.set_xlabel('total votes (1x = CCAI budget)'); ax.legend(fontsize=7); ax.set_title('rank-aware routing vs uniform')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_cert3_planning_routing.png'), dpi=150)
print('ok')
