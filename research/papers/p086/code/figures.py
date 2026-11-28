"""Figures for the paper (reads CSV/JSON in results/, writes PNGs to results/)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import data as D

R = os.path.join(os.path.dirname(__file__), '..', 'results')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID = '#222222', '#666666', '#e4e4e4'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 2})


def load(sc):
    return np.genfromtxt(os.path.join(R, f'fleet_{sc}.csv'), delimiter=',', names=True)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for i, sc in enumerate(['boom', 'plateau', 'retrenchment']):
        d = load(sc)
        ax.plot(d['year'], d['capex'], color=C[i], marker='o', ms=4, label=sc)
    ax.axvline(2026.5, color=MUTED, lw=0.8, ls=':')
    ax.text(2026.6, ax.get_ylim()[1] * 0.92, 'scenarios', color=MUTED, fontsize=8)
    ax.set_title('Capex, four hyperscalers ($bn)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for i, sc in enumerate(['boom', 'plateau']):
        d = load(sc)
        ax.plot(d['year'], d['dep_acc'], color=C[i], ls='--', label=f'{sc}: straight-line')
        ax.plot(d['year'], d['dep_econ'], color=C[i], label=f'{sc}: economic')
    ax.set_title('Depreciation, base case ($bn)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig1_capex_depreciation.png'), dpi=160)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    d = np.genfromtxt(os.path.join(R, 'phi_curves.csv'), delimiter=',', names=True)
    for i, g in enumerate([0.2, 0.35, 0.5]):
        m = np.isclose(d['g'], g)
        ax.plot(d['n'][m], d['phi'][m], color=C[i], label=f'g={g} (T={d["T"][m][0]:.1f}y)')
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlabel('steady capex growth rate n (log/yr)'); ax.set_ylabel('Φ = economic / reported depreciation')
    ax.set_title('Proposition 1–2: bias grows with investment growth', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    gc = np.genfromtxt(os.path.join(R, 'gstar_curve.csv'), delimiter=',', names=True)
    for i, L in enumerate([5.0, 5.5, 6.0]):
        m = np.isclose(gc['L'], L)
        ax.plot(gc['n'][m], gc['gstar'][m], color=C[i], label=f'L={L}y')
    lo, hi = np.log(2) / 3.13, np.log(2) / 1.54
    ax.axhspan(lo, hi, color=C[3], alpha=0.18, lw=0)
    ax.axhline(np.log(2) / 2.07, color=C[3], lw=1.2)
    ax.text(0.02, np.log(2) / 2.07 + 0.008, 'Epoch AI ML-GPU price-performance (95% CI band)', fontsize=7.5, color=INK)
    for y, nn in json.load(open(os.path.join(R, 'extras.json')))['capex_growth_log'].items():
        if int(y) >= 2024:
            ax.axvline(nn, color=MUTED, lw=0.6, ls=':'); ax.text(nn + (0.01 if y != '2024' else -0.01), 0.12, y, fontsize=7, color=MUTED, ha='left' if y != '2024' else 'right')
    ax.set_ylim(0.1, 0.5)
    ax.set_xlabel('capex growth rate n (log/yr)'); ax.set_ylabel('break-even obsolescence g*')
    ax.set_title('Corollary 2: straight-line is unbiased only if g < g*(n,L)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig2_phi_gstar.png'), dpi=160)


def fig3():
    m = np.genfromtxt(os.path.join(R, 'montecarlo.csv'), delimiter=',', names=True)
    s = json.load(open(os.path.join(R, 'montecarlo_summary.json')))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.hist(m['gap2026'], bins=40, color=C[0], edgecolor='white', lw=0.5)
    ax.axvline(np.median(m['gap2026']), color=INK, lw=1)
    ax.set_xlabel('2026 depreciation gap, economic − reported ($bn)'); ax.set_ylabel('draws')
    ax.set_title(f'Monte Carlo, N={s["N_valid"]}', loc='left', fontsize=10)
    ax = axes[1]
    sp = s['spearman_gap2026']; keys = sorted(sp, key=lambda k: abs(sp[k]))
    ax.barh(keys, [sp[k] for k in keys], color=C[0], height=0.5)
    for k in keys:
        ax.text(sp[k] + (0.02 if sp[k] >= 0 else -0.02), k, f'{sp[k]:.2f}', va='center',
                ha='left' if sp[k] >= 0 else 'right', fontsize=8, color=INK)
    ax.set_xlim(-0.2, 1.05); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title('Spearman rank correlation with 2026 gap', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig3_montecarlo.png'), dpi=160)


def fig4():
    raw = open(os.path.join(R, 'heatmap_gap_ratio_2026.csv')).read().strip().split('\n')
    Ls = [float(x) for x in raw[0].split(',')[1:]]
    gs = [float(r.split(',')[0]) for r in raw[1:]]
    Z = np.array([[float(x) for x in r.split(',')[1:]] for r in raw[1:]])
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#e9e9e6', '#eb6834'])
    vmax = np.abs(Z).max()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(Z, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls)
    ax.set_yticks(range(len(gs))); ax.set_yticklabels(gs)
    ax.grid(False)
    for i in range(len(gs)):
        for j in range(len(Ls)):
            ax.text(j, i, f'{Z[i,j]*100:.0f}', ha='center', va='center', fontsize=7, color=INK)
    ax.set_xlabel('accounting life L of servers (years)'); ax.set_ylabel('obsolescence rate g (per year)')
    ax.set_title('2026 gap as % of reported depreciation', loc='left', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig4_heatmap.png'), dpi=160)


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, sc in enumerate(['boom', 'plateau', 'retrenchment']):
        d = load(sc)
        axes[0].plot(d['year'], d['gap'], color=C[i], label=sc)
        axes[1].plot(d['year'], d['value_gap'], color=C[i], label=sc)
    cal = json.load(open(os.path.join(R, 'calibration.json')))
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_title('Flow gap: economic − reported depreciation ($bn)', loc='left', fontsize=10)
    axes[1].set_title('Stock gap: book − economic value ($bn)', loc='left', fontsize=10)
    for a in axes: a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig5_gaps.png'), dpi=160)


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures written')
