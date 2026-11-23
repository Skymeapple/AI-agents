"""Empirical part: definition-robust frontier concentration, disclosure-censoring bounds,
frontier growth rates, open-weight gap/lag, country shares. Outputs -> ../results/"""
import json, numpy as np, os
from collections import Counter, defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, actor, hhi, R

rng = np.random.default_rng(0)
ALL = load('2010-01-01')
K = [r for r in ALL if r['C']]
for r in K: r['L'] = np.log10(r['C'])
GRID = np.round(np.arange(2016.0, 2026.8334, 1/12), 4)  # monthly to end Oct 2026
out = {}

def envelope(t):
    v = [r['L'] for r in K if r['t'] <= t]
    return max(v) if v else np.nan

ENV = {tau: envelope(tau) for tau in GRID}

def frontier_set(tau, delta, W, rows=K):
    M = ENV[tau]
    return [r for r in rows if tau - W < r['t'] <= tau and r['L'] >= M - delta]

def metrics(S, mode='lab'):
    c = Counter(actor(r['org'], mode) for r in S)
    if not c: return dict(n=0, neff=np.nan, top=np.nan)
    h = hhi(c)
    return dict(n=len(c), neff=1/h, top=max(c.values())/sum(c.values()))

# ---------- 1. definition sweep ----------
deltas = [0.3, 0.5, 1.0, 1.5]; Ws = [0.5, 1.0, 2.0]
sweep = {}
for mode in ['lab', 'bloc']:
    for d in deltas:
        for W in Ws:
            key = f'{mode}_d{d}_W{W}'
            sweep[key] = [metrics(frontier_set(tau, d, W), mode) for tau in GRID]
# Epoch "frontier model" flag (top-10 by compute at release) as alternative definition
for W in Ws:
    sweep[f'lab_epochflag_W{W}'] = [metrics([r for r in ALL if r['epoch_frontier'] and tau - W < r['t'] <= tau]) for tau in GRID]

def at(key, year):  # value at end of a calendar year (December grid point) or last grid point
    i = int(np.argmin(np.abs(GRID - (year + 11/12)))) if year < 2026 else len(GRID) - 1
    return sweep[key][i]

table = {}
for key in sweep:
    table[key] = {str(y): {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)) for k, v in at(key, y).items()}
                  for y in [2018, 2020, 2022, 2023, 2024, 2025, 2026]}
out['definition_sweep_yearend'] = table

# Rank-reversal check: sign of change 2022->2026 in N_eff across definitions
signs = {}
for key in sweep:
    a, b = at(key, 2022)['neff'], at(key, 2026)['neff']
    signs[key] = None if np.isnan(a) or np.isnan(b) else float(np.sign(b - a))
out['neff_change_sign_2022_to_2026'] = signs
vals = [v for v in signs.values() if v is not None]
out['neff_change_sign_counts'] = dict(Counter(vals))

# ---------- 2. disclosure censoring: missingness & partial identification ----------
def is_genai(r):
    return any(x in r['domain'] for x in ['Language', 'Multimodal'])
miss = defaultdict(lambda: [0, 0])
for r in ALL:
    if r['t'] < 2018 or not is_genai(r): continue
    cls = 'open' if r['open'] else ('closed' if r['access'] else 'unknown_access')
    y = r['date'].year
    miss[(y, cls)][0] += 1; miss[(y, cls)][1] += (r['C'] is None)
out['missing_compute_share_lang_mm'] = {f'{y}_{c}': dict(n=v[0], missing=v[1], share=round(v[1]/v[0], 3)) for (y, c), v in sorted(miss.items())}

U = [r for r in ALL if r['C'] is None and is_genai(r) and 'Industry' in (r['cat'] or '')]
bounds = {d: dict(lb=[], inc=[], ub=[]) for d in deltas}
for d in deltas:
    for tau in GRID:
        S = frontier_set(tau, d, 1.0)
        known = set(actor(r['org']) for r in S)
        recent_frontier_actors = set(actor(r['org']) for r in K if tau - 2.0 < r['t'] <= tau and r['L'] >= ENV[tau] - d - 0.0)
        # incumbents: actors with a delta-frontier model (relative to the envelope at its own release) in last 24 months
        inc_actors = set(actor(r['org']) for r in K if tau - 2.0 < r['t'] <= tau and r['L'] >= envelope(r['t']) - d)
        Uw = [r for r in U if tau - 1.0 < r['t'] <= tau]
        inc = known | (set(actor(r['org']) for r in Uw) & inc_actors)
        ub = known | set(actor(r['org']) for r in Uw)
        bounds[d]['lb'].append(len(known)); bounds[d]['inc'].append(len(inc)); bounds[d]['ub'].append(len(ub))
out['partial_id_last'] = {str(d): {k: v[-1] for k, v in bounds[d].items()} for d in deltas}
out['partial_id_2024end'] = {str(d): {k: v[int(np.argmin(np.abs(GRID-2024.9167)))] for k, v in bounds[d].items()} for d in deltas}
# which incumbents are added by imputation at last grid point (delta=1)
tau = GRID[-1]; d = 1.0
S = frontier_set(tau, d, 1.0); known = set(actor(r['org']) for r in S)
inc_actors = set(actor(r['org']) for r in K if tau - 2.0 < r['t'] <= tau and r['L'] >= envelope(r['t']) - d)
Uw = [r for r in U if tau - 1.0 < r['t'] <= tau]
out['last_known_frontier_actors_d1'] = sorted(known)
out['last_imputed_incumbents_d1'] = sorted((set(actor(r['org']) for r in Uw) & inc_actors) - known)
out['last_unknown_compute_industry_genai_models_12m'] = len(Uw)

# ---------- 3. growth rates ----------
def ols(x, y):
    X = np.vstack([np.ones_like(x), x]).T; b = np.linalg.lstsq(X, y, rcond=None)[0]; return b
def boot_slope(x, y, B=2000):
    n = len(x); s = []
    for _ in range(B):
        i = rng.integers(0, n, n); s.append(ols(x[i], y[i])[1])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
growth = {}
for name, lo, hi in [('2018_2026', 2018, 2026.84), ('2018_2022', 2018, 2022.0), ('2022_2026', 2022, 2026.84)]:
    F = [r for r in K if r['epoch_frontier'] and lo <= r['t'] < hi]
    x = np.array([r['t'] for r in F]); y = np.array([r['L'] for r in F])
    b = ols(x, y); ci = boot_slope(x, y)
    growth[name] = dict(n=len(F), slope_oom_per_yr=round(float(b[1]), 3), ci95=[round(ci[0], 3), round(ci[1], 3)],
                        factor_per_yr=round(10**b[1], 2))
    Fc = [r for r in F if r['cost']]
    if len(Fc) >= 5:
        xc = np.array([r['t'] for r in Fc]); yc = np.log10([r['cost'] for r in Fc]); bc = ols(xc, yc); cic = boot_slope(xc, yc)
        growth[name].update(cost_n=len(Fc), cost_slope_oom_per_yr=round(float(bc[1]), 3), cost_ci95=[round(cic[0], 3), round(cic[1], 3)],
                            cost_factor_per_yr=round(10**bc[1], 2))
# records (running max)
rec = []; m = -1
for r in K:
    if r['t'] >= 2016 and r['L'] > m: rec.append(r); m = r['L']
    elif r['L'] > m: m = r['L']
growth['records_since_2016'] = [dict(model=r['model'], org=r['org'], date=str(r['date']), log10C=round(r['L'], 2)) for r in rec]
costs = sorted([r for r in K if r['cost']], key=lambda r: -r['cost'])[:8]
growth['top_costs_2023usd'] = [dict(model=r['model'], date=str(r['date']), cost=round(r['cost'] / 1e6, 1)) for r in costs]
out['growth'] = growth
g_hat = growth['2018_2026']['slope_oom_per_yr']

# ---------- 4. open-weight gap and lag ----------
gap, lag, pred = [], [], []
env_series = sorted(((r['t'], r['L']) for r in K), key=lambda z: z[0])
def first_time_env_reached(L):
    m = -1
    for t, l in env_series:
        m = max(m, l)
        if m >= L: return t
    return np.nan
for tau in GRID:
    O = [r['L'] for r in K if r['open'] and r['t'] <= tau]
    if not O: gap.append(np.nan); lag.append(np.nan); continue
    o = max(O); gap.append(ENV[tau] - o); lag.append(tau - first_time_env_reached(o))
gap, lag = np.array(gap), np.array(lag)
sel = GRID >= 2020
out['open_gap'] = dict(
    gap_oom_yearend={str(y): round(float(gap[int(np.argmin(np.abs(GRID-(y+11/12))))]), 2) for y in range(2018, 2026)} | {'2026-10': round(float(gap[-1]), 2)},
    lag_years_yearend={str(y): round(float(lag[int(np.argmin(np.abs(GRID-(y+11/12))))]), 2) for y in range(2018, 2026)} | {'2026-10': round(float(lag[-1]), 2)},
    mean_gap_2020_2026=round(float(np.nanmean(gap[sel])), 2), mean_lag_2020_2026=round(float(np.nanmean(lag[sel])), 2),
    corr_gap_vs_ghat_lag=round(float(np.corrcoef(gap[sel], g_hat * lag[sel])[0, 1]), 3),
    mean_abs_err_gap_minus_ghat_lag=round(float(np.nanmean(np.abs(gap[sel] - g_hat * lag[sel]))), 3))
best_open = max((r for r in K if r['open']), key=lambda r: r['L'])
out['open_gap']['best_open_model'] = dict(model=best_open['model'], org=best_open['org'], date=str(best_open['date']), log10C=round(best_open['L'], 2))

# ---------- 5. country shares (delta=1, W=1) ----------
cs = {}
for y in [2020, 2022, 2024, 2025, 2026]:
    i = int(np.argmin(np.abs(GRID - (y + 11/12)))) if y < 2026 else len(GRID) - 1
    S = frontier_set(GRID[i], 1.0, 1.0)
    c = Counter('USA' if r['country'].startswith('United States') else ('China' if r['country'].startswith('China') else 'Other') for r in S)
    ca = Counter()
    for a, cc in {(actor(r['org']), ('USA' if r['country'].startswith('United States') else ('China' if r['country'].startswith('China') else 'Other'))) for r in S}: ca[cc] += 1
    cs[str(y)] = dict(models=dict(c), actors=dict(ca), n_models=len(S))
out['country_d1_W1'] = cs
out['n_rows_total'] = len(ALL); out['n_with_compute'] = len(K)
out['data_last_date'] = str(ALL[-1]['date'])
json.dump(out, open(os.path.join(R, 'empirical.json'), 'w'), indent=1, default=str)

# ---------- figures ----------
plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(figsize=(7, 4))
Kp = [r for r in K if r['t'] >= 2016]
ax.scatter([r['t'] for r in Kp if not r['open']], [r['L'] for r in Kp if not r['open']], s=10, c='#c0392b', alpha=.6, label='closed / unreleased / API')
ax.scatter([r['t'] for r in Kp if r['open']], [r['L'] for r in Kp if r['open']], s=10, c='#2471a3', alpha=.6, label='open weights')
ax.step(GRID, [ENV[t] for t in GRID], where='post', c='k', lw=1.2, label='frontier envelope M(t)')
ax.plot(GRID, [ENV[t] - 1 for t in GRID], 'k--', lw=.8, label='M(t) - 1 OOM')
ax.set_xlabel('publication date'); ax.set_ylabel('log10 training compute (FLOP)'); ax.legend(fontsize=7); ax.set_title('Epoch notable models with compute estimates (n=%d since 2016)' % len(Kp))
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig1_envelope.png'), dpi=160)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for d, col in zip(deltas, ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']):
    axes[0].plot(GRID, [m['n'] for m in sweep[f'lab_d{d}_W1.0']], c=col, label=f'δ={d}')
    axes[1].plot(GRID, [m['neff'] for m in sweep[f'lab_d{d}_W1.0']], c=col, label=f'δ={d}')
axes[0].plot(GRID, [m['n'] for m in sweep['lab_epochflag_W1.0']], 'k:', label='Epoch top-10 flag')
axes[1].plot(GRID, [m['neff'] for m in sweep['lab_epochflag_W1.0']], 'k:', label='Epoch top-10 flag')
axes[0].set_ylabel('distinct actors in frontier set'); axes[1].set_ylabel('N_eff = 1/HHI (model counts)')
for a in axes: a.set_xlabel('date (12-month window)'); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig2_definition_sweep.png'), dpi=160)

fig, ax = plt.subplots(figsize=(7, 3.6))
for d, col in zip([0.5, 1.0], ['#d95f02', '#7570b3']):
    ax.fill_between(GRID, bounds[d]['lb'], bounds[d]['ub'], color=col, alpha=.15)
    ax.plot(GRID, bounds[d]['lb'], c=col, lw=1, label=f'δ={d}: lower (disclosed only)')
    ax.plot(GRID, bounds[d]['inc'], c=col, lw=1.5, ls='--', label=f'δ={d}: incumbent imputation')
ax.set_xlabel('date (12-month window)'); ax.set_ylabel('distinct frontier actors'); ax.legend(fontsize=7); ax.set_xlim(2019, 2026.9)
ax.set_title('Partial identification under non-random compute disclosure')
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig3_partial_id.png'), dpi=160)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(GRID, gap, c='#2471a3', label='gap G(t): log10 M(t) - log10 O(t) [OOM]')
ax.plot(GRID, g_hat * lag, c='#c0392b', ls='--', label=f'ĝ·Λ(t), ĝ={g_hat} OOM/yr')
ax2 = ax.twinx(); ax2.plot(GRID, lag, c='gray', lw=.8, alpha=.7); ax2.set_ylabel('lag Λ(t) [years] (gray)')
ax.set_xlabel('date'); ax.set_ylabel('OOM'); ax.legend(fontsize=7, loc='upper left'); ax.set_xlim(2018, 2026.9)
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig4_open_gap.png'), dpi=160)
print(json.dumps({k: out[k] for k in ['growth', 'open_gap', 'partial_id_last', 'partial_id_2024end', 'neff_change_sign_counts', 'missing_compute_share_lang_mm', 'country_d1_W1', 'last_known_frontier_actors_d1', 'last_imputed_incumbents_d1']}, indent=1, default=str))
