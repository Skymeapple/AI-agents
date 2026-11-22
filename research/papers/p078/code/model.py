"""Theory part: numerical illustration of Propositions 1-5 calibrated to empirical.json.
All budget quantities are expressed RELATIVE to the cost of the current frontier run, so no
firm-level budget data are assumed."""
import json, os, numpy as np
from collections import Counter
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, actor, R

E = json.load(open(os.path.join(R, 'empirical.json')))
LN10 = np.log(10)
G = E['growth']['2018_2026']
g = G['slope_oom_per_yr'] * LN10                     # compute envelope growth, nats/yr
gam = G['cost_slope_oom_per_yr'] * LN10              # frontier cost growth, nats/yr
gam_lo, gam_hi = [x * LN10 for x in G['cost_ci95']]
h_implied = g - gam                                  # implied compute-per-dollar improvement
out = dict(g_nats=round(g, 3), gamma_nats=round(gam, 3), gamma_ci_nats=[round(gam_lo, 3), round(gam_hi, 3)],
           h_implied_nats=round(h_implied, 3), h_implied_doubling_years=round(np.log(2) / h_implied, 2))

# Prop 1 / 2: tenure of a coalition with annual training budget X times the current frontier run cost
def tenure(X, delta, mu, gamma=gam):
    if gamma <= mu: return np.inf
    return max(0.0, (np.log(X) + delta * LN10) / (gamma - mu))
tab = []
for X in [0.1, 0.3, 1, 3, 10, 30]:
    for delta in [0.0, 0.5, 1.0]:
        for mu in [0.0, np.log(1.25), np.log(1.5)]:
            tab.append(dict(X=X, delta=delta, mu_growth_factor=round(np.exp(mu), 2),
                            tenure_years=round(tenure(X, delta, mu), 2),
                            tenure_lo=round(tenure(X, delta, mu, gam_lo), 2), tenure_hi=round(tenure(X, delta, mu, gam_hi), 2)))
out['tenure_table'] = tab
out['pooling_years_per_doubling_mu0'] = dict(point=round(np.log(2) / gam, 3), lo=round(np.log(2) / gam_hi, 3), hi=round(np.log(2) / gam_lo, 3))
out['pooling_k_equal'] = {str(k): round(np.log(k) / gam, 2) for k in [2, 5, 10, 27, 100]}
out['required_budget_growth_for_permanent_membership'] = round(np.exp(gam), 2)

# Prop 3: bounded vs unbounded exclusivity margin
lam = E['open_gap']['mean_lag_2020_2026']
out['diffusion_bound_oom'] = round(G['slope_oom_per_yr'] * lam, 2)

# Prop 4: compute cap -> catch-up time for an actor with budget ratio X (relative to cost of a cap-level run at t_c)
def catchup(X, delta, mu, h):
    return max(0.0, (-np.log(X) - delta * LN10) / (mu + h))
out['cap_catchup'] = {f'X={X}': {f'mu={round(np.exp(mu),2)}': round(catchup(X, 0.0, mu, h_implied), 2) for mu in [0.0, np.log(1.25), np.log(1.5)]}
                      for X in [0.01, 0.03, 0.1, 0.3]}

# Prop 5: identification of the number of frontier actors under non-random disclosure
ALL = load('2018-01-01')
def is_genai(r): return any(x in r['domain'] for x in ['Language', 'Multimodal'])
w = [r for r in ALL if r['t'] > ALL[-1]['t'] - 1.0 and is_genai(r) and not r['open'] and r['access']]
q = sum(r['C'] is not None for r in w) / len(w)
per_actor = Counter(actor(r['org']) for r in w)
top = [c for a, c in per_actor.most_common(8)]
m = float(np.median(top))
out['closed_last12m'] = dict(n_models=len(w), disclosure_rate_q=round(q, 3), median_models_per_top8_actor=m,
                             top8=per_actor.most_common(8))
p_obs = 1 - (1 - q) ** m
def max_N_consistent(n_obs, p, alpha=0.05):
    N = n_obs
    while stats.binom.cdf(n_obs, N + 1, p) >= alpha: N += 1
    return N
out['identification'] = dict(p_actor_observed=round(p_obs, 3),
                             expected_observed_if_N=[dict(N=N, E_obs=round(N * p_obs, 2), P_obs_le1=round(stats.binom.cdf(1, N, p_obs), 3)) for N in [1, 2, 3, 5, 8, 10]],
                             max_N_consistent_with_obs1_at_5pct=max_N_consistent(1, p_obs),
                             max_N_consistent_with_obs1_at_5pct_q2021=max_N_consistent(1, 1 - (1 - 0.8) ** m))
# Fisher test: missingness closed vs open, 2025-2026
ms = E['missing_compute_share_lang_mm']
a = ms['2025_closed']['missing'] + ms['2026_closed']['missing']; b = ms['2025_closed']['n'] + ms['2026_closed']['n'] - a
c = ms['2025_open']['missing'] + ms['2026_open']['missing']; d = ms['2025_open']['n'] + ms['2026_open']['n'] - c
orr, p = stats.fisher_exact([[a, b], [c, d]])
out['fisher_missing_closed_vs_open_2025_26'] = dict(table=[[a, b], [c, d]], odds_ratio=round(orr, 2), p=float(f'{p:.2e}'))
# logistic trend of closed missingness 2018-2026
yrs, ys = [], []
for r in ALL:
    if is_genai(r) and not r['open'] and r['access']:
        yrs.append(r['t']); ys.append(int(r['C'] is None))
yrs = np.array(yrs); ys = np.array(ys)
X = np.vstack([np.ones_like(yrs), yrs - 2022]).T
beta = np.zeros(2)
for _ in range(50):
    pr = 1 / (1 + np.exp(-X @ beta)); W = pr * (1 - pr)
    beta += np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (ys - pr))
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
out['logit_closed_missing_trend'] = dict(n=len(ys), slope_per_yr=round(beta[1], 3), se=round(se[1], 3), odds_mult_per_yr=round(np.exp(beta[1]), 2))
json.dump(out, open(os.path.join(R, 'model.json'), 'w'), indent=1, default=str)

# Figure 5: tenure vs budget ratio; Figure 6: identification curve
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
Xs = np.logspace(-1.5, 2, 200)
for delta, col in zip([0, 0.5, 1.0], ['#1b9e77', '#d95f02', '#7570b3']):
    axes[0].plot(Xs, [tenure(x, delta, 0) for x in Xs], c=col, label=f'δ={delta}, flat budget')
    axes[0].plot(Xs, [tenure(x, delta, np.log(1.5)) for x in Xs], c=col, ls='--', label=f'δ={delta}, budget +50%/yr')
axes[0].set_xscale('log'); axes[0].set_xlabel('annual budget / current frontier-run cost (X)'); axes[0].set_ylabel('years of δ-frontier membership')
axes[0].legend(fontsize=6); axes[0].set_title('Prop. 1-2: tenure is logarithmic in budget', fontsize=9)
Ns = np.arange(1, 16)
for qq, col, lab in [(q, '#c0392b', f'q={q:.2f} (closed, last 12m)'), (0.8, '#2471a3', 'q=0.80 (≈2021 level)')]:
    pp = 1 - (1 - qq) ** m
    axes[1].plot(Ns, [stats.binom.cdf(1, N, pp) for N in Ns], 'o-', ms=3, c=col, label=lab)
axes[1].axhline(0.05, c='k', lw=.6, ls=':'); axes[1].set_xlabel('true number of frontier actors N'); axes[1].set_ylabel('P(≤1 actor observed with disclosed compute)')
axes[1].legend(fontsize=7); axes[1].set_title('Prop. 5: identification under disclosure censoring', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig5_model.png'), dpi=160)
print(json.dumps(out, indent=1, default=str))
