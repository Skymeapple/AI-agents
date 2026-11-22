"""Empirical analysis for 'Compute thresholds, capability diffusion and the adaptation-rate condition'.

Inputs : data/epoch_large_scale.csv  (reconstructed by code/parse_epoch.py from Epoch AI's
         Large-Scale AI Models dataset, retrieved 2026-11-22)
Outputs: results/*.json, results/*.csv, results/fig*.png

All quantities are in log10 FLOP units; rates are in log10 units (orders of magnitude, OOM) per year.
Run from the paper root:  python3 code/analysis.py
"""
import csv, json, math, datetime, collections, os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261122)
os.makedirs('results', exist_ok=True)
T0 = 2023.0          # time origin for regressions
FIT_YEARS = (2022, 2025)   # complete calendar years used for fitting
NBOOT = 400

def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

def frac_year(d):
    dt = datetime.date.fromisoformat(d)
    return dt.year + (dt.timetuple().tm_yday - 0.5) / (366 if dt.year % 4 == 0 else 365)

rows = list(csv.DictReader(open('data/epoch_large_scale.csv', encoding='utf-8')))
M = []
for r in rows:
    c = fnum(r['Training compute (FLOP)'])
    M.append(dict(model=r['Model'], org=r['Organization'], date=r['Publication date'],
                  t=frac_year(r['Publication date']), year=int(r['Publication date'][:4]),
                  x=math.log10(c) if c else None, conf=r['Confidence'],
                  country=r['Country (of organization)'], domain=r['Domain'],
                  access=r['Model accessibility']))
out = {}

# ---------------------------------------------------------------- 1. descriptives
years = list(range(2020, 2027))
desc = []
for y in years:
    ys = [m for m in M if m['year'] == y]
    known = [m for m in ys if m['x'] is not None]
    row = dict(year=y, n_listed=len(ys), n_with_compute=len(known),
               missing_share=round(1 - len(known) / len(ys), 3) if ys else None)
    for thr in (23, 24, 25, 26):
        sel = [m for m in known if m['x'] >= thr]
        row[f'n_ge_1e{thr}'] = len(sel)
        row[f'orgs_ge_1e{thr}'] = len(set(m['org'] for m in sel))
    row['max_log10'] = round(max([m['x'] for m in known]), 3) if known else None
    us = sum(1 for m in known if m['x'] >= 25 and 'United States' in m['country'])
    cn = sum(1 for m in known if m['x'] >= 25 and 'China' in m['country'])
    row['ge_1e25_US'] = us; row['ge_1e25_CN'] = cn
    desc.append(row)
with open('results/descriptives_by_year.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(desc[0].keys())); w.writeheader(); w.writerows(desc)
out['descriptives'] = desc
out['n_records'] = len(M); out['n_with_compute'] = sum(m['x'] is not None for m in M)

# ---------------------------------------------------------------- 2. frontier growth g
def frontier_slope(ms, k=1):
    """OLS slope of the mean log10 compute of the top-k models per complete year."""
    ys, vs = [], []
    for y in range(2021, 2026):
        xs = sorted([m['x'] for m in ms if m['year'] == y and m['x'] is not None], reverse=True)
        if len(xs) >= k:
            ys.append(y + 0.5 - T0); vs.append(np.mean(xs[:k]))
    return np.polyfit(ys, vs, 1)   # (slope, intercept at T0)

def records_slope(ms):
    """OLS slope through running-record models (new compute maximum when released), 2019-2026."""
    srt = sorted([m for m in ms if m['x'] is not None], key=lambda m: m['t'])
    best, pts = -1, []
    for m in srt:
        if m['x'] > best:
            best = m['x']; pts.append((m['t'], m['x']))
    pts = np.array(pts)
    return np.polyfit(pts[:, 0] - T0, pts[:, 1], 1)[0], pts

g_top1 = frontier_slope(M, 1)[0]; g_top5, mu0_top5 = frontier_slope(M, 5); g_rec, rec_pts = records_slope(M)
out['frontier_growth'] = dict(top1_per_year=g_top1, top5_per_year=g_top5, record_setters=g_rec,
                              top5_multiplier_per_year=10 ** g_top5, top5_level_at_2023_0=mu0_top5)


# ---------------------------------------------------------------- 3. Poisson intensity models
# Release intensity n(t, x) [models per year per OOM] is modelled as log10 n = eta(t, x; theta).
#   E : exponential tail, fixed shape     eta = c + gamma*t - beta*(x-25)
#   EQ: E plus curvature                  eta = c + gamma*t - beta*(x-25) + q*(x-25)^2
#   G : frontier-relative Gaussian shape  eta = c + lam*t + b1*d + b2*d^2,  d = mu(t) - x,
#       mu(t) = mu0 + g*t taken from the top-5 frontier regression on the same sample.
XLO, XHI, BW = 23.0, 27.5, 0.25
xedges = np.arange(XLO, XHI + 1e-9, BW); xmid = 0.5 * (xedges[:-1] + xedges[1:])
LN10 = math.log(10)

def bin_counts(ms, yrs=FIT_YEARS):
    tab = []
    for y in range(yrs[0], yrs[1] + 1):
        xs = [m['x'] for m in ms if m['year'] == y and m['x'] is not None]
        h, _ = np.histogram(xs, bins=xedges)
        for xm, n in zip(xmid, h):
            tab.append((y + 0.5 - T0, xm, n))
    return np.array(tab, dtype=float)

def feats(kind, t, x, fr):
    if kind == 'E':
        return np.stack([np.ones_like(t), t, -(x - 25)], 1)
    if kind == 'EQ':
        return np.stack([np.ones_like(t), t, -(x - 25), (x - 25) ** 2], 1)
    if kind == 'G':
        d = fr[1] + fr[0] * t - x
        return np.stack([np.ones_like(t), t, d, d ** 2], 1)
    raise ValueError(kind)

def fit(kind, tab, fr):
    t, x, n = tab[:, 0], tab[:, 1], tab[:, 2]
    X = feats(kind, t, x, fr)
    def nll(p):
        eta = (X @ p) * LN10 + math.log(BW)
        return np.sum(np.exp(eta) - n * eta)
    def grad(p):
        eta = (X @ p) * LN10 + math.log(BW)
        return X.T @ (np.exp(eta) - n) * LN10
    p0 = np.zeros(X.shape[1]); p0[0] = 0.5
    res = minimize(nll, p0, jac=grad, method='BFGS', options=dict(maxiter=2000, gtol=1e-8))
    mu = np.exp((X @ res.x) * LN10 + math.log(BW))
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = 2 * np.sum(np.where(n > 0, n * np.log(n / mu), 0.0) - (n - mu))
    return dict(kind=kind, p=res.x, nll=res.fun, dev=dev, k=X.shape[1], aic=2 * res.fun + 2 * X.shape[1],
                fr=fr)

XGRID = np.arange(18.0, 32.0, 0.01)
DSUPPORT = 3.25   # deepest distance below the frontier observed in the fit sample (2025: ~26.2 - 23)
def count_above(fitd, r, t_abs, plateau=False):
    """Expected number of releases in the year centred at t_abs with log10 compute >= r.
    plateau=True (model G only) holds the release density constant at its value at
    d_cap = min(argmax_d density, DSUPPORT) for all d > d_cap, i.e. it forbids the fitted Gaussian
    from extrapolating a *decline* in model counts below the observed support (a lower-bound
    assumption: in practice small models are more numerous than large ones)."""
    t = t_abs - T0
    xs = XGRID[XGRID >= r]
    if len(xs) == 0:
        return 0.0
    if plateau:
        c, lam, b1, b2 = fitd['p']; g, mu0 = fitd['fr']
        dpk = b1 / (-2 * b2) if b2 < 0 else DSUPPORT
        dcap = min(dpk, DSUPPORT)
        d = np.minimum(mu0 + g * t - xs, dcap)
        return float(np.sum(10 ** (c + lam * t + b1 * d + b2 * d ** 2)) * 0.01)
    X = feats(fitd['kind'], np.full_like(xs, t), xs, fitd['fr'])
    return float(np.sum(10 ** (X @ fitd['p'])) * 0.01)

def cnt(F, kind, r, t):
    """kind in {'E','EQ','G','Gp'}; 'Gp' = model G with the plateau extension."""
    if kind == 'Gp':
        return count_above(F['G'], r, t, plateau=True)
    return count_above(F[kind], r, t)

def fit_all(ms, yrs=FIT_YEARS):
    g, mu0 = frontier_slope(ms, 5)
    tab = bin_counts(ms, yrs)
    return {k: fit(k, tab, (g, mu0)) for k in ('E', 'EQ', 'G')}, tab

FITS, tab = fit_all(M)
def summarize(F, tab_=None):
    tab_ = tab if tab_ is None else tab_
    e, eq, gg = F['E'], F['EQ'], F['G']
    return dict(
        E=dict(c=e['p'][0], gamma=e['p'][1], beta=e['p'][2], dev=e['dev'], aic=e['aic']),
        EQ=dict(c=eq['p'][0], gamma=eq['p'][1], beta=eq['p'][2], q=eq['p'][3], dev=eq['dev'], aic=eq['aic'],
                lr_vs_E=2 * (e['nll'] - eq['nll'])),
        G=dict(c=gg['p'][0], lam=gg['p'][1], b1=gg['p'][2], b2=gg['p'][3], dev=gg['dev'], aic=gg['aic'],
               g=gg['fr'][0], mu0=gg['fr'][1]),
        n_bins=len(tab_), n_models_fit=int(tab_[:, 2].sum()))
out['fits_main'] = summarize(FITS)

# ---------------------------------------------------------------- 4. bootstrap
def boot_samples(ms, nb=NBOOT, yrs=FIT_YEARS):
    sel = [m for m in ms if m['x'] is not None and yrs[0] <= m['year'] <= yrs[1]]
    byy = collections.defaultdict(list)
    for m in sel:
        byy[m['year']].append(m)
    for _ in range(nb):
        smp = []
        for y, lst in byy.items():
            ny = RNG.poisson(len(lst))           # Poisson noise on annual totals
            idx = RNG.integers(0, len(lst), ny)
            smp += [lst[i] for i in idx]
        yield smp

BOOT = []
for smp in boot_samples(M):
    F, _ = fit_all(smp)
    BOOT.append(F)
def ci(v):
    v = np.asarray(v, float)
    return [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))]
bs = dict(gamma=ci([F['E']['p'][1] for F in BOOT]), beta=ci([F['E']['p'][2] for F in BOOT]),
          q=ci([F['EQ']['p'][3] for F in BOOT]), g=ci([F['G']['fr'][0] for F in BOOT]),
          lam=ci([F['G']['p'][1] for F in BOOT]), b1=ci([F['G']['p'][2] for F in BOOT]),
          b2=ci([F['G']['p'][3] for F in BOOT]),
          frac_b2_negative=float(np.mean([F['G']['p'][3] < 0 for F in BOOT])),
          frac_q_negative=float(np.mean([F['EQ']['p'][3] < 0 for F in BOOT])))
out['bootstrap_ci90'] = bs

# ---------------------------------------------------------------- 5. robustness variants
def impute(ms, mode, rng):
    res = []
    for m in ms:
        if m['x'] is not None or not (FIT_YEARS[0] <= m['year'] <= FIT_YEARS[1]):
            res.append(m); continue
        obs = [k['x'] for k in ms if k['year'] == m['year'] and k['x'] is not None]
        mm = dict(m)
        mm['x'] = obs[rng.integers(0, len(obs))] if mode == 'mar' else rng.uniform(np.percentile(obs, 75), max(obs))
        res.append(mm)
    return res
variants = {
    'confident_or_likely_only': summarize(*fit_all([m for m in M if m['conf'] in ('Confident', 'Likely')])),
    'language_domain_only': summarize(*fit_all([m for m in M if m['domain'].startswith('Language')])),
    'years_2023_2025': summarize(*fit_all(M, (2023, 2025))),
    'years_2022_2026partial': summarize(*fit_all(M, (2022, 2026))),
}
for mode in ('mar', 'top_heavy'):
    vs = [summarize(fit_all(impute(M, mode, RNG))[0]) for _ in range(30)]
    variants[f'impute_{mode}'] = dict(
        E=dict(gamma=float(np.mean([v['E']['gamma'] for v in vs])), beta=float(np.mean([v['E']['beta'] for v in vs]))),
        G=dict(lam=float(np.mean([v['G']['lam'] for v in vs])), b2=float(np.mean([v['G']['b2'] for v in vs])),
               g=float(np.mean([v['G']['g'] for v in vs]))))
out['variants'] = variants

# ---------------------------------------------------------------- 6. algorithmic efficiency a (cited input)
# Ho et al. (2024): compute to reach fixed LM performance halves every ~8 months (95% CI ~5-14 months).
def a_from_halving(months):
    return math.log10(2) * 12.0 / months
A_CENTRAL, A_LO, A_HI = a_from_halving(8), a_from_halving(14), a_from_halving(5)
mu_ln = 0.5 * (math.log(5) + math.log(14)); sd_ln = (math.log(14) - math.log(5)) / (2 * 1.96)
a_draw = math.log10(2) * 12.0 / np.exp(RNG.normal(mu_ln, sd_ln, len(BOOT)))

# ---------------------------------------------------------------- 7. adaptation-rate condition
# Hazard anchor: GPT-4-level effective compute: kappa0 = log10(2.1e25) at 2023-03-15 (Epoch estimate).
# Risk-relevant raw-compute level: r_h(t) = kappa0 + (h - a)(t - TK).
KAPPA0 = math.log10(2.1e25); TK = frac_year('2023-03-15')
def r_level(h, a, t):
    return KAPPA0 + (h - a) * (t - TK)

g_pt = FITS['G']['fr'][0]
asym = dict(
    light_tail_hstar_point=A_CENTRAL + g_pt,
    light_tail_hstar_ci90=ci([a_draw[i] + BOOT[i]['G']['fr'][0] for i in range(len(BOOT))]),
    exp_tail_hstar_point=A_CENTRAL + FITS['E']['p'][1] / FITS['E']['p'][2],
    exp_tail_hstar_ci90=ci([a_draw[i] + BOOT[i]['E']['p'][1] / BOOT[i]['E']['p'][2] for i in range(len(BOOT))]),
)
# finite-horizon h*: smallest drift h such that the zero-miss rule's burden in 2032 does not exceed 2025's
HGRID = np.round(np.arange(-0.5, 2.51, 0.02), 3)
def finite_hstar(F, a, kind, y0=2025.5, y1=2032.5, normalize=False):
    """Smallest h with burden(2032) <= burden(2025). normalize=True (G-family only) divides out the
    frontier-relative population growth 10^(lam*t), isolating the effect of threshold drift."""
    adj = 10 ** (F['G']['p'][1] * (y1 - y0)) if normalize else 1.0
    for h in HGRID:
        b0 = cnt(F, kind, r_level(h, a, y0), y0)
        b1 = cnt(F, kind, r_level(h, a, y1), y1)
        if b1 <= b0 * adj:
            return float(h)
    return float('nan')
fh = {}
for kind in ('E', 'EQ', 'G', 'Gp'):
    pt = finite_hstar(FITS, A_CENTRAL, kind)
    draws = [finite_hstar(BOOT[i], a_draw[i], kind) for i in range(0, len(BOOT), 2)]
    fh[kind] = dict(point=pt, ci90=ci([d for d in draws if not math.isnan(d)]),
                    effective_frontier_point=A_CENTRAL + g_pt)
pt = finite_hstar(FITS, A_CENTRAL, 'Gp', normalize=True)
draws = [finite_hstar(BOOT[i], a_draw[i], 'Gp', normalize=True) for i in range(0, len(BOOT), 2)]
fh['Gp_normalized'] = dict(point=pt, ci90=ci([d for d in draws if not math.isnan(d)]))
fh['share_boot_lambda_negative'] = float(np.mean([F['G']['p'][1] < 0 for F in BOOT]))
asym['finite_horizon_2025_2032'] = fh
out['adaptation_condition'] = asym

# ---------------------------------------------------------------- 8. backtest on observed models
def backtest(a, h, rule):
    rows_ = []
    for y in range(2021, 2027):
        ys = [m for m in M if m['year'] == y and m['x'] is not None]
        risky = [m for m in ys if m['x'] >= r_level(h, a, m['t'])]
        cov = [m for m in ys if m['x'] >= rule(m['t'])]
        miss = [m for m in risky if m['x'] < rule(m['t'])]
        rows_.append(dict(year=y, covered=len(cov), covered_orgs=len(set(m['org'] for m in cov)),
                          risky=len(risky), missed=len(miss), over=len([m for m in cov if m not in risky]),
                          missed_models=[m['model'] for m in miss][:12]))
    return rows_
rules = {
    'static_1e25': lambda t: 25.0,
    'static_1e26': lambda t: 26.0,
    'frontier_minus_1OOM': lambda t: mu0_top5 + g_top5 * (t - T0) - 1.0,
    'eff_indexed_central': lambda t: r_level(0.0, A_CENTRAL, t),
}
bt = {}
for aname, a in (('a_lo', A_LO), ('a_central', A_CENTRAL), ('a_hi', A_HI)):
    for h in (0.0, 0.25):
        for rn, rf in rules.items():
            bt[f'{aname}|h={h}|{rn}'] = backtest(a, h, rf)
out['backtest'] = bt

# composition of covered vs missed sets, 2024-2025, a central, h=0
def comp(ms):
    n = len(ms)
    if n == 0:
        return dict(n=0)
    us = sum('United States' in m['country'] for m in ms); cn = sum('China' in m['country'] for m in ms)
    ow = sum(m['access'].startswith('Open weights') for m in ms)
    return dict(n=n, share_US=us / n, share_CN=cn / n, share_other=1 - (us + cn) / n, share_open_weights=ow / n)
sel = [m for m in M if m['x'] is not None and 2024 <= m['year'] <= 2025]
risky = [m for m in sel if m['x'] >= r_level(0.0, A_CENTRAL, m['t'])]
out['composition_2024_2025_central'] = dict(
    covered_1e25=comp([m for m in sel if m['x'] >= 25]),
    risky_missed_by_1e25=comp([m for m in risky if m['x'] < 25]),
    risky_missed_by_1e26=comp([m for m in risky if m['x'] < 26]),
    all_risky=comp(risky))
with open('results/backtest_central.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['a', 'h', 'rule', 'year', 'covered', 'covered_orgs', 'risky', 'missed', 'over_included'])
    for k, v in bt.items():
        an, hh, rn = k.split('|')
        for r in v:
            w.writerow([an, hh[2:], rn, r['year'], r['covered'], r['covered_orgs'], r['risky'], r['missed'], r['over']])

# ---------------------------------------------------------------- 9. forward projection with uncertainty
PY = np.arange(2025, 2033)
def proj_rule(F, kind, a, h, rule, g, mu0):
    res = []
    for y in PY:
        t = y + 0.5
        r = r_level(h, a, t)
        tau = rule(t, a, h, g, mu0)
        cov = cnt(F, kind, tau, t)
        risky = cnt(F, kind, r, t)
        miss = max(risky - cnt(F, kind, max(tau, r), t), 0.0)
        res.append((cov, risky, miss, mu0 + g * (t - T0) - r))
    return np.array(res)
prules = {
    'static_1e25': lambda t, a, h, g, mu0: 25.0,
    'static_1e26': lambda t, a, h, g, mu0: 26.0,
    'frontier_minus_1OOM': lambda t, a, h, g, mu0: mu0 + g * (t - T0) - 1.0,
    'zero_miss_indexed': lambda t, a, h, g, mu0: r_level(h, a, t),
}
proj = {}
for kind in ('Gp', 'E'):
    for h in (0.0, 0.5, 1.0):
        for rn, rf in prules.items():
            D = []
            for i in range(0, len(BOOT), 2):
                F = BOOT[i]
                D.append(proj_rule(F, kind, a_draw[i], h, rf, F['G']['fr'][0], F['G']['fr'][1]))
            D = np.array(D)
            P = proj_rule(FITS, kind, A_CENTRAL, h, rf, g_top5, mu0_top5)
            proj[f'{kind}|h={h}|{rn}'] = dict(
                years=PY.tolist(), covered_point=P[:, 0].tolist(), missed_point=P[:, 2].tolist(),
                risky_point=P[:, 1].tolist(), depth_below_frontier_of_risk_level=P[:, 3].tolist(),
                covered_ci90=[np.percentile(D[:, :, 0], 5, 0).tolist(), np.percentile(D[:, :, 0], 95, 0).tolist()],
                missed_ci90=[np.percentile(D[:, :, 2], 5, 0).tolist(), np.percentile(D[:, :, 2], 95, 0).tolist()])
out['projection'] = proj
# cumulative releases above 1e26 FLOP, 2025-2030 inclusive (model-based; cf. Cottier & Owen 2025)
cum = {}
for kind in ('Gp', 'E', 'EQ'):
    cum[kind] = sum(cnt(FITS, kind, 26.0, y + 0.5) for y in range(2025, 2031))
out['cumulative_above_1e26_2025_2030'] = cum
with open('results/projection_table.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tail_model', 'h', 'rule', 'year', 'covered', 'covered_p05', 'covered_p95', 'missed', 'missed_p05', 'missed_p95', 'risky'])
    for k, P in proj.items():
        kind, hh, rn = k.split('|')
        for j, y in enumerate(P['years']):
            w.writerow([kind, hh[2:], rn, y, round(P['covered_point'][j], 2), round(P['covered_ci90'][0][j], 2),
                        round(P['covered_ci90'][1][j], 2), round(P['missed_point'][j], 2), round(P['missed_ci90'][0][j], 2),
                        round(P['missed_ci90'][1][j], 2), round(P['risky_point'][j], 2)])

# ---------------------------------------------------------------- 10. 2030 burden-miss frontier
par = {}
for h in (0.0, 0.5, 1.0):
    t = 2030.5; r = r_level(h, A_CENTRAL, t)
    taus = np.linspace(22, 29.5, 151)
    cov = [cnt(FITS, 'Gp', tau, t) for tau in taus]
    risky = cnt(FITS, 'Gp', r, t)
    miss = [max(risky - cnt(FITS, 'Gp', max(tau, r), t), 0) for tau in taus]
    par[f'h={h}'] = dict(tau=taus.tolist(), covered=cov, missed=miss, risky=risky)
out['pareto_2030_Gp'] = par

json.dump(out, open('results/results.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C1, C2, C3, C4 = '#2b6cb0', '#c05621', '#2f855a', '#6b46c1'
fig, ax = plt.subplots(figsize=(6.4, 3.6))
pts = np.array([(m['t'], m['x']) for m in M if m['x'] is not None])
ax.scatter(pts[:, 0], pts[:, 1], s=7, color='#718096', alpha=0.6, label='Epoch large-scale models (compute known)')
tt = np.linspace(2021, 2027, 50)
for a_, lab, ls in ((A_LO, '14-mo halving', ':'), (A_CENTRAL, '8-mo halving', '-'), (A_HI, '5-mo halving', '--')):
    ax.plot(tt, r_level(0, a_, tt), color=C2, ls=ls, lw=1.2, label=f'GPT-4-equivalent raw compute ({lab})')
ax.plot(tt, mu0_top5 + g_top5 * (tt - T0), color=C3, lw=1, label='top-5 frontier trend')
ax.axhline(25, color=C1, lw=1); ax.axhline(26, color=C1, lw=1, ls='--')
ax.text(2021.05, 25.05, 'EU AI Act 1e25', color=C1, fontsize=7); ax.text(2021.05, 26.05, 'SB 53 1e26', color=C1, fontsize=7)
ax.set_xlabel('publication date'); ax.set_ylabel('log10 training compute (FLOP)')
ax.set_xlim(2021, 2027); ax.set_ylim(22.9, 27.4); ax.legend(fontsize=6, loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig('results/fig1_models_thresholds.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
cols = {2022: '#a0aec0', 2023: C1, 2024: C3, 2025: C2}
for y, col in cols.items():
    xs = np.sort([m['x'] for m in M if m['year'] == y and m['x'] is not None])[::-1]
    ax.step(xs, np.arange(1, len(xs) + 1), where='post', color=col, label=f'{y} observed')
    xg = np.linspace(23, xs.max() + 0.4, 60)
    ax.plot(xg, [count_above(FITS['G'], x, y + 0.5) for x in xg], color=col, ls='--', lw=1)
    ax.plot(xg, [count_above(FITS['E'], x, y + 0.5) for x in xg], color=col, ls=':', lw=0.8)
ax.set_yscale('log'); ax.set_ylim(0.3, 300); ax.set_xlabel('log10 training compute x'); ax.set_ylabel('# models in year with compute ≥ x')
ax.legend(fontsize=7, frameon=False); ax.set_title('observed (solid), model G (dashed), model E (dotted)', fontsize=8)
fig.tight_layout(); fig.savefig('results/fig2_tail_fit.png', dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.1), sharey=True)
for ax, h in zip(axs, (0.0, 0.5, 1.0)):
    for rn, col in (('static_1e25', C1), ('static_1e26', '#63b3ed'), ('frontier_minus_1OOM', C4), ('zero_miss_indexed', C2)):
        P = proj[f'Gp|h={h}|{rn}']
        ax.plot(P['years'], np.maximum(P['covered_point'], 1e-2), color=col, label=f'{rn} covered')
        if rn != 'zero_miss_indexed':
            ax.plot(P['years'], np.maximum(P['missed_point'], 1e-2), color=col, ls=':', label=f'{rn} missed')
    ax.set_yscale('log'); ax.set_title(f'h = {h} OOM/yr', fontsize=8); ax.set_xlabel('year'); ax.set_ylim(1e-2, 3e4)
axs[0].set_ylabel('expected models per year (model G)'); axs[2].legend(fontsize=5, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig('results/fig3_projection.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.3))
for h, col in ((0.0, C2), (0.5, C4), (1.0, C3)):
    P = par[f'h={h}']
    ax.plot(np.array(P['covered']), np.array(P['missed']) + 1e-3, color=col, label=f'h={h} (risk-relevant: {P["risky"]:.0f})')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('models covered in 2030 (burden)'); ax.set_ylabel('risk-relevant models missed in 2030')
ax.legend(fontsize=7, frameon=False); ax.set_xlim(0.1, 3e4); ax.set_ylim(1e-2, 3e4)
fig.tight_layout(); fig.savefig('results/fig4_frontier2030.png', dpi=200); plt.close(fig)

print(json.dumps({k: out[k] for k in ('n_records', 'n_with_compute', 'frontier_growth', 'fits_main', 'bootstrap_ci90', 'adaptation_condition')}, indent=1, default=float))
