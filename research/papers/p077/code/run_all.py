"""Reproduce every number and figure in paper.md.  Run:  python3 run_all.py  (from code/)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
import params as P
import model as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)
R = {}
RS = M.rho_star()
R['rho_star'] = RS
R['sigma_emp_rel'] = P.SIGMA_EMP_REL
R['sigma_emp_abs'] = P.SIGMA_EMP_ABS
R['smuggled_frac_median'] = P.SMUG_MED / P.S_REF

# ---------------------------------------------------------------- 1. baseline point values
pts = {}
for t in [2023.0, 2026.0, 2028.0, 2030.0]:
    row = {}
    for rule in ['capability', 'flopcap', 'pause', 'frontier']:
        s = float(M.sqc(t, rule))
        row[rule] = dict(SQC=s,
                         sigma_req=float(M.sigma_required(t, rule)),
                         rho_at_emp=float(M.rho(t, rule, P.SIGMA_EMP_REL)),
                         potential_sites=float(M.potential_sites(t, rule)))
    row['stock'] = float(M.stock(t))
    pts[str(t)] = row
R['points'] = pts
R['decay_rates'] = {r: float(M.decay_rate(r)) for r in ['capability', 'flopcap', 'frontier']}
R['inspection_nmax_IAEA'] = float(M.inspection_nmax())
R['inspection_nmax_IAEA_timeliness1mo'] = float(M.inspection_nmax(tau=1/12))
R['iaea_safeguards_budget_eur'] = P.IAEA_RB*P.IAEA_SG_SHARE
R['sqc_chipflop_base'] = M.chip_flop()

# horizons (baseline)
hz = {}
for rule in ['capability', 'flopcap', 'pause']:
    hz[rule] = {f'{s:g}': float(M.horizon(rule, s)) for s in [0.02, 0.005, 0.001, 1e-4, 1e-5]}
# frontier: date at which rho crosses rho* upward
hz['frontier'] = {f'{s:g}': float(M.horizon('frontier', s)) for s in [0.02, 0.005, 0.001]}
R['horizons_baseline'] = hz

# ---------------------------------------------------------------- Figure 1: SQC vs detectable floor
t = np.linspace(2023, 2032, 400)
fig, ax = plt.subplots(figsize=(7.2, 6.0))
cols = {'capability': '#c0392b', 'flopcap': '#8e44ad', 'pause': '#d68910', 'frontier': '#1f618d'}
labs = {'capability': 'Fixed capability (GPT-4 level, eroded by alg. progress)',
        'flopcap': 'Fixed physical cap $10^{25}$ FLOP',
        'pause': 'Pause at 2026 frontier (eroded after 2026)',
        'frontier': 'Frontier-trend cap (grows 4x/yr)'}
for rule in ['capability', 'flopcap', 'pause', 'frontier']:
    ax.semilogy(t, M.sqc(t, rule), color=cols[rule], lw=2, label=labs[rule])
for s, ls in [(P.SIGMA_EMP_REL, '-'), (0.001, '--'), (1e-5, ':')]:
    ax.semilogy(t, RS*s*M.stock(t), color='k', ls=ls, lw=1,
                label=f'Detectable diversion floor, $\\sigma$={s:.2g}')
ax.axhline(8, color='grey', lw=0.8)
ax.text(2023.1, 10, 'one 8-GPU server', fontsize=8, color='grey')
ax.set_xlabel('Year'); ax.set_ylabel('H100-equivalents')
ax.set_title('Significant Quantity of Compute vs. what stock accounting can detect\n(90-day window, MFU 0.4, $P_D$=0.9 at $\\alpha$=0.05)', fontsize=10)
ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2); ax.set_ylim(1, 1e9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_sqc_vs_floor.png'), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 2. Monte Carlo over trend parameters
N = 200_000
h = np.exp(rng.normal(np.log(P.H_ALG_MED), np.log(P.H_ALG_HI/P.H_ALG_LO)/(2*1.645), N))
gS = rng.uniform(P.G_S_LO, P.G_S_HI, N)
gF = rng.uniform(P.G_F_LO, P.G_F_HI, N)
u = rng.uniform(P.U_LO, P.U_HI, N)
tau = rng.choice(np.array([30, 90, 180, 365])/365, N)
sig = 10**rng.uniform(-5, np.log10(0.05), N)   # log-uniform from 0.001% to 5%
lam_cap = np.log(2)/h + np.log(gS)
lam_fr = np.log(gS) - np.log(gF)
def sqc_vec(rule, tt):
    F = M.threshold_flop(tt, rule, h=h, gF=gF) if rule != 'flopcap' else 1e25*np.ones(N)
    return F/(P.R_H100*u*tau*M.SEC_PER_YR)
r1_cap = sqc_vec('capability', 2026.0)/(sig*P.S_REF)
H_cap = 2026.0 + np.log(r1_cap/RS)/lam_cap
r1_fr = (P.F0*gF**(2026.0-P.T0)/(P.R_H100*u*tau*M.SEC_PER_YR))/(sig*P.S_REF)
mc = {}
mc['lambda_capability_q'] = np.percentile(lam_cap, [5, 50, 95]).tolist()
mc['lambda_frontier_q'] = np.percentile(lam_fr, [5, 50, 95]).tolist()
mc['P_lambda_frontier_negative'] = float(np.mean(lam_fr < 0))
mc['factor_per_year_capability_q'] = np.exp(np.percentile(lam_cap, [5, 50, 95])).tolist()
for s_target in [0.02, 0.001, 1e-4]:
    r1 = sqc_vec('capability', 2026.0)/(s_target*P.S_REF)
    Hc = 2026.0 + np.log(r1/RS)/lam_cap
    mc[f'horizon_capability_sigma{s_target:g}_q'] = np.percentile(Hc, [5, 50, 95]).tolist()
# server-crossing date: SQC(t) = 8 chips, fixed capability
Hserver = P.T0 + h*np.log2((P.F0/(P.R_H100*u*tau*M.SEC_PER_YR))/8)
mc['server_crossing_capability_q'] = np.percentile(Hserver, [5, 50, 95]).tolist()
Hserver_base = P.T0 + P.H_ALG_MED*np.log2((P.F0/M.chip_flop())/8)
R['server_crossing_capability_base'] = float(Hserver_base)

# PRCC sensitivity of horizon (capability) to inputs
def prcc(X, y):
    Xr = np.column_stack([rankdata(c) for c in X.T]); yr = rankdata(y)
    out = []
    for j in range(Xr.shape[1]):
        others = np.delete(Xr, j, axis=1); A = np.column_stack([np.ones(len(yr)), others])
        rx = Xr[:, j] - A @ np.linalg.lstsq(A, Xr[:, j], rcond=None)[0]
        ry = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
        out.append(float(np.corrcoef(rx, ry)[0, 1]))
    return out
# forward-looking target: horizon of the 2026-pause rule, sigma log-uniform in [1e-6, 7e-3]
sig_p = 10**rng.uniform(-6, np.log10(7e-3), N)
Fp = P.F0*gF**(2026.0-P.T0)
r1_p = (Fp/(P.R_H100*u*tau*M.SEC_PER_YR))/(sig_p*P.S_REF)
H_p = 2026.0 + np.log(r1_p/RS)/lam_cap
idx = rng.choice(N, 40_000, replace=False)
X = np.column_stack([h, gS, gF, u, tau, np.log10(sig_p)])[idx]
mc['prcc_horizon_pause'] = dict(zip(['h_alg', 'g_S', 'g_F', 'MFU', 'tau', 'log10_sigma'], prcc(X, H_p[idx])))
for s_target in [1e-3, 1e-4]:
    r1 = (Fp/(P.R_H100*u*tau*M.SEC_PER_YR))/(s_target*P.S_REF)
    mc[f'horizon_pause_sigma{s_target:g}_q'] = np.percentile(2026.0 + np.log(r1/RS)/lam_cap, [5, 50, 95]).tolist()
r1 = (P.F0*gF**(2026.0-P.T0)/(P.R_H100*u*tau*M.SEC_PER_YR))/(0.02*P.S_REF)
mc['frontier_rho2026_sigma0.02_q'] = np.percentile(r1, [5, 50, 95]).tolist()
R['monte_carlo'] = mc

fig, ax = plt.subplots(figsize=(7.2, 4.4))
sg = np.logspace(-6, np.log10(0.05), 60)
for rule, c in [('capability', cols['capability']), ('pause', cols['pause']), ('flopcap', cols['flopcap'])]:
    ys = np.array([M.horizon(rule, s) for s in sg])
    ax.semilogx(sg, ys, color=c, lw=2, label=labs[rule])
# MC band for capability
qs = []
for s in sg:
    r1 = sqc_vec('capability', 2026.0)/(s*P.S_REF)
    qs.append(np.percentile(2026.0 + np.log(r1/RS)/lam_cap, [5, 95]))
qs = np.array(qs)
ax.fill_between(sg, qs[:, 0], qs[:, 1], color=cols['capability'], alpha=0.15, label='Fixed capability, 90% MC band')
ax.axvline(P.SIGMA_EMP_REL, color='k', ls='--', lw=1); ax.text(P.SIGMA_EMP_REL*1.1, 2014, 'empirical\n$\\sigma\\approx$2%', fontsize=8)
ax.axhline(2026, color='grey', lw=0.8)
ax.set_xlabel('Relative inventory uncertainty $\\sigma$ (1-sigma, fraction of stock)')
ax.set_ylabel('Accountancy horizon $t^*$ (year)')
ax.set_title('Last year in which a diversion of one SQC meets the IAEA-style detection goal', fontsize=10)
ax.set_ylim(2012, 2040); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_horizon_vs_sigma.png'), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 3. Log-time: value of institutional improvement
Ms = np.logspace(0, 3, 50)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ext = {}
for rule, lam in [('capability', M.decay_rate('capability')), ('flopcap', M.decay_rate('flopcap'))]:
    ax.semilogx(Ms, np.log(Ms)/lam*12, lw=2, color=cols[rule], label=f'{labs[rule]} ($\\lambda$={lam:.2f}/yr)')
    ext[rule] = {f'M={m}': float(np.log(m)/lam*12) for m in [3.16, 10, 100, 1000]}
ax.set_xlabel('Multiplicative improvement M in accountancy SNR\n(budget, pooling $\\sqrt{m}$, measurement precision)')
ax.set_ylabel('Horizon extension (months)')
ax.set_title('Institutions buy log-time against eroding thresholds', fontsize=10)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_logtime.png'), dpi=160); plt.close(fig)
R['horizon_extension_months'] = ext
R['pooling_m10_extension_months_capability'] = float(np.log(np.sqrt(10))/M.decay_rate('capability')*12)

# ---------------------------------------------------------------- 4. Monte Carlo validation of accountancy test + protracted diversion
def mc_detect(D, sigma_abs, k_periods, n=200_000, alpha=P.ALPHA, test='cumuf'):
    """Diverter removes D chips spread equally over k balance periods; each period's MUF has
    independent N(0, sigma_abs^2) error. Tests: per-period (Bonferroni-free, any alarm) or CUMUF."""
    e = rng.normal(0, sigma_abs, (n, k_periods))
    muf = e + D/k_periods
    if test == 'cumuf':
        stat = muf.sum(1)/(sigma_abs*np.sqrt(k_periods))
        return float(np.mean(stat > norm.ppf(1-alpha)))
    # per-period, with overall false alarm alpha (Sidak)
    a1 = 1-(1-alpha)**(1/k_periods)
    return float(np.mean((muf/sigma_abs > norm.ppf(1-a1)).any(1)))
val = {}
for rr in [1.0, 2.0, RS, 4.0]:
    val[f'rho={rr:.3f}'] = dict(analytic=float(norm.cdf(rr - norm.ppf(1-P.ALPHA))),
                                mc_k1=mc_detect(rr, 1.0, 1))
prot = {}
for k in [1, 3, 12]:
    prot[f'k={k}'] = dict(cumuf=mc_detect(RS, 1.0, k, test='cumuf'),
                          per_period=mc_detect(RS, 1.0, k, test='period'),
                          analytic_cumuf=float(norm.cdf(RS/np.sqrt(k) - norm.ppf(1-P.ALPHA))))
R['mc_validation'] = val; R['protracted'] = prot

# ---------------------------------------------------------------- 5. deterrence stakes
R['rho_star_by_G_over_C'] = {f'{k:g}': float(M.rho_star_deterrence(k)) for k in [1, 9, 99, 999, 1e4]}

# ---------------------------------------------------------------- 6. jurisdictional decomposition
jur = {}
for share in [0.05, 0.25, 0.5, 0.75]:
    jur[f'share={share}'] = {rule: float(M.sqc(2026.0, rule)/(P.SIGMA_EMP_REL*share*P.S_REF))
                             for rule in ['capability', 'pause', 'frontier']}
R['jurisdiction_rho_2026'] = jur

# ---------------------------------------------------------------- 7. inspection capacity (facility regime) Figure 4
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for rule in ['capability', 'pause', 'frontier']:
    ax.semilogy(t, M.potential_sites(t, rule), color=cols[rule], lw=2, label=labs[rule])
nmax = M.inspection_nmax()
ax.axhline(nmax, color='k', ls='--', lw=1)
ax.text(2023.1, nmax*1.3, f'IAEA-scale inspection capacity ({nmax:.0f} sites at $P_D$=0.9)', fontsize=8)
ax.set_xlabel('Year'); ax.set_ylabel('Potential simultaneous violating runs  S/SQC')
ax.set_title('Facility-inspection regime: candidate sites vs. inspection capacity', fontsize=10)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_inspection.png'), dpi=160); plt.close(fig)
R['budget_multiple_needed_capability'] = {str(y): float(M.potential_sites(y, 'capability')/nmax) for y in [2026.0, 2028.0, 2030.0]}
R['budget_multiple_needed_pause'] = {str(y): float(M.potential_sites(y, 'pause')/nmax) for y in [2026.0, 2028.0, 2030.0]}

# ---------------------------------------------------------------- 8. hardware-enabled mechanisms Figure 5
fig, ax = plt.subplots(figsize=(6.8, 4.2))
tt = np.linspace(2026, 2036, 1001)
hem = {}
cases = [('pause', 0.02, 0.0, '-'), ('pause', 0.002, 0.0, '--'), ('pause', 0.002, 1e-3, ':'),
         ('frontier', 0.02, 0.0, '-'), ('frontier', 0.02, 1e-2, ':')]
for rule, sL, eps, ls in cases:
    r = M.sqc(tt, rule)/M.uncontrolled_noise(tt, sL, eps)
    ax.semilogy(tt, r, color=cols[rule], ls=ls, lw=1.6,
                label=f'{rule}: legacy $\\sigma_L$={sL:g}, tamper $\\epsilon$={eps:g}')
    ok = r >= RS
    hem[f'{rule}_sL{sL:g}_eps{eps:g}'] = dict(first_ok=float(tt[ok][0]) if ok.any() else None,
                                     last_ok=float(tt[ok][-1]) if ok.any() else None,
                                     rho_2026=float(r[0]), rho_2030=float(r[400]), rho_2036=float(r[-1]))
# no-HEM comparison for pause with sigma 0.002
r_nohem = M.sqc(tt, 'pause')/(0.002*M.stock(tt))
hem['pause_noHEM_sigma0.002_last_ok'] = float(tt[r_nohem >= RS][-1]) if (r_nohem >= RS).any() else None
hem['lambda_pause_HEM_eps0'] = float(np.log(2)/P.H_ALG_MED - 1/5.0)
ax.axhline(RS, color='k', lw=1); ax.text(2026.1, RS*1.2, '$\\rho^*$', fontsize=9)
ax.set_xlabel('Year'); ax.set_ylabel('Accountancy ratio $\\rho$')
ax.set_title('HEMs on all chips shipped from 2026 (best case); legacy stock retires (mean life 5 yr)', fontsize=9)
ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_hem.png'), dpi=160); plt.close(fig)
R['hem'] = hem
R['frontier_share_of_stock'] = {str(y): float(M.sqc(y, 'frontier')/M.stock(y)) for y in [2023.0, 2026.0, 2028.0, 2030.0]}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(R, f, indent=2)
print(json.dumps(R, indent=1))
