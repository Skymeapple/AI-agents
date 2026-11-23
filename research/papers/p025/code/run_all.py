"""Run every experiment in the paper. Usage (from p025/):  python code/run_all.py
Outputs go to results/. Total CPU time is a few minutes. Seeds are fixed.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from klm import klm_time
from model import (agent_time_closed, parity_horizon, required_l0, simulate_agent,
                   budgeted_success, human_budgeted_success)

RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)
T0 = time.time()
OUT = {}

# ----------------------------------------------------------------------------------
# Sourced constants (see paper, Section 4 and Table 1)
P_HUMAN = 0.7236            # OSWorld human success (Xie et al., 2024)
HUMAN_MEDIAN_OSWORLD = 120  # ~2 min median human time on OSWorld 1.0 (Yuan et al., 2026)
SIGMA_H = 0.5               # ASSUMPTION: lognormal spread of human times around task median
COLORS = ['#2a6f97', '#c44536', '#6a994e', '#bc6c25', '#7b2cbf', '#555555']

tasks = json.load(open(os.path.join(HERE, '..', 'data', 'oswh_tasks.json')))

# ===================================== E1 ===========================================
def e1():
    rows = []
    k = [klm_time(t) for t in tasks]
    feas = [i for i, x in enumerate(k) if not x['infeasible']]
    tot = np.array([k[i]['total'] for i in feas])
    ns = np.array([len(tasks[i]['single']) for i in feas])
    ng = np.array([len(tasks[i]['grouped']) for i in feas])
    apps = sorted(set(tasks[i]['app'] for i in feas))
    for app in apps + ['ALL']:
        idx = [j for j, i in enumerate(feas) if app == 'ALL' or tasks[i]['app'] == app]
        rows.append(dict(app=app, n_tasks=len(idx), median_actions=float(np.median(ns[idx])),
                         mean_actions=float(np.mean(ns[idx])), median_groups=float(np.median(ng[idx])),
                         median_klm_s=float(np.median(tot[idx])), mean_klm_s=float(np.mean(tot[idx])),
                         median_s_per_action=float(np.median(tot[idx] / ns[idx]))))
    with open(os.path.join(RES, 'e1_klm_by_app.csv'), 'w') as f:
        f.write(','.join(rows[0].keys()) + '\n')
        for r in rows:
            f.write(','.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')
    # sensitivity of the median KLM time
    sens = {}
    for name, kw in [('baseline', {}), ('K=0.20', dict(K=0.20)), ('K=0.50', dict(K=0.50)),
                     ('M per action', dict(M_MODE='single')), ('SCROLL=2s', dict(SCROLL_S=2.0)),
                     ('WAIT=5s', dict(WAIT_S=5.0)), ('READ 300wpm', dict(READ_WPM=300.0))]:
        tt = np.array([klm_time(tasks[i], **kw)['total'] for i in feas])
        sens[name] = dict(median=float(np.median(tt)), mean=float(np.mean(tt)),
                          scale_to_2min=float(HUMAN_MEDIAN_OSWORLD / np.median(tt)))
    OUT['e1'] = dict(n_feasible=len(feas), n_infeasible=len(tasks) - len(feas), by_app=rows,
                     sensitivity=sens,
                     share_motor=float(np.sum([k[i]['motor'] for i in feas]) / tot.sum()),
                     share_mental=float(np.sum([k[i]['mental'] for i in feas]) / tot.sum()),
                     share_reading=float(np.sum([k[i]['reading'] for i in feas]) / tot.sum()))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(np.log10(tot), bins=30, color=COLORS[0])
    ax[0].axvline(np.log10(np.median(tot)), color='k', ls='--', lw=1, label=f'median KLM {np.median(tot):.0f}s')
    ax[0].axvline(np.log10(120), color=COLORS[1], ls=':', lw=1.5, label='reported human median ~120s')
    ax[0].set_xlabel('log10 expert KLM time (s)'); ax[0].set_ylabel('tasks'); ax[0].legend(fontsize=8)
    ax[0].set_title('(a) Expert time per OSWorld task (n=%d)' % len(feas), fontsize=10)
    ax[1].scatter(ns, tot, s=8, alpha=0.6, color=COLORS[0])
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('human reference actions n'); ax[1].set_ylabel('expert KLM time (s)')
    b = np.polyfit(np.log(ns), np.log(tot), 1)
    xs = np.logspace(0, np.log10(ns.max()), 50)
    ax[1].plot(xs, np.exp(b[1]) * xs ** b[0], 'k--', lw=1, label=f'slope {b[0]:.2f}')
    ax[1].legend(fontsize=8); ax[1].set_title('(b) Time grows ~linearly in actions', fontsize=10)
    OUT['e1']['loglog_slope'] = float(b[0])
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig1_klm.png'), dpi=150); plt.close()
    return feas, tot, ns, ng

feas, T_klm, N_act, N_grp = e1()
SCALE = HUMAN_MEDIAN_OSWORLD / np.median(T_klm)
T_typ = T_klm * SCALE        # "typical human": KLM shape rescaled to the reported 2-min median
H_EXP = float(np.median(T_klm / N_act))   # expert seconds per action (median)
H_TYP = H_EXP * SCALE

# ===================================== Agents =======================================
# Calibration (paper Table 1):
#  A25  : Agent-S2-like (2025). Mean 48.5 s/step over a 50-step task (>40 min, Abhyankar
#         et al.), later steps ~3x slower -> gamma = 0.04, l0 = 24 s, a = 1 s; step
#         inflation kappa = 2.0 (between 1.4x and 2.7x reported); overall success ~41%.
#  A26o : 2026 frontier CUA, slower/more accurate: CADWorld Opus-4.8 row 270.6 s / 33.5
#         steps = 8.1 s/step, kappa = 33.5/14.6 = 2.3. Hazard from the Opus-4.8 pair
#         (83.5% OSWorld-Verified at ~30 steps; 20.6% OSWorld 2.0 at ~318 steps).
#  A26g : 2026 frontier CUA, faster: CADWorld GPT-5.4 row 213.5 s / 44.5 steps = 4.8 s/step,
#         kappa = 44.5/14.6 = 3.0, same hazard (ASSUMPTION).
#  F    : HYPOTHETICAL fast agent (not a measured system): 2 s/step, kappa 1.5, but lower
#         task success (p0 0.80).
H_HAZ = np.log(0.835 / 0.206) / (318 - 30)
P0_26 = 0.835 * np.exp(H_HAZ * 30)

def calib_l0(step_mean, K, gamma, a):
    # mean step time over K steps = l0*(1+gamma*(K-1)/2) + a
    return (step_mean - a) / (1 + gamma * (K - 1) / 2)

AGENTS = {
    'A25 (S2-like, 2025)': dict(kappa=2.0, l0=24.0, gamma=0.04, a=1.0, p0=0.90, h=None, target=0.414, eps=0.05, rho=2.0, noise=np.exp(0.3)),
    'A26o (Opus-4.8-like)': dict(kappa=2.3, l0=None, step_mean=8.1, Kref=33.5, gamma=0.01, a=1.0, p0=P0_26, h=H_HAZ, eps=0.05, rho=2.0, noise=np.exp(0.3)),
    'A26g (GPT-5.4-like)': dict(kappa=3.0, l0=None, step_mean=4.8, Kref=44.5, gamma=0.01, a=1.0, p0=P0_26, h=H_HAZ, eps=0.05, rho=2.0, noise=np.exp(0.3)),
    'F (hypothetical fast)': dict(kappa=1.5, l0=1.0, gamma=0.0, a=1.0, p0=0.80, h=0.0, eps=0.05, rho=2.0, noise=np.exp(0.3)),
}
for nm, ag in AGENTS.items():
    if ag['l0'] is None:
        ag['l0'] = calib_l0(ag['step_mean'], ag['Kref'], ag['gamma'], ag['a'])
    if ag['h'] is None:  # calibrate hazard so mean success over OSWorld feasible tasks = target
        lo, hi = 0.0, 1.0
        K = np.ceil(ag['kappa'] * N_act)
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.mean(ag['p0'] * np.exp(-mid * K)) > ag['target']:
                lo = mid
            else:
                hi = mid
        ag['h'] = (lo + hi) / 2
OUT['agents'] = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                     for kk, vv in v.items()} for k, v in AGENTS.items()}
OUT['human'] = dict(scale_typ_over_klm=float(SCALE), h_expert_s_per_action=H_EXP,
                    h_typical_s_per_action=H_TYP, sigma_h=SIGMA_H, p_h=P_HUMAN,
                    hazard_2026=float(H_HAZ), p0_2026=float(P0_26))

# ===================================== E2 ===========================================
def e2():
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (lab, Th) in enumerate([('expert (KLM)', T_klm), ('typical (2-min median)', T_typ)]):
        d = {}
        for kappa in [1.0, 1.5, 2.3, 3.0]:
            req = Th / (kappa * N_act)              # required (l0 + a), gamma = 0
            d[f'kappa={kappa}'] = dict(median_req_step_s=float(np.median(req)),
                                       p10=float(np.percentile(req, 10)), p90=float(np.percentile(req, 90)))
            if kappa in (1.0, 2.3):
                xs = np.sort(req)
                ax[j].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=COLORS[0 if kappa == 1.0 else 1],
                           label=f'required step time, kappa={kappa}')
        for nm, ag in AGENTS.items():
            K = np.ceil(ag['kappa'] * N_act)
            TA = agent_time_closed(K, ag['l0'], ag['gamma'], ag['a'])
            d[nm] = dict(frac_tasks_time_parity=float(np.mean(TA <= Th)),
                         median_time_ratio=float(np.median(TA / Th)),
                         mean_step_s_on_osworld=float(np.mean(TA / K)))
        for k2, (nm, ag) in enumerate(AGENTS.items()):
            ax[j].axvline(d[nm]['mean_step_s_on_osworld'], color=COLORS[2 + k2 % 4], ls=':', lw=1.2, label=nm.split(' (')[0] + ' mean step')
        ax[j].set_xscale('log'); ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[j].set_xlabel('seconds per agent step'); ax[j].set_ylabel('fraction of tasks')
        ax[j].set_title(f'({"ab"[j]}) Step-time budget for parity, {lab} human', fontsize=9)
        ax[j].legend(fontsize=6.5)
        res[lab] = d
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig2_step_budget.png'), dpi=150); plt.close()
    OUT['e2'] = res

e2()

# ===================================== E3 ===========================================
def e3():
    rng = np.random.default_rng(0)
    taus = np.logspace(-1, 2, 61)
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (lab, Th) in enumerate([('typical', T_typ), ('expert', T_klm)]):
        Sh = human_budgeted_success(taus, P_HUMAN, SIGMA_H)
        axs[j].plot(taus, Sh, 'k-', lw=2, label='human (OSWorld, p=0.724)')
        d = {}
        for k, (nm, ag) in enumerate(AGENTS.items()):
            times, succ = simulate_agent(N_act, Th, ag, rng, reps=200)
            S = budgeted_success(times, succ, Th, taus)
            K = np.ceil(ag['kappa'] * N_act)
            wes_plus = float(np.mean(succ.mean(0) * np.minimum(N_act / K, 1)))
            t1 = budgeted_success(times, succ, Th, [1.0, 2.0, 5.0, 10.0])
            # tau at which agent reaches the human's S(1)
            target = human_budgeted_success([1.0], P_HUMAN, SIGMA_H)[0]
            above = np.where(S >= target)[0]
            d[nm] = dict(raw_success=float(succ.mean()), S_tau1=float(t1[0]), S_tau2=float(t1[1]),
                         S_tau5=float(t1[2]), S_tau10=float(t1[3]), wes_plus_like=wes_plus,
                         median_time_ratio=float(np.median(times / Th[None, :])),
                         tau_parity=float(taus[above[0]]) if len(above) else None)
            axs[j].plot(taus, S, color=COLORS[k], label=nm)
        d['human'] = dict(raw_success=P_HUMAN, S_tau1=float(human_budgeted_success([1.0], P_HUMAN, SIGMA_H)[0]),
                          S_tau2=float(human_budgeted_success([2.0], P_HUMAN, SIGMA_H)[0]))
        names = list(AGENTS)
        d['rank_by_raw_success'] = sorted(names, key=lambda n: -d[n]['raw_success'])
        d['rank_by_S_tau1'] = sorted(names, key=lambda n: -d[n]['S_tau1'])
        d['rank_by_wes_plus'] = sorted(names, key=lambda n: -d[n]['wes_plus_like'])
        res[lab] = d
        axs[j].axvline(1, color='gray', lw=0.8, ls='--')
        axs[j].set_xscale('log'); axs[j].set_xlabel(r'time budget $\tau$ (multiples of human median time)')
        axs[j].set_ylabel(r'$S(\tau)$ = P(success within budget)')
        axs[j].set_title(f'({"ab"[j]}) Time-budgeted success, {lab} human reference', fontsize=9)
        axs[j].legend(fontsize=6.5, loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig3_budgeted_success.png'), dpi=150); plt.close()
    OUT['e3'] = res

e3()

# ===================================== E4 ===========================================
def e4():
    """Horizon dependence (Prop. 1) and the on-time threshold (Prop. 3)."""
    rng = np.random.default_rng(1)
    ns = np.unique(np.round(np.logspace(0.3, 3.3, 40)).astype(int))
    ag = dict(kappa=2.3, l0=3.0, a=1.0)   # A26g-like latency with A26o-like step inflation
    res = {'n_star': {}, 'ontime': {}}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    C0 = HUMAN_MEDIAN_OSWORLD * OUT['e1']['share_reading']  # human fixed overhead ~ reading share
    for k, g in enumerate([0.0, 0.001, 0.003, 0.01, 0.04]):
        ratio = agent_time_closed(ag['kappa'] * ns, ag['l0'], g, ag['a']) / (H_TYP * ns)
        axs[0].plot(ns, ratio, color=COLORS[k], label=f'gamma={g}')
    for l0 in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for g in [0.0, 0.001, 0.003, 0.01, 0.04]:
            res['n_star'][f'l0={l0},gamma={g}'] = dict(
                typical=float(parity_horizon(H_TYP, ag['kappa'], l0, g, ag['a'])),
                expert=float(parity_horizon(H_EXP, ag['kappa'], l0, g, ag['a'])))
    axs[0].axhline(1, color='k', lw=0.8)
    for xh, lab in [(120 / H_TYP, 'OSWorld-1 median'), (5760 / H_TYP, 'OSWorld-2 median')]:
        axs[0].axvline(xh, color='gray', ls=':', lw=1); axs[0].text(xh * 1.08, 8, lab, fontsize=7, rotation=90)
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel('human actions n'); axs[0].set_ylabel('agent time / typical human time')
    axs[0].set_title('(a) Time ratio vs horizon (kappa=2.3, l0=3s), varying gamma', fontsize=9)
    axs[0].legend(fontsize=7)
    # threshold: per-step mean mu relative to budget per agent step b = tau*h/kappa
    # vary the agent's base latency so that its mean step time is a multiple c of the budget
    for k, c in enumerate([0.8, 0.95, 1.05, 1.25]):
        ag2 = dict(kappa=2.0, gamma=0.0, a=0.0, p0=1.0, h=0.0, eps=0.15, rho=2.0, noise=1.0)
        # mean executed steps per nominal step = 1 + eps*rho
        ag2['l0'] = c * H_TYP / (ag2['kappa'] * (1 + ag2['eps'] * ag2['rho']))
        on = []
        for n in ns:
            nn = np.full(400, n)
            times, succ = simulate_agent(nn, H_TYP * nn, ag2, rng, reps=1)
            on.append(np.mean(times[0] <= H_TYP * nn))
        axs[1].plot(ns, on, color=COLORS[k], label=f'mean step = {c} x budget')
        res['ontime'][f'c={c}'] = dict(zip([int(x) for x in ns], [float(x) for x in on]))
    axs[1].set_xscale('log'); axs[1].set_xlabel('human actions n'); axs[1].set_ylabel('P(agent finishes within human time)')
    axs[1].set_title('(b) On-time probability sharpens into a threshold', fontsize=9)
    axs[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig4_horizon.png'), dpi=150); plt.close()
    OUT['e4'] = res

e4()

# ===================================== E5 ===========================================
def e5():
    """Verification/best-of-N under a deadline (Prop. 4). Stylised, parameters are ASSUMPTIONS."""
    Ns = np.arange(1, 17)
    n = 20; kappa = 1.5; K = kappa * n
    eps1, alpha = 0.03, 1.0          # per-step unrecoverable error with N candidates: eps1 / N^alpha
    l0, beta, a = 3.0, 0.15, 1.0     # judge/aggregation overhead grows with N
    Th = H_TYP * n
    res = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    p = (1 - eps1 / Ns ** alpha) ** K
    ax.plot(Ns, p, 'k-', lw=2, label=r'success, no deadline ($\tau=\infty$)')
    for k, tau in enumerate([0.75, 1.0, 1.5, 3.0]):
        mu = K * (l0 * (1 + beta * (Ns - 1)) + a)
        F = 0.5 * (1 + np.vectorize(lambda x: __import__('math').erf(x))((np.log(tau * Th) - np.log(mu)) / (0.3 * np.sqrt(2))))
        S = p * F
        ax.plot(Ns, S, color=COLORS[k], label=fr'$\tau$={tau}: N*={Ns[np.argmax(S)]}')
        res[f'tau={tau}'] = dict(N_star=int(Ns[np.argmax(S)]), S_max=float(S.max()))
    res['tau=inf'] = dict(N_star=int(Ns[np.argmax(p)]), S_max=float(p.max()))
    ax.set_xlabel('candidates per step N'); ax.set_ylabel(r'$S(\tau)$')
    ax.set_title('Deadline-optimal verification breadth', fontsize=9); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig5_verification.png'), dpi=150); plt.close()
    OUT['e5'] = res

e5()

# ===================================== E6 ===========================================
def e6():
    """Scenario projection of the date of fluency parity. All improvement rates are
    ASSUMPTIONS drawn from wide priors; the output is conditional on them."""
    rng = np.random.default_rng(2)
    D = 20000
    lu = lambda lo, hi, size: np.exp(rng.uniform(np.log(lo), np.log(hi), size))
    l0_0 = rng.uniform(3.5, 7.0, D)        # s/step model latency, late 2026 (CADWorld-derived range)
    a = 1.0
    kap0 = rng.uniform(2.0, 3.0, D)
    gam0 = lu(0.001, 0.02, D)
    r_l = lu(1.25, 4.0, D)                 # annual speed-up of per-step latency
    r_k = lu(1.0, 2.0, D)                  # annual shrink factor of excess steps (kappa-1)
    r_g = lu(1.0, 3.0, D)                  # annual shrink of context-growth coefficient
    halv = lu(0.5, 2.0, D)                 # years for the per-step failure hazard to halve
    years = np.arange(0, 15.01, 1 / 12)
    START = 2026 + 10.5 / 12
    classes = {'2 min (OSWorld-1 median)': 120.0, '15 min': 900.0, '1.6 h (OSWorld-2 median)': 5760.0}
    res = {}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for hk, (hname, href) in enumerate([('typical', H_TYP), ('expert', H_EXP)]):
        res[hname] = {}
        for ck, (cname, T) in enumerate(classes.items()):
            Th = T if hname == 'typical' else T / SCALE
            n = Th / href
            target = P_HUMAN * 0.5          # human S(1) at the per-task median time
            t_par = np.full(D, np.inf); t_time = np.full(D, np.inf)
            for t in years:
                l0 = l0_0 / r_l ** t
                kap = 1 + (kap0 - 1) / r_k ** t
                gam = gam0 / r_g ** t
                K = kap * n
                TA = K * (l0 + a) + l0 * gam * K * (K - 1) / 2
                haz = H_HAZ * 0.5 ** (t / halv)
                ps = np.minimum(P0_26 + (0.99 - P0_26) * (1 - 0.5 ** (t / halv)), 0.99) * np.exp(-haz * K)
                from scipy.stats import norm
                ontime = norm.cdf(np.log(Th / TA) / 0.3)
                Sa = ps * ontime
                t_par = np.where(np.isinf(t_par) & (Sa >= target), t, t_par)
                t_time = np.where(np.isinf(t_time) & (TA <= Th), t, t_time)
            fin = np.isfinite(t_par)
            q = np.percentile(np.where(fin, t_par, 99), [10, 50, 90])
            qt = np.percentile(np.where(np.isfinite(t_time), t_time, 99), [10, 50, 90])
            res[hname][cname] = dict(
                n_human_actions=float(n),
                parity_year_p10=float(START + q[0]) if q[0] < 99 else None,
                parity_year_p50=float(START + q[1]) if q[1] < 99 else None,
                parity_year_p90=float(START + q[2]) if q[2] < 99 else None,
                frac_parity_within_15y=float(fin.mean()),
                frac_parity_already=float(np.mean(t_par == 0)),
                time_only_year_p50=float(START + qt[1]) if qt[1] < 99 else None)
            # sensitivity: Spearman rank correlation of parameters with parity time
            tp = np.where(fin, t_par, 15.5)
            res[hname][cname]['spearman'] = {nm: float(spearmanr(v, tp)[0]) for nm, v in
                                             [('l0_2026', l0_0), ('kappa_2026', kap0), ('gamma_2026', gam0),
                                              ('latency_speedup', r_l), ('kappa_shrink', r_k),
                                              ('gamma_shrink', r_g), ('hazard_halving_yrs', halv)]}
            if hname == 'typical':
                xs = np.sort(np.where(fin, t_par, np.nan)); xs = xs[~np.isnan(xs)]
                ax.plot(START + xs, np.arange(1, len(xs) + 1) / D, color=COLORS[ck], label=cname)
    ax.set_xlabel('year'); ax.set_ylabel('P(fluency parity reached)')
    ax.set_title('Scenario CDF of fluency-parity date (typical human ref.)', fontsize=9)
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig6_projection.png'), dpi=150); plt.close()
    OUT['e6'] = res

def e6_bounds():
    """Deterministic pessimistic / central / optimistic scenarios + p_h sensitivity."""
    from scipy.stats import norm
    START = 2026 + 10.5 / 12
    scen = {'pessimistic': dict(l0=7.0, kap=3.0, gam=0.02, rl=1.25, rk=1.0, rg=1.0, halv=2.0),
            'central': dict(l0=5.0, kap=2.5, gam=0.005, rl=2.0, rk=1.4, rg=1.7, halv=1.0),
            'optimistic': dict(l0=3.5, kap=2.0, gam=0.001, rl=4.0, rk=2.0, rg=3.0, halv=0.5)}
    years = np.arange(0, 30.01, 1 / 12)
    out = {}
    for sn, p in scen.items():
        for hname, href, sc in [('typical', H_TYP, 1.0), ('expert', H_EXP, SCALE)]:
            for cname, T in [('2min', 120.0), ('15min', 900.0), ('1.6h', 5760.0)]:
                for ph in [0.5, P_HUMAN, 0.9]:
                    Th = T / sc; n = Th / href; yr = None
                    for t in years:
                        l0 = p['l0'] / p['rl'] ** t; kap = 1 + (p['kap'] - 1) / p['rk'] ** t
                        gam = p['gam'] / p['rg'] ** t; K = kap * n
                        TA = K * (l0 + 1.0) + l0 * gam * K * (K - 1) / 2
                        haz = H_HAZ * 0.5 ** (t / p['halv'])
                        ps = min(P0_26 + (0.99 - P0_26) * (1 - 0.5 ** (t / p['halv'])), 0.99) * np.exp(-haz * K)
                        if ps * norm.cdf(np.log(Th / TA) / 0.3) >= ph * 0.5:
                            yr = START + t; break
                    out[f'{sn}|{hname}|{cname}|p_h={ph}'] = yr
    OUT['e6_bounds'] = out

e6_bounds()

e6()

OUT['runtime_s'] = time.time() - T0
json.dump(OUT, open(os.path.join(RES, 'results.json'), 'w'), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:20000])
print('runtime', OUT['runtime_s'])
