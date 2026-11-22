"""Analysis + figures from ../results/all_runs.json -> ../results/{summary.json, fig*.png, tables.md}"""
import json, os, collections
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
R = json.load(open(os.path.join(RES, 'all_runs.json')))
T = 400
def zipf(alpha, T=T):
    f = np.arange(1, T + 1) ** (-(1 + alpha)); return f / f.sum()
def group(exp, keys):
    d = collections.defaultdict(list)
    for r in R:
        if r['exp'] == exp: d[tuple(r[k] for k in keys)].append(r)
    return d
S = {}; md = []
plt.rcParams.update({'font.size': 9})

# ---------- E1: capacity regime ----------
g = group('E1', ['family', 'h'])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
S['E1'] = {}
for fam, c in [('random', 'C0'), ('lowrank', 'C3')]:
    hs = sorted(h for f, h in g if f == fam)
    N = np.array([g[(fam, h)][0]['params'] for h in hs])
    K = np.array([np.mean([r['learned'] for r in g[(fam, h)]]) for h in hs])
    Ksd = np.array([np.std([r['learned'] for r in g[(fam, h)]]) for h in hs])
    ax[0].errorbar(N, K, yerr=Ksd, fmt='o-', color=c, label=f'{fam} skills')
    ok = (K > 2) & (K < 0.9 * T)
    slope, icpt = np.polyfit(np.log(N[ok]), np.log(K[ok]), 1)
    S['E1'][fam] = dict(h=hs, params=N.tolist(), K=K.tolist(), Ksd=Ksd.tolist(), loglog_slope=float(slope),
                        params_per_skill=(N[ok] / K[ok]).tolist(),
                        wscore=[float(np.mean([r['wscore'] for r in g[(fam, h)]])) for h in hs],
                        uscore=[float(np.mean([r['uscore'] for r in g[(fam, h)]])) for h in hs])
    md.append(f'| {fam} | ' + ' | '.join(f'{n} / {k:.1f}' for n, k in zip(N, K)) + f' | {slope:.2f} |')
ax[0].axhline(T, ls=':', c='gray'); ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('student parameters N'); ax[0].set_ylabel('skills learned K (score>0.75)'); ax[0].legend()
ax[0].set_title('(a) capacity regime (unlimited teacher queries)')
p = zipf(0.5)
for h, c in zip([8, 16, 32, 48], ['C0', 'C1', 'C2', 'C3']):
    sc = np.mean([r['score_t'] for r in g[('random', h)]], 0)
    ax[1].plot(np.arange(1, T + 1), sc, '.', ms=2, color=c, label=f'h={h}')
ax[1].set_xscale('log'); ax[1].set_xlabel('skill frequency rank'); ax[1].set_ylabel('skill score')
ax[1].set_title('(b) per-skill score, random family'); ax[1].legend(markerscale=4)
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig1_capacity.png'), dpi=160); plt.close()
# frequency ordering: Spearman between rank and score
from scipy.stats import spearmanr
S['E1']['spearman_rank_vs_score'] = {h: float(spearmanr(np.arange(T), np.mean([r['score_t'] for r in g[('random', h)]], 0))[0]) for h in [8, 16, 32, 48]}
# convergence check
conv = {r['h']: r['learned'] for r in R if r['exp'] == 'E1conv'}
S['E1']['conv_90k_vs_45k'] = {h: dict(K90k=conv[h], K45k_seed0=[r['learned'] for r in g[('random', h)] if r['seed'] == 0][0]) for h in conv}

# ---------- head/tail divergence (E1 random + E4) ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
pts = []
for alpha, c in [(0.25, 'C2'), (0.5, 'C0'), (1.0, 'C3')]:
    pz = zipf(alpha); cum = np.cumsum(pz)
    K = np.arange(0, T + 1); A = np.concatenate([[0], cum]); U = K / T
    ax[0].plot(U, A, '-', color=c, label=f'theory top-K, α={alpha}')
    runs = [r for r in R if (r['exp'] == 'E4' and r['alpha'] == alpha) or (r['exp'] == 'E1' and r['family'] == 'random' and alpha == 0.5)]
    for r in runs:
        ax[0].plot(r['uscore'], r['wscore'], 'o', color=c, ms=4)
        # top-K prediction from the student's own K
        pts.append(dict(alpha=alpha, h=r['h'], K=r['learned'], U=r['uscore'], A=r['wscore'], A_pred=float(A[r['learned']]), U_pred=r['learned'] / T))
ax[0].set_xlabel('uniform-over-skills score U (tail benchmark)'); ax[0].set_ylabel('frequency-weighted score A (aggregate)')
ax[0].set_title('(a) aggregate vs tail retention'); ax[0].legend(fontsize=7)
# theory: compression ratio vs tail retention at aggregate tolerance (infinite skills)
d = np.linspace(0.01, 3, 300)
for alpha, c in [(0.25, 'C2'), (0.5, 'C0'), (1.0, 'C3')]:
    ax[1].plot(1 + d, (1 + d) ** (1 / alpha), color=c, label=f'α={alpha}')
ax[1].set_yscale('log'); ax[1].set_xlabel('allowed aggregate-error ratio  E_S/E_T = 1+δ')
ax[1].set_ylabel('skill compression r = K_T/K_S  (= 1/tail retention)'); ax[1].legend(); ax[1].set_title('(b) Prop. 2')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig2_head_tail.png'), dpi=160); plt.close()
S['head_tail_points'] = pts
S['head_tail_A_abs_err'] = float(np.mean([abs(q['A'] - q['A_pred']) for q in pts]))
S['prop2_table'] = {f'alpha={a}': {f'delta={dd}': float((1 + dd) ** (1 / a)) for dd in [0.25, 0.5, 1.0, 2.0]} for a in [0.25, 0.5, 1.0]}

# ---------- E2: data regime ----------
g2 = group('E2', ['h', 'D', 'method'])
Ds = [1000, 3000, 10000, 30000, 100000, 300000]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
S['E2'] = {}
for i, h in enumerate([16, 32, 128]):
    S['E2'][h] = {}
    for meth, c in [('hard', 'C1'), ('soft', 'C0')]:
        K = [np.mean([r['learned'] for r in g2[(h, D, meth)]]) for D in Ds]
        W = [np.mean([r['wscore'] for r in g2[(h, D, meth)]]) for D in Ds]
        Kinf = np.mean([r['learned'] for r in g2[(h, None, meth)]])
        ax[i].plot(Ds, K, 'o-', color=c, label=f'{meth} labels')
        ax[i].axhline(Kinf, color=c, ls=':', lw=1)
        S['E2'][h][meth] = dict(D=Ds, K=K, wscore=W, K_unlimited=float(Kinf))
    if h == 128:
        tk = [r for r in R if r['exp'] == 'E2T']
        tk = sorted([r for r in tk if r['D'] is not None], key=lambda r: r['D'])
        ax[i].plot([r['D'] for r in tk], [r['learned'] for r in tk], 's--', color='C2', label='soft (trained teacher)')
        S['E2']['teacher_soft'] = dict(D=[r['D'] for r in tk], K=[r['learned'] for r in tk], wscore=[r['wscore'] for r in tk])
    ax[i].set_xscale('log'); ax[i].set_yscale('log'); ax[i].set_title(f'width h={h}'); ax[i].set_xlabel('training inputs D')
ax[0].set_ylabel('skills learned K'); ax[2].legend(fontsize=7)
# fit K_D = (D/(n Z))^(1/(1+alpha)) on h=128 (non-capacity-limited) points with 3<K<0.8*K_unlimited
Z = (np.arange(1, T + 1) ** -1.5).sum()
def fit_n(meth):
    Kinf = S['E2'][128][meth]['K_unlimited']
    xs = [(D, K) for D, K in zip(Ds, S['E2'][128][meth]['K']) if 3 < K < 0.8 * Kinf]
    # threshold rule: K = (D/(n Z))^(1/1.5) -> n = D / (Z K^1.5); geometric mean over points
    ns = [D / (Z * K ** 1.5) for D, K in xs]
    return float(np.exp(np.mean(np.log(ns)))), xs
n_h, xh = fit_n('hard'); n_s, xs_ = fit_n('soft')
S['E2']['n_hard'] = n_h; S['E2']['n_soft'] = n_s; S['E2']['gamma'] = n_h / n_s
S['E2']['fit_points'] = dict(hard=xh, soft=xs_)
Dg = np.logspace(3, 5.5, 50)
for meth, n, c in [('hard', n_h, 'C1'), ('soft', n_s, 'C0')]:
    for a in ax: a.plot(Dg, (Dg / (n * Z)) ** (1 / 1.5), color=c, lw=0.8, alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig3_data_regime.png'), dpi=160); plt.close()
# predicted crossover D* where hard-label K_D reaches capacity ceiling K_cap(h) (10k-step ceiling)
S['E2']['crossover'] = {}
for h in [16, 32, 128]:
    Kc = S['E2'][h]['soft']['K_unlimited']
    Dstar = n_h * Z * Kc ** 1.5
    gap = [s - hd for s, hd in zip(S['E2'][h]['soft']['K'], S['E2'][h]['hard']['K'])]
    rel = [ (s - hd) / max(s, 1) for s, hd in zip(S['E2'][h]['soft']['K'], S['E2'][h]['hard']['K'])]
    S['E2']['crossover'][h] = dict(K_cap=Kc, Dstar_pred=float(Dstar), gap=gap, rel_gap=rel)

# ---------- E3: label noise ----------
g3 = group('E3', ['beta', 'method'])
betas = [0.5, 1.0, 2.0, 4.0, 8.0]
S['E3'] = {}
for b in betas:
    kh = np.mean([r['learned'] for r in g3[(b, 'hard')]]); ks = np.mean([r['learned'] for r in g3[(b, 'soft')]])
    wh = np.mean([r['wscore'] for r in g3[(b, 'hard')]]); ws = np.mean([r['wscore'] for r in g3[(b, 'soft')]])
    S['E3'][b] = dict(K_hard=float(kh), K_soft=float(ks), w_hard=float(wh), w_soft=float(ws),
                      gamma_implied=float((ks / max(kh, 1)) ** 1.5))
# Bayes conditional entropy per beta (Monte Carlo, fixed world seed 0)
import sys; sys.path.insert(0, HERE); import skills as SK, torch
for b in betas:
    w = SK.World(beta=b, seed=0); lp = w.test_logp; S['E3'][b]['bayes_entropy_nats'] = float(-(lp.exp() * lp).sum(-1).mean())
fig, ax = plt.subplots(figsize=(4.2, 3.2))
H = [S['E3'][b]['bayes_entropy_nats'] for b in betas]
ax.plot(H, [S['E3'][b]['K_soft'] for b in betas], 'o-', c='C0', label='soft (oracle)')
ax.plot(H, [S['E3'][b]['K_hard'] for b in betas], 'o-', c='C1', label='hard')
for b, x in zip(betas, H): ax.annotate(f'β={b}', (x, S['E3'][b]['K_soft']), fontsize=7)
ax.set_xlabel('Bayes conditional entropy H(y|t,z) [nats]'); ax.set_ylabel('skills learned K (D=10^4, h=128)'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig4_noise.png'), dpi=160); plt.close()

S['teacher'] = [dict(learned=r['learned'], wscore=r['wscore'], uscore=r['uscore']) for r in R if r['exp'] == 'teacher']
json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k not in ('head_tail_points',)}, indent=1, default=float)[:6000])

# ---------- teacher-relative compression in the testbed (E1 random students vs trained teacher) ----------
tr = [r for r in R if r['exp'] == 'teacher'][0]
tmask = np.array(tr['score_t']) > 0.75
S['compression_vs_teacher'] = []
for h in [8, 12, 16, 24, 32, 48]:
    for r in g[('random', h)]:
        smask = np.array(r['score_t']) > 0.75
        S['compression_vs_teacher'].append(dict(h=h, seed=r['seed'], ratio_params=tr['params'] / r['params'],
            A_student=r['wscore'], A_teacher=tr['wscore'], teacher_skills=int(tmask.sum()),
            frac_teacher_skills_retained=float((smask & tmask).sum() / tmask.sum()),
            err_ratio=(1 - r['wscore']) / (1 - tr['wscore'])))
S['teacher_params'] = tr['params']

# ---------- E5: capacity ceiling with long training ----------
e5p = os.path.join(RES, 'e5_runs.json')
if os.path.exists(e5p):
    E5 = json.load(open(e5p)); d5 = collections.defaultdict(list)
    for r in E5: d5[(r['h'], r['D'], r['method'])].append(r)
    S['E5'] = {f'h={h},D={D},{m}': dict(K=[r['learned'] for r in v], wscore=[round(r['wscore'], 4) for r in v],
               K_mean=float(np.mean([r['learned'] for r in v])), w_mean=float(np.mean([r['wscore'] for r in v])))
               for (h, D, m), v in sorted(d5.items(), key=lambda kv: (kv[0][0], kv[0][1] or 1e12, kv[0][2]))}
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for i, h in enumerate([8, 16]):
        xs = [10000, 100000, 1e7]
        for m, c in [('hard', 'C1'), ('soft', 'C0')]:
            ys = [np.mean([r['learned'] for r in d5[(h, D, m)]]) for D in [10000, 100000, None]]
            ax[i].plot(xs, ys, 'o-', c=c, label=m)
        ax[i].set_xscale('log'); ax[i].set_xticks(xs); ax[i].set_xticklabels(['1e4', '1e5', 'unlim.'])
        ax[i].set_title(f'h={h}, 45k steps'); ax[i].set_xlabel('training inputs D')
    ax[0].set_ylabel('skills learned K'); ax[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig5_ceiling.png'), dpi=160); plt.close()
json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)
print(json.dumps(S.get('E5', {}), indent=0))
for q in S['compression_vs_teacher']: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
