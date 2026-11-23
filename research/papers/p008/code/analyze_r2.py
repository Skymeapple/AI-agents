"""Round-2 analyses (revision). Reads results/all_runs.json, e5_runs.json, e6_runs_{A,B}.json, e6_models/*.pt,
data_ext/popqa/popQA.tsv. Writes results/summary_r2.json and fig6_prop2_finiteM.png, fig7_e6_ceiling.png."""
import json, os, glob, itertools, collections, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); sys.path.insert(0, HERE)
R = json.load(open(os.path.join(RES, 'all_runs.json')))
T = 400; S = {}
def zipf(a, M=T): f = np.arange(1, M + 1) ** (-(1 + a)); return f / f.sum()
rng = np.random.default_rng(0)
def boot_ci(x, n=10000, stat=np.mean):
    x = np.asarray(x, float); b = [stat(rng.choice(x, len(x))) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

# ======== 1. Prop. 2, finite M: validation against all capacity-regime runs ========
def Etop(K, a): p = zipf(a); return p[K:].sum()
def finiteM(r, kappa, a): return (r ** a - kappa ** a) / (1 - kappa ** a)
tr = [r for r in R if r['exp'] == 'teacher'][0]
LT = np.array(tr['score_t']) > 0.75; KT = int(LT.sum()); ET = 1 - tr['wscore']
rows = []
for r in R:
    if r['exp'] == 'E1' and r['family'] == 'random':
        KS = r['learned']; rr = KT / KS; obs = (1 - r['wscore']) / ET
        rows.append(dict(h=r['h'], seed=r['seed'], K_S=KS, r=rr, obs=obs, asym=rr ** .5,
                         finiteM=finiteM(rr, KT / T, .5), topK_exact=Etop(KS, .5) / Etop(KT, .5)))
S['prop2_vs_teacher'] = dict(K_T=KT, E_T=ET, max_learned_index_teacher=int(np.where(LT)[0].max() + 1), rows=rows)
# all ordered pairs of capacity-regime students with K_S < K_T (same alpha)
pts = []
for r in R:
    if (r['exp'] == 'E1' and r['family'] == 'random') or r['exp'] == 'E4':
        pts.append((r.get('alpha', .5), r['learned'], 1 - r['wscore']))
pairs = []
for (a1, k1, e1), (a2, k2, e2) in itertools.permutations(pts, 2):
    if a1 == a2 and 3 <= k1 < k2:
        rr = k2 / k1
        pairs.append(dict(alpha=a1, r=rr, kappa=k2 / T, obs=e1 / e2, asym=rr ** a1, fin=finiteM(rr, k2 / T, a1)))
# plus teacher pairs
for q in rows: pairs.append(dict(alpha=.5, r=q['r'], kappa=KT / T, obs=q['obs'], asym=q['asym'], fin=q['finiteM']))
le = lambda k: np.array([np.log(p[k] / p['obs']) for p in pairs])
S['prop2_pairs'] = dict(n=len(pairs), median_abs_logerr_asym=float(np.median(np.abs(le('asym')))),
                        median_abs_logerr_finiteM=float(np.median(np.abs(le('fin')))),
                        median_signed_logerr_finiteM=float(np.median(le('fin'))),
                        median_signed_logerr_asym=float(np.median(le('asym'))),
                        frac_finiteM_closer=float(np.mean(np.abs(le('fin')) < np.abs(le('asym')))))
# admissible compression table at finite kappa
S['prop2_table_finite'] = {f'alpha={a}': {f'kappa={k}': {f'delta={d}': float(((1 + d) * (1 - k ** a) + k ** a) ** (1 / a))
                           for d in [0.25, 0.5, 1.0, 2.0]} for k in [0.0, 0.01, 0.1, 0.3, 0.6]} for a in [0.25, 0.5, 1.0]}
fig, ax = plt.subplots(figsize=(4.4, 3.6))
for a, c in [(0.25, 'C2'), (0.5, 'C0'), (1.0, 'C3')]:
    P = [p for p in pairs if p['alpha'] == a]
    ax.scatter([p['obs'] for p in P], [p['asym'] for p in P], s=6, marker='x', color=c, alpha=.4)
    ax.scatter([p['obs'] for p in P], [p['fin'] for p in P], s=6, color=c, label=f'α={a}')
lim = [1, 60]; ax.plot(lim, lim, 'k:', lw=1); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('observed error ratio E_S/E_T'); ax.set_ylabel('predicted (dots: finite-M; x: asymptotic r^α)')
ax.legend(fontsize=7); ax.set_title('Prop. 2: finite-M vs asymptotic')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig6_prop2_finiteM.png'), dpi=160); plt.close()

# ======== 2. gamma: point-wise, seed-resampled, free-exponent ========
Ds = [1000, 3000, 10000, 30000, 100000, 300000]; Z = (np.arange(1, T + 1) ** -1.5).sum()
cell = collections.defaultdict(list)
for r in R:
    if r['exp'] == 'E2' and r['h'] == 128: cell[(r['D'], r['method'])].append(r['learned'])
def fit(Kd, meth, free=False):
    Kinf = np.mean(cell[(None, meth)])
    xs = [(D, Kd[D]) for D in Ds if 3 < Kd[D] < 0.8 * Kinf]
    n = np.exp(np.mean([np.log(D / (Z * K ** 1.5)) for D, K in xs]))
    sl = np.polyfit(np.log([x[0] for x in xs]), np.log([x[1] for x in xs]), 1)[0] if free and len(xs) > 1 else None
    return n, sl, xs
gam = []; sl_h = []; sl_s = []
for choice in itertools.product([0, 1], repeat=2 * len(Ds)):
    Kh = {D: cell[(D, 'hard')][choice[i]] for i, D in enumerate(Ds)}
    Ks = {D: cell[(D, 'soft')][choice[len(Ds) + i]] for i, D in enumerate(Ds)}
    nh, a1, _ = fit(Kh, 'hard', True); ns, a2, _ = fit(Ks, 'soft', True); gam.append(nh / ns); sl_h.append(a1); sl_s.append(a2)
Km = {m: {D: np.mean(cell[(D, m)]) for D in Ds} for m in ['hard', 'soft']}
pw = {D: float((Km['soft'][D] / Km['hard'][D]) ** 1.5) for D in Ds if Km['hard'][D] >= 3}
S['gamma'] = dict(pointwise_by_D=pw, seed_resampled_fit=dict(median=float(np.median(gam)), p2_5=float(np.percentile(gam, 2.5)),
                  p97_5=float(np.percentile(gam, 97.5)), min=float(min(gam)), max=float(max(gam)), n=len(gam)),
                  free_slope_hard=[float(np.min(sl_h)), float(np.max(sl_h))], free_slope_soft=[float(np.min(sl_s)), float(np.max(sl_s))])
# soft-vs-hard at matched K: D needed by hard / D needed by soft, via log-log interpolation (exponent-free gamma)
def Dneed(meth, K):
    xs = np.array(Ds, float); ks = np.array([Km[meth][D] for D in Ds])
    ok = ks > 0; return float(np.exp(np.interp(np.log(K), np.log(ks[ok]), np.log(xs[ok]))))
S['gamma']['matched_K_data_ratio'] = {K: Dneed('hard', K) / Dneed('soft', K) for K in [10, 20, 40, 70]}
# E3 accuracy-based K (argmax agreement with Bayes >= 0.9) and gamma
g3 = collections.defaultdict(list)
for r in R:
    if r['exp'] == 'E3': g3[(r['beta'], r['method'])].append(r)
S['E3_acc'] = {}
for b in [0.5, 1.0, 2.0, 4.0, 8.0]:
    d = {}
    for m in ['hard', 'soft']:
        d[f'Kacc_{m}'] = float(np.mean([(np.array(r['acc_t']) >= 0.9).sum() for r in g3[(b, m)]]))
        d[f'K_{m}'] = float(np.mean([r['learned'] for r in g3[(b, m)]]))
    d['gamma_acc'] = float((d['Kacc_soft'] / max(d['Kacc_hard'], 1)) ** 1.5); d['gamma_kl'] = float((d['K_soft'] / max(d['K_hard'], 1)) ** 1.5)
    S['E3_acc'][b] = d

# ======== 3. P1 calibration: bits per skill at the learned threshold ========
import torch, torch.nn.functional as F, skills as SK
w = SK.World(seed=0); P = w.test_logp.exp()
kl0 = (P * (w.test_logp + np.log(4))).sum(-1).view(T, -1).mean(1)
g = torch.Generator().manual_seed(1); cal = []
for eps in [0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2]:
    fr = []; ms = []
    for rep in range(5):
        Wc = w.W - w.W.mean(1, keepdim=True)                # identifiable part ((C-1)d = 12 dof, var 0.75 each)
        xi = torch.randn(w.W.shape, generator=g); xi = (xi - xi.mean(1, keepdim=True))
        What = (1 - eps) * Wc + np.sqrt(eps * (1 - eps)) * xi   # Gaussian RD test channel at relative MSE eps
        lq = F.log_softmax(w.beta * torch.einsum('ncd,nd->nc', What[w.test_t], w.test_z), -1)
        kl = (P * (w.test_logp - lq)).sum(-1).view(T, -1).mean(1); s = (1 - kl / kl0).clamp(0, 1)
        fr.append((s > 0.75).float().mean().item()); ms.append(s.mean().item())
    cal.append(dict(eps=eps, frac_learned=float(np.mean(fr)), mean_score=float(np.mean(ms)), bits_per_skill=float(12 * 0.5 * np.log2(1 / eps))))
eps_star = float(np.interp(0.5, [c['frac_learned'] for c in cal][::-1], [c['eps'] for c in cal][::-1]))
S['P1_calibration'] = dict(table=cal, eps_at_half_learned=eps_star, R_bits_per_skill=float(6 * np.log2(1 / eps_star)))

# ======== 4. PopQA popularity (public data; Mallen et al. 2023) ========
pq = os.path.join(HERE, '..', 'data_ext', 'popqa', 'popQA.tsv')
if os.path.exists(pq):
    import pandas as pd
    d = pd.read_csv(pq, sep='\t'); x = np.sort(d.s_pop.values.astype(float))[::-1]; c = np.cumsum(x) / x.sum()
    rk = np.arange(1, len(x) + 1)
    S['popqa'] = dict(n_items=int(len(x)), quantiles={q: float(np.percentile(x, q)) for q in [10, 50, 90, 99]},
        view_share_of_top={f: float(c[int(f * len(x)) - 1]) for f in [0.01, 0.05, 0.1, 0.2, 0.5]},
        local_rank_slope={f'{lo}-{hi}': float(np.polyfit(np.log(rk[lo:hi]), np.log(x[lo:hi]), 1)[0]) for lo, hi in [(10, 1000), (1000, 5000), (5000, 12000)]})
    # error-ratio arithmetic with the empirical popularity vector (general-p version of Prop. 2)
    pv = x / x.sum(); E = lambda f: pv[int(f * len(x)):].sum()
    S['popqa']['err_ratio_teacher_top50pct'] = {f: float(E(f) / E(0.5)) for f in [0.05, 0.1, 0.2, 0.3]}

# ======== 5. E6 long training ========
E6 = []
for fn in ['e6_runs_A.json', 'e6_runs_B.json']:
    fp = os.path.join(RES, fn)
    if os.path.exists(fp): E6 += json.load(open(fp))
E5 = json.load(open(os.path.join(RES, 'e5_runs.json')))
if E6:
    d6 = collections.defaultdict(list)
    for r in E6: d6[(r['h'], r['steps'], r['method'])].append(r)
    for r in E5:   # E5 world seed == train seed; seed-0 runs are identical in config to world-0 runs
        if r['D'] is None and r['seed'] == 0 and r['h'] == 16 and len(d6[(16, 45000, r['method'])]) < 5:
            d6[(16, 45000, r['method'])].append(r)
    out = {}
    for (h, st, m), v in sorted(d6.items()):
        K = [r['learned'] for r in v]; A = [r['wscore'] for r in v]
        out[f'h={h},steps={st},{m}'] = dict(n=len(v), K=K, K_mean=float(np.mean(K)), K_ci=boot_ci(K) if len(K) > 2 else None,
            A_mean=float(np.mean(A)), A=A, params=v[0]['params'],
            traj=[r.get('traj') for r in v])
    S['E6'] = out
    # paired comparisons: soft vs hard, teacher vs hard (by seed), with exact sign test / Welch
    from scipy.stats import ttest_ind, mannwhitneyu
    cmp = {}
    for (h, st) in sorted({(k[0], k[1]) for k in d6}):
        H = [r['learned'] for r in d6[(h, st, 'hard')]]
        for m in ['soft', 'soft_teacher']:
            O = [r['learned'] for r in d6[(h, st, m)]]
            if len(H) >= 2 and len(O) >= 2:
                diff = np.mean(O) - np.mean(H)
                bd = [np.mean(rng.choice(O, len(O))) - np.mean(rng.choice(H, len(H))) for _ in range(10000)]
                cmp[f'h={h},steps={st},{m}-hard'] = dict(diff=float(diff), ratio=float(np.mean(O) / np.mean(H)),
                    rel_gap=float(diff / np.mean(O)), ci95=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                    welch_p=float(ttest_ind(O, H, equal_var=False).pvalue) if np.std(O) + np.std(H) > 0 else None,
                    mw_p=float(mannwhitneyu(O, H, alternative='two-sided').pvalue))
    S['E6_compare'] = cmp
    # teacher-distilled students vs teacher (replaces oracle-trained Table 2)
    tab = []
    for (h, st, m), v in sorted(d6.items()):
        if m != 'soft_teacher': continue
        for r in v:
            sm = np.array(r['score_t']) > 0.75; KS = int(sm.sum())
            tab.append(dict(h=h, steps=st, seed=r['seed'], params=r['params'], A=r['wscore'], K=KS,
                            err_ratio=(1 - r['wscore']) / ET, retained=float((sm & LT).sum() / KT), outside_teacher=int((sm & ~LT).sum()),
                            pred_asym=(KT / KS) ** .5 if KS else None, pred_finiteM=finiteM(KT / KS, KT / T, .5) if KS else None))
    S['E6_teacher_table'] = tab
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for i, h in enumerate([16, 32, 48]):
        for m, c, off in [('hard', 'C1', -0.03), ('soft', 'C0', 0), ('soft_teacher', 'C2', 0.03)]:
            sts = sorted({k[1] for k in d6 if k[0] == h and k[2] == m})
            # include 10k-step E2 unlimited (world seed = train seed) for reference
            xs, ys, lo, hi = [], [], [], []
            for st in sts:
                K = [r['learned'] for r in d6[(h, st, m)]]
                xs.append(st * (1 + off)); ys.append(np.mean(K)); lo.append(np.mean(K) - min(K)); hi.append(max(K) - np.mean(K))
            if xs: ax[i].errorbar(xs, ys, yerr=[lo, hi], fmt='o-', color=c, label=m, capsize=2)
            if m != 'soft_teacher':
                e2 = [r['learned'] for r in R if r['exp'] == 'E2' and r['h'] == h and r['D'] is None and r['method'] == m]
                if e2: ax[i].plot([1e4 * (1 + off)], [np.mean(e2)], 's', mfc='none', color=c)
        ax[i].set_xscale('log'); ax[i].set_xlabel('training steps (unlimited data)'); ax[i].set_title(f'h={h}')
    ax[0].set_ylabel('skills learned K (range over seeds)'); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig7_e6_ceiling.png'), dpi=160); plt.close()

# ======== 6. effective bits per parameter via post-training quantisation (E6 final models) ========
mods = sorted(glob.glob(os.path.join(RES, 'e6_models', 'A_h*_s*_100000.pt')))
if mods:
    qres = collections.defaultdict(list)
    for fp in mods:
        name = os.path.basename(fp); h = int(name.split('_h')[1].split('_')[0]); meth = name.split(f'_h{h}_')[1].rsplit('_s', 1)[0]
        sd = torch.load(fp); m = SK.MLP(w.code, w.d, w.C, h); m.load_state_dict(sd)
        K0 = int((SK.evaluate(w, m)['score_t'] > 0.75).sum()); row = dict(h=h, method=meth, K_full=K0, params=SK.n_params(m))
        for b in [2, 3, 4, 5, 6, 8]:
            q = {}
            for k, v in sd.items():
                if k == 'code': q[k] = v; continue
                s = v.abs().max() / (2 ** (b - 1) - 1); q[k] = torch.round(v / s) * s
            m.load_state_dict(q); row[f'K_{b}bit'] = int((SK.evaluate(w, m)['score_t'] > 0.75).sum())
        bmin = next((b for b in [2, 3, 4, 5, 6, 8] if row[f'K_{b}bit'] >= 0.95 * K0), None)
        row['b_min95'] = bmin; row['bits_per_learned_skill'] = bmin * row['params'] / K0 if bmin else None
        qres[(h, meth)].append(row)
    S['quantisation'] = {f'h={h},{m}': v for (h, m), v in qres.items()}
    allb = [r['bits_per_learned_skill'] for v in qres.values() for r in v if r['bits_per_learned_skill']]
    S['P1_calibration']['measured_bits_per_learned_skill'] = dict(min=float(min(allb)), median=float(np.median(allb)), max=float(max(allb)))
    S['P1_calibration']['looseness_factor_median'] = float(np.median(allb) / S['P1_calibration']['R_bits_per_skill'])

json.dump(S, open(os.path.join(RES, 'summary_r2.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k not in ('E6', 'prop2_vs_teacher', 'E6_teacher_table', 'quantisation')}, indent=1, default=float)[:9000])
