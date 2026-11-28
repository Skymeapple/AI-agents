"""Revision analyses (round 1).
 (1) Exp A: union-bound-corrected certificates; diagnosis of the 'zero-false-accept' observation.
 (2) Exp B: per-seed lambda slopes (only 2 seeds -> report both).
 (3) Exp C: large-universe test of capacity invariance and the C^-(s-1) rate, with cross-fitted
     (held-out) operating-point selection, seed bootstrap CIs, measured open-world hallucination,
     and comparison to store-and-filter theory and the any-model bound (Thm 3).
Outputs results/analysis_rev.txt, results/exp_c_summary.json, results/fig_exp_c.png"""
import json, math, collections, numpy as np
from scipy.stats import beta as _beta
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import certified_bits, D
R = '../results/'
lines = []
def P(x=''): print(x); lines.append(str(x))
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- (1) Experiment A
A = json.load(open(R + 'exp_a.json')); A1 = [r for r in A if r['cfg']['block'] == 'A1']
n = 2.0 ** D; NT = 200000
P("== (1a) Exp A1 certificates with union bound over (g, a) and Bonferroni-CP over thresholds ==")
P("B >= Eq.(1) - 1 - t - log2((k+1)(n+1)), t = 7 (holds w.p. >= 1 - 2^-7 simultaneously over all (g,a));")
P("eps replaced by one-sided Clopper-Pearson upper bound at level 0.05/43 (simultaneous over the 43 thresholds).")
def eps_ucb(e, alpha):
    x = round(e * NT)
    return 1.0 if x >= NT else float(_beta.ppf(1 - alpha, x + 1, NT - x))
best = collections.defaultdict(list)
for r in A1:
    k = r['cfg']['k']; pen = 1 + 7 + math.log2((k + 1) * (n + 1)); nt = len(r['curve'])
    b = max(max(certified_bits(c['g'], k, eps_ucb(c['eps'], 0.05 / nt), 16) - pen, 0) for c in r['curve'])
    best[r['cfg']['h']].append(b / r['P'])
for h, v in sorted(best.items()):
    P(f"h={h}: max certified bits/param {np.max(v):.2f}, median {np.median(v):.2f}  (penalty ~{1+7+math.log2(4097*(n+1)):.0f} bits)")

P("\n== (1b) Diagnosis of the strictest threshold (accept only stored keys with conf >= max stored conf) ==")
P("h  k    w    | tau_max      saturated(=1.0) | g  | eps at tau_max | fresh keys scoring >= best stored key (of 200k) | Thm1 bits needed for (g, eps)")
diag = []
for r in sorted(A1, key=lambda r: (r['cfg']['h'], r['cfg']['k'], r['cfg']['w'], r['cfg']['seed'])):
    cur = [c for c in r['curve'] if c['tau'] <= 1.0 and c['g'] + c['wrong_stored'] >= 1]
    c = max(cur, key=lambda c: c['tau'])
    need = certified_bits(c['g'], r['cfg']['k'], max(c['eps'], 3 / NT), 16)
    diag.append(dict(h=r['cfg']['h'], k=r['cfg']['k'], w=r['cfg']['w'], tau=c['tau'], sat=c['tau'] >= 1.0, g=c['g'], eps=c['eps']))
    if r['cfg']['seed'] == 0:
        P(f"{r['cfg']['h']:<3}{r['cfg']['k']:<5}{r['cfg']['w']:<5}| {c['tau']:.9f}  {str(c['tau'] >= 1.0):<5} | {c['g']:<3}| {c['eps']:.2e} | {round(c['eps']*NT)} | {need:.0f}")
eps_top = np.array([d['eps'] for d in diag])
P(f"Across 60 A1 runs: eps at the strictest stored-key threshold: median {np.median(eps_top):.1e}, range {eps_top.min():.1e}-{eps_top.max():.1e}; "
  f"runs with saturated tau=1.0: {sum(d['sat'] for d in diag)}; runs with eps=0 there: {int((eps_top == 0).sum())}")

# ---------------------------------------------------------------- (2) Experiment B per-seed slopes
B = json.load(open(R + 'exp_b.json'))
P("\n== (2) Exp B: lambda-slope of score-optimal hallucination per seed (lam in [1,100]) ==")
LAMS = [1, 3, 9, 30, 100]
for h in [32, 64, 128]:
    sl = []
    for sd in [0, 1]:
        rs = [r for r in B if r['cfg']['h'] == h and r['cfg']['seed'] == sd]
        hs = []
        for lam in LAMS:
            bestc = max((c for r in rs for c in r['curve']), key=lambda c: c['know'] - lam * c['wrong'])
            hs.append(bestc['wrong'])
        sl.append(np.polyfit(np.log(LAMS), np.log(np.maximum(hs, 1e-9)), 1)[0])
    P(f"h={h}: per-seed slopes {sl[0]:.2f}, {sl[1]:.2f}")

# ---------------------------------------------------------------- (3) Experiment C
C_ = json.load(open(R + 'exp_c.json'))
V = 16; LC = [1, 3, 9, 30]
alpha = 2.03
def metrics(c, U_out, eps_key):
    e = c[eps_key]
    know = c['know_in'] + U_out * e / V
    hall = c['wrong_in'] + U_out * e * (1 - 1 / V)
    return know, hall, e
_cache = {}
def arrs(r, eps_key):
    key = (id(r['curve']), eps_key)
    if key not in _cache:
        cur = sorted(r['curve'], key=lambda c: c['tau'])
        tau = np.array([c['tau'] for c in cur]); e = np.array([c[eps_key] for c in cur])
        know = np.array([c['know_in'] for c in cur]) + r['U_out'] * e / V
        hall = np.array([c['wrong_in'] for c in cur]) + r['U_out'] * e * (1 - 1 / V)
        hun = np.array([c['wrong_unseen'] for c in cur]) + r['U_out'] * e * (1 - 1 / V)
        _cache[key] = (tau, know, hall, e, hun)
    return _cache[key]
def at(r, T, eps_key):
    tau, know, hall, e, hun = arrs(r, eps_key)
    j = np.minimum(np.searchsorted(tau, np.asarray(T) - 1e-15, side='left'), len(tau) - 1)
    return know[j], hall[j], e[j], hun[j]
def select(runs, lam, eps_key):
    """choose (w, tau) maximising mean penalised score over `runs` (operating point = first grid tau >= candidate)"""
    byw = collections.defaultdict(list)
    for r in runs: byw[r['cfg']['w']].append(r)
    bestv, bestwt = -1e9, None
    for w, rs in byw.items():
        T = np.unique(np.concatenate([arrs(r, eps_key)[0] for r in rs]))
        v = np.mean([at(r, T, eps_key)[0] - lam * at(r, T, eps_key)[1] for r in rs], axis=0)
        j = int(np.argmax(v))
        if v[j] > bestv: bestv, bestwt = v[j], (w, T[j])
    return bestwt
def evaluate(runs, wt, lam, eps_key):
    w, tau = wt; out = []
    for r in runs:
        if r['cfg']['w'] != w: continue
        k_, h_, e, hu = at(r, tau, eps_key)
        out.append(dict(know=float(k_), h=float(h_), eps=float(e), h_unseen=float(hu), score=float(k_ - lam * h_)))
    return out
def crossfit(runs, lam, seeds):
    """2-fold over seeds: select (w,tau) on fold-1 seeds with fresh sample A, evaluate on fold-2 seeds with sample B; swap."""
    s1 = set(seeds[: len(seeds) // 2]); s2 = set(seeds[len(seeds) // 2:]); res = []
    for a, b in [(s1, s2), (s2, s1)]:
        wt = select([r for r in runs if r['cfg']['seed'] in a], lam, 'epsA')
        res += evaluate([r for r in runs if r['cfg']['seed'] in b], wt, lam, 'epsB')
    return res
summ = {}
P("\n== (3) Exp C: universe 2^20 facts + infinite tail; cross-fitted (held-out) operating points; 6 seeds ==")
for s in [1.2, 1.5]:
    rs_s = [r for r in C_ if r['cfg']['s'] == s]
    widths = sorted(set(r['cfg']['h'] for r in rs_s))
    Ps = {h: [r for r in rs_s if r['cfg']['h'] == h][0]['P'] for h in widths}
    r0 = rs_s[0]
    P(f"\n-- s={s}: U_out (never-trained tail mass) = {r0['U_out']:.4f}; mass of in-universe facts never seen in training "
      f"(median over runs) = {np.median([r['frac_mass_unseen'] for r in rs_s]):.4f}; facts seen >=1 (median) = {int(np.median([r['n_seen'] for r in rs_s]))}")
    P("h   P      lam | know          h (total)          h_open (unseen+tail)  eps* (held-out)   [mean ± 95% seed-bootstrap CI]")
    for lam in LC:
        for h in widths:
            runs = [r for r in rs_s if r['cfg']['h'] == h]
            seeds = sorted(set(r['cfg']['seed'] for r in runs))
            ev = crossfit(runs, lam, seeds)
            bs = collections.defaultdict(list)
            for _ in range(200):
                sd = list(rng.choice(seeds, len(seeds), replace=True))
                # bootstrap: resample seeds, relabel duplicates, re-run cross-fit
                rr = []
                for j, q in enumerate(sd):
                    for r in runs:
                        if r['cfg']['seed'] == q:
                            r2 = dict(r); r2['cfg'] = dict(r['cfg'], seed=100 + j); rr.append(r2)  # shares curve -> cache
                e2 = crossfit(rr, lam, list(range(100, 100 + len(sd))))
                for key in ['know', 'h', 'eps', 'h_unseen']: bs[key].append(np.mean([x[key] for x in e2]))
            m = {key: float(np.mean([x[key] for x in ev])) for key in ['know', 'h', 'eps', 'h_unseen']}
            ci = {key: (float(np.percentile(bs[key], 2.5)), float(np.percentile(bs[key], 97.5))) for key in bs}
            summ[f"{s}|{h}|{lam}"] = dict(s=s, h=h, P=Ps[h], lam=lam, mean=m, ci=ci, boot={k_: list(map(float, v)) for k_, v in bs.items()})
            P(f"{h:<4}{Ps[h]:<7}{lam:<4}| {m['know']:.3f} [{ci['know'][0]:.3f},{ci['know'][1]:.3f}]  {m['h']:.2e} [{ci['h'][0]:.1e},{ci['h'][1]:.1e}]  "
              f"{m['h_unseen']:.2e}  {m['eps']:.4f} [{ci['eps'][0]:.4f},{ci['eps'][1]:.4f}]")
# slopes vs P with bootstrap CIs
P("\nSlopes vs parameter count P (log-log, widths 16..128), mean and 95% bootstrap CI; prediction for h: -(s-1); for eps*: 0")
from theory import solve as solve_sf
slopes = {}
for s in [1.2, 1.5]:
    for lam in LC:
        ks = sorted([v for v in summ.values() if v['s'] == s and v['lam'] == lam], key=lambda v: v['P'])
        x = np.log([v['P'] for v in ks])
        out = {}
        for key in ['h', 'eps', 'h_unseen']:
            pt = np.polyfit(x, np.log([max(v['mean'][key], 1e-12) for v in ks]), 1)[0]
            bsl = [np.polyfit(x, np.log([max(v['boot'][key][b], 1e-12) for v in ks]), 1)[0] for b in range(200)]
            out[key] = (pt, np.percentile(bsl, 2.5), np.percentile(bsl, 97.5))
        # theory at C = alpha P (store-and-filter, infinite Zipf), same widths
        th = [solve_sf(alpha * v['P'], s, lam, 4) for v in ks]
        th_h = np.polyfit(x, np.log([t['halluc'] for t in th]), 1)[0]
        th_e = np.polyfit(x, np.log([t['eps'] for t in th]), 1)[0]
        slopes[f"{s}|{lam}"] = dict(nn=out, th_h=th_h, th_e=th_e, th_eps=[t['eps'] for t in th], th_hv=[t['halluc'] for t in th])
        P(f"s={s} lam={lam:<3}: h slope {out['h'][0]:+.3f} [{out['h'][1]:+.3f},{out['h'][2]:+.3f}] | eps* slope {out['eps'][0]:+.3f} "
          f"[{out['eps'][1]:+.3f},{out['eps'][2]:+.3f}] | h_open slope {out['h_unseen'][0]:+.3f} | theory(C=2.03P): h {th_h:+.3f}, eps {th_e:+.3f}; "
          f"theory eps* = {', '.join(f'{t:.4f}' for t in slopes[f'{s}|{lam}']['th_eps'])}")
# dose response: difference of slopes s=1.5 minus s=1.2
P("\nDose-response: h-slope(s=1.5) - h-slope(s=1.2)  (prediction -0.30)")
for lam in LC:
    a = summ_s = None
    d_pt = slopes[f"1.5|{lam}"]['nn']['h'][0] - slopes[f"1.2|{lam}"]['nn']['h'][0]
    # bootstrap of the difference (independent resamples)
    ks5 = sorted([v for v in summ.values() if v['s'] == 1.5 and v['lam'] == lam], key=lambda v: v['P'])
    ks2 = sorted([v for v in summ.values() if v['s'] == 1.2 and v['lam'] == lam], key=lambda v: v['P'])
    x = np.log([v['P'] for v in ks5])
    diffs = [np.polyfit(x, np.log([max(v['boot']['h'][b], 1e-12) for v in ks5]), 1)[0]
             - np.polyfit(x, np.log([max(v['boot']['h'][b], 1e-12) for v in ks2]), 1)[0] for b in range(200)]
    P(f"lam={lam}: {d_pt:+.3f} [{np.percentile(diffs, 2.5):+.3f},{np.percentile(diffs, 97.5):+.3f}]")
# lambda slopes with CIs per width
P("\nLambda-slope of held-out h* (lam 1..30), per width, mean [95% CI]; store-and-filter theory at C=2.03P")
for s in [1.2, 1.5]:
    for h in sorted(set(v['h'] for v in summ.values())):
        ks = sorted([v for v in summ.values() if v['s'] == s and v['h'] == h], key=lambda v: v['lam'])
        x = np.log([v['lam'] for v in ks])
        pt = np.polyfit(x, np.log([v['mean']['h'] for v in ks]), 1)[0]
        bsl = [np.polyfit(x, np.log([max(v['boot']['h'][b], 1e-12) for v in ks]), 1)[0] for b in range(200)]
        th = np.polyfit(x, np.log([solve_sf(alpha * ks[0]['P'], s, l, 4)['halluc'] for l in LC]), 1)[0]
        P(f"s={s} h={h}: {pt:+.2f} [{np.percentile(bsl,2.5):+.2f},{np.percentile(bsl,97.5):+.2f}]  theory {th:+.2f}")

json.dump(dict(summary={k: {kk: vv for kk, vv in v.items() if kk != 'boot'} for k, v in summ.items()},
               slopes={k: dict(nn={kk: list(map(float, vv)) for kk, vv in v['nn'].items()}, th_h=v['th_h'], th_e=v['th_e'],
                               th_eps=v['th_eps'], th_hv=v['th_hv']) for k, v in slopes.items()}),
          open(R + 'exp_c_summary.json', 'w'), indent=1)
open(R + 'analysis_rev.txt', 'w').write("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for s, mk in [(1.2, 'o'), (1.5, 's')]:
    for lam, c in zip(LC, plt.cm.viridis(np.linspace(0, 0.85, len(LC)))):
        ks = sorted([v for v in summ.values() if v['s'] == s and v['lam'] == lam], key=lambda v: v['P'])
        Pp = [v['P'] for v in ks]
        for j, key in enumerate(['h', 'eps']):
            m = np.array([v['mean'][key] for v in ks]); lo = np.array([v['ci'][key][0] for v in ks]); hi = np.array([v['ci'][key][1] for v in ks])
            ax[j].errorbar(Pp, m, yerr=[m - lo, hi - m], marker=mk, color=c, ls='-' if s == 1.2 else '--', capsize=2,
                           label=f"s={s}, λ={lam}" if j == 0 else None)
            th = slopes[f"{s}|{lam}"]['th_hv' if key == 'h' else 'th_eps']
            ax[j].plot(Pp, th, color=c, ls=':', lw=1)
    ks = sorted([v for v in summ.values() if v['s'] == s and v['lam'] == 9], key=lambda v: v['P'])
    ax[2].errorbar([v['P'] for v in ks], [v['mean']['know'] for v in ks],
                   yerr=[[v['mean']['know'] - v['ci']['know'][0] for v in ks], [v['ci']['know'][1] - v['mean']['know'] for v in ks]],
                   marker=mk, label=f"s={s}, λ=9", capsize=2)
for a in ax[:2]: a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('parameters P')
ax[0].set_ylabel('held-out hallucination h* (incl. open-world tail)'); ax[0].legend(fontsize=6, ncol=2)
ax[1].set_ylabel('held-out false-accept ε*'); ax[1].set_title('dotted: store-and-filter theory, C=2.03P')
ax[2].set_xscale('log'); ax[2].set_xlabel('parameters P'); ax[2].set_ylabel('knowledge'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(R + 'fig_exp_c.png', dpi=150)
