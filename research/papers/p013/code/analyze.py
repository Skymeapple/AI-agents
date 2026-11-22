"""Analysis + figures for Experiments A and B (reads results/exp_a.json, exp_b.json)."""
import json, math, numpy as np, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '../results/'
lines = []
def P(x=''): print(x); lines.append(str(x))

# ------------------------------------------------------------------ Experiment A
A = json.load(open(R + 'exp_a.json'))
A1 = [r for r in A if r['cfg']['block'] == 'A1']; A2 = [r for r in A if r['cfg']['block'] == 'A2']
P("== Experiment A1: V=16 (H=4 bits), softmax head with ABSTAIN class ==")
P("h  k    w     | default thr: g/k   eps     wrong | max certified bits/param (mean over 2 seeds)")
grp = collections.defaultdict(list)
for r in A1:
    c = r['cfg']; grp[(c['h'], c['k'], c['w'])].append(r)
capacity = collections.defaultdict(list)
for key in sorted(grp):
    rs = grp[key]; h, k, w = key
    g = np.mean([r['default']['g'] for r in rs]) / k; e = np.mean([r['default']['eps'] for r in rs])
    wr = np.mean([r['default']['wrong_stored'] for r in rs])
    cb = np.mean([r['best']['cert_bits'] / r['P'] for r in rs])
    capacity[h].append(max(r['best']['cert_bits'] for r in rs) / rs[0]['P'])
    P(f"{h:<3}{k:<5}{w:<6}| {g:.3f}  {e:.4f}  {wr:.0f} | {cb:.2f}")
alpha = {h: float(np.max(v)) for h, v in capacity.items()}
P(f"Capacity estimate alpha_h = max certified bits/param over runs: {alpha}")
allmax = [r['best']['cert_bits'] / r['P'] for r in A1]
P(f"Certified bits/param across all A1 runs: median {np.median(allmax):.2f}, max {np.max(allmax):.2f}")

# robustness: recompute certified bits with a one-sided 95% Clopper-Pearson upper bound on eps
from scipy.stats import beta as _beta
from common import certified_bits
def eps_ucb(e, n=200000):
    x = round(e * n)
    return 1.0 if x >= n else float(_beta.ppf(0.95, x + 1, n - x))
cp = collections.defaultdict(list)
for r in A1:
    best = max(certified_bits(c['g'], r['cfg']['k'], eps_ucb(c['eps']), r['cfg']['V']) for c in r['curve'])
    cp[r['cfg']['h']].append(best / r['P'])
P("Robustness (95% Clopper-Pearson upper bound on eps): max certified bits/param " +
  ", ".join(f"h={h}: {np.max(v):.2f} (median {np.median(v):.2f})" for h, v in sorted(cp.items())))

# zero-hallucination test: largest recall with NO false accept among 200k fresh keys
P("\nZero-observed-false-accept operating points (eps_hat=0 of 200k => eps<1.5e-5 @95%):")
zrows = collections.defaultdict(list)
for r in A1:
    zs = [c['g'] for c in r['curve'] if c['eps'] == 0.0]
    zrows[(r['cfg']['h'], r['cfg']['k'])].append(max(zs) if zs else 0)
for key in sorted(zrows):
    P(f"h={key[0]} k={key[1]}: max g with zero FP = {max(zrows[key])} ({max(zrows[key])/key[1]:.3f} of k); median over runs {np.median(zrows[key]):.0f}")

# efficiency of trained nets vs. the bound as a function of filter strength f
P("\nCertified bits/param achieved vs filter strength f=log2(1/eps) (A1, max over runs in bin; only points with g>=50):")
for h in [32, 64]:
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9)]
    row = []
    for lo, hi in bins:
        v = [c['cert_bits'] / r['P'] for r in A1 if r['cfg']['h'] == h for c in r['curve']
             if c['g'] >= 50 and lo <= math.log2(1 / max(c['eps'], 3 / 200000)) < hi]
        row.append(f"[{lo},{hi}): {max(v):.2f}" if v else f"[{lo},{hi}): -")
    P(f"h={h}: " + "; ".join(row))

P("\n== Experiment A2: value-entropy sweep (k=2048, h=32, bit-output head) ==")
P("H bits | P    | max cert bits/param (mean over w,seeds) | best | value-bits share at best point")
a2 = collections.defaultdict(list)
for r in A2: a2[r['cfg']['V']].append(r)
a2sum = {}
for V in sorted(a2):
    rs = a2[V]; H = math.log2(V)
    vals = [r['best']['cert_bits'] / r['P'] for r in rs]
    b = max(rs, key=lambda r: r['best']['cert_bits'])['best']
    e = max(b['eps'], 3 / 200000); share = H / (H + math.log2(1 / e))
    a2sum[V] = (np.mean(vals), np.max(vals))
    P(f"{H:.0f}      | {rs[0]['P']} | {np.mean(vals):.2f} | {np.max(vals):.2f} | {share:.2f} (g={b['g']}, eps={b['eps']:.4f})")

# Fig A
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cols = {1024: 'C0', 2048: 'C1', 4096: 'C2'}
for j, h in enumerate([32, 64]):
    for r in A1:
        if r['cfg']['h'] != h: continue
        k = r['cfg']['k']
        f = [math.log2(1 / max(c['eps'], 3 / 200000)) for c in r['curve']]
        g = [c['g'] for c in r['curve']]
        ax[j].scatter(f, g, s=4, color=cols[k], alpha=0.5)
    Ph = [r['P'] for r in A1 if r['cfg']['h'] == h][0]
    fs = np.linspace(0, 16, 100); C = alpha[h] * Ph
    for k, c in cols.items():
        ax[j].plot(fs, np.minimum(k, C / (4 + fs)), color=c, lw=1.5, label=f"k={k}: min(k, Ĉ/(H+f))")
    ax[j].set_xlabel('f = log2(1/ε)  (membership bits per fact)'); ax[j].set_ylabel('g = facts answered correctly')
    ax[j].set_title(f'h={h} (P={Ph}), Ĉ={alpha[h]:.2f}·P bits'); ax[j].legend(fontsize=7)
Hs = [math.log2(V) for V in sorted(a2sum)]
ax[2].plot(Hs, [a2sum[V][0] for V in sorted(a2sum)], 'o-', label='mean over w, seeds')
ax[2].plot(Hs, [a2sum[V][1] for V in sorted(a2sum)], 's--', label='best run')
ax[2].set_xlabel('value entropy H (bits)'); ax[2].set_ylabel('certified bits / parameter'); ax[2].set_ylim(0, None)
ax[2].set_title('A2: value bits vs membership bits exchange'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(R + 'fig_exp_a.png', dpi=150)

# ------------------------------------------------------------------ Experiment B
try:
    B = json.load(open(R + 'exp_b.json'))
except FileNotFoundError:
    B = None
if B:
    from exp_b_penalty import NU, S, V, LAMS, zipf_p
    p = zipf_p(); Ucum = 1 - np.cumsum(p); H = math.log2(V)
    def solve_finite(C, lam):
        beta = lam * (1 - 1 / V) - 1 / V
        ks = np.arange(1, min(NU, int(C / H)) + 1); U = Ucum[ks - 1]
        eps = np.minimum(1, 2.0 ** (H - C / ks)) if beta > 0 else np.ones_like(U)
        if beta <= 0: ks = ks[-1:]; U = U[-1:]; eps = eps[-1:]
        J = (1 - U) - beta * U * eps; i = int(np.argmax(J))
        if J[i] <= 0: return dict(h=0.0, eps=0.0, know=0.0)
        return dict(h=(1 - 1 / V) * U[i] * eps[i], eps=float(eps[i]), know=float(1 - U[i]), k=int(ks[i]))
    P("\n== Experiment B: penalised grading, Zipf(s=%.1f) over %d facts, V=%d ==" % (S, NU, V))
    P("Per (h, lam): best (w, threshold) per seed, mean over 2 seeds;  theory uses C = alpha*P (alpha from A1)")
    alpha_glob = float(np.median(list(alpha.values())))
    P(f"alpha used for prediction: {alpha_glob:.2f} bits/param")
    P("h   P      lam  | NN: know   halluc   eps*     | theory: know   halluc   eps*")
    tabB = collections.defaultdict(dict)
    byh = collections.defaultdict(list)
    for r in B: byh[r['cfg']['h']].append(r)
    for h in sorted(byh):
        Ph = byh[h][0]['P']
        for lam in LAMS:
            res = []
            for seed in [0, 1]:
                best = None
                for r in byh[h]:
                    if r['cfg']['seed'] != seed: continue
                    for c in r['curve']:
                        sc = c['know'] - lam * c['wrong']
                        if best is None or sc > best[0]: best = (sc, c, r['cfg']['w'])
                res.append(best)
            kn = np.mean([b[1]['know'] for b in res]); hl = np.mean([b[1]['wrong'] for b in res]); ep = np.mean([b[1]['eps'] for b in res])
            th = solve_finite(alpha_glob * Ph, lam)
            tabB[h][lam] = dict(know=kn, halluc=hl, eps=ep, w=[b[2] for b in res], th=th, P=Ph)
            P(f"{h:<4}{Ph:<7}{lam:<5}| {kn:.3f}  {hl:.5f}  {ep:.5f}  | {th['know']:.3f}  {th['h']:.5f}  {th['eps']:.5f}   (w*={[b[2] for b in res]})")
    # slopes vs lam
    P("\nlog-log slope of h* vs lam over lam in [1,100]:")
    for h in sorted(tabB):
        L = [l for l in LAMS if 1 <= l <= 100]
        y = [tabB[h][l]['halluc'] for l in L]; yt = [tabB[h][l]['th']['h'] for l in L]
        if min(y) > 0:
            P(f"h={h}: NN slope {np.polyfit(np.log(L), np.log(y), 1)[0]:.2f}; theory slope {np.polyfit(np.log(L), np.log(yt), 1)[0]:.2f}")
        else:
            P(f"h={h}: NN reaches zero measured hallucination at some lam; values {y}")
    P("\nRatio of eps* across model sizes at fixed lam (capacity-invariance prediction: ratio ~1):")
    hs = sorted(tabB)
    for lam in [1, 3, 9, 30]:
        P(f"lam={lam}: eps* = " + ", ".join(f"h={h}: {tabB[h][lam]['eps']:.5f}" for h in hs) +
          "  | theory: " + ", ".join(f"{tabB[h][lam]['th']['eps']:.5f}" for h in hs))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for h, c in zip(hs, ['C0', 'C1', 'C2']):
        L = [l for l in LAMS if l > 0]
        ax[0].loglog(L, [max(tabB[h][l]['halluc'], 1e-7) for l in L], 'o-', color=c, label=f'NN h={h} (P={tabB[h][1]["P"]})')
        ax[0].loglog(L, [max(tabB[h][l]['th']['h'], 1e-7) for l in L], ':', color=c)
        ax[1].loglog(L, [max(tabB[h][l]['eps'], 1e-6) for l in L], 'o-', color=c)
        ax[1].loglog(L, [max(tabB[h][l]['th']['eps'], 1e-6) for l in L], ':', color=c)
        ax[2].semilogx(L, [tabB[h][l]['know'] for l in L], 'o-', color=c)
        ax[2].semilogx(L, [tabB[h][l]['th']['know'] for l in L], ':', color=c)
    ax[0].set_xlabel('penalty λ'); ax[0].set_ylabel('optimal hallucination rate'); ax[0].legend(fontsize=7); ax[0].set_title('solid: trained MLPs, dotted: theory')
    ax[1].set_xlabel('penalty λ'); ax[1].set_ylabel('optimal false-accept rate ε*')
    ax[2].set_xlabel('penalty λ'); ax[2].set_ylabel('knowledge (mass answered correctly)')
    plt.tight_layout(); plt.savefig(R + 'fig_exp_b.png', dpi=150)
    json.dump({str(h): {str(l): v for l, v in d.items()} for h, d in tabB.items()}, open(R + 'exp_b_summary.json', 'w'), indent=1, default=float)
open(R + 'analysis_tables.txt', 'w').write("\n".join(lines))
