"""Information-theoretic relaxation (Theorem 3) for ARBITRARY B-bit closed-book models.

Theorem 3 (paper): facts (X_i, Y_i) independent, X_i uniform keys, Y_i uniform on [V].  For any
model M (any architecture, any learner) with H(M) <= B, let eps = Pr[M accepts a fresh key independent
of M] and, for fact i, P_i = (Pr[reject x_i], Pr[accept & wrong], Pr[accept & correct]).  Then
    sum_i KL( P_i || Q(eps) ) <= I(M; D) <= B,     Q(eps) = (1-eps, eps(1-1/V), eps/V)   [bits].
So max_{P, eps} sum_i p_i [ r_i - lam (q_i - r_i) ]  s.t. sum_i KL <= C  is an UPPER bound on the
penalised score of every C-bit model.  For fixed eps it is convex; the solution is an exponential tilt
(Gibbs) P_i,o ∝ Q_o exp(a_{i,o}/nu), a_i = (0, -lam p_i, +p_i), with nu set by the budget.

Outputs results/relax_tables.txt, results/relax.json, results/fig_relax.png"""
import json, math, numpy as np
from scipy.special import zeta, logsumexp
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory import solve as solve_sf, eps_asymptotic

LN2 = math.log(2)

def fact_bins(s, N=None, I0=1000, per_dec=60, Imax=1e19):
    """Representative query probability p_b and multiplicity m_b for facts 1..N (N=None: infinite Zipf).
    Exact for i<=I0, log-spaced bins after; bin p = exact bin mass / count. Last element: lump tail."""
    Z = zeta(s) if N is None else zeta(s) - zeta(s, N + 1.0)
    top = Imax if N is None else float(N)
    i = np.arange(1, min(I0, top) + 1, dtype=float)
    p = list(i ** -s / Z); m = list(np.ones_like(i))
    if top > I0:
        edges = np.unique(np.floor(np.geomspace(I0 + 1, top + 1, int(per_dec * math.log10(top / I0)) + 2)))
        lo, hi = edges[:-1], edges[1:]
        mass = (zeta(s, lo) - zeta(s, hi)) / Z; cnt = hi - lo
        p += list(mass / cnt); m += list(cnt)
    tail = 0.0 if N is not None else float(zeta(s, top + 1.0) / Z)   # treated at q=eps (zero cost; 2nd-order exact)
    return np.array(p), np.array(m), tail

def gibbs(p, eps, V, lam, nu, zero_h=False):
    """Per-fact optimal 3-outcome distribution and its KL (bits). Returns r, wrong, kl."""
    lq = np.array([math.log(1 - eps), math.log(eps * (1 - 1 / V)) if not zero_h else -np.inf, math.log(eps / V)])
    a = np.stack([np.zeros_like(p), -lam * p, p], 1) / nu
    L = lq[None, :] + a
    lz = logsumexp(L, axis=1, keepdims=True)
    logP = L - lz; P = np.exp(logP)
    with np.errstate(invalid='ignore'):
        kl = np.where(P > 0, P * (logP - lq[None, :]), 0.0).sum(1) / LN2
    return P[:, 2], P[:, 1], kl

def relax_at_eps(p, m, tail, C, eps, V, lam, zero_h=False):
    """Max score at fixed eps with total information C bits (bisection on temperature nu)."""
    def cost(lognu):
        r, w, kl = gibbs(p, eps, V, lam, math.exp(lognu), zero_h); return float((m * kl).sum())
    lo, hi = math.log(1e-30), math.log(1e3)
    if cost(hi) > C: return None
    if cost(lo) <= C: hi = lo
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        if cost(mid) > C: lo = mid
        else: hi = mid
    r, w, kl = gibbs(p, eps, V, lam, math.exp(hi), zero_h)
    know = float((m * p * r).sum()) + (0 if zero_h else tail * eps / V)
    hall = float((m * p * w).sum()) + (0 if zero_h else tail * eps * (1 - 1 / V))
    return dict(know=know, halluc=hall, score=know - lam * hall, eps=eps, nu=math.exp(hi),
                bits=float((m * kl).sum()))

def relax_opt(s, H, lam, C, N=None, zero_h=False, bins=None):
    V = 2.0 ** H
    p, m, tail = bins if bins is not None else fact_bins(s, N)
    obj = lambda le: -(relax_at_eps(p, m, tail, C, 10 ** le, V, lam, zero_h) or dict(score=-1e9))['score']
    grid = np.linspace(-14, -1e-3, 22)
    vals = [obj(g) for g in grid]; j = int(np.argmin(vals))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    res = minimize_scalar(obj, bounds=(a, b), method='bounded', options=dict(xatol=3e-4))
    return relax_at_eps(p, m, tail, C, 10 ** res.x, V, lam, zero_h)

def rate_row(key):
    s, H, lam, C = key
    sf = solve_sf(C, s, lam, H); rx = relax_opt(s, H, lam, C, bins=fact_bins(s))
    return dict(s=s, H=H, lam=lam, C=C, sf_eps=sf['eps'], sf_h=sf['halluc'], sf_loss=1 - sf['score'],
                sf_know=sf['knowledge'], rx_eps=rx['eps'], rx_h=rx['halluc'], rx_loss=1 - rx['score'],
                rx_know=rx['know'], rx_bits=rx['bits'])

LAMG = np.geomspace(0.3, 3000, 14)
def frontier_row(key):
    s, H, C = key
    bins = fact_bins(s); sf = solve_sf(C, s, 9, H); K0 = sf['knowledge']
    Js = [relax_opt(s, H, l, C, bins=bins)['score'] for l in LAMG]
    hlb = max(max((K0 - J) / l for J, l in zip(Js, LAMG)), 0.0)
    return dict(s=s, H=H, C=C, K0=K0, h_sf=sf['halluc'], h_lb=hlb)

def zero_row(key):
    s, H, N = key
    bins = fact_bins(s, N=N)
    z = relax_opt(s, H, 0.0, 1e6, N=N, zero_h=True, bins=bins)
    hi = relax_opt(s, H, 1e4, 1e6, N=N, bins=bins)
    # achievable: store top-k values + exact membership inside the N-universe: k(H + log2(e N/k)) <= C
    Zs = zeta(s) - zeta(s, N + 1.0)
    ks = np.unique(np.floor(np.geomspace(1, min(N, 1e6), 3000)))
    cost = ks * H + ks * np.log2(math.e * N / ks)
    kbest = ks[cost <= 1e6].max()
    Ksf = float((zeta(s) - zeta(s, kbest + 1)) / Zs)
    return dict(s=s, H=H, N=N, K0=z['know'], Ksf=Ksf, Khi=hi['know'], hhi=hi['halluc'])

if __name__ == '__main__':
    out = dict(rate=[], zero=[], frontier=[])
    lines = []
    def P_(x=''): print(x); lines.append(str(x))
    Cs = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    P_("Table R1: information-theoretic upper bound (any C-bit model, Thm 3) vs store-and-filter (SF, achievable)")
    P_("s    H  lam | C      | SF: eps*     h*        loss=1-J | RELAX: eps*   h*        loss=1-J | loss ratio SF/RELAX")
    import itertools, multiprocessing as mp
    pool = mp.Pool(4)
    keys = list(itertools.product([1.1, 1.2, 1.5, 2.0], [4, 10], [1, 9, 100], Cs))
    out['rate'] = pool.map(rate_row, keys)
    for row in out['rate']:
        s, H, lam, C = row['s'], row['H'], row['lam'], row['C']
        if lam == 9 or C in (1e4, 1e7, 1e10):
            P_(f"{s:<4} {H:<2} {lam:<3} | {C:.0e} | {row['sf_eps']:.2e}  {row['sf_h']:.3e}  {row['sf_loss']:.4f} | "
               f"{row['rx_eps']:.2e}  {row['rx_h']:.3e}  {row['rx_loss']:.4f} | {row['sf_loss']/row['rx_loss']:.3f}")
    P_("\nTable R2: log-log slopes vs C over C=1e6..1e10 (prediction -(s-1)); eps* drift = max/min over same range")
    P_("s    H  lam | slope loss SF | slope loss RELAX | slope h RELAX | slope h SF | eps*_RELAX max/min | eps*_SF max/min")
    for s in [1.1, 1.2, 1.5, 2.0]:
        for H in [4, 10]:
            for lam in [1, 9, 100]:
                rr = [r for r in out['rate'] if r['s'] == s and r['H'] == H and r['lam'] == lam and r['C'] >= 1e6]
                x = np.log([r['C'] for r in rr])
                sl = lambda key: np.polyfit(x, np.log([r[key] for r in rr]), 1)[0]
                e1 = [r['rx_eps'] for r in rr]; e2 = [r['sf_eps'] for r in rr]
                P_(f"{s:<4} {H:<2} {lam:<3} | {sl('sf_loss'):.3f} | {sl('rx_loss'):.3f} | {sl('rx_h'):.3f} | {sl('sf_h'):.3f} | "
                   f"{max(e1)/min(e1):.3f} | {max(e2)/min(e2):.3f}   (pred {-(s-1):.2f})")

    # ---- hallucination lower bound for ANY model at the knowledge level of the optimal SF policy
    P_("\nTable R3: min hallucination of ANY C-bit model that knows as much as the optimal SF policy (lam=9):")
    P_("h_LB(K0) = max_lam' (K0 - J'_relax(lam'))/lam'   [weak duality], K0 = SF knowledge")
    P_("s    H  | C     | K0       | h_SF      | h_LB(any model) | ratio")
    out['frontier'] = pool.map(frontier_row, list(itertools.product([1.2, 1.5], [4, 10], [1e5, 1e6, 1e7, 1e8, 1e9])))
    for r in out['frontier']:
        P_(f"{r['s']:<4} {r['H']:<2} | {r['C']:.0e} | {r['K0']:.5f} | {r['h_sf']:.3e} | {r['h_lb']:.3e} | {r['h_sf']/max(r['h_lb'],1e-300):.2f}")
    for s in [1.2, 1.5]:
        for H in [4, 10]:
            rr = [r for r in out['frontier'] if r['s'] == s and r['H'] == H and r['h_lb'] > 0]
            if len(rr) >= 3:
                sl = np.polyfit(np.log([r['C'] for r in rr]), np.log([r['h_lb'] for r in rr]), 1)[0]
                P_(f"slope of h_LB vs C (s={s}, H={H}): {sl:.3f}  (pred {-(s-1):.2f})")

    # ---- zero-hallucination knowledge in a finite queryable universe of N facts
    P_("\nTable R4: max knowledge of ANY C-bit model with ZERO hallucination (Cor. 3), Zipf(s) over N facts")
    P_("C=1e6 bits. Compare: SF exact-membership k(H+log2(N/k)) construction, and RELAX at lam=1e4 (h tiny but >0)")
    P_("s    H  | N      | K_0(any model, h=0) | K_SFexact | K_relax(lam=1e4) | h at lam=1e4")
    out['zero'] = pool.map(zero_row, list(itertools.product([1.2, 1.5], [4, 10], [1e5, 1e6, 1e8, 1e10, 1e12, 1e15])))
    for r in out['zero']:
        P_(f"{r['s']:<4} {r['H']:<2} | {r['N']:.0e} | {r['K0']:.4f} | {r['Ksf']:.4f} | {r['Khi']:.4f} | {r['hhi']:.2e}")
    json.dump(out, open('../results/relax.json', 'w'))
    open('../results/relax_tables.txt', 'w').write("\n".join(lines))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s, c in zip([1.1, 1.2, 1.5, 2.0], ['C0', 'C1', 'C2', 'C3']):
        rr = [r for r in out['rate'] if r['s'] == s and r['H'] == 10 and r['lam'] == 9]
        ax[0].loglog([r['C'] for r in rr], [r['sf_loss'] for r in rr], 'o-', color=c, label=f's={s} SF')
        ax[0].loglog([r['C'] for r in rr], [r['rx_loss'] for r in rr], 's--', color=c, label=f's={s} bound')
        ax[1].semilogx([r['C'] for r in rr], [r['rx_eps'] for r in rr], 's--', color=c)
        ax[1].semilogx([r['C'] for r in rr], [r['sf_eps'] for r in rr], 'o-', color=c)
    ax[0].set_xlabel('capacity C (bits)'); ax[0].set_ylabel('penalised loss 1-J (λ=9, H=10)'); ax[0].legend(fontsize=6)
    ax[0].set_title('Any model (dashed bound) vs store-and-filter')
    ax[1].set_yscale('log'); ax[1].set_xlabel('capacity C (bits)'); ax[1].set_ylabel('background false-accept ε*')
    ax[1].set_title('ε* capacity-invariant in both')
    for s, H, c in [(1.2, 10, 'C1'), (1.5, 10, 'C2'), (1.2, 4, 'C4')]:
        rr = [r for r in out['zero'] if r['s'] == s and r['H'] == H]
        ax[2].semilogx([r['N'] for r in rr], [r['K0'] for r in rr], 'o-', color=c, label=f's={s},H={H}: h=0 bound')
        ax[2].semilogx([r['N'] for r in rr], [r['Khi'] for r in rr], 's:', color=c, label='λ=1e4 bound')
    ax[2].set_xlabel('queryable universe N'); ax[2].set_ylabel('max knowledge (C=1e6 bits)'); ax[2].legend(fontsize=6)
    ax[2].set_title('Exact zero costs log N per fact')
    plt.tight_layout(); plt.savefig('../results/fig_relax.png', dpi=150)
