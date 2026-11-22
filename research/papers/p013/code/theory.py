"""Theorem 2 (capacity-constrained optimal abstention under penalised grading).
Facts i=1,2,... queried with Zipf probabilities p_i = i^-s / zeta(s); values uniform on V=2^H.
A model with C bits stores the top-k facts (H bits each) plus a filter with false-accept
rate eps = 2^-f (f bits per stored fact): k (H + f) <= C  [tight by Thm 1 + Dietzfelbinger-Pagh].
Score: +1 correct, 0 abstain, -lam wrong.  beta = lam (1-1/V) - 1/V = expected loss of a guess.
J(k) = M(k) - beta U(k) eps(k),  eps(k) = min(1, 2^(H - C/k)).   Outputs results/theory_*.json + figures."""
import json, math, numpy as np
from scipy.special import zeta
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def U_of_k(k, s):      # residual (unstored) mass for top-k storage, k may be real (interpolated)
    return zeta(s, np.asarray(k, float) + 1.0) / zeta(s)

def solve(C, s, lam, H, ngrid=4000):
    V = 2.0 ** H; beta = lam * (1 - 1 / V) - 1 / V
    kmax = C / H
    if beta <= 0:   # guessing is profitable: never abstain, spend all bits on values
        k = math.floor(kmax); U = float(U_of_k(k, s))
        return dict(k=k, eps=1.0, f=0.0, U=U, halluc=(1 - 1 / V) * U, coverage=1.0,
                    knowledge=1 - U, score=1 - U + U * (1 / V - lam * (1 - 1 / V)), beta=beta)
    ks = np.unique(np.floor(np.geomspace(1, kmax, ngrid)))
    eps = np.minimum(1.0, 2.0 ** (H - C / ks)); U = U_of_k(ks, s)
    J = (1 - U) - beta * U * eps
    i = int(np.argmax(J))
    # local integer refinement
    lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    kk = np.arange(lo, hi + 1) if hi - lo < 200000 else ks[max(i-1,0):i+2]
    e2 = np.minimum(1.0, 2.0 ** (H - C / kk)); U2 = U_of_k(kk, s); J2 = (1 - U2) - beta * U2 * e2
    j = int(np.argmax(J2)); k, e, Uk, Jk = float(kk[j]), float(e2[j]), float(U2[j]), float(J2[j])
    if Jk <= 0:   # trivial all-abstain policy is optimal
        return dict(k=0, eps=0.0, f=float('inf'), U=1.0, halluc=0.0, coverage=0.0, knowledge=0.0, score=0.0, beta=beta)
    return dict(k=k, eps=e, f=-math.log2(e), U=Uk, halluc=(1 - 1 / V) * Uk * e,
                coverage=(1 - Uk) + Uk * e, knowledge=1 - Uk, score=Jk, beta=beta)

def eps_asymptotic(s, lam, H):
    """Fixed point eps = (s-1) / (beta [ln2 (H + log2 1/eps) - (s-1)])  (Zipf tail, k m(k) ~ (s-1) U(k))."""
    V = 2.0 ** H; beta = lam * (1 - 1 / V) - 1 / V
    if beta <= 0: return 1.0
    e = 0.1
    for _ in range(200):
        den = beta * (math.log(2) * (H - math.log2(e)) - (s - 1))
        e = min(1.0, (s - 1) / den) if den > 0 else 1.0
    return e

if __name__ == '__main__':
    out = {}
    lams = [0.0, 0.1, 0.3, 1, 3, 9, 30, 100, 1000]
    Cs = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    rows = []
    for s in [1.05, 1.1, 1.2, 1.5, 2.0]:
        for H in [4, 10, 20]:
            for lam in lams:
                for C in Cs:
                    r = solve(C, s, lam, H); r.update(s=s, H=H, lam=lam, C=C, eps_asym=eps_asymptotic(s, lam, H))
                    rows.append(r)
    out['grid'] = rows
    json.dump(out, open('../results/theory_grid.json', 'w'))

    # ---- summary tables printed + saved
    lines = []
    def P(x): print(x); lines.append(x)
    P("Table T1: optimal false-accept rate eps* vs capacity C (s=1.1, H=10). Capacity-invariance prediction.")
    P("lam | " + " | ".join(f"C=1e{int(math.log10(C))}" for C in Cs) + " | asymptotic")
    for lam in [1, 3, 9, 30, 100]:
        rr = [r for r in rows if r['s'] == 1.1 and r['H'] == 10 and r['lam'] == lam]
        P(f"{lam} | " + " | ".join(f"{r['eps']:.4f}" for r in rr) + f" | {rr[0]['eps_asym']:.4f}")
    P("\nTable T2: optimal hallucination rate h* (s=1.1, H=10)")
    P("lam | " + " | ".join(f"C=1e{int(math.log10(C))}" for C in Cs))
    for lam in lams:
        rr = [r for r in rows if r['s'] == 1.1 and r['H'] == 10 and r['lam'] == lam]
        P(f"{lam} | " + " | ".join(f"{r['halluc']:.4f}" for r in rr))
    P("\nTable T3: log-log slope of h* vs C (fit over C=1e7..1e11, lam=9) vs predicted -(s-1)")
    for s in [1.05, 1.1, 1.2, 1.5, 2.0]:
        for H in [4, 10, 20]:
            rr = [r for r in rows if r['s'] == s and r['H'] == H and r['lam'] == 9 and r['C'] >= 1e7]
            x = np.log([r['C'] for r in rr]); y = np.log([r['halluc'] for r in rr])
            P(f"s={s} H={H}: slope={np.polyfit(x, y, 1)[0]:.3f} (pred {-(s-1):.3f})")
    P("\nTable T4: log-log slope of h* vs lam (lam=3..1000, C=1e9)")
    for s in [1.1, 1.5]:
        for H in [4, 10, 20]:
            rr = [r for r in rows if r['s'] == s and r['H'] == H and r['lam'] >= 3 and r['C'] == 1e9]
            x = np.log([r['lam'] for r in rr]); y = np.log([r['halluc'] for r in rr])
            P(f"s={s} H={H}: slope={np.polyfit(x, y, 1)[0]:.3f}")
    # accuracy of the asymptotic eps
    errs = [abs(math.log(r['eps']) - math.log(r['eps_asym'])) for r in rows if r['lam'] >= 1 and r['C'] >= 1e8 and r['eps'] < 1]
    P(f"\nMedian |log eps* - log eps_asym| over lam>=1, C>=1e8: {np.median(errs):.3f}; max {np.max(errs):.3f}")

    # ---- price of hallucination frontier at fixed C: knowledge vs hallucination as f varies
    fr = {}
    for H in [4, 10, 20]:
        for s in [1.1, 1.5]:
            C = 1e9; fs = np.linspace(0, 40, 161); ks = C / (H + fs); U = U_of_k(ks, s)
            fr[f"H{H}_s{s}"] = dict(f=fs.tolist(), knowledge=(1 - U).tolist(),
                                    halluc=((1 - 2.0 ** -H) * U * 2.0 ** -fs).tolist(), k=ks.tolist())
    json.dump(fr, open('../results/theory_frontier.json', 'w'))
    P("\nTable T5: price of lowering hallucination at C=1e9 bits: knowledge (answered-correctly mass) at given eps")
    for key, d in fr.items():
        f = np.array(d['f']); kn = np.array(d['knowledge']); hl = np.array(d['halluc'])
        sel = [0, 20, 40, 80, 160]
        P(key + ": " + "; ".join(f"f={f[i]:.0f}: know={kn[i]:.4f}, h={hl[i]:.2e}" for i in sel))

    P("\nTable T6: share of capacity spent on the abstention filter, f*/(H+f*), at C=1e9")
    P("s   H  | " + " | ".join(f"lam={l}" for l in [1, 3, 9, 30, 100]))
    for s_ in [1.1, 1.5, 2.0]:
        for H_ in [4, 10, 20]:
            rr = [r for r in rows if r['s'] == s_ and r['H'] == H_ and r['C'] == 1e9 and r['lam'] in (1, 3, 9, 30, 100)]
            P(f"{s_} {H_:<3}| " + " | ".join(f"{r['f']/(H_+r['f']):.2f}" for r in rr))
    open('../results/theory_tables.txt', 'w').write("\n".join(lines))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for lam, c in zip([1, 3, 9, 30, 100], plt.cm.viridis(np.linspace(0, 0.9, 5))):
        rr = [r for r in rows if r['s'] == 1.1 and r['H'] == 10 and r['lam'] == lam]
        ax[0].loglog([r['C'] for r in rr], [r['halluc'] for r in rr], 'o-', color=c, label=f"λ={lam}")
        ax[1].semilogx([r['C'] for r in rr], [r['eps'] for r in rr], 'o-', color=c)
        ax[1].axhline(rr[0]['eps_asym'], color=c, ls=':')
    ax[0].set_xlabel('capacity C (bits)'); ax[0].set_ylabel('optimal hallucination rate h*'); ax[0].legend(fontsize=8)
    ax[0].set_title('h* > 0 for every finite C (s=1.1, H=10)')
    ax[1].set_yscale('log'); ax[1].set_xlabel('capacity C (bits)'); ax[1].set_ylabel('optimal false-accept rate ε*')
    ax[1].set_title('ε* is capacity-invariant (dotted: fixed point)')
    for key, d in fr.items():
        ax[2].loglog(d['halluc'], 1 - np.array(d['knowledge']), label=key)
    ax[2].set_xlabel('hallucination rate'); ax[2].set_ylabel('unanswered-correctly mass 1-M(k)')
    ax[2].set_title('Price frontier at C=1e9 bits'); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig('../results/fig_theory.png', dpi=150)
