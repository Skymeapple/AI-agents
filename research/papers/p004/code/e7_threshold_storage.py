"""E7: When does Prop. 2's invariance fail? Threshold storage (analytic + numeric).

Generalized storage law: each exposure yields an independent Bernoulli(s)
'storage event'; a fact is stored iff it accumulates >= m events (m=1 is Prop. 2).
Plain corpus: events ~ Poisson(s*lambda), lambda = N p_i.
Source-grounded r-fold paraphrase: each human exposure gives Bin(1+r, s) events,
so the total is compound Poisson (computed exactly by Panjer's recursion).
We compute the multiplier M by inverting the plain accuracy curve and show that
for m >= 2 it depends on the frequency distribution (alpha) and corpus size N,
and that rare facts can get M > 1/s (the m=1 ceiling).
"""
import json, os, numpy as np
from scipy import optimize, stats

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
K, KCLS = 20000, 10

def zipf(alpha, K=K):
    w = np.arange(1, K + 1, dtype=float) ** (-alpha); return w / w.sum()

def p_unstored_plain(lam, s, m):
    return stats.poisson.cdf(m - 1, s * lam)

def p_unstored_amp(lam, s, m, r):
    """P(compound Poisson(lam, Bin(1+r,s)) < m) via Panjer recursion, vectorized over lam."""
    lam = np.atleast_1d(lam).astype(float)
    f = stats.binom.pmf(np.arange(r + 2), r + 1, s)          # severity pmf on 0..r+1
    g = np.zeros((m, lam.size))
    g[0] = np.exp(-lam * (1 - f[0]))
    for k in range(1, m):
        acc = np.zeros(lam.size)
        for j in range(1, min(k, r + 1) + 1):
            acc += j * f[j] * g[k - j]
        g[k] = lam / k * acc
    return g.sum(0)

def acc_plain(N, s, m, p):
    return 1 - (1 - 1 / KCLS) * np.sum(p * p_unstored_plain(N * p, s, m))

def acc_amp(N, s, m, r, p):
    return 1 - (1 - 1 / KCLS) * np.sum(p * p_unstored_amp(N * p, s, m, r))

def n_equiv(target, s, m, p):
    f = lambda lg: acc_plain(np.exp(lg), s, m, p) - target
    return float(np.exp(optimize.brentq(f, np.log(1e-3), np.log(1e14), xtol=1e-12)))

def multiplier(N, s, m, r, p):
    return n_equiv(acc_amp(N, s, m, r, p), s, m, p) / N

if __name__ == "__main__":
    res = dict(invariance=[], per_lambda=[])
    s, r = 0.1, 7
    for m in [1, 2, 3]:
        for alpha in [0.6, 1.0, 1.4]:
            for N in [1e4, 1e5, 1e6]:
                M = multiplier(N, s, m, r, zipf(alpha))
                res["invariance"].append(dict(m=m, s=s, r=r, alpha=alpha, N=N, M=M))
                print(f"m={m} alpha={alpha} N={N:.0e}  M={M:.3f}")
    # per-fact multiplier as a function of expected exposures lambda (single fact)
    for m in [1, 2, 3]:
        for lam in [0.1, 1, 10, 100]:
            pa = p_unstored_amp(lam, s, m, r)[0]
            f = lambda lg: p_unstored_plain(np.exp(lg), s, m) - pa
            lam_eq = float(np.exp(optimize.brentq(f, np.log(1e-6), np.log(1e8))))
            res["per_lambda"].append(dict(m=m, lam=lam, M=lam_eq / lam))
            print(f"m={m} lambda={lam}: per-fact M={lam_eq / lam:.2f}  (m=1 value {(1 - (1 - s) ** (1 + r)) / s:.2f}, 1/s={1 / s:.0f})")
    json.dump(res, open(os.path.join(OUT, "e7_threshold.json"), "w"), indent=1)
