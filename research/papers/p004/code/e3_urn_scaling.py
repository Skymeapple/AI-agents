"""E3: Verified self-training as a generalized Polya urn (single prompt / tabular learner).
NOTE: for speed the pool is frozen inside blocks of n/200 proposals between checkpoints,
so this is an approximate (blocked) simulation of the urn; it is adequate for exponents.

Pool holds labelled examples for one prompt (human seeds: c0 correct, w0 wrong spread over
k-1 wrong labels). Each step the learner samples a label from the pool's empirical
distribution (a tabular max-likelihood learner sampling at T=1), a verifier accepts with
prob a (correct) / b (wrong), accepted labels are accumulated.
Mean-field prediction (Proposition 4): odds(correct) grows like t^{(a-b)/a}, so the wrong
share decays as t^{-(1-b/a)}; with a=b the expected share is a martingale (no gain).
We fit the decay exponent from Monte Carlo and compare with 1-b/a.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(3)
T, REPS = 200_000, 200

def sim(a, b, c0=3, w0=3):
    # vectorised over replicates; wrong labels pooled (share only matters for odds)
    C = np.full(REPS, float(c0)); W = np.full(REPS, float(w0))
    checkpoints = np.unique(np.logspace(2, np.log10(T), 25).astype(int))
    out = []; t = 0
    for cp in checkpoints:
        n = cp - t; t = cp
        # simulate n proposals in blocks for speed (pool frozen inside small blocks)
        blk = max(1, n // 200)
        done = 0
        while done < n:
            k = min(blk, n - done)
            p = C / (C + W)
            nc = rng.binomial(k, p)
            C += rng.binomial(nc, a); W += rng.binomial(k - nc, b)
            done += k
        out.append((int(cp), float(np.mean(W / (C + W)))))
    return out

res = {}
fits = []
for a, b in [(1, 0), (1, 0.25), (1, 0.5), (1, 0.75), (1, 1.0), (0.5, 0.25)]:
    o = sim(a, b)
    res[f"{a},{b}"] = o
    t = np.array([x[0] for x in o]); e = np.array([x[1] for x in o])
    m = t >= 2000
    slope = np.polyfit(np.log(t[m]), np.log(e[m]), 1)[0]
    fits.append(dict(a=a, b=b, fitted_exponent=float(-slope), predicted=1 - b / a,
                     wrong_share_at_T=float(e[-1]), wrong_share_t0=0.5))
    print(fits[-1])
json.dump(dict(curves=res, fits=fits, T=T, reps=REPS), open(os.path.join(OUT, "e3_urn.json"), "w"), indent=1)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for k, o in res.items():
    ax.loglog([x[0] for x in o], [x[1] for x in o], "o-", ms=3, label=f"(a,b)=({k})")
ax.set_xlabel("synthetic proposals t"); ax.set_ylabel("mean wrong-label share in pool")
ax.set_title("Verified self-training urn: error ~ t^-(1-b/a)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_urn.png"), dpi=150)
