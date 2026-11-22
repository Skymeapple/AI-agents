"""E1: Upper bound (Theorem 1) vs. achievable horizon (Proposition 2) for a single
bounded-logit softmax readout head that must return the LATEST version of a key.
E1b: recency-retention frontier on random Poisson revision streams, attention head
(exact-match + linear recency, bounded logits) vs. delta-rule fast-weight memory.
All numbers are produced by this script; outputs go to ../results/.
"""
import numpy as np, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

def L_of(eps): return np.log((1 - eps) / eps)

def upper_bound_R(B, T, eps):
    """Theorem 1: max number of chained versions R with  R*L <= 2B - log(T-1)."""
    return (2 * B - np.log(T - 1)) / L_of(eps)

def worst_weight_latest(B, lam, A, Delta, T):
    """Exact worst case (over the family F(T,A,Delta)) of the attention weight on the
    latest version for head s = clip(B - lam*age, -B, B) on matches, -B elsewhere.
    Worst case: latest at age A, older versions packed at gaps Delta,2Delta,... up to T."""
    ages_old = A + Delta * np.arange(1, int((T - 1 - A) // Delta) + 1)
    s_lat = max(B - lam * A, -B)
    s_old = np.maximum(B - lam * ages_old, -B)
    n_dis = T - 1 - len(ages_old)
    m = max(s_lat, s_old.max() if len(s_old) else -B, -B)
    num = np.exp(s_lat - m)
    den = num + np.exp(s_old - m).sum() + n_dis * np.exp(-B - m)
    return num / den

def achievable_A(B, T, eps, Delta):
    """Largest A such that some lam gives worst-case weight >= 1-eps (grid over lam)."""
    best = -1
    for lam in np.geomspace(1e-4, 50, 400) / Delta:
        # binary search largest A for this lam (weight is decreasing in A)
        lo, hi = -1, T - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if worst_weight_latest(B, lam, mid, Delta, T) >= 1 - eps: lo = mid
            else: hi = mid
        best = max(best, lo)
    return best

res = {"E1": []}
eps = 0.05
for T in [10**3, 10**5, 10**7]:
    for B in [4, 6, 8, 10, 12, 16, 20]:
        for Delta in [16]:
            Rub = upper_bound_R(B, T, eps)
                        A_ach = achievable_A(B, min(T, 10**7), eps, Delta)
            res["E1"].append(dict(T=T, B=B, Delta=Delta, eps=eps, R_ub=float(Rub),
                                  A_ub_over_Delta=float(np.floor(Rub)) if Rub >= 1 else -1  # Thm1: A/Delta < floor(R_ub),
                                  A_ach_over_Delta=float(A_ach / Delta) if A_ach >= 0 else -1))
            print(res["E1"][-1])

# ---- E1b: frontier on Poisson streams -------------------------------------------------
T = 4096; Bk = 10.0
def attention_eval(lam, n_trials=4000):
    """Two query types: 'hot' key revised every ~Delta_hot steps (latest age ~U[0,Delta_hot]);
    'cold' key written once at age ~U[0,T). Distractors at logit -Bk (orthogonal keys),
    matches at clip(Bk - lam*age). Return accuracy (argmax-decoding of attention mixture
    == latest value, values random orthogonal => correct iff weight on latest > max other weight
    AND weight >= 0.5 for a soft-decoding criterion; we report P(weight_latest >= 0.5))."""
    out = {}
    for kind in ["hot", "cold"]:
        ok = 0
        for _ in range(n_trials):
            if kind == "hot":
                Dh = 32
                a0 = rng.integers(0, Dh)
                ages = a0 + Dh * np.arange(0, (T - 1 - a0) // Dh + 1)
                ages = ages + rng.integers(-Dh // 4, Dh // 4 + 1, size=len(ages)) * (np.arange(len(ages)) > 0)
                ages = np.sort(np.clip(ages, a0, T - 1))
            else:
                ages = np.array([rng.integers(0, T - 1)])
            s = np.maximum(Bk - lam * ages, -Bk)
            nd = T - len(ages)
            m = s.max()
            w = np.exp(s - m); den = w.sum() + nd * np.exp(-Bk - m)
            ok += (w[0] / den) >= 0.5
        out[kind] = ok / n_trials
    return out

frontier = []
for lam in np.geomspace(1e-4, 1.0, 25):
    r = attention_eval(lam, 1500)
    frontier.append(dict(lam=float(lam), hot=r["hot"], cold=r["cold"]))
    print("lam", lam, r)
res["E1b_frontier"] = frontier

# delta-rule memory on the same task: M distinct keys in dim d, random unit keys,
# W <- W + (v - W k) k^T ; decode nearest of value codebook.
def delta_eval(M, d, revisions_per_key=8, nV=64, trials=20):
    accs = []
    for _ in range(trials):
        K = rng.standard_normal((M, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
        Vb = rng.standard_normal((nV, d)); Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
        W = np.zeros((d, d)); latest = np.zeros(M, int)
        order = rng.permutation(np.repeat(np.arange(M), revisions_per_key))
        for k in order:
            v = rng.integers(nV); latest[k] = v
            W += np.outer(Vb[v] - W @ K[k], K[k])
        pred = np.argmax((K @ W.T) @ Vb.T, axis=1)
        accs.append(np.mean(pred == latest))
    return float(np.mean(accs))
delta = []
d = 64
for M in [8, 16, 32, 48, 64, 96, 128, 256]:
    for R in [1, 8, 32]:
        a = delta_eval(M, d, R, trials=10 if M > 64 else 20)
        delta.append(dict(M=M, d=d, R=R, acc=a)); print("delta", M, R, a)
res["E1c_delta"] = delta
json.dump(res, open(os.path.join(OUT, "theory_sim.json"), "w"), indent=1)

print('saved json; run make_fig1.py for the figure')
