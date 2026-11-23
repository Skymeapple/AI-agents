"""Revision (round 1) of Experiment A: Monte Carlo uncertainty, a finer N grid, the decomposition of the
closed-form rule's error, and a check of the law at the corpus vocabulary sizes used in Section 5.4.

1. Additive memory, V=512: N grid spaced by 2^(1/8) around the threshold, 200 trials per cell
   (each trial = fresh keys, codes, values; all N items queried). N* (50% accuracy) by log-interpolation,
   with a 95% bootstrap CI over trials. Compared with Eq. (1) (quadrature) and three closed forms:
     rule2      : N* = d_k / (2 lnV (1/d_v + 1/V))                      (paper Eq. 2)
     rule2_s    : keeps the target's self-term s^2/d_v in the competitor variance
     rule2_med  : as rule2_s but replaces sqrt(2 lnV) by the exact median of the max of V-1 std normals
2. Large vocabulary: V = 41,509 (prose word vocabulary) with (a) uniform values, (b) Zipf-distributed values
   drawn from the empirical prose unigram distribution; d_k=16, d_v=64 and d_k=32, d_v=64.
Output: results/sim_capacity_ci.json
"""
import json, os, re, glob, collections, numpy as np
from scipy.stats import norm
from capacity_sim import theory_acc

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "..", "results")
rng = np.random.default_rng(7)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def trial_acc(N, dk, dv, V, p=None, U=None):
    if U is None:
        U = unit(rng.standard_normal((V, dv)).astype(np.float32))
    K = unit(rng.standard_normal((N, dk)).astype(np.float32))
    vals = rng.integers(0, V, N) if p is None else rng.choice(V, N, p=p)
    M = U[vals].T @ K                       # d_v x d_k, additive outer-product state
    R = K @ M.T                             # N x d_v
    pred = np.empty(N, dtype=np.int64)
    for i in range(0, N, 2048):
        pred[i:i + 2048] = np.argmax(R[i:i + 2048] @ U.T, axis=1)
    return float((pred == vals).mean())


def nstar(Ns, accs):
    pts = sorted(zip(Ns, accs))
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 >= 0.5 > a1:
            f = (a0 - 0.5) / (a0 - a1)
            return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0))))
    return None


def curve_with_ci(dk, dv, V, Ns, trials, p=None, U_fixed=None, B=1000):
    A = np.array([[trial_acc(N, dk, dv, V, p, U_fixed) for _ in range(trials)] for N in Ns])  # len(Ns) x trials
    ns = nstar(Ns, A.mean(1))
    boots = []
    for _ in range(B):
        idx = rng.integers(0, trials, trials)
        v = nstar(Ns, A[:, idx].mean(1))
        if v is not None:
            boots.append(v)
    return A.mean(1).tolist(), ns, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def closed_forms(dk, dv, V):
    c = 1 / (dk * dv) + 1 / (V * dk)
    r2 = 1 / (2 * np.log(V) * c)
    r2s = 1 + (1 / (2 * np.log(V)) - 1 / dv) / c
    m = norm.ppf(0.5 ** (1 / (V - 1)))      # median of max of V-1 standard normals
    r2m = 1 + (1 / m ** 2 - 1 / dv) / c
    return {"rule2": float(r2), "rule2_s": float(r2s), "rule2_med": float(r2m), "median_max": float(m),
            "sqrt_2lnV": float(np.sqrt(2 * np.log(V)))}


def eq1_nstar(dk, dv, V):
    grid = np.exp(np.linspace(np.log(4), np.log(20000), 400))
    return nstar(list(grid), [theory_acc(1.0, n - 1, dk, dv, V) for n in grid])


def main():
    out = {"V512": [], "largeV": []}
    V = 512
    cells = [(8, 64), (16, 64), (32, 64), (64, 64), (16, 16), (16, 32), (16, 128), (16, 256)]
    for dk, dv in cells:
        cf = closed_forms(dk, dv, V)
        center = cf["rule2_med"]
        Ns = sorted(set(int(round(center * 2 ** (e / 8))) for e in range(-12, 13)))
        Ns = [n for n in Ns if n >= 2]
        acc, ns, ci = curve_with_ci(dk, dv, V, Ns, trials=200)
        e1 = eq1_nstar(dk, dv, V)
        rec = {"dk": dk, "dv": dv, "V": V, "Ns": Ns, "acc_sim": acc, "Nstar_sim": ns, "Nstar_sim_ci95": ci,
               "Nstar_eq1": e1, **cf}
        for k in ["Nstar_eq1", "rule2", "rule2_s", "rule2_med"]:
            rec["relerr_" + k] = (rec[k] - ns) / ns
        out["V512"].append(rec)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("Ns", "acc_sim")}, flush=True)
    # large vocabulary, uniform and Zipf values
    docs = [re.findall(r"[a-z]+", open(f, encoding="latin-1").read().lower())
            for f in sorted(glob.glob(os.path.join(ROOT, "..", "data", "gutenberg", "*.txt")))]
    cnt = collections.Counter(t for d in docs for t in d)
    VL = len(cnt); freq = np.array(sorted(cnt.values(), reverse=True), dtype=float); pz = freq / freq.sum()
    for dk, dv in [(16, 64), (32, 64)]:
        cf = closed_forms(dk, dv, VL)
        for dist, p in [("uniform", None), ("zipf_prose_unigram", pz)]:
            center = cf["rule2_med"]
            Ns = sorted(set(int(round(center * 2 ** (e / 6))) for e in range(-12, 13)))
            acc, ns, ci = curve_with_ci(dk, dv, VL, Ns, trials=40, p=p, B=500)
            rec = {"dk": dk, "dv": dv, "V": VL, "values": dist, "Ns": Ns, "acc_sim": acc, "Nstar_sim": ns,
                   "Nstar_sim_ci95": ci, "Nstar_eq1": eq1_nstar(dk, dv, VL), **cf,
                   "sum_p2": float((pz ** 2).sum()) if p is not None else 1 / VL}
            out["largeV"].append(rec)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("Ns", "acc_sim")}, flush=True)
    json.dump(out, open(os.path.join(RES, "sim_capacity_ci.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
