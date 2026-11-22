"""E2c: under vanilla softmax policy gradient (population limit of GRPO) a rare exploit
is starved unless its mass exceeds a threshold.  We locate the threshold h0* by bisection
on random 20-mode universes and compare it with the 'initial logit-velocity race'
heuristic:  exploit wins  iff  h0 (eta - fbar0) > max_c m_c (f_c - fbar0)."""
import json, os
import numpy as np
from exp2_tabular_grpo import vanilla_flow

R = os.path.join(os.path.dirname(__file__), "..", "results")


def make(h0, seed, eta):
    rng = np.random.default_rng(seed)
    c0 = rng.uniform(0.1, 0.4)
    fc = rng.uniform(0.5, 0.8, 4); mc = rng.dirichlet(np.ones(4)) * c0
    fw = rng.uniform(0.05, 0.45, 15); mw = rng.dirichlet(np.ones(15)) * (1 - c0 - h0)
    f = np.concatenate([fc, fw, [eta]]); w = np.concatenate([mc, mw, [h0]])
    y = np.zeros(20, bool); y[:4] = True
    return w, y, f


def collapses(h0, seed, eta, T):
    w, y, f = make(h0, seed, eta)
    h = vanilla_flow(w, y, f, T=T, log_every=T)
    return h[-1, 3] > 0.5


def heuristic(seed, eta, h0_guess=0.01):
    # solve h0 (eta - fbar0(h0)) = max_c m_c (f_c - fbar0(h0)) by fixed point
    h0 = h0_guess
    for _ in range(50):
        w, y, f = make(h0, seed, eta); fb = w @ f
        rhs = np.max(w[:4] * (f[:4] - fb)); h0 = rhs / (eta - fb)
    return h0


if __name__ == "__main__":
    rows = []
    for seed in range(20):
        eta = [0.85, 0.9, 0.95, 1.0][seed % 4]
        res = {}
        for T in [5000, 20000, 80000]:
            lo, hi = 1e-5, 0.3
            if collapses(lo, seed, eta, T):
                res[T] = lo; continue
            for _ in range(14):
                mid = np.sqrt(lo * hi)
                if collapses(mid, seed, eta, T): hi = mid
                else: lo = mid
            res[T] = float(np.sqrt(lo * hi))
        rows.append({"seed": seed, "eta": eta, "h0_star_T5k": res[5000], "h0_star_T20k": res[20000],
                     "h0_star_T80k": res[80000], "heuristic": float(heuristic(seed, eta))})
        print(rows[-1])
    lt = np.log10([r["h0_star_T20k"] for r in rows]); lh = np.log10([r["heuristic"] for r in rows])
    summ = {"corr_log10": float(np.corrcoef(lt, lh)[0, 1]),
            "mean_abs_log10_err": float(np.mean(np.abs(lt - lh))),
            "median_ratio_T80k_over_T5k": float(np.median([r["h0_star_T80k"] / r["h0_star_T5k"] for r in rows]))}
    print(summ)
    json.dump({"rows": rows, "summary": summ}, open(os.path.join(R, "exp2c.json"), "w"), indent=2)
