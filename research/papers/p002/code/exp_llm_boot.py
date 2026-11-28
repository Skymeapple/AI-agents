"""Revision r2 (reviewer W4): problem-level bootstrap for the sign of the beta-binomial (BB) and power-law forecast
errors at k = 10^4 from n = 16 on the Large Language Monkeys MATH/Pythia data.
Point estimate = mean error over the 10 counterfactual subsets of exp_llm.py (same BB fitter, exp_baselines.fit).
Each bootstrap replicate resamples the 128 problems with replacement (the same resample for the n = 16 counts and for
the 10,000-sample truth) and refits on all 10 subsets.  Output: results/llm_boot.csv (percentile 95% intervals)."""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import powerlaw_passk
from exp_baselines import fit, predict
from exp_llm import subsets, unbiased, N
RES = os.path.join(os.path.dirname(__file__), "..", "results")
NB, K, n = 200, 10000, 16


def job(args):
    m, c, mi = args
    subs = subsets(c, n, 1000 * mi + n)
    rng = np.random.default_rng(7 + mi)
    errs = []
    for b in range(NB + 1):
        idx = np.arange(len(c)) if b == 0 else rng.integers(0, len(c), len(c))
        tr = unbiased(c[idx], N, K)
        eb, ep = [], []
        for X in subs:
            th, _, _ = fit(X[idx], np.full(len(idx), n), 1, False, np.random.default_rng(b))
            eb.append(float(predict(th, 1, False, K)) - tr)
            ep.append(float(powerlaw_passk(X[idx], n, [K])[0]) - tr)
        errs.append((b, np.mean(eb), np.mean(ep)))
    e = np.array(errs)
    return dict(model=m, err_BB=e[0, 1], BB_lo=np.percentile(e[1:, 1], 2.5), BB_hi=np.percentile(e[1:, 1], 97.5),
                frac_BB_pos=float(np.mean(e[1:, 1] > 0)),
                err_PL=e[0, 2], PL_lo=np.percentile(e[1:, 2], 2.5), PL_hi=np.percentile(e[1:, 2], 97.5))


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "llm_monkeys_math_pythia_counts.csv"))
    models = sorted(d.model.unique(), key=lambda m: float(m.split()[1].rstrip("MB")) * (1e-3 if m.endswith("M") else 1))
    C = {m: d[d.model == m].sort_values("problem_idx").n_correct.values.astype(int) for m in models}
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        out = pool.map(job, [(m, C[m], mi) for mi, m in enumerate(models)])
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, "llm_boot.csv"), index=False, float_format="%.4f")
    print(df.to_string())
