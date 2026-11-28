"""Revision r2: prompt-level bootstrap of the forecast error at the cap (k = 10^4; 6,000 for cygnet; 4,200 = largest
recorded run for Llama-3-8B-Instruct, where every prompt succeeded) from the first n = 60 attempts, on the BoN data.
Each replicate resamples the 159 prompts with replacement (same resample for counts and realised first-success
truth) and refits the beta-binomial (exp_baselines.fit) and the power law.  Output: results/bon_boot.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ttc import powerlaw_passk
from exp_baselines import fit, predict
RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
NB, n = 200, 60
d = pd.read_csv(os.path.join(DATA, "bon_text_first_success.csv"), dtype={"first60": str})
rows = []
for model, g in d.groupby("model"):
    K = min(10000, int(g.n_recorded.max()))
    X = np.array([s[:n].count("1") for s in g.first60]); T = g.first_success.values.astype(float)
    hit = np.where(np.isnan(T), False, T <= K)
    rng = np.random.default_rng(11)
    e = []
    for b in range(NB + 1):
        idx = np.arange(len(X)) if b == 0 else rng.integers(0, len(X), len(X))
        th, _, _ = fit(X[idx], np.full(len(idx), n), 1, False, np.random.default_rng(b))
        tr = hit[idx].mean()
        e.append((float(predict(th, 1, False, K)) - tr, float(powerlaw_passk(X[idx], n, [K])[0]) - tr))
    e = np.array(e)
    rows.append(dict(model=model, k=K, err_BB=e[0, 0], BB_lo=np.percentile(e[1:, 0], 2.5), BB_hi=np.percentile(e[1:, 0], 97.5),
                     err_PL=e[0, 1], PL_lo=np.percentile(e[1:, 1], 2.5), PL_hi=np.percentile(e[1:, 1], 97.5)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "bon_boot.csv"), index=False, float_format="%.4f")
