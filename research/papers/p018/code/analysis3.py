"""Gap-closing (crossing-time) predictor for STABLE conjectures (record age >= 2), and the
K2/tiny-graph tightness diagnostic.  Uses results/per_order_maxima.npy.
Run: python3 analysis3.py
"""
import os, json
import numpy as np
from analysis import templates, describe, BIG, RES, auc

M = np.load(os.path.join(RES, "per_order_maxima.npy"))
T = templates(); nT = len(T)
q = np.array([t[5] for t in T], float)
res = {}
for k in (6, 7, 8, 9, 10):
    E = M[:, 2:k + 1].max(1)
    jfirst = np.argmax(M[:, 2:k + 1] == E[:, None], axis=1) + 2
    age = k - jfirst
    # orders <= k at which the bound is attained
    tight_orders = [(np.where(M[i, 2:k + 1] == E[i])[0] + 2) for i in range(nT)]
    tiny_tight_only = np.array([to.max() <= 3 for to in tight_orders])
    f = np.zeros(nT, int)
    for j in list(range(k + 1, 11)) + BIG:
        hit = (M[:, j] > E) & (f == 0); f[hit] = j
    ref = f > 0
    jj = np.array([k - 2, k - 1, k], float)
    slope = ((M[:, k - 2:k + 1] - M[:, k - 2:k + 1].mean(1, keepdims=True)) * (jj - jj.mean())).sum(1) / 2.0
    gap = E - M[:, k]
    stable = age >= 2
    closing = slope > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = np.where(closing, np.floor(gap / np.where(slope > 0, slope, 1)) + 1, np.inf)
    horizon = 16 - k
    pred = closing & (t_cross <= horizon)
    s = stable
    r = {"n_stable": int(s.sum()), "n_stable_refuted_observed": int((ref & s).sum()),
         "stable_tiny_tight_only": int((tiny_tight_only & s).sum()),
         "refuted_rate_stable_tiny_tight_only": float(ref[s & tiny_tight_only].mean()) if (s & tiny_tight_only).any() else None,
         "refuted_rate_stable_tight_at_order>=4": float(ref[s & ~tiny_tight_only].mean()) if (s & ~tiny_tight_only).any() else None,
         "gap_closing_rule(slope>0 & predicted crossing <=16)": {
             "TP": int((pred & ref & s).sum()), "FP": int((pred & ~ref & s).sum()),
             "FN": int((~pred & ref & s).sum()), "TN": int((~pred & ~ref & s).sum())},
         "AUC_score(slope/(gap+1))_stable": auc((slope / (gap + q))[s], ref[s]),
         "refuted_rate_stable_slope<=0": float(ref[s & ~closing].mean()),
         "refuted_rate_stable_slope>0": float(ref[s & closing].mean()) if (s & closing).any() else None,
         "n_stable_slope<=0": int((s & ~closing).sum()),
         "n_stable_slope>0": int((s & closing).sum())}
    if k <= 8:
        exh = ref & (f <= 10)
        r["exhaustive_label_by10"] = {"n_ref": int((exh & s).sum()),
                                      "rate_slope<=0": float(exh[s & ~closing].mean()),
                                      "rate_slope>0": float(exh[s & closing].mean()) if (s & closing).any() else None}
    # survivors of all filters: stable, slope<=0, tight at order>=4 -> residual observed false rate
    clean = s & ~closing & ~tiny_tight_only
    r["clean_filter(stable & slope<=0 & tight at order>=4)"] = {"n": int(clean.sum()), "n_refuted_observed": int(ref[clean].sum()),
                                                               "rate": float(ref[clean].mean()) if clean.any() else None,
                                                               "examples_refuted": [describe(T[i], int(E[i])) + f"  [first refuted at order {f[i]}]" for i in np.where(clean & ref)[0][:12]]}
    res[k] = r
json.dump(res, open(os.path.join(RES, "summary3.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
