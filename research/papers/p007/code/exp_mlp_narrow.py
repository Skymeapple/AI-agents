"""Round 3 (review_r2 W2): why do narrow ReLU networks compete at widths h <= k_s?

Control experiment with the round-2 protocol (exp_mlp2.py: d=64, q=8, k_s=6, orthonormal
teacher, 2 learning rates selected on validation, 4 seeds, n=4096), widths h in {2,4,6} only,
for two teachers:
  k_p = 6 : shared + private features (the round-2 teacher; replication at narrow width)
  k_p = 0 : shared features only.  If the competition at h <= k_s is caused by private
            features loading the narrow shared layer, it must vanish here.
Ratio = unified / specialist noiseless test MSE (geometric mean over seeds, bootstrap 95% CI).
Writes ../results/mlp_narrow_results.json     (~10 min, 1 thread)
"""
import json, os, itertools
import numpy as np
import exp_mlp2 as E

E.HS = [2, 4, 6]; E.HMAX = 6
N, SEEDS = 4096, [0, 1, 2, 3]
res = dict(d=E.d, q=E.q, ks=E.ks, n=N, hs=E.HS, seeds=SEEDS, runs=[], summary={})
for kp, M in itertools.product([6, 0], [2, 4, 8]):
    E.kp = kp
    LU, LS = [], []
    for seed in SEEDS:
        Lu, Ls, _, _ = E.run(M, N, 1000 * seed + 7 * M + N)
        LU.append(Lu); LS.append(Ls)
        res["runs"].append(dict(kp=kp, M=M, seed=seed, Lu=Lu.tolist(), Ls=Ls.tolist()))
    lr = np.log(np.array(LU) / np.array(LS))  # [seed, h]
    rng = np.random.default_rng(0)
    bs = np.exp(np.array([lr[rng.integers(0, len(SEEDS), len(SEEDS))].mean(0) for _ in range(2000)]))
    res["summary"][f"kp={kp},M={M}"] = dict(ratio=np.exp(lr.mean(0)).tolist(), ci_lo=np.percentile(bs, 2.5, axis=0).tolist(),
                                            ci_hi=np.percentile(bs, 97.5, axis=0).tolist(),
                                            per_seed_min=np.exp(lr).min(0).tolist(), per_seed_max=np.exp(lr).max(0).tolist(),
                                            Lu=np.mean(LU, 0).tolist(), Ls=np.mean(LS, 0).tolist())
    print(f"kp={kp} M={M}", "ratio h=2,4,6:", np.exp(lr.mean(0)).round(3).tolist(), "seed range", np.exp(lr).min(0).round(2).tolist(), np.exp(lr).max(0).round(2).tolist(), flush=True)
    json.dump(res, open(os.path.join(E.OUT, "mlp_narrow_results.json"), "w"), indent=1)
