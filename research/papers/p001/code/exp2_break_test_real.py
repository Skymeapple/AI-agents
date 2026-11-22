"""Exp 2: Is there within-sample deceleration of the *reducible* loss in the Chinchilla data?
(i) Nested break model: L = E + (A N^-a + B D^-b) * max(1, C/Cb)^delta  (delta>0 = slowdown)
    profiled over Cb, bootstrap CI for delta.
(ii) Backtest: fit the base law on runs with C < Cs, forecast runs with C >= Cs; report mean signed
     log-residual (negative = the law over-predicts loss = runs did better than forecast)."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

rng = np.random.default_rng(1)
N, D, C, L = load_chinchilla()
k = epoch_keep(L)
N, D, C, L = N[k], D[k], C[k], L[k]
lN, lD, lC, lL = np.log(N), np.log(D), np.log(C), np.log(L)
base = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
p0 = np.array([np.log(base["A"]), np.log(base["B"]), np.log(base["E"]), base["alpha"], base["beta"]])


def obj_break(q, lN, lD, lC, lL, lCb, delta_h=1e-3):
    a, b, e, al, be, dl = q
    h = np.maximum(0.0, lC - lCb)
    T = np.stack([a - al * lN + dl * h, b - be * lD + dl * h, np.full_like(lN, e)])
    lp = logsumexp(T, axis=0)
    w = np.exp(T - lp)
    r = lp - lL
    hp = np.clip(r, -delta_h, delta_h)
    g = np.array([(hp * w[0]).sum(), (hp * w[1]).sum(), (hp * w[2]).sum(),
                  -(hp * w[0] * lN).sum(), -(hp * w[1] * lD).sum(), (hp * (w[0] + w[1]) * h).sum()])
    return SCALE * huber(r, delta_h).sum(), SCALE * g


def fit_break(lN, lD, lC, lL, lCb, init):
    best = None
    for dl0 in [0.0, 0.05, -0.05]:
        x0 = np.r_[init, dl0]
        r = minimize(obj_break, x0, args=(lN, lD, lC, lL, lCb), method="BFGS", jac=True, options=OPT)
        if best is None or r.fun < best.fun:
            best = r
    return best


Cb_grid = np.log(np.quantile(C, [0.3, 0.4, 0.5, 0.6, 0.7]))


def profile(idx, init):
    fb = minimize(obj_base, init, args=(lN[idx], lD[idx], lL[idx]), method="BFGS", jac=True, options=OPT)
    out = []
    for lCb in Cb_grid:
        r = fit_break(lN[idx], lD[idx], lC[idx], lL[idx], lCb, fb.x)
        out.append((r.fun, r.x[5], lCb))
    j = int(np.argmin([o[0] for o in out]))
    return fb.fun, out[j], out


f0, best, allb = profile(np.arange(len(L)), p0)
print("base obj", f0, "best break obj", best[0], "delta", best[1], "Cb", np.exp(best[2]))
per_cb = [dict(Cb=float(np.exp(o[2])), delta=float(o[1]), obj=float(o[0])) for o in allb]

B = 300
deltas, deltas_fixed = [], {float(np.exp(c)): [] for c in Cb_grid}
for i in range(B):
    idx = rng.integers(0, len(L), len(L))
    _, bb, allbb = profile(idx, p0)
    deltas.append(bb[1])
    for o in allbb:
        deltas_fixed[float(np.exp(o[2]))].append(o[1])
deltas = np.array(deltas)
res_break = dict(
    base_obj=float(f0), break_obj=float(best[0]), delta_hat=float(best[1]), Cb_hat=float(np.exp(best[2])),
    per_Cb=per_cb,
    delta_profiled_ci80=[float(np.percentile(deltas, 10)), float(np.percentile(deltas, 90))],
    delta_profiled_ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    frac_boot_delta_pos=float((deltas > 0).mean()),
    delta_fixed_Cb_ci95={str(k2): [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k2, v in deltas_fixed.items()},
    gamma_base=float(base["alpha"] * base["beta"] / (base["alpha"] + base["beta"])),
)
print(json.dumps(res_break, indent=1))

# (ii) backtest
bt = []
for qs in [0.5, 0.6, 0.7, 0.8]:
    Cs_ = np.quantile(C, qs)
    tr, te = C < Cs_, C >= Cs_
    f = fit_base(N[tr], D[tr], L[tr], grid=coarse_grid() + [list(p0)])
    r = lL[te] - pred_log_base(f["raw"], lN[te], lD[te])
    # bootstrap the training set to get a forecast interval for the mean residual
    mr = []
    for i in range(200):
        ii = np.where(tr)[0][rng.integers(0, tr.sum(), tr.sum())]
        fb = minimize(obj_base, f["raw"], args=(lN[ii], lD[ii], lL[ii]), method="BFGS", jac=True, options=OPT)
        mr.append(np.mean(lL[te] - pred_log_base(fb.x, lN[te], lD[te])))
    bt.append(dict(split_quantile=qs, C_split=float(Cs_), n_train=int(tr.sum()), n_test=int(te.sum()),
                   mean_log_resid=float(r.mean()), median_log_resid=float(np.median(r)),
                   mean_resid_ci90=[float(np.percentile(mr, 5)), float(np.percentile(mr, 95))],
                   train_fit={kk: float(f[kk]) for kk in ["A", "B", "E", "alpha", "beta"]}))
    print(bt[-1])
res_break["backtest"] = bt
json.dump(res_break, open(f"{RES}/exp2_break_real.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].hist(deltas, bins=30, color="C0", alpha=.8); ax[0].axvline(0, color="k", lw=1)
ax[0].set_xlabel(r"bootstrap $\hat\delta$ (profiled $C_b$)"); ax[0].set_ylabel("count")
ax[0].set_title("exponent change in reducible loss", fontsize=9)
xs = [b["C_split"] for b in bt]
ax[1].errorbar(xs, [b["mean_log_resid"] for b in bt],
               yerr=np.array([[b["mean_log_resid"] - b["mean_resid_ci90"][0], b["mean_resid_ci90"][1] - b["mean_log_resid"]] for b in bt]).T,
               fmt="o", capsize=3)
ax[1].axhline(0, color="k", lw=1); ax[1].set_xscale("log")
ax[1].set_xlabel("train/test compute split"); ax[1].set_ylabel("mean log-residual on held-out\n(higher-compute) runs")
ax[1].set_title("backtest: does the law over/under-predict?", fontsize=9)
plt.tight_layout(); plt.savefig(f"{RES}/fig2_break_real.png", dpi=150)
