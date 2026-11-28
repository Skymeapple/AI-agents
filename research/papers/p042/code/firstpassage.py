"""Secondary, model-free estimate of the rarity exponent: per-feature first-passage time.
For every feature, T_rec = geometric midpoint of the checkpoint interval in which max-cos first reaches THR.
Features are binned by log10 p (8 features per bin in rank order); a bin is used only if its median T_rec is
uncensored (i.e. the median feature crossed after the first and by the last checkpoint). OLS of
log10 median T_rec on log10 median p gives slope -kappa; beta_fp = 1/kappa.
Usage: python3 firstpassage.py  -> ../results/firstpassage.json, ../results/fig6_firstpassage.png"""
import os, json
import numpy as np
import analyze_lib as A
plt = A.plt

runs = ["E1_a1.2_s0", "E1_a1.2_s1", "E1_a1.5_s0", "E1_a1.5_s1", "E1_a1.8_s0", "E1_a1.8_s1", "E3_lr0.0003", "E3_lr0.003",
        "E3_B64", "E3_B1024", "E3_D64", "E3_D256", "E3_L02.0", "E3_L08.0", "E5_adam_b20.99", "E5_adam_b20.9999",
        "E5_sgd_lr0.1", "E5_sgd_lr0.3"]
out = {}; fig, ax = plt.subplots(figsize=(6, 4.3))
show = {"E1_a1.5_s0": "Adam lr=1e-3 (baseline)", "E3_lr0.0003": "Adam lr=3e-4", "E3_lr0.003": "Adam lr=3e-3",
        "E5_adam_b20.99": "Adam β2=0.99", "E5_sgd_lr0.3": "SGD lr=0.3"}
for n in runs:
    d = A.load(n); T = d["T"].astype(float); rec = d["maxcos"] >= A.THR          # ckpt x feature
    first = np.where(rec.any(0), rec.argmax(0), -1)
    Trec = np.where(first > 0, np.sqrt(T[np.maximum(first, 1) - 1] * T[np.maximum(first, 1)]), np.nan)
    Trec = np.where(first == 0, 0.0, Trec); Trec = np.where(first < 0, np.inf, Trec)   # left / right censored
    p = d["p"]; w = 8; xs, ys = [], []
    for i in range(0, len(p) - w + 1, w):
        med = np.median(Trec[i:i + w])
        if np.isfinite(med) and med > 0:
            xs.append(np.log10(np.median(p[i:i + w]))); ys.append(np.log10(med))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5:
        out[n] = dict(n_bins=int(len(xs)), note="too few uncensored bins"); continue
    sl, ic = np.polyfit(xs, ys, 1); res = ys - (sl * xs + ic)
    bs = []
    for _ in range(500):
        idx = A.rng.integers(0, len(xs), len(xs))
        if np.ptp(xs[idx]) > 0: bs.append(np.polyfit(xs[idx], ys[idx], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out[n] = dict(n_bins=int(len(xs)), kappa=float(-sl), kappa_CI95=[float(-hi), float(-lo)], beta_fp=float(-1 / sl),
                  beta_fp_CI95=[float(-1 / lo), float(-1 / hi)], r2=float(1 - res.var() / ys.var()),
                  frac_never=float((first < 0).mean()), frac_at_first_ckpt=float((first == 0).mean()))
    print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[n].items()}, flush=True)
    if n in show:
        l, = ax.plot(xs, ys, "o", ms=3, label=f"{show[n]}: κ={-sl:.2f}"); ax.plot(xs, sl * xs + ic, "-", lw=.8, c=l.get_color())
ax.set_xlabel("log10 feature frequency p"); ax.set_ylabel("log10 tokens to first recovery (bin median)")
ax.set_title("first-passage time vs frequency:  T_rec ∝ p^(-κ)", fontsize=10); ax.legend(fontsize=7)
fig.savefig(os.path.join(A.R, "fig6_firstpassage.png"), dpi=130, bbox_inches="tight")
json.dump(out, open(os.path.join(A.R, "firstpassage.json"), "w"), indent=1)
