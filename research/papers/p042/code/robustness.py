"""Robustness of the exponent beta to the analysis choices (recovery threshold THR, transient cut TMIN),
and Fig. 4 (beta and m* across training configurations). Usage: python3 robustness.py"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze_lib as A

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
runs = ["E1_a1.2_s0", "E1_a1.5_s0", "E1_a1.8_s0", "E3_lr0.0003", "E3_lr0.003", "E5_adam_b20.99", "E5_adam_b20.9999",
        "E5_sgd_lr0.1", "E5_sgd_lr0.3"]
out = {}
for n in runs:
    d = A.load(n); out[n] = {}
    for thr in (0.8, 0.9, 0.95):
        for tmin in (30_000, 100_000, 300_000):
            lp, lT, y = A.stack(d, tmin=tmin, thr=thr)
            out[n][f"thr{thr}_tmin{tmin}"] = round(A.fit_beta(lp, lT, y)["beta"], 3)
    v = list(out[n].values()); out[n]["range"] = [min(v), max(v)]
    print(n, out[n]["range"], flush=True)
json.dump(out, open(os.path.join(R, "robustness.json"), "w"), indent=1)

s = json.load(open(os.path.join(R, "summary.json")))["E3"]
order = ["E5_sgd_lr0.1", "E5_sgd_lr0.3", "E3_lr0.003", "E3_L08.0", "E3_D256", "E5_adam_b20.9999", "E3_D64", "E3_L02.0",
         "E3_B64", "E3_B1024", "E1_a1.5_s0", "E3_lr0.0003", "E5_adam_b20.99"]
lab = {"E5_sgd_lr0.1": "SGD lr=0.1", "E5_sgd_lr0.3": "SGD lr=0.3", "E3_lr0.003": "Adam lr=3e-3", "E3_L08.0": "L0=8,k=16",
       "E3_D256": "d=256", "E5_adam_b20.9999": "Adam β2=0.9999", "E3_D64": "d=64", "E3_L02.0": "L0=2,k=4",
       "E3_B64": "batch 64", "E3_B1024": "batch 1024", "E1_a1.5_s0": "baseline (Adam)", "E3_lr0.0003": "Adam lr=3e-4",
       "E5_adam_b20.99": "Adam β2=0.99"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
y = np.arange(len(order))
b = [s[k]["beta"] for k in order]; lo = [s[k]["beta"] - s[k]["beta_CI95"][0] for k in order]; hi = [s[k]["beta_CI95"][1] - s[k]["beta"] for k in order]
cols = ["#c0392b" if k.startswith("E5_sgd") else "#2c7fb8" for k in order]
ax[0].barh(y, b, xerr=[lo, hi], color=cols, alpha=.85); ax[0].axvline(1, c="k", ls="--", lw=.8)
ax[0].set_yticks(y); ax[0].set_yticklabels([lab[k] for k in order], fontsize=8)
ax[0].set_xlabel("β  (P(recovered) depends on log p + β log T)"); ax[0].set_title("rarity exponent β (95% feature-bootstrap CI)", fontsize=10)
ax[1].barh(y, [s[k]["mstar_tokens"] for k in order], color=cols, alpha=.85); ax[1].set_xscale("log")
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel("m* (expected occurrences at 50% recovery, β fixed to 1)")
ax[1].set_title("occurrence threshold m*", fontsize=10)
fig.savefig(os.path.join(R, "fig4_beta.png"), dpi=130, bbox_inches="tight")
