"""Fig. 7 (revision 1): delta_hat as a function of the breakpoint on the real data, with compute-bin cluster
bootstrap 95% intervals (Exp. 9), and the run count per compute decade."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, load_chinchilla, epoch_keep
d = json.load(open(f"{RES}/exp9_real_inference.json"))
N, D, C, L = load_chinchilla(); C = C[epoch_keep(L)]
pc = d["profiled_wide_grid"]["per_Cb"]; ci = d["cluster_bootstrap"]["C"]["fixed_Cb_ci95"]
x = np.array([r["Cb"] for r in pc]); y = np.array([r["delta_hat"] for r in pc])
lo = np.array([c["ci95"][0] for c in ci]); hi = np.array([c["ci95"][1] for c in ci])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=3)
ax[0].axhline(0, color="k", lw=.8); ax[0].set_xscale("log")
ax[0].set_xlabel("breakpoint $C_b$ (FLOP)"); ax[0].set_ylabel(r"$\hat\delta$ (positive = slowdown above $C_b$)")
ax[0].set_title("fixed-breakpoint estimates, cluster-bootstrap 95% CI", fontsize=9)
bins = np.arange(18, 22.6, 0.5)
ax[1].hist(np.log10(C), bins=bins, color="C1", alpha=.8)
ax[1].set_xlabel("log10 training FLOP"); ax[1].set_ylabel("number of runs")
ax[1].set_title("where the 240 runs are", fontsize=9)
plt.tight_layout(); plt.savefig(f"{RES}/fig7_real_profile.png", dpi=150)
