"""Plot marginal distributions of the identified set (results/abc_accepted.npz) against prior boxes."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibrate import PRIOR, KEYS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
Z = np.load(os.path.join(RES, "abc_accepted.npz"))
fig, ax = plt.subplots(3, 4, figsize=(13, 8))
for i, k in enumerate(KEYS):
    a = ax.flat[i]
    a.hist(Z[k], bins=40, range=PRIOR[k], color="#2c7fb8", alpha=0.8)
    a.set_title(k); a.set_xlim(*PRIOR[k]); a.set_yticks([])
plt.suptitle("Identified set: parameter vectors reproducing all six Bastani et al. (2025) arm means within tolerance\n(x-range = search box; flat histogram = not identified)")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_identified_set.png"), dpi=140)
