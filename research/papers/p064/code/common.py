import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)

# reference categorical palette (fixed order), light mode
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False,
})

# Baseline calibration: software developers (BLS EP 2025-35 table; see bls_data.py)
SD_MU = 0.043                                   # total occupational separations rate
SD_G = np.log(1892.6 / 1717.8) / 10             # projected employment growth, per year
TAU = 5.0                                       # assumed time-to-seniority (sensitivity in exp2)
P = 0.8                                         # assumed completion prob. (cancels in most results)


def save(fig, name):
    path = os.path.join(RES, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
