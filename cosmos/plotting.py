"""One plotting style for every figure: thin marks, recessive axes, fixed hue order."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import OUT  # noqa: E402

# Categorical slots, always used in this order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def figure(ncols: int = 1, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, height))
    return fig, axes


def save(fig, name: str) -> str:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(OUT.parent.parent))
