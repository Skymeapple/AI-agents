"""E3: Translate data multipliers into years of delay of the human-data horizon.

Sourced inputs (see paper, Sec. 6):
  - effective public human text stock ~300T tokens, 90% CI 100T-1000T, and full
    utilisation projected between 2026 and 2032 (Villalobos et al., 2024).
  - frontier training compute growth 4-5x / year (Sevilla & Roldan, Epoch AI, 2024).
  - compute-optimal data scales as C^0.5 (Hoffmann et al., 2022) => dataset growth
    g_D = sqrt(4..5) = 2.0..2.24x / year (derived, assumes compute-optimal scaling).
  - up to ~4 epochs of repetition cost little loss (Muennighoff et al., 2023).
Model: delay Dt = ln(M) / ln(g_D) for an effective-data multiplier M.
Multipliers M come from the model in this paper: knowledge component M <= 1/s
(source-grounded amplification), M = 1 + s_g(1-s)/s for a teacher of retention s_g.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
gD = {"compute 4x/yr": 2.0, "compute 5x/yr": np.sqrt(5.0)}
Ms = np.logspace(0, 3, 200)
rows = []
for s in [0.5, 0.2, 0.1, 0.03, 0.01]:
    for sg in [None, 0.5]:
        M = 1 / s if sg is None else 1 + sg * (1 - s) / s
        for name, g in gD.items():
            rows.append(dict(s=s, s_gen=("source-grounded (1.0)" if sg is None else sg), M=M, growth=name,
                             delay_years=float(np.log(M) / np.log(g))))
# repetition (4 epochs) for reference
for name, g in gD.items():
    rows.append(dict(s=None, s_gen="4-epoch repetition", M=4.0, growth=name, delay_years=float(np.log(4) / np.log(g))))
# stock uncertainty expressed as a multiplier relative to the 300T central estimate
stock = {"100T (5th pct)": 100 / 300, "1000T (95th pct)": 1000 / 300}
for k, v in stock.items():
    for name, g in gD.items():
        rows.append(dict(s=None, s_gen="stock " + k, M=v, growth=name, delay_years=float(np.log(v) / np.log(g))))
json.dump(rows, open(os.path.join(OUT, "e3_horizon.json"), "w"), indent=1)
for r in rows:
    print(r)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for name, g in gD.items():
    ax.plot(Ms, np.log(Ms) / np.log(g), label=f"dataset growth {g:.2f}x/yr ({name})")
for s in [0.5, 0.1, 0.01]:
    ax.axvline(1 / s, ls=":", c="gray"); ax.text(1 / s * 1.05, 0.3, f"1/s, s={s}", rotation=90, fontsize=7)
ax.axvline(4, ls="--", c="k", lw=.8); ax.text(4.2, 6.5, "4 epochs", rotation=90, fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("effective-data multiplier M"); ax.set_ylabel("delay of data horizon (years)")
ax.legend(fontsize=7); ax.set_title("Horizon delay from a data multiplier")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_horizon.png"), dpi=150)
