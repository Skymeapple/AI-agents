"""E4: Translate data multipliers into years of delay of the human-data horizon.

Model: delay Dt = ln(M) / ln(g_D) for an effective-data multiplier M, where g_D is
the yearly growth of compute-optimal dataset size.
Sourced inputs (cited in the paper):
  - stock ~300T tokens, 90% CI 100T-1000T; full use projected 2026-2032, an 80%
    interval (Villalobos et al., 2024).
  - frontier training compute growth 4-5x / year (Sevilla & Roldan, 2024).
  - compute-optimal data ~ C^0.5 (Hoffmann et al., 2022) => g_D = 2.0..2.24x / year.
Multipliers, each labelled by provenance:
  [measured-ext] reported in a cited paper (NOT a human-data multiplier in our strict
                 sense unless stated; speed-ups to a target loss are upper-bound-ish proxies)
  [measured-here] measured in this paper's E5 (tiny transformer, planted facts)
  [model]        ceilings from Prop. 2 for hypothetical retention s
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
gD = {"compute 4x/yr": 2.0, "compute 5x/yr": np.sqrt(5.0)}
e5 = json.load(open(os.path.join(OUT, "e5_analysis.json")))
e5max = max(m["M"] for m in e5["multipliers"] if m["kind"] == "paraphrase")
e5sat = [m["M"] for m in e5["multipliers"] if m["kind"] == "paraphrase" and m["alpha"] == 1.0 and m["N0"] == 2000 and m["r"] == 7][0]
entries = [
    ("[measured-ext] 4-epoch repetition ~ fresh data (Muennighoff et al., 2023)", 4.0),
    ("[measured-ext] WRAP rephrasing, ~3x pre-training speed-up (Maini et al., 2024)", 3.0),
    ("[measured-ext] 1/3 rephrased mix, 5x speed-up to same val. loss, low end (Kang et al., 2025)", 5.0),
    ("[measured-ext] 1/3 rephrased mix, 10x speed-up, high end (Kang et al., 2025)", 10.0),
    ("[measured-here] E5 tiny LM, paraphrase saturation at coverage (alpha=1, N0=2000, r=7)", e5sat),
    ("[measured-here] E5 tiny LM, largest paraphrase multiplier observed", e5max),
    ("[model] source-grounded paraphrase ceiling, s=0.5", 2.0),
    ("[model] source-grounded paraphrase ceiling, s=0.1", 10.0),
    ("[model] teacher s_g=0.5, learner s=0.1", 1 + 0.5 * 0.9 / 0.1),
    ("[model] source-grounded paraphrase ceiling, s=0.01", 100.0),
    ("[sourced] stock 1000T (upper end of 90% CI) vs 300T central", 1000 / 300),
]
rows = [dict(source=k, M=float(M), growth=name, delay_years=float(np.log(M) / np.log(g))) for k, M in entries for name, g in gD.items()]
json.dump(rows, open(os.path.join(OUT, "e4_horizon.json"), "w"), indent=1)
for k, M in entries:
    print(f"{k:95s} M={M:6.2f}  delay {np.log(M)/np.log(2):.2f} / {np.log(M)/np.log(np.sqrt(5)):.2f} yr")
Ms = np.logspace(0, 3, 200)
fig, ax = plt.subplots(figsize=(6, 3.8))
for name, g in gD.items():
    ax.plot(Ms, np.log(Ms) / np.log(g), label=f"dataset growth {g:.2f}x/yr ({name})")
for k, M in entries[:6]:
    ax.axvline(M, ls=":", c="gray", lw=.7)
ax.text(3.1, 7, "measured anchors (ext. + E5)", fontsize=7, rotation=90)
ax.set_xscale("log"); ax.set_xlabel("effective-data multiplier M"); ax.set_ylabel("delay of data horizon (years)")
ax.legend(fontsize=7); ax.set_title("Horizon delay from a data multiplier")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_horizon.png"), dpi=150)
