"""Table 5 / fig7 for the Othello section, from results/othello/*.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import OUT
J = lambda f: json.load(open(os.path.join(OUT, f))) if os.path.exists(os.path.join(OUT, f)) else None
COLS = ["all", "empty", "occupied", "age 0", "age 1-3", "age 4-7", "age 8-15", "age 16+"]
rows = []
for tag, f, mf in [("6k", "analysis_6k.json", "mlp_probe_gpt_6k.json"), ("long", "analysis_long.json", "mlp_probe_gpt_long.json")]:
    a, m = J(f), J(mf)
    if a is None: continue
    steps = a["train_steps"] if tag == "6k" else 6000 + a["train_steps"]
    for mod in ["trained", "untrained_twin"]:
        if mod == "untrained_twin" and tag == "long": continue
        b = a[mod]["best_layer"]
        lab = f"{'trained' if mod == 'trained' else 'untrained twin'} ({steps} steps), linear probe L{b}, legal top-1 {a[mod]['legal_top1']:.3f}" if mod == "trained" else f"untrained twin, linear probe L{b}, legal top-1 {a[mod]['legal_top1']:.3f}"
        rows.append((lab, a[mod]["layers"][str(b)]))
        if m and mod in m:
            rows.append((lab.split(", linear")[0] + ", MLP probe (last layer)", m[mod]))
w = J("window_mlp.json")
for k in ["0", "1", "4", "8", "16"]:
    if w and k in w: rows.append((f"floor: last {k} moves + t" if k != "0" else "floor: t only", w[k]))
p0, p8 = J("placement_k0.json"), J("placement.json")
if p0: rows.append(("floor: placement + t", p0["placement+last0"]))
if p8 and "placement+last8" in p8: rows.append(("floor: placement + last 8 moves + t", p8["placement+last8"]))
L = ["| model / floor | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
for lab, d in rows:
    L.append(f"| {lab} | " + " | ".join(f"{d[c][0]:.3f}" if c in d else "–" for c in COLS) + " |")
n = rows[0][1]
L.append("\nCounts (square-positions): " + ", ".join(f"{c}: {n[c][1]}" for c in COLS if c in n))
open(os.path.join(OUT, "..", "summary_othello.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
xs = COLS[3:]
for lab, d in rows:
    if "MLP" in lab: continue
    style = "--" if lab.startswith("floor") else "-"
    ax.plot(range(len(xs)), [d[c][0] for c in xs], style, marker="o", ms=3, label=lab.split(", legal")[0])
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_ylabel("accuracy on occupied squares (mine/theirs)")
ax.set_xlabel("square age (moves since last change)"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "..", "fig7_othello_age.png"), dpi=160)
