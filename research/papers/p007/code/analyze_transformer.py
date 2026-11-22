"""Summarise transformer runs (exp_transformer.py) -> results/transformer_summary.json + figure.

Interaction for modality m at (D, size, seed): Delta_m = L_unified,m - L_specialist,m
(test cross-entropy, nats/token).  Primary: each modality at its own best-val
checkpoint ('test_per_mod_stop'); secondary: the unified model's single
best-mean-val checkpoint ('test').  Means over seeds with the seed-to-seed
standard error (when >= 2 seeds).
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
runs = [json.loads(l) for l in open(os.path.join(R, "transformer_runs.jsonl"))]
idx = {(tuple(r["mods"]), r["D"], r["d_model"], r["seed"]): r for r in runs}


def spec(m, D, dm, seed, key):
    r = idx.get(((m,), D, dm, seed))
    return None if r is None else r[key][m]


summ = {"grid": {}, "budget": {}, "n_runs": len(runs)}
for key in ["test_per_mod_stop", "test"]:
    G = {}
    for D in [25000, 100000]:
        for dm in [16, 32, 64, 128]:
            per = defaultdict(list)
            for seed in [0, 1]:
                u = idx.get((("image", "audio"), D, dm, seed))
                if u is None:
                    continue
                for m in ["image", "audio"]:
                    s = spec(m, D, dm, seed, "test_per_mod_stop")  # specialists: own stop
                    if s is not None:
                        per[m].append((u[key][m], s))
            if not per:
                continue
            ent = dict(n_params=idx[(("image",), D, dm, 0)]["n_params"] if (("image",), D, dm, 0) in idx else None)
            for m, v in per.items():
                v = np.array(v); d_ = v[:, 0] - v[:, 1]
                ent[m] = dict(L_u=v[:, 0].mean(), L_s=v[:, 1].mean(), delta=d_.mean(),
                              delta_se=(d_.std(ddof=1) / np.sqrt(len(d_))) if len(d_) > 1 else None,
                              ratio=(v[:, 0] / v[:, 1]).mean(), n_seeds=len(d_))
            if "image" in ent and "audio" in ent and ent["image"]["n_seeds"] == ent["audio"]["n_seeds"]:
                ent["mean_delta"] = (ent["image"]["delta"] + ent["audio"]["delta"]) / 2
            G[f"D={D},d={dm}"] = ent
    summ["grid"][key] = G

# P4: modality budget at d_model=32, D=25k
order = ["image", "audio", "text", "fashion"]
B = {}
for M in [1, 2, 3, 4]:
    mods = tuple(order[:M])
    rows = defaultdict(list)
    for seed in [0, 1]:
        u = idx.get((mods, 25000, 32, seed))
        if u is None:
            continue
        for m in mods:
            s = spec(m, 25000, 32, seed, "test_per_mod_stop")
            if s is not None:
                rows[m].append((u["test_per_mod_stop"][m], s))
    if rows:
        B[str(M)] = {m: dict(L_u=float(np.mean([a for a, _ in v])), L_s=float(np.mean([b for _, b in v])),
                             delta=float(np.mean([a - b for a, b in v])), n_seeds=len(v))
                     for m, v in rows.items()}
        B[str(M)]["mean_delta"] = float(np.mean([B[str(M)][m]["delta"] for m in rows]))
summ["budget"] = B
json.dump(summ, open(os.path.join(R, "transformer_summary.json"), "w"), indent=1, default=float)

# figure
fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
G = summ["grid"]["test_per_mod_stop"]
for ax, m in zip(axs[:2], ["image", "audio"]):
    for D, c in [(25000, "C0"), (100000, "C3")]:
        xs, ys, es = [], [], []
        for dm in [16, 32, 64, 128]:
            e = G.get(f"D={D},d={dm}", {}).get(m)
            if e:
                xs.append(G[f"D={D},d={dm}"]["n_params"] or dm); ys.append(e["delta"]); es.append(e["delta_se"] or 0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=c, ms=4, label=f"D={D // 1000}k tokens/modality")
    ax.axhline(0, color="k", lw=.8, ls="--"); ax.set_xscale("log")
    ax.set_xlabel("parameters"); ax.set_ylabel("L_unified - L_specialist (nats/token)")
    ax.set_title(f"{m} (unified = image+audio)", fontsize=9); ax.legend(fontsize=7)
ax = axs[2]
for m, c in zip(order, ["C0", "C1", "C2", "C4"]):
    xs = [int(M) for M in B if m in B[M]]; ys = [B[str(M)][m]["delta"] for M in xs]
    if xs:
        ax.plot(xs, ys, "o-", color=c, label=m)
ax.axhline(0, color="k", lw=.8, ls="--"); ax.set_xlabel("number of modalities M (d_model=32, D=25k)")
ax.set_ylabel("L_unified - L_specialist"); ax.set_xticks([1, 2, 3, 4]); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "figH_transformer.png"), dpi=150)
print(json.dumps(summ, indent=1, default=lambda x: round(float(x), 4))[:6000])
