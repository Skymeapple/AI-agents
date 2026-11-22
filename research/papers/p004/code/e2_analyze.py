"""Analyze E2 outputs: tables, human-equivalent N_eff, figure."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = json.load(open(os.path.join(OUT, "e2_selftrain.json")))
runs, curves, grids = R["runs"], R["curves"], R["grids"]

def n_equiv(kind, acc):
    g = np.array(grids[kind], float); c = np.mean(curves[kind], 0)
    c = np.maximum.accumulate(c)  # monotone envelope
    if acc <= c[0]: return float(g[0]) if acc >= c[0] - 1e-9 else float("nan")
    if acc >= c[-1]: return float(">%d" % 0 or 0) if False else float("inf")
    return float(np.exp(np.interp(acc, c, np.log(g))))

def key(r): return (r["kind"], r["n0"], r["a"], r["b"], r["mode"])
agg = {}
for r in runs:
    agg.setdefault(key(r), []).append(r)
table = []
for k, rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][4], kv[0][1], -kv[0][2], kv[0][3])):
    A = np.array([x["accs"] for x in rs])
    a0, af = A[:, 0].mean(), A[:, -1].mean()
    row = dict(kind=k[0], n0=k[1], a=k[2], b=k[3], mode=k[4],
               acc0=round(float(a0), 4), acc_final=round(float(af), 4), acc_final_sd=round(float(A[:, -1].std()), 4),
               precision=round(float(np.mean([x["precision"] for x in rs])), 3),
               n_accepted=int(np.mean([x["n_accepted"] for x in rs])),
               verifier_bits=round(float(np.mean([x["bits"] for x in rs])), 1),
               N_eff=n_equiv(k[0], af), q_star=round(k[3] / (k[2] + k[3]), 3))
    row["N_eff_over_n0"] = row["N_eff"] / k[1]
    table.append(row)
json.dump(table, open(os.path.join(OUT, "e2_table.json"), "w"), indent=1)
hdr = "kind mode n0 a b q*=b/(a+b) acc0 acc_final(sd) precision accepted verifier_bits N_eff N_eff/n0"
lines = [hdr]
for t in table:
    lines.append(f"{t['kind']:9s} {t['mode']:10s} {t['n0']:4d} {t['a']:.1f} {t['b']:.2f} {t['q_star']:.3f} "
                 f"{t['acc0']:.3f} {t['acc_final']:.3f}({t['acc_final_sd']:.3f}) {t['precision']:.3f} "
                 f"{t['n_accepted']:6d} {t['verifier_bits']:8.0f} {t['N_eff']:8.0f} {t['N_eff_over_n0']:6.2f}")
open(os.path.join(OUT, "e2_table.txt"), "w").write("\n".join(lines))
print("\n".join(lines))
print("plain curves:", {k: np.round(np.mean(v, 0), 3).tolist() for k, v in curves.items()})

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for j, (kind, n0) in enumerate([("skill", 64), ("knowledge", 800)]):
    for (a, b) in [(1, 0), (1, 0.05), (1, 0.15), (1, 0.3), (1, 0.6), (1, 1.0), (0.7, 0.1)]:
        rs = agg.get((kind, n0, a, b, "accumulate"))
        if not rs: continue
        A = np.mean([x["accs"] for x in rs], 0)
        ax[j].plot(A, "o-", ms=3, label=f"a={a}, b={b}")
    for (a, b) in [(1, 0.6), (1, 1.0)]:
        rs = agg.get((kind, n0, a, b, "replace"))
        if rs:
            ax[j].plot(np.mean([x["accs"] for x in rs], 0), "x--", ms=4, label=f"REPLACE a={a}, b={b}")
    ax[j].set_title(f"({'ab'[j]}) {kind}, n0={n0} human samples"); ax[j].set_xlabel("self-training round")
    ax[j].set_ylabel("test accuracy"); ax[j].legend(fontsize=6)
# phase plot: gain vs b/(a) with initial q
for kind, mk in [("skill", "o"), ("knowledge", "s")]:
    for t in table:
        if t["kind"] != kind or t["mode"] != "accumulate": continue
        ax[2].scatter(t["verifier_bits"] / 1000, t["acc_final"] - t["acc0"], marker=mk,
                      c=[[t["b"] / t["a"] / 1.0 * 0.9, 0.2, 1 - min(t["b"] / t["a"], 1) * 0.9]], s=25)
ax[2].axhline(0, c="k", lw=.6)
ax[2].set_xlabel("external information from verifier (kbits, sum of I(C;V))"); ax[2].set_ylabel("accuracy gain after 8 rounds")
ax[2].set_title("(c) gain vs verifier information\n(o skill, s knowledge; red = larger b/a)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_selftrain.png"), dpi=150)
