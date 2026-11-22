"""Analyze E6: decorrelated verifier grid and verifier-cost experiments."""
import json, os, numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
runs = json.load(open(os.path.join(OUT, "e6_verifier_design.json")))["runs"]
lines, rep = [], {}

def partial_rank_corr(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(u, v):
        A = np.vstack([v, np.ones_like(v)]).T
        return u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])

PRED = {"bits_cond": lambda d: d["bits_cond"], "bits_marg": lambda d: d["bits_marg"],
        "-b/a": lambda d: -d["b"] / d["a"], "a-b": lambda d: d["a"] - d["b"], "precision": lambda d: d["precision"]}
rng = np.random.default_rng(0)
rep["A"] = {}
for kind in ["skill", "knowledge"]:
    R = [x for x in runs if x["vtype"] == "ab" and x["kind"] == kind]
    settings = sorted(set((x["p1"], x["p2"]) for x in R))
    seeds = sorted(set(x["seed"] for x in R))
    by = {(x["p1"], x["p2"], x["seed"]): x for x in R}
    def cell_stats(seed_list):
        out = []
        for (a, b) in settings:
            xs = [by[(a, b, s)] for s in seed_list]
            out.append(dict(a=a, b=b, gain=np.mean([x["accs"][-1] - x["accs"][0] for x in xs]),
                            bits_cond=np.mean([x["bits_cond"] for x in xs]), bits_marg=np.mean([x["bits_marg"] for x in xs]),
                            precision=np.mean([x["precision"] for x in xs]), acc0=np.mean([x["accs"][0] for x in xs]),
                            accf=np.mean([x["accs"][-1] for x in xs]), sd=np.std([x["accs"][-1] for x in xs], ddof=1)))
        return out
    cells = cell_stats(seeds)
    gain = [c["gain"] for c in cells]
    res = {}
    boots = {k: [] for k in PRED}; diff = []
    for _ in range(2000):
        bs = list(rng.choice(seeds, len(seeds)))
        cb = cell_stats(bs); gb = [c["gain"] for c in cb]
        rr = {k: spearmanr([f(c) for c in cb], gb)[0] for k, f in PRED.items()}
        for k in PRED: boots[k].append(rr[k])
        diff.append(rr["bits_cond"] - rr["-b/a"])
    for k, f in PRED.items():
        rho = spearmanr([f(c) for c in cells], gain)[0]
        lo, hi = np.nanpercentile(boots[k], [5, 95])
        res[k] = dict(rho=float(rho), ci90=[float(lo), float(hi)])
        lines.append(f"A {kind:9s} {k:10s} rho={rho:.3f} seed-bootstrap 90% CI [{lo:.3f},{hi:.3f}]")
    pc = partial_rank_corr([c["bits_cond"] for c in cells], gain, [-c["b"] / c["a"] for c in cells])
    pb = partial_rank_corr([-c["b"] / c["a"] for c in cells], gain, [c["bits_cond"] for c in cells])
    res["partial_bits_given_ratio"] = pc; res["partial_ratio_given_bits"] = pb
    res["P(rho_bits_cond > rho_-b/a)"] = float(np.mean(np.array(diff) > 0))
    lines.append(f"A {kind:9s} partial rank corr: bits|b/a = {pc:.3f}; b/a|bits = {pb:.3f}; bootstrap P(rho_bits>rho_b/a)={res['P(rho_bits_cond > rho_-b/a)']:.3f}")
    res["cells"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()} for c in cells]
    for c in cells:
        lines.append(f"    a={c['a']:.2f} b={c['b']:.3f} b/a={c['b']/c['a']:.1f} acc0={c['acc0']:.3f} accF={c['accf']:.3f}({c['sd']:.3f}) gain={c['gain']:+.3f} "
                     f"prec={c['precision']:.3f} kbits_cond={c['bits_cond']/1e3:.2f} kbits_marg={c['bits_marg']/1e3:.2f}")
    rep["A"][kind] = res

# Part B
rep["B"] = {}
for vt, kind, key in [("cover", "knowledge", "p1"), ("quant", "skill", "p1")]:
    R = [x for x in runs if x["vtype"] == vt]
    rows = []
    for v in sorted(set(str(x[key]) for x in R), key=lambda s: (s == "None", float(s) if s != "None" else 0)):
        xs = [x for x in R if str(x[key]) == v]
        row = dict(level=v, cost_bits=float(np.mean([x["verifier_bits_cost"] for x in xs])),
                   acc0=float(np.mean([x["accs"][0] for x in xs])), accF=float(np.mean([x["accs"][-1] for x in xs])),
                   accF_sd=float(np.std([x["accs"][-1] for x in xs], ddof=1)),
                   a_emp=float(np.mean([x["a_emp"] for x in xs])), b_emp=float(np.mean([x["b_emp"] for x in xs])),
                   precision=float(np.mean([x["precision"] for x in xs])), kbits_cond=float(np.mean([x["bits_cond"] for x in xs]) / 1e3))
        if vt == "cover":
            row["coverage_bound"] = float(np.mean([x["coverage_bound"] for x in xs]))
            if v == "0.0": row["cost_bits"] = 0.0
        else:
            row["verifier_agreement"] = float(np.mean([x["verifier_agreement"] for x in xs]))
        row["gain"] = row["accF"] - row["acc0"]
        row["gain_per_kbit_cost"] = row["gain"] / (row["cost_bits"] / 1e3) if row["cost_bits"] > 0 else None
        rows.append(row)
        lines.append(f"B {vt:5s} {kind:9s} level={v:5s} cost={row['cost_bits']:6.0f} bits acc0={row['acc0']:.3f} accF={row['accF']:.3f}({row['accF_sd']:.3f}) "
                     f"gain={row['gain']:+.3f} a={row['a_emp']:.3f} b={row['b_emp']:.3f} prec={row['precision']:.3f} "
                     + (f"bound={row['coverage_bound']:.3f}" if vt == "cover" else f"verifier_agree={row['verifier_agreement']:.3f}"))
    rep["B"][vt] = rows
json.dump(rep, open(os.path.join(OUT, "e6_analysis.json"), "w"), indent=1)
open(os.path.join(OUT, "e6_summary.txt"), "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for kind, mk in [("skill", "o"), ("knowledge", "s")]:
    for c in rep["A"][kind]["cells"]:
        col = plt.cm.viridis(c["b"] / c["a"] / 0.8)
        ax[0].scatter(c["bits_cond"] / 1e3, c["gain"], marker=mk, color=col, s=15 + 50 * c["a"])
ax[0].set_xlabel("verifier information, conditional q_j (kbits)"); ax[0].set_ylabel("accuracy gain (8 rounds)")
ax[0].set_title("(a) decorrelated grid: size = a, colour = b/a\n(o skill n0=64, s knowledge n0=200)")
for vt, kind, mk in [("cover", "knowledge", "s-"), ("quant", "skill", "o-")]:
    rows = rep["B"][vt]
    ax[1].plot([r["cost_bits"] for r in rows], [r["gain"] for r in rows], mk, label=f"{kind} ({'lookup' if vt=='cover' else 'quantized-program'} verifier)")
ax[1].set_xscale("symlog", linthresh=100); ax[1].set_xlabel("verifier description length (bits)"); ax[1].set_ylabel("accuracy gain")
ax[1].legend(fontsize=7); ax[1].set_title("(b) gain vs verifier cost")
rows = rep["B"]["cover"]
ax[2].plot([float(r["level"]) for r in rows], [r["accF"] for r in rows], "s-", label="knowledge: final acc")
ax[2].plot([float(r["level"]) for r in rows], [r["coverage_bound"] for r in rows], "k--", label="bound: human ∪ verifier coverage")
ax[2].set_xlabel("verifier coverage kappa (fraction of facts stored)"); ax[2].set_ylabel("Zipf-weighted accuracy"); ax[2].legend(fontsize=7)
ax[2].set_title("(c) knowledge: accuracy capped by coverage")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_verifier_design.png"), dpi=150)
