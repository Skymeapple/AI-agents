"""Revision r1 re-analysis of E2 (trained 1/2/4-layer Transformers, delta-rule model).
* pools the 300 evaluation streams of every seed and reports Wilson 95% intervals;
* Cochran-Armitage-style trend test (logistic regression slope on log2 R, pooled streams)
  for every row of the age-0 table, plus a check of strict monotonicity of the means;
* compares the original 1-layer models (tf1) with 1-layer models trained on the
  R-covering distribution (tf1c, train_toy.py --cover).
Writes results/e2_r1_stats.json and results/e2_r1_tables.md.
"""
import json, os, numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = {}
for fn in ["train_toy.json", "tt_delta0.json", "tt_delta1.json", "tt_tf.json", "tt_tf4b.json", "tt_tf1c.json"]:
    if os.path.exists(os.path.join(OUT, fn)): R.update(json.load(open(os.path.join(OUT, fn))))
EB = 300
names = {"tf1": "Transformer L=1", "tf1c": "Transformer L=1, R-covering training", "tf2": "Transformer L=2",
         "tf4": "Transformer L=4", "delta": "Delta-rule"}
EVAL_R = [1, 2, 4, 8, 16, 32, 64]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def trend_test(Rs, ks, ns):
    """Logistic regression of success on log2 R (pooled Bernoulli outcomes); Wald test of slope."""
    import numpy.linalg as la
    x = np.concatenate([np.full(n, np.log2(r)) for r, n in zip(Rs, ns)])
    y = np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for k, n in zip(ks, ns)])
    X = np.c_[np.ones_like(x), x]; b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-12
        H = X.T @ (X * W[:, None]); g = X.T @ (y - p); b = b + la.solve(H, g)
    se = np.sqrt(la.inv(H)[1, 1]); z = b[1] / se
    return float(b[1]), float(se), float(2 * stats.norm.sf(abs(z)))


out = {}; md = []
models = [m for m in names if any(v["model"] == m for v in R.values())]
for m in models:
    runs = [v for v in R.values() if v["model"] == m]
    out[m] = {"n_seeds": len(runs)}
    keys = runs[0]["grid"].keys()
    for key in keys:
        ks = [int(round(r["grid"][key] * EB)) for r in runs if key in r["grid"]]
        k, n = sum(ks), EB * len(ks); lo, hi = wilson(k, n)
        out[m][key] = dict(acc=k / n, lo=lo, hi=hi, k=k, n=n, per_seed=[x / EB for x in ks])

md.append("### Age-0 R-sweep, pooled over seeds, Wilson 95% CI; trend = logistic slope on log2 R (p-value); mono = means strictly decreasing\n")
md.append("| model | N | " + " | ".join(f"R={r}" for r in EVAL_R) + " | slope (p) | mono |")
md.append("|---|---|" + "---|" * len(EVAL_R) + "---|---|")
for af in ["0.0", "0.25"]:
    if af == "0.25":
        md.append("\n### Latest version at age N/4 (same format)\n")
        md.append("| model | N | " + " | ".join(f"R={r}" for r in EVAL_R) + " | slope (p) | mono |")
        md.append("|---|---|" + "---|" * len(EVAL_R) + "---|---|")
    for m in models:
        for N in [32, 128, 512]:
            cells, Rs, ks, ns = [], [], [], []
            for r in EVAL_R:
                key = f"N{N}_R{r}_fixage{af}"
                if key in out[m]:
                    c = out[m][key]; cells.append(f"{c['acc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}]")
                    Rs.append(r); ks.append(c["k"]); ns.append(c["n"])
                else: cells.append("–")
            sl, se, pv = trend_test(Rs, ks, ns)
            accs = [k / n for k, n in zip(ks, ns)]; mono = all(a > b for a, b in zip(accs, accs[1:]))
            out[m][f"trend_N{N}_fixage{af}"] = dict(slope=sl, se=se, p=pv, strictly_monotone=mono)
            md.append(f"| {names[m]} | {N} | " + " | ".join(cells) + f" | {sl:.2f} ({pv:.1e}) | {'yes' if mono else 'no'} |")
md.append("\n### Never-revised fact (R = 1) vs age, pooled, Wilson 95% CI\n")
md.append("| model | N | age 0 | age N/4 | age N/2 | age 0.95N |"); md.append("|---|---|---|---|---|---|")
for m in models:
    for N in [32, 128, 512]:
        ks_ = [f"N{N}_R1_fixage0.0", f"N{N}_R1_fixage0.25", f"N{N}_R1_age0.5", f"N{N}_R1_age0.95"]
        md.append(f"| {names[m]} | {N} | " + " | ".join(f"{out[m][k]['acc']:.2f} [{out[m][k]['lo']:.2f},{out[m][k]['hi']:.2f}]" if k in out[m] else "–" for k in ks_) + " |")
json.dump(out, open(os.path.join(OUT, "e2_r1_stats.json"), "w"), indent=1)
open(os.path.join(OUT, "e2_r1_tables.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
