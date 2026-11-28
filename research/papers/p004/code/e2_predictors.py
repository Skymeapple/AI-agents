"""Re-analysis of E2 (reviewer point M4): does verifier information predict gain
better than simpler functions of (a,b)? Pooled and within-budget Spearman."""
import json, os, numpy as np
from scipy.stats import spearmanr
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
t = [x for x in json.load(open(os.path.join(OUT, "e2_table.json"))) if x["mode"] == "accumulate"]
preds = {"sum I(C;V)": lambda x: x["verifier_bits"], "-b/a": lambda x: -x["b"] / x["a"],
         "a-b": lambda x: x["a"] - x["b"], "precision": lambda x: x["precision"]}
lines = []
for k in ["skill", "knowledge"]:
    r = [x for x in t if x["kind"] == k]
    gain = [x["acc_final"] - x["acc0"] for x in r]
    n0s = sorted(set(x["n0"] for x in r))
    for name, f in preds.items():
        rho = spearmanr([f(x) for x in r], gain)[0]
        within = [spearmanr([f(x) for x in r if x["n0"] == n], [x["acc_final"] - x["acc0"] for x in r if x["n0"] == n])[0] for n in n0s]
        lines.append(f"{k:9s} {name:12s} pooled rho={rho:.3f}  within-n0 mean rho={np.mean(within):.3f}  ({', '.join('%.3f' % w for w in within)})")
open(os.path.join(OUT, "e2_predictors.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
