"""Rank correlation between verifier information (bits) and accuracy gain in E2."""
import json, os
from scipy.stats import spearmanr
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
t = json.load(open(os.path.join(OUT, "e2_table.json")))
lines = []
for k in ["skill", "knowledge"]:
    groups = sorted(set(x["n0"] for x in t if x["kind"] == k)) + ["all"]
    for n0 in groups:
        r = [x for x in t if x["kind"] == k and x["mode"] == "accumulate" and (n0 == "all" or x["n0"] == n0)]
        rho, p = spearmanr([x["verifier_bits"] for x in r], [x["acc_final"] - x["acc0"] for x in r])
        lines.append(f"{k} n0={n0} cells={len(r)} spearman_rho={rho:.3f} p={p:.2g}")
open(os.path.join(OUT, "e2_spearman.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
