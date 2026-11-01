"""Task 1.1: verifier map from two independent ratings, with agreement statistics.

Stand-in for 'two independent experts': two separately prompted model raters
(no shared context, no access to the study's own assessment). Agreement is
measured; disagreements are adjudicated by the mean. Replace with human
experts before relying on the ranking.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

D = Path(__file__).resolve().parent / "data" / "verifier_map"  # shipped with the package


def quadratic_kappa(a, b, k=5):
    a, b = np.array(a) - 1, np.array(b) - 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def load():
    A = {r[0]: tuple(map(float, r[1:])) for r in csv.reader(open(D / "rater_A.csv"))}
    B = {r["id"]: (r["verifier_strength"], r["cost_per_check_usd"], r["latency_hours"], r["best_verifier"])
         for r in json.load(open(D / "rater_B.json"))}
    return A, B


def run():
    A, B = load()
    ids = list(B)
    sa = [int(A[i][0]) for i in ids]; sb = [int(B[i][0]) for i in ids]
    ca = [A[i][1] for i in ids]; cb = [B[i][1] for i in ids]
    la = [A[i][2] for i in ids]; lb = [B[i][2] for i in ids]
    rows = []
    for i in ids:
        s = (A[i][0] + B[i][0]) / 2
        c = (A[i][1] + B[i][1]) / 2
        l = (A[i][2] + B[i][2]) / 2
        # autonomy readiness: strong verifier, cheap and fast checks (higher = readier)
        ready = s - 0.5 * c / 2 - 0.5 * l / 2
        rows.append({"id": i, "strength": s, "cost_exp": c, "latency_exp": l, "readiness": round(ready, 2),
                     "best_verifier": B[i][3], "strength_disagreement": abs(A[i][0] - B[i][0])})
    rows.sort(key=lambda r: -r["readiness"])
    return {
        "agreement": {"strength_quadratic_kappa": round(float(quadratic_kappa(sa, sb)), 3),
                      "strength_exact_match": sum(x == y for x, y in zip(sa, sb)) / len(ids),
                      "cost_spearman": round(float(spearmanr(ca, cb).statistic), 3),
                      "latency_spearman": round(float(spearmanr(la, lb).statistic), 3)},
        "map": rows,
        "note": "Raters are two independently prompted model instances standing in for human experts.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
