"""Exp 5: What can suites of published dangerous-capability benchmark sizes rule out?

Analytic (exact Clopper-Pearson) one-sided 95% upper bounds on the suite-level
pass@1 rate for item counts of public benchmarks (Cybench: 40 tasks; VCT: 322
questions; WMDP: 3,668 questions; item counts are from the cited papers), for
observed rates 0%, 5%, 20%, treating items as independent and, alternatively,
with the median item-level (family vs task) design effect measured in Exp 1 on METR
TH1.1 (transferring it to other suites is an assumption, labelled as such). Output: results/exp5_suite_sizes.csv
"""
import csv
import json
import os

import numpy as np
from scipy.stats import beta

from common import RESULTS

SUITES = {"Cybench (40 tasks)": 40, "VCT (322 questions)": 322, "WMDP (3,668 questions)": 3668}


def ucb(x, n, a=0.05):
    return 1.0 if x >= n else float(beta.ppf(1 - a, x + 1, n - x))


def main():
    # item-level design effect: (family-cluster SE / task-level SE)^2, median over models.
    # This isolates between-item clustering (task families), excluding repeat-run correlation.
    with open(os.path.join(RESULTS, "exp1_clustering.csv")) as f:
        rr = list(csv.DictReader(f))
    deff = float(np.median([(float(r["se_family"]) / float(r["se_task"])) ** 2 for r in rr]))
    rows = []
    for name, n in SUITES.items():
        for rate in (0.0, 0.05, 0.20):
            x = rate * n
            neff = n / deff
            rows.append(dict(suite=name, n_items=n, observed_rate=rate,
                             ucb95_independent=ucb(x, n),
                             deff_assumed=deff, n_eff=neff,
                             ucb95_clustered=ucb(rate * neff, neff)))
    with open(os.path.join(RESULTS, "exp5_suite_sizes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
