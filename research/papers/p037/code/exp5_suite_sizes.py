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

# rev. 1: VCT removed (its answer format makes the chance level item-dependent, so a
# zero-success bound is not interpretable); WMDP is additionally chance-corrected below.
SUITES = {"Cybench (40 tasks)": 40, "WMDP (3,668 questions)": 3668}


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
    # (rev. 1) chance correction for a four-option multiple-choice suite (WMDP): the
    # above-chance rate kappa = (acc - c)/(1 - c), c = 0.25; upper bound from the exact
    # upper bound on accuracy. Zero-success bounds above do not apply to MC suites.
    cc = []
    c = 0.25
    n = SUITES["WMDP (3,668 questions)"]
    for acc in (0.25, 0.30, 0.40):
        for lab, nn in (("independent", n), ("clustered", n / deff)):
            ua = ucb(acc * nn, nn)
            cc.append(dict(suite="WMDP (4-option MC)", accuracy=acc, items=lab, n_eff=nn,
                           ucb95_accuracy=ua, ucb95_above_chance=max(0.0, (ua - c) / (1 - c))))
    with open(os.path.join(RESULTS, "exp5_chance_corrected.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cc[0].keys()))
        w.writeheader()
        w.writerows(cc)
    for r in cc:
        print(r)
    # (rev. 1) per-task attempt counts in the range used by published cyber evaluations
    # (repeated-sampling k between 3 and 30, as tabulated by Wei et al. 2025): exact 95%
    # upper bound on pass@K after zero successes (Prop. 3a) and the largest K for which a
    # non-randomized level-0.05 test can rule out pass@K >= 0.1 at all (Prop. 3b).
    ct = []
    for m in (3, 5, 10, 12, 30, 100, 300):
        r = dict(m=m)
        for K in (1, 10, 100):
            r[f"ucb95_passK{K}_zero_successes"] = 1 - 0.05 ** (K / m)
        r["max_K_ruleout_tau0.1"] = int(np.floor(m * np.log(1 / 0.9) / np.log(1 / 0.05)))
        r["max_K_ruleout_tau0.5"] = int(np.floor(m * np.log(1 / 0.5) / np.log(1 / 0.05)))
        ct.append(r)
    with open(os.path.join(RESULTS, "exp5_attempt_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ct[0].keys()))
        w.writeheader()
        w.writerows(ct)
    for r in ct:
        print(r)


if __name__ == "__main__":
    main()
