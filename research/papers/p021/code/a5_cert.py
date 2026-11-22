"""Analysis 5: how many tasks does it take to certify a short-task failure rate below eps?

Uses the Kish design effect deff = 1 + (m-1) * rho with the empirical intra-task correlation
rho measured in a2 (frontier agents, tasks < 15 min), and an exact Clopper-Pearson upper
bound on the effective sample. Outputs results/certification.csv and prints a table.
"""
import csv
import numpy as np
from scipy.stats import beta
from common import *

rows = load_runs()
dec = list(csv.DictReader(open(os.path.join(RESULTS, "decomp_T15.csv"))))
front = [r for r in dec if r["date"] >= "2025-04-01"]
rho_emp = float(np.median([float(r["icc"]) for r in front]))
T = task_table(rows, "Claude Opus 4.5 (Inspect)")
n_short15 = int((T["minutes"] < 15).sum()); n_short4 = int((T["minutes"] < 4).sum())
print(f"median frontier ICC (T<15) = {rho_emp:.3f}; tasks <15 min: {n_short15}, <4 min: {n_short4}")


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else beta.ppf(conf, k + 1, n - k)  # one-sided


def n_eff_needed(eps, r, conf=0.95):
    """Smallest effective n such that observing round(r*n) failures gives one-sided
    upper bound < eps (expected-case planning)."""
    n = 1
    while True:
        if cp_upper(int(round(r * n)), n, conf) < eps:
            return n
        n = int(np.ceil(n * 1.02)) + 1
        if n > 10 ** 8:
            return np.nan


out = []
for eps in (0.05, 0.01, 0.001):
    for frac in (0.0, 0.25, 0.5):
        ne = n_eff_needed(eps, frac * eps)
        for m in (1, 4, 8):
            for rho in (0.0, 0.5, rho_emp, 1.0):
                deff = 1 + (m - 1) * rho
                N = int(np.ceil(ne * deff / m))
                out.append(dict(eps=eps, true_over_eps=frac, runs_per_task=m, rho=round(rho, 3),
                                n_eff=ne, tasks_needed=N, runs_needed=N * m))
with open(os.path.join(RESULTS, "certification.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
for r in out:
    if r["runs_per_task"] in (1, 8) and r["rho"] in (0.0, round(rho_emp, 3), 1.0):
        print(r)
