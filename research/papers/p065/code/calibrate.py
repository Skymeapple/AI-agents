"""Inverse calibration of (a, kappa) to the two reported moments of Bastani et al. (PNAS 2025):
   GPT Base : practice +48% vs control, exam -17% vs control
   GPT Tutor: practice +127% vs control, exam ~ 0 (not significantly different) vs control
(Moments as reported in the paper / Knowledge at Wharton summary.)
Unobserved structural parameters (s0, eta*T, q) are swept as a sensitivity grid.
Outputs: results/calibration.csv, results/calibration_summary.json
"""
import json, itertools, os
import numpy as np
from model import step, practice

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TARGETS = {"GPT_Base": (1.48, 0.83), "GPT_Tutor": (2.27, 1.00)}
T = 10
A = np.linspace(0, 1, 401)
K = np.linspace(0, 2.0, 401)


def moments(s0, eta, q, a, k):
    """vectorised over a,k meshes; constant a and kappa across the session."""
    s = np.full(np.broadcast(a, k).shape, s0, float)
    sc = s0
    Psum = np.zeros_like(s); Pc = 0.0
    for t in range(T):
        Psum += practice(s, a, q); Pc += sc
        s = step(s, a, eta, k)
        sc = sc + eta * (1 - sc)
    return (Psum / T) / (Pc / T), s / sc


def main():
    os.makedirs(OUT, exist_ok=True)
    Am, Km = np.meshgrid(A, K, indexing="ij")
    rows = []
    for s0, etaT, q in itertools.product([0.2, 0.3, 0.4, 0.5], [0.2, 0.4, 0.8], [0.7, 0.85, 1.0]):
        eta = etaT / T
        pr, er = moments(s0, eta, q, Am, Km)
        for arm, (tp, te) in TARGETS.items():
            err = (np.log(pr) - np.log(tp)) ** 2 + (np.log(er) - np.log(te)) ** 2
            i, j = np.unravel_index(np.argmin(err), err.shape)
            ok = err[i, j] < 1e-4
            rows.append(dict(arm=arm, s0=s0, etaT=etaT, q=q, a=float(A[i]), kappa=float(K[j]),
                             fit_err=float(err[i, j]), feasible=bool(ok),
                             practice_ratio=float(pr[i, j]), exam_ratio=float(er[i, j])))
    import csv
    with open(os.path.join(OUT, "calibration.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    summ = {}
    for arm in TARGETS:
        fe = [r for r in rows if r["arm"] == arm and r["feasible"]]
        summ[arm] = dict(n_cells=sum(r["arm"] == arm for r in rows), n_feasible=len(fe),
                         kappa_min=min(r["kappa"] for r in fe) if fe else None,
                         kappa_median=float(np.median([r["kappa"] for r in fe])) if fe else None,
                         kappa_max=max(r["kappa"] for r in fe) if fe else None,
                         a_min=min(r["a"] for r in fe) if fe else None,
                         a_median=float(np.median([r["a"] for r in fe])) if fe else None,
                         a_max=max(r["a"] for r in fe) if fe else None)
    # retention ratio R = (exam change)/(practice change), reported moments
    summ["retention_ratio_reported"] = {arm: (te - 1) / (tp - 1) for arm, (tp, te) in TARGETS.items()}
    with open(os.path.join(OUT, "calibration_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
