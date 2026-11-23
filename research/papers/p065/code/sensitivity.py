"""Robustness checks (run after experiments.py).

S1  Closed-form last-period offloading rule (Prop. 3) vs the dynamic programme's last-period policy.
S2  Sensitivity of the headline cohort effects to (T, eta, kappa_base, prior-skill distribution):
      loss_open  = 1 - mean s_T(Open AI, w=0.3, q=0.9) / mean s_T(full engagement)
      loss_open7 = same with w=0.7
      gain_fade  = mean s_T(Tutor k(s)+fading) / mean s_T(full engagement) - 1
      corr_grade_open = corr(course grade, s_T) under Open AI, w=0.3, q=0.9
S3  Guardrail non-monotonicity: share of (beta, c, s0) configurations in which the private-best response
    yields LOWER terminal skill at a higher kappa (kappa grid, q=0.9, w=0.3).
Outputs: ../results/sensitivity.json, ../results/sensitivity_table.csv
"""
import json, os, itertools, csv
import numpy as np
from model import solve_student, simulate, S_GRID

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
RNG = np.random.default_rng(7)
res = {}

# ---------------- S1
def s1():
    mism, tot = 0, 0
    for q, k, w, beta, c in itertools.product([0.3, 0.6, 0.9], [0.0, 0.3, 0.6], [0.0, 0.3, 0.7],
                                               [0.25, 0.5, 1.0], [0.005, 0.02]):
        T, eta, L = 20, 0.04, 1.0
        pol = solve_student(T, q, eta, k, w=w, L=L, beta=beta, c=c, a_grid=np.array([0.0, 1.0]))
        gain = (1 - w) * (q - S_GRID) / T + c - beta * (w + L) * eta * (1 - k) * (1 - S_GRID)
        pred = (gain > 1e-12).astype(float)
        # exclude knife-edge grid points
        ok = np.abs(gain) > 1e-9
        mism += int(np.sum(pol[-1][ok] != pred[ok])); tot += int(ok.sum())
    res["S1"] = dict(grid_points_checked=tot, mismatches=mism)
    print("S1", res["S1"])

# ---------------- S2
def cohort(T, eta, kb, dist, N=2000, q=0.9):
    s0 = RNG.beta(2, 3, N) if dist == "beta23" else (RNG.beta(5, 5, N) if dist == "beta55" else RNG.beta(1.5, 4, N))
    betas = np.array([0.1, 0.3, 0.5, 0.7, 0.9]); cs = np.array([0.005, 0.01, 0.02])
    bi = RNG.integers(0, 5, N); ci = RNG.integers(0, 3, N)
    k0f, k1f = 1.4, 0.8
    regs = {"bench": (kb, 0, 0.3, lambda t, S, A: A == 0), "open3": (kb, 0, 0.3, None),
            "open7": (kb, 0, 0.7, None),
            "fade": (k0f, k1f, 0.3, lambda t, S, A: (A == 0) | (k0f - k1f * S > 1))}
    out = {}
    for name, (k0, k1, w, al) in regs.items():
        sT = np.zeros(N); Pb = np.zeros(N)
        for i, b in enumerate(betas):
            for j, c in enumerate(cs):
                m = (bi == i) & (ci == j)
                pol = solve_student(T, q, eta, k0, k1, w=w, L=1.0, beta=b, c=c, allowed=al)
                r = simulate(s0[m], pol, T, q, eta, k0, k1)
                sT[m] = r["sT"]; Pb[m] = r["Pbar"]
        grade = (1 - w) * Pb + w * sT + RNG.normal(0, 0.05, N)
        out[name] = (sT.mean(), np.corrcoef(grade, sT)[0, 1])
    return dict(loss_open=1 - out["open3"][0] / out["bench"][0], loss_open7=1 - out["open7"][0] / out["bench"][0],
                gain_fade=out["fade"][0] / out["bench"][0] - 1, corr_grade_open=out["open3"][1],
                corr_grade_bench=out["bench"][1])


def s2():
    rows = []
    base = dict(T=20, eta=0.04, kb=0.305, dist="beta23")
    variants = [base] + [dict(base, T=t) for t in [10, 40]] + [dict(base, eta=e) for e in [0.02, 0.08]] + \
               [dict(base, kb=k) for k in [0.0, 0.61]] + [dict(base, dist=d) for d in ["beta55", "beta154"]]
    for v in variants:
        r = cohort(v["T"], v["eta"] if v["T"] == 20 else v["eta"] * 20 / v["T"], v["kb"], v["dist"])
        row = dict(v, eta_effective=(v["eta"] if v["T"] == 20 else v["eta"] * 20 / v["T"]), **{k: float(x) for k, x in r.items()})
        rows.append(row); print("S2", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()})
    with open(os.path.join(OUT, "sensitivity_table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    res["S2"] = rows

# ---------------- S3
def s3():
    ks = np.linspace(0, 0.95, 20); T, eta = 20, 0.04
    n_nonmono, n = 0, 0; examples = []; curves = []
    for beta, c, s0 in itertools.product([0.25, 0.5, 1.0], [0.005, 0.01, 0.02], [0.1, 0.3, 0.5]):
        sT = []
        for k in ks:
            pol = solve_student(T, 0.9, eta, k, w=0.3, L=1.0, beta=beta, c=c)
            sT.append(simulate(s0, pol, T, 0.9, eta, k)["sT"][0])
        sT = np.array(sT); n += 1
        curves.append(dict(beta=beta, c=c, s0=s0, kappa=ks.tolist(), sT=sT.tolist(), sT_at_k0=float(sT[0]),
                           min_sT=float(sT.min()), kappa_at_min=float(ks[np.argmin(sT)]), sT_at_k095=float(sT[-1])))
        d = np.diff(sT)
        if np.any(d < -1e-6):
            n_nonmono += 1
            i = int(np.argmin(d)); examples.append(dict(beta=beta, c=c, s0=s0, kappa_from=float(ks[i]),
                                                        kappa_to=float(ks[i + 1]), drop=float(-d[i])))
    res["S3"] = dict(configs=n, non_monotone_configs=n_nonmono, largest_drops=sorted(examples, key=lambda e: -e["drop"])[:5],
                  curves=curves)
    print("S3", res["S3"])


if __name__ == "__main__":
    s1(); s3(); s2()
    with open(os.path.join(OUT, "sensitivity.json"), "w") as f:
        json.dump(res, f, indent=1)
