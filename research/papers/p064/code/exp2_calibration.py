"""Calibrated detection lags and locked-in deficits for 11 BLS occupations, with Monte Carlo
over the unobserved parameters (time-to-seniority tau, extensive shock x, intensive loss d)."""
import csv, json
import numpy as np
from scipy.stats import spearmanr
from common import *
from model import *
from bls_data import BLS

rng = np.random.default_rng(20261122)
EPS = 0.05
rows = []
for name, soc, e25, e35, ex, tr, tot in BLS:
    mu = tot / 100.0
    g = np.log(e35 / e25) / 10.0
    lam = mu + g
    r = dict(occupation=name, soc=soc, mu=mu, g=g, R_required=required_R(g, TAU, mu))
    for x in [0.13, 0.16, 0.19]:
        for d in [0.0, 0.17]:
            xe = x_eff(x, d)
            te = detection_time(EPS, xe, TAU, mu, g)
            r[f"t5_x{x}_d{d}"] = te
            r[f"locked_x{x}_d{d}"] = xe * te
            r[f"peak_x{x}_d{d}"] = xe * (1 - np.exp(-lam * te))
    r["halflife_to_longrun"] = TAU + np.log(2) / lam
    # Monte Carlo
    N = 20000
    tau = rng.uniform(2, 8, N); x = rng.uniform(0.13, 0.19, N); d = rng.uniform(0, 0.17, N)
    xe = x_eff(x, d)
    te = tau - np.log(1 - EPS / xe) / lam
    lk = xe * te
    pk = xe * (1 - np.exp(-lam * te))
    for k, v in [("t5", te), ("locked", lk), ("peak", pk)]:
        r[f"{k}_mc_p05"], r[f"{k}_mc_p50"], r[f"{k}_mc_p95"] = np.percentile(v, [5, 50, 95])
    if name == "Software developers":
        r["spearman_t5"] = {k: float(spearmanr(v, te).correlation) for k, v in [("tau", tau), ("x", x), ("d", d)]}
    rows.append(r)

keys = list(rows[0].keys()); keys.remove("spearman_t5") if "spearman_t5" in keys else None
with open(os.path.join(RES, "exp2_calibration.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "spearman_t5"])
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k != "spearman_t5"})
json.dump(rows, open(os.path.join(RES, "exp2_calibration.json"), "w"), indent=1, default=float)

# compact table for the paper
print(f"{'occupation':36s} mu    g      R*   t5(x.19) t5(x.19,d.17) locked(x.19,d.17) MC t5 [5,50,95]")
for r in rows:
    print(f"{r['occupation']:36s} {r['mu']:.3f} {r['g']:+.4f} {r['R_required']:.2f} {r['t5_x0.19_d0.0']:6.1f} {r['t5_x0.19_d0.17']:9.1f} {r['locked_x0.19_d0.17']:9.2f}"
          f"   [{r['t5_mc_p05']:.1f}, {r['t5_mc_p50']:.1f}, {r['t5_mc_p95']:.1f}]")
print("spearman (software devs):", rows[0]["spearman_t5"])

# figure: detection time by occupation with MC interval
order = np.argsort([r["t5_mc_p50"] for r in rows])
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ys = np.arange(len(rows))
for yi, i in enumerate(order):
    r = rows[i]
    ax.plot([r["t5_mc_p05"], r["t5_mc_p95"]], [yi, yi], color=C[0], lw=2, solid_capstyle="round")
    ax.plot(r["t5_mc_p50"], yi, "o", color=C[0], ms=7, mec="white", mew=1.5)
    ax.plot(r["t5_x0.19_d0.0"], yi, "D", color=C[1], ms=5, mec="white", mew=1)
ax.set_yticks(ys); ax.set_yticklabels([rows[i]["occupation"] for i in order], fontsize=8)
ax.set_xlabel("years after intake shock until senior shortfall reaches 5% of demand")
ax.plot([], [], "o", color=C[0], label="Monte Carlo median (bar: 5-95%)")
ax.plot([], [], "D", color=C[1], label="point scenario: x=0.19, d=0, tau=5")
ax.legend(fontsize=7, loc="lower right"); ax.set_xlim(0, None)
ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "fig2_detection_lag_by_occupation.png")
