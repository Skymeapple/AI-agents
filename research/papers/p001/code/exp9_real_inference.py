"""Exp 9 (revision 1): re-examination of the within-sample break analysis on the Chinchilla reconstruction.
Responds to reviewer W2, M4, M5 and the noise question in W3.

(a) Where the data actually are: run counts above compute thresholds.
(b) How much of the residual scatter is colour-quantisation (digitisation) error, and how much of the
    remaining scatter is rough (run-level) vs smooth (misspecification) -- nearest-neighbour decomposition.
(c) Profiled break fit with a WIDER breakpoint grid (10th-90th compute percentiles, 9 points).
(d) Calibrated tests of delta = 0 (one-sided T1 and two-sided T2) by parametric bootstrap from the fitted
    null: (i) iid Gaussian noise, (ii) wild cluster bootstrap of residuals (Rademacher signs shared within
    0.1-dex compute bins), (iii) same with clusters = model-size groups.
(e) Cluster (pairs) bootstrap of delta_hat, profiled and at fixed Cb, clusters = compute bins / size groups.
(f) Power of the calibrated test on the ACTUAL 240-run design (min. detectable delta), wild-cluster noise.
(g) Sensitivity: all 245 runs.
"""
import json, os, time
import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp
from breaktest import profiled, wilson
from common import load_chinchilla, epoch_keep, RES, pred_log_base

t0 = time.time()
N, D, C, L = load_chinchilla()
keep = epoch_keep(L)
N5, D5, C5, L5 = N, D, C, L
N, D, C, L = N[keep], D[keep], C[keep], L[keep]
lN, lD, lC, lL = np.log(N), np.log(D), np.log(C), np.log(L)
base = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
p0 = np.array([np.log(base["A"]), np.log(base["B"]), np.log(base["E"]), base["alpha"], base["beta"]])
GRID = np.quantile(lC, np.linspace(0.1, 0.9, 9))
cl_C = np.unique(np.round(np.log10(C), 1), return_inverse=True)[1]
cl_N = np.unique(np.round(np.log10(N) / 0.05).astype(int), return_inverse=True)[1]
out = {}

# (a)
out["support"] = {f">={t:.0e}": int((C >= t).sum()) for t in [1e20, 3e20, 1e21, 3e21, 5e21, 1e22]}
out["support"]["n"] = len(C)
out["support"]["log10C_quantiles_10_90"] = [float(x) for x in np.log10(np.quantile(C, [0.1, 0.5, 0.9]))]

# (b) quantisation of the colour-coded loss
u = np.unique(L)
lu = np.log(u)
gap_lo = np.r_[np.nan, np.diff(lu)]; gap_hi = np.r_[np.diff(lu), np.nan]
width = np.nanmean(np.stack([gap_lo, gap_hi]), axis=0)  # local spacing of the colour levels (log-loss)
idx = np.searchsorted(u, L)
w_run = width[idx]
sd_quant = float(np.sqrt(np.mean(w_run ** 2 / 12.0)))  # uniform rounding error within one level
res = lL - pred_log_base(p0, lN, lD)
# nearest neighbour in (log10 N, log10 C) space
X = np.stack([np.log10(N), np.log10(C)], 1)
dmat = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dmat, np.inf)
nn = dmat.argmin(1)
var_rough = float(np.mean((res - res[nn]) ** 2) / 2.0)
out["noise_decomposition"] = dict(
    n_unique_loss_levels=int(len(u)), median_level_spacing_log=float(np.median(np.diff(lu))),
    sd_quantisation_log=sd_quant, residual_sd_total=float(res.std()),
    rough_sd_nearest_neighbour=float(np.sqrt(var_rough)),
    smooth_sd=float(np.sqrt(max(res.var() - var_rough, 0.0))),
    rough_sd_after_removing_quantisation=float(np.sqrt(max(var_rough - sd_quant ** 2, 0.0))),
    median_nn_distance_dex=float(np.median(dmat.min(1))))
print(out["support"], out["noise_decomposition"], flush=True)

# (c) profiled fit on real data with the wide grid
INITS = [p0]
r_real = profiled(lN, lD, lC, lL, GRID, INITS)
out["profiled_wide_grid"] = dict(
    grid_Cb=[float(np.exp(g)) for g in GRID], delta_hat=r_real["delta_hat"], Cb_hat=float(np.exp(r_real["lCb_hat"])),
    Cb_hat_at_grid_edge=bool(r_real["j_hat"] in (0, len(GRID) - 1)), T1=r_real["T1"], T2=r_real["T2"],
    per_Cb=[dict(Cb=float(np.exp(r[2])), delta_hat=r[1], obj_gain=float(r_real["f0"] - r[0])) for r in r_real["rows"]])
print(out["profiled_wide_grid"], flush=True)
mu0 = pred_log_base(r_real["x0"], lN, lD)
res0 = lL - mu0
sig0 = float(res0.std())


def sim_null(seed, kind, mu=None, delta=0.0, lCb=None):
    rng = np.random.default_rng(seed)
    m = mu0 if mu is None else mu
    if kind == "gauss":
        e = sig0 * rng.standard_normal(len(m))
    else:
        cl = cl_C if kind == "wildC" else cl_N
        s = rng.choice([-1.0, 1.0], cl.max() + 1)[cl]
        e = res0 * s
    return m + e


def stat_job(a):
    seed, kind, delta, lCb_true = a
    if delta == 0.0:
        y = sim_null(seed, kind)
    else:
        x = r_real["x0"]; h = np.maximum(0.0, lC - lCb_true)
        mu = logsumexp(np.stack([x[0] - x[3] * lN + delta * h, x[1] - x[4] * lD + delta * h,
                                 np.full_like(lN, x[2])]), axis=0)
        y = sim_null(seed, kind, mu=mu)
    r = profiled(lN, lD, lC, y, GRID, [r_real["x0"]])
    return r["T1"], r["T2"], r["delta_hat"]


def boot_job(a):
    seed, kind = a
    rng = np.random.default_rng(seed)
    cl = cl_C if kind == "C" else cl_N
    pick = rng.integers(0, cl.max() + 1, cl.max() + 1)
    ii = np.concatenate([np.where(cl == c)[0] for c in pick])
    r = profiled(lN[ii], lD[ii], lC[ii], lL[ii], GRID, [p0])
    return r["delta_hat"], [row[1] for row in r["rows"]]


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        # (d)
        out["calibrated_tests"] = {}
        crit = {}
        for kind in ["gauss", "wildC", "wildN"]:
            sims = np.array(pool.map(stat_job, [(100000 * (1 + ["gauss", "wildC", "wildN"].index(kind)) + i, kind, 0.0, None)
                                                for i in range(300)]))
            crit[kind] = float(np.quantile(sims[:, 0], 0.95))
            out["calibrated_tests"][kind] = dict(
                p_one_sided=float((1 + (sims[:, 0] >= r_real["T1"]).sum()) / (1 + len(sims))),
                p_two_sided=float((1 + (sims[:, 1] >= r_real["T2"]).sum()) / (1 + len(sims))),
                crit95_T1=crit[kind], null_delta_hat_p2_5_97_5=[float(np.percentile(sims[:, 2], 2.5)),
                                                                float(np.percentile(sims[:, 2], 97.5))])
            print(kind, out["calibrated_tests"][kind], f"{time.time()-t0:.0f}s", flush=True)
        # (e)
        out["cluster_bootstrap"] = {}
        for kind in ["C", "N"]:
            bs = pool.map(boot_job, [(200000 + 1000 * (kind == "N") + i, kind) for i in range(200)])
            dh = np.array([b[0] for b in bs]); df = np.array([b[1] for b in bs])
            out["cluster_bootstrap"][kind] = dict(
                n_clusters=int((cl_C if kind == "C" else cl_N).max() + 1),
                profiled_delta_ci95=[float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))],
                fixed_Cb_ci95=[dict(Cb=float(np.exp(g)), ci95=[float(np.percentile(df[:, j], 2.5)), float(np.percentile(df[:, j], 97.5))])
                               for j, g in enumerate(GRID)])
            print(kind, out["cluster_bootstrap"][kind], f"{time.time()-t0:.0f}s", flush=True)
        # (f) power on the actual design, wild-cluster (compute-bin) noise
        out["power_actual_design"] = []
        for lCb_true in [GRID[4], GRID[6]]:
            for d in [0.01, 0.02, 0.04, 0.08]:
                sims = np.array(pool.map(stat_job, [(300000 + int(d * 1e4) * 100 + int(lCb_true) + i, "wildC", d, lCb_true)
                                                    for i in range(100)]))
                k = int((sims[:, 0] > crit["wildC"]).sum())
                out["power_actual_design"].append(dict(Cb_true=float(np.exp(lCb_true)), delta=d, power=k / 100,
                                                       ci95=wilson(k, 100)))
                print(out["power_actual_design"][-1], f"{time.time()-t0:.0f}s", flush=True)
    # (g) all 245 runs
    lN5, lD5, lC5, lL5 = np.log(N5), np.log(D5), np.log(C5), np.log(L5)
    f245 = json.load(open(f"{RES}/exp1_fit.json")).get("fit245")
    inits = [p0] + ([[np.log(f245["A"]), np.log(f245["B"]), np.log(f245["E"]), f245["alpha"], f245["beta"]]] if f245 else [])
    G5 = np.quantile(lC5, np.linspace(0.1, 0.9, 9))
    r5 = profiled(lN5, lD5, lC5, lL5, G5, inits)
    out["all245"] = dict(delta_hat=r5["delta_hat"], Cb_hat=float(np.exp(r5["lCb_hat"])), T1=r5["T1"],
                         per_Cb=[dict(Cb=float(np.exp(r[2])), delta_hat=r[1]) for r in r5["rows"]])
    print(out["all245"], flush=True)
    json.dump(out, open(f"{RES}/exp9_real_inference.json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s")
