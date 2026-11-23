"""Exp 15 (revision 2): a real out-of-sample forecast-residual test and audit horizon on a second public
suite, Gadre et al. (2024) (github.com/mlfoundations/scaling, exp_data/models; extracted to
data/gadre2024/models.csv).  Also: empirical anchors for correlated (per-configuration) noise.

For each training distribution (C4, RedPajama, RefinedWeb), evaluated in distribution (c4_val,
paloma_redpajama, paloma_falcon-refinedweb):
  * fit the 5-parameter law (Huber 1e-3 on log-loss, Hoffmann-style initial grid) to the 32 small runs
    (10.6M-412M parameters, token multipliers 0.25-32, C up to ~6e20 FLOP);
  * forecast the held-out 1.4B (M=1 and one over-trained run) and 6.9B (M=1) runs;
  * forecast uncertainty from (i) a pairs bootstrap over runs (300) and (ii) a leave-one-size-out jackknife
    of the forecast (structural sensitivity), and compare with the realised residual;
  * compute-equivalent penalty P_hat = ((L_obs - E)/(L_hat - E))^(1/gamma) with bootstrap interval;
  * audit horizon of the small sweep on its own compute-optimal frontier.
Correlated-noise anchors:
  (a) cross-distribution shared residual component per (size, multiplier) configuration;
  (b) hyper-parameter sensitivity in grid.json (warm-up sweep; head-count at fixed width/depth/tokens);
  (c) Chinchilla: between-compute-bin variance component of the 240-run residuals.
"""
import csv, json, time, re
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from common import RES, ROOT, obj_base, OPT, INIT_GRID, pred_log_base, load_chinchilla, epoch_keep
import horizon as H

rows = list(csv.DictReader(open(f"{ROOT}/data/gadre2024/models.csv")))
EVAL = {"c4_original": "c4_val", "rpj": "paloma_redpajama", "rw_original": "paloma_falcon-refinedweb"}
CG = np.logspace(18, 30, 241)
BIG = ("open_lm_1b", "open_lm_7b")


def fit_grid(lN, lD, lL, inits=INIT_GRID):
    return min((minimize(obj_base, np.array(x0, float), args=(lN, lD, lL), method="BFGS", jac=True, options=OPT)
                for x0 in inits), key=lambda r: r.fun).x


def fit_local(args):
    lN, lD, lL, x0, seed = args
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(lL), len(lL))
    inits = [x0, x0 + np.array([0.5, -0.5, 0.03, 0.03, -0.03]), x0 + np.array([-0.5, 0.5, -0.03, -0.03, 0.03])]
    return fit_grid(lN[ii], lD[ii], lL[ii], inits)




if __name__ == "__main__":
    t0 = time.time()
    out = dict(source="Gadre et al. 2024, github.com/mlfoundations/scaling commit a003c49", datasets={})
    resid_by_cfg = {}
    with Pool(4) as pool:
        for ds, ev in EVAL.items():
            R = [r for r in rows if r["dataset"] == ds]
            small = [r for r in R if r["model"] not in BIG]
            big = [r for r in R if r["model"] in BIG]
            N = np.array([float(r["params"]) for r in small]); Dt = np.array([float(r["tokens"]) for r in small])
            L = np.array([float(r[ev]) for r in small])
            lN, lD, lL = np.log(N), np.log(Dt), np.log(L)
            th = fit_grid(lN, lD, lL)
            res = lL - pred_log_base(th, lN, lD)
            for r, e in zip(small, res):
                resid_by_cfg.setdefault((r["model"], float(r["multiplier"])), {})[ds] = float(e)
            boots = np.array(pool.map(fit_local, [(lN, lD, lL, th, 777 + i) for i in range(300)]))
            sizes = sorted(set(r["model"] for r in small))
            jk = []
            for s_ in sizes:
                m = np.array([r["model"] != s_ for r in small])
                jk.append(fit_grid(lN[m], lD[m], lL[m], [th]))
            gam = H.gam(th)
            d = dict(eval=ev, n_fit=len(small), fit=dict(A=float(np.exp(th[0])), B=float(np.exp(th[1])),
                     E=float(np.exp(th[2])), alpha=float(th[3]), beta=float(th[4]), gamma=float(gam)),
                     residual_sd_fit=float(res.std()), max_C_fit=float((6 * N * Dt).max()), targets=[])
            for r in big:
                n_, t_ = float(r["params"]), float(r["tokens"]); Lobs = float(r[ev]); C = 6 * n_ * t_
                lp = float(pred_log_base(th, np.log([n_]), np.log([t_]))[0])
                lpb = np.array([pred_log_base(b, np.log([n_]), np.log([t_]))[0] for b in boots])
                lpj = np.array([pred_log_base(b, np.log([n_]), np.log([t_]))[0] for b in jk])
                E = np.exp(th[2])
                kappa = (Lobs - E) / (np.exp(lp) - E)
                Pb = []
                for b in boots:
                    Eb = np.exp(b[2]); Lb = np.exp(pred_log_base(b, np.log([n_]), np.log([t_]))[0])
                    Pb.append(((Lobs - Eb) / (Lb - Eb)) ** (1 / H.gam(b)) if Lobs > Eb else 0.0)
                Pb = np.array(Pb)
                d["targets"].append(dict(
                    name=r["name"], params=n_, tokens=t_, C=C, multiplier=float(r["multiplier"]), lr=float(r["lr"]),
                    wd=float(r["wd"]), L_obs=Lobs, L_pred=float(np.exp(lp)), log_residual=float(np.log(Lobs) - lp),
                    s_boot=float(lpb.std()), boot_q025_975=[float(x) for x in np.quantile(np.log(Lobs) - lpb, [0.025, 0.975])],
                    jackknife_residual_range=[float((np.log(Lobs) - lpj).min()), float((np.log(Lobs) - lpj).max())],
                    z_boot=float((np.log(Lobs) - lp) / lpb.std()),
                    P_hat=float(kappa ** (1 / gam)) if kappa > 0 else 0.0,
                    P_boot_q05_95=[float(x) for x in np.quantile(Pb, [0.05, 0.95])],
                    P_min50_from_s_boot=float(H.p_min(th, C, lpb.std()))))
            # audit horizon of this sweep (law uncertainty only)
            s_curve = np.array([np.log(H.Lstar(b, CG)) for b in boots]).std(0)
            d["horizon_C_H_boot"] = H.horizon(th, CG, s_curve)
            d["P_min50_frontier"] = {f"{c:.0e}": float(H.p_min(th, c, s_curve[int(np.argmin(np.abs(np.log(CG / c))))]))
                                     for c in [1e21, 1e22, 1e23, 1e24, 1e25, 1e26]}
            out["datasets"][ds] = d
            print(ds, json.dumps({k: v for k, v in d.items() if k != "targets"}), flush=True)
            for tg in d["targets"]:
                print("   ", tg["name"], f"C={tg['C']:.2e} res={tg['log_residual']:+.4f} s_boot={tg['s_boot']:.4f} "
                      f"z={tg['z_boot']:+.2f} jk={tg['jackknife_residual_range']} P_hat={tg['P_hat']:.2f} "
                      f"P90={tg['P_boot_q05_95']} Pmin50={tg['P_min50_from_s_boot']:.2f}", flush=True)

    # ---- correlated-noise anchors ----
    anchors = {}
    cfgs = [v for v in resid_by_cfg.values() if len(v) == 3]
    Rm = np.array([[v[d] for d in EVAL] for v in cfgs])  # configs x 3 datasets
    tot = Rm.var(ddof=1, axis=0).mean()
    shared = np.cov(Rm.T)[np.triu_indices(3, 1)].mean()
    anchors["gadre_cross_dataset"] = dict(n_configs=len(cfgs), residual_sd_per_dataset=float(np.sqrt(tot)),
                                          shared_component_sd=float(np.sqrt(max(shared, 0))),
                                          mean_pairwise_corr=float(np.corrcoef(Rm.T)[np.triu_indices(3, 1)].mean()))
    g = json.load(open(f"{ROOT}/data/gadre2024/grid.json"))
    parsed = []
    for k, v in g.items():
        p = k.split("-")
        mm = re.match(r"d=(\d+)_l=(\d+)_h=(\d+)", p[0])
        parsed.append(dict(d=int(mm.group(1)), l=int(mm.group(2)), h=int(mm.group(3)), bs=p[1], tokens=p[5], warmup=int(p[6]), loss=v))
    gaps = []  # best vs second-best warm-up, same architecture/tokens
    for key in set((q["d"], q["l"], q["h"], q["bs"], q["tokens"]) for q in parsed):
        ls = sorted(np.log(q["loss"]) for q in parsed if (q["d"], q["l"], q["h"], q["bs"], q["tokens"]) == key)
        if len(ls) >= 2:
            gaps.append(ls[1] - ls[0])
    hs = []  # SD across head counts at fixed width/depth/tokens/bs, each at its best warm-up
    for key in set((q["d"], q["l"], q["bs"], q["tokens"]) for q in parsed):
        best = {}
        for q in parsed:
            if (q["d"], q["l"], q["bs"], q["tokens"]) == key:
                best[q["h"]] = min(best.get(q["h"], 9e9), np.log(q["loss"]))
        if len(best) >= 2:
            hs.append(np.std(list(best.values()), ddof=1))
    anchors["gadre_grid_warmup_best_vs_second"] = dict(n=len(gaps), median=float(np.median(gaps)),
                                                       q25_75=[float(x) for x in np.quantile(gaps, [0.25, 0.75])])
    anchors["gadre_grid_headcount_sd"] = dict(n=len(hs), median=float(np.median(hs)),
                                              q25_75=[float(x) for x in np.quantile(hs, [0.25, 0.75])])
    N, D, C, L = load_chinchilla(); keep = epoch_keep(L)
    N, D, C, L = N[keep], D[keep], C[keep], L[keep]
    base = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
    p0 = np.array([np.log(base["A"]), np.log(base["B"]), np.log(base["E"]), base["alpha"], base["beta"]])
    r = np.log(L) - pred_log_base(p0, np.log(N), np.log(D))
    cl = np.unique(np.round(np.log10(C), 1), return_inverse=True)[1]
    ks = np.unique(cl); n_k = np.array([(cl == k).sum() for k in ks]); m_k = np.array([r[cl == k].mean() for k in ks])
    w_var = sum(((r[cl == k] - r[cl == k].mean()) ** 2).sum() for k in ks) / (len(r) - len(ks))
    n0 = (len(r) - (n_k ** 2).sum() / len(r)) / (len(ks) - 1)
    b_ms = (n_k * (m_k - r.mean()) ** 2).sum() / (len(ks) - 1)
    anchors["chinchilla_compute_bin_component"] = dict(n_bins=int(len(ks)), within_sd=float(np.sqrt(w_var)),
                                                       between_sd=float(np.sqrt(max((b_ms - w_var) / n0, 0))))
    out["correlated_noise_anchors"] = anchors
    print(json.dumps(anchors, indent=1))
    json.dump(out, open(f"{RES}/exp15_gadre.json", "w"), indent=2)
    print("done", f"{time.time()-t0:.0f}s")
