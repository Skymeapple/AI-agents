"""Exp 13 (revision 2): the audit horizon of the public Chinchilla sweep.

For a frontier run at compute C whose loss is compared with the forecast of the law fitted to the 240
Chinchilla runs, how large a compute penalty P could go undetected?  We measure the forecast uncertainty
s(C) of ln L*(C) in five ways:
  (i)   iid pairs bootstrap (Exp. 1 replicates, 300);
  (ii)  cluster pairs bootstrap over 0.1-dex compute bins (300);
  (iii) cluster pairs bootstrap over model-size groups (300);
  (iv)  delta method, iid noise at the residual SD;
  (v)   delta method, cluster-robust sandwich over compute bins.
and (vi) cluster bootstrap on all 245 runs.  From s(C) we compute the minimum detectable penalty P_min(C)
(50% and 80% power, one-sided 5%) and the audit horizon C_H (P_min = 2^(1/gamma) at 50% power).
We also compute, without the Gaussian/linear approximation, the bootstrap upper 95% bound on P implied by
a frontier observation that exactly equals the point forecast.
"""
import json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from common import load_chinchilla, epoch_keep, RES, obj_base, OPT, pred_log_base
import horizon as H

t0 = time.time()
N, D, C, L = load_chinchilla()
keep = epoch_keep(L)
base = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
p0 = np.array([np.log(base["A"]), np.log(base["B"]), np.log(base["E"]), base["alpha"], base["beta"]])
CG = np.logspace(20, 30, 201)
REPORT = [1e22, 1e23, 1e24, 1e25, 1e26, 5e26, 1e27, 1e28]
INITS = [p0, p0 + np.array([0.5, -0.5, 0.02, 0.02, -0.02]), p0 + np.array([-0.5, 0.5, -0.02, -0.02, 0.02])]


def fit(lN, lD, lL):
    return min((minimize(obj_base, x0, args=(lN, lD, lL), method="BFGS", jac=True, options=OPT) for x0 in INITS),
               key=lambda r: r.fun).x


def boot(args):
    idx_sets, seed, sub = args
    rng = np.random.default_rng(seed)
    lN, lD, lL = sub
    k = len(idx_sets)
    pick = rng.integers(0, k, k)
    ii = np.concatenate([idx_sets[j] for j in pick])
    return fit(lN[ii], lD[ii], lL[ii])


def summarise(th_hat, boots, label):
    """Forecast SD of ln L*(C) across bootstrap replicates and derived quantities."""
    lf = np.array([np.log(H.Lstar(b, CG)) for b in boots])
    s = lf.std(0)
    out = dict(label=label, n_boot=len(boots), E_sd=float(np.exp(boots[:, 2]).std()),
               E_q025_975=[float(x) for x in np.quantile(np.exp(boots[:, 2]), [0.025, 0.975])])
    out["horizon_C_H"] = H.horizon(th_hat, CG, s)
    out["horizon_C_H_80pct_power"] = H.horizon(th_hat, CG, s, power=0.8)
    rows = []
    Lhat = H.Lstar(th_hat, CG)
    for c in REPORT:
        i = int(np.argmin(np.abs(np.log(CG / c))))
        # nonlinear bootstrap bound: P_b = C / C'_b with L*_b(C'_b) = Lhat(C).  If Lhat <= E_b the observed loss is
        # unreachable under replicate b (C'_b = inf), i.e. the run would be infinitely MORE efficient: P_b = 0.
        Pb = []
        for b in boots:
            Lb = H.Lstar(b, CG)
            if Lhat[i] <= np.exp(b[2]):
                Pb.append(0.0)
            elif Lhat[i] >= Lb[0]:
                Pb.append(CG[i] / CG[0])
            else:
                Pb.append(CG[i] / np.exp(np.interp(-Lhat[i], -Lb, np.log(CG))))
        Pb = np.minimum(np.array(Pb), 1e300)  # inf -> 1e300 so that quantiles are defined; reported as 'inf'
        rows.append(dict(C=c, rho=float(H.rho(th_hat, c)), s_log=float(s[i]),
                         P_min_50=float(H.p_min(th_hat, c, s[i])), P_min_80=float(H.p_min(th_hat, c, s[i], power=0.8)),
                         P_upper95_bootstrap=(float(np.quantile(Pb, 0.95)) if np.quantile(Pb, 0.95) < 1e299 else "inf"),
                         P_lower05_bootstrap=float(np.quantile(Pb, 0.05)), frac_unreachable=float(np.mean(Pb == 0.0))))
    out["rows"] = rows
    return out, s


if __name__ == "__main__":
    res = {}
    gam = H.gam(p0)
    res["gamma"] = gam
    curves = {}
    with Pool(4) as pool:
        for tag, sel in [("240", keep), ("245", np.arange(len(L)))]:
            n_, d_, c_, l_ = N[sel], D[sel], C[sel], L[sel]
            lN, lD, lL = np.log(n_), np.log(d_), np.log(l_)
            th = fit(lN, lD, lL)
            clC = np.unique(np.round(np.log10(c_), 1), return_inverse=True)[1]
            clN = np.unique(np.round(np.log10(n_) / 0.05).astype(int), return_inverse=True)[1]
            schemes = [("cluster_compute_bins", clC)] if tag == "245" else \
                [("iid", np.arange(len(l_))), ("cluster_compute_bins", clC), ("cluster_size_groups", clN)]
            for name, cl in schemes:
                sets = [np.where(cl == g)[0] for g in np.unique(cl)]
                boots = np.array(pool.map(boot, [(sets, 5000 + i, (lN, lD, lL)) for i in range(300)]))
                o, s = summarise(th, boots, f"{tag}:{name}")
                res[f"{tag}:{name}"] = o; curves[f"{tag}:{name}"] = s
                print(o["label"], "C_H=%.2e" % o["horizon_C_H"], [(f"{r['C']:.0e}", round(r["P_min_50"], 2),
                      round(r["P_upper95_bootstrap"],2), r["frac_unreachable"]) for r in o["rows"]], f"{time.time()-t0:.0f}s", flush=True)
            if tag == "240":
                # delta method
                r = lL - pred_log_base(th, lN, lD)
                J = H.jac_log(th, lN, lD)
                Hi = np.linalg.inv(J.T @ J)
                cov_iid = Hi * r.var()
                meat = sum(np.outer(J[cl_].T @ r[cl_], J[cl_].T @ r[cl_]) for cl_ in
                           [np.where(clC == g)[0] for g in np.unique(clC)])
                cov_cl = Hi @ meat @ Hi
                for name, cv in [("delta_iid", cov_iid), ("delta_cluster_compute_bins", cov_cl)]:
                    s = H.forecast_sd(th, cv, CG)
                    o = dict(label=f"240:{name}", E_sd=float(np.exp(th[2]) * np.sqrt(cv[2, 2])),
                             horizon_C_H=H.horizon(th, CG, s), horizon_C_H_80pct_power=H.horizon(th, CG, s, power=0.8),
                             rows=[dict(C=c, s_log=float(H.forecast_sd(th, cv, c)[0]),
                                        P_min_50=float(H.p_min(th, c, H.forecast_sd(th, cv, c)[0])),
                                        P_min_80=float(H.p_min(th, c, H.forecast_sd(th, cv, c)[0], power=0.8)))
                                   for c in REPORT])
                    res[o["label"]] = o; curves[o["label"]] = s
                    print(o["label"], "C_H=%.2e" % o["horizon_C_H"], [(f"{r['C']:.0e}", round(r["P_min_50"], 2))
                                                                     for r in o["rows"]], flush=True)
                # with an additional frontier-run measurement noise of 1e-3 (cluster bootstrap s)
                s_tot = np.sqrt(curves["240:cluster_compute_bins"] ** 2 + 1e-3 ** 2)
                res["240:cluster_compute_bins+frontier_noise_1e-3"] = dict(
                    horizon_C_H=H.horizon(th, CG, s_tot),
                    rows=[dict(C=c, P_min_50=float(H.p_min(th, c, s_tot[int(np.argmin(np.abs(np.log(CG / c))))])))
                          for c in REPORT])
                th240 = th
    json.dump(res, open(f"{RES}/exp13_chinchilla_horizon.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k, sty in [("240:iid", "-"), ("240:cluster_compute_bins", "-"), ("240:cluster_size_groups", "-"),
                   ("240:delta_iid", "--"), ("245:cluster_compute_bins", ":")]:
        s = curves[k]
        ax[0].loglog(CG, s, sty, label=k)
        ax[1].loglog(CG, H.p_min(th240, CG, s), sty, label=k)
    ax[0].loglog(CG, np.log1p(H.rho(th240, CG) * (2 ** gam - 1)), "k-", lw=2, label="signal of a 2x penalty")
    ax[0].loglog(CG, np.log1p(H.rho(th240, CG) * (10 ** gam - 1)), "k--", lw=2, label="signal of a 10x penalty")
    ax[0].set_xlabel("frontier compute C (FLOP)"); ax[0].set_ylabel("log-loss scale")
    ax[0].set_title("forecast SD s(C) vs the signal of a compute penalty", fontsize=9)
    ax[1].axhline(2 ** (1 / gam), color="grey", lw=.8)
    ax[1].text(1.5e20, 2 ** (1 / gam) * 1.2, r"$2^{1/\gamma}$ (defines $C_H$)", fontsize=7)
    ax[1].set_xlabel("frontier compute C (FLOP)"); ax[1].set_ylabel(r"$P_{\min}$ (50% power, 5% one-sided)")
    ax[1].set_title("smallest detectable compute penalty", fontsize=9); ax[1].set_ylim(1, 1e4)
    for a_ in ax:
        a_.axvspan(1.4e18, 1.3e22, color="C7", alpha=.12); a_.axvline(4.6e26, color="r", lw=.7)
    ax[1].legend(fontsize=6); ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{RES}/fig8_chinchilla_horizon.png", dpi=150)
    print("done", f"{time.time()-t0:.0f}s")
