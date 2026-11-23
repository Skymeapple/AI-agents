"""Exp 14 (revision 2): audit horizon as a function of the sweep design, and Monte Carlo validation of the
closed-form resolution P_min(C) = [1 + (exp(z s(C)) - 1)/rho(C)]^(1/gamma) for the forecast-residual test.

(a) Design law (delta method, OLS sandwich): sweep of K iso-FLOP budgets x M sizes over [C_hi/1e3, C_hi];
    iid noise sigma_w, per-budget offset sigma_b; frontier run(s) carry their own noise
    (sigma_b^2 + sigma_w^2/m).  Reports s(C), C_H and the audit reach log10(C_H / C_hi).
(b) Monte Carlo: sweep 1e20-1e23 (K=12, M=20); least-squares fit (5-parameter law); plug-in delta-method
    forecast SD at the fitted parameters; m in {1,3} frontier runs at 1e25, 1e26, 1e27 with a compute penalty
    P in {1, 2, 5, 10} or a constant-exponent break delta in {0.02, 0.05} starting at C_b = 1e23.
    Reports empirical size/power of the one-sided 5% residual test against the closed-form prediction,
    and the empirical vs delta-method SD of the forecast error.
Truth: the 240-run Chinchilla refit (Exp. 1).
"""
import json, time, os
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm
from common import RES, obj_base, OPT
from breaktest import wilson
import horizon as H

P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
TH = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])
GAM = H.gam(TH)
CG = np.logspace(19, 34, 301)
Z = norm.ppf(0.95)


def design_law():
    rows = []
    for Chi in [1e21, 1e22, 1e23, 1e24, 1e25]:
        for K, M in [(12, 20), (6, 8), (48, 5)]:
            lN, lD, lC, g = H.iso_design(Chi / 1e3, Chi, K, M)
            for sw in [1e-4, 3e-4, 1e-3, 3e-3, 7.5e-3]:
                for sb in [0.0, 1e-3, 3e-3]:
                    cov = H.cov_theta(TH, lN, lD, g, sw, sb)
                    s_fc = H.forecast_sd(TH, cov, CG)
                    for m in [1, 3]:
                        s = np.sqrt(s_fc ** 2 + sb ** 2 + sw ** 2 / m)
                        CH = H.horizon(TH, CG, s)
                        i26 = int(np.argmin(np.abs(np.log(CG / (Chi * 1e3)))))
                        rows.append(dict(C_hi=Chi, K=K, M=M, n=K * M, sigma_w=sw, sigma_b=sb, m=m,
                                         E_sd=float(np.exp(TH[2]) * np.sqrt(cov[2, 2])),
                                         C_H=CH, reach_decades=float(np.log10(CH / Chi)),
                                         s_at_Chi_x1e3=float(s[i26]),
                                         P_min50_at_Chi_x1e3=float(H.p_min(TH, CG[i26], s[i26])),
                                         P_min80_at_Chi_x1e3=float(H.p_min(TH, CG[i26], s[i26], power=0.8))))
    return rows


# ---------------- Monte Carlo ----------------
LN, LD, LC, GR = H.iso_design(1e20, 1e23, 12, 20)
CF = [1e25, 1e26, 1e27]
PEN = [1.0, 2.0, 5.0, 10.0]
DEL = [0.02, 0.05]
CB = 1e23
INITS = [TH, TH + np.array([0.4, -0.4, 0.02, 0.02, -0.02])]


def ls_obj(p, lN, lD, lL):  # least squares via a Huber threshold far above the noise
    return obj_base(p, lN, lD, lL, delta=10.0)


def mc_one(args):
    sw, sb, seed = args
    rng = np.random.default_rng(seed)
    mu = H.log_pred(TH, LN, LD)
    y = mu + sw * rng.standard_normal(len(mu)) + sb * rng.standard_normal(12)[GR]
    fit = min((minimize(ls_obj, x0, args=(LN, LD, y), method="BFGS", jac=True, options=OPT) for x0 in INITS),
              key=lambda r: r.fun)
    th = fit.x
    resid = y - H.log_pred(th, LN, LD)
    # plug-in: noise components estimated from the sweep residuals (one-way random-effects moments)
    bm = np.array([resid[GR == k].mean() for k in range(12)])
    sw_hat = float(np.sqrt(np.mean([(resid[GR == k] - bm[k]).var(ddof=1) for k in range(12)])))
    sb_hat = float(np.sqrt(max(bm.var(ddof=1) - sw_hat ** 2 / 20, 0.0)))
    cov = H.cov_theta(th, LN, LD, GR, sw_hat, sb_hat)
    out = []
    for cf in CF:
        lf_hat = float(np.log(H.Lstar(th, cf)))
        lf_true = float(np.log(H.Lstar(TH, cf)))
        s_fc = float(H.forecast_sd(th, cov, cf)[0])
        for m in [1, 3]:
            e_f = sb * rng.standard_normal() + sw * rng.standard_normal(m).mean()
            s_tot = np.sqrt(s_fc ** 2 + sb_hat ** 2 + sw_hat ** 2 / m)
            obs = {}
            for Pn in PEN:
                obs[f"P{Pn:g}"] = float(np.log(H.Lstar(TH, cf / Pn))) + e_f
            for d in DEL:
                R = H.Lstar(TH, cf) - np.exp(TH[2])
                obs[f"d{d:g}"] = float(np.log(np.exp(TH[2]) + R * (cf / CB) ** d)) + e_f
            out.append(dict(cf=cf, m=m, fc_err=lf_hat - lf_true, s_fc=s_fc, s_tot=float(s_tot),
                            rej={k: bool(v - lf_hat > Z * s_tot) for k, v in obs.items()}))
    return out


def predicted_power(cf, m, sw, sb, key):
    cov = H.cov_theta(TH, LN, LD, GR, sw, sb)
    s = np.sqrt(H.forecast_sd(TH, cov, cf)[0] ** 2 + sb ** 2 + sw ** 2 / m)
    if key.startswith("P"):
        Pn = float(key[1:])
    else:
        Pn = (cf / CB) ** (float(key[1:]) / GAM)
    return float(H.residual_power(TH, cf, s, Pn)), float(s)


if __name__ == "__main__":
    t0 = time.time()
    res = dict(gamma=GAM, design_law=design_law())
    print("design law done", f"{time.time()-t0:.0f}s", flush=True)
    for r in res["design_law"]:
        if r["K"] == 12 and r["m"] == 1:
            print(f"C_hi={r['C_hi']:.0e} sw={r['sigma_w']:g} sb={r['sigma_b']:g} C_H={r['C_H']:.2e} "
                  f"reach={r['reach_decades']:.2f} Pmin(Chi*1e3)={r['P_min50_at_Chi_x1e3']:.2f}")
    mc = []
    NREP = 400
    with Pool(os.cpu_count()) as pool:
        for sw, sb in [(3e-4, 0.0), (1e-3, 0.0), (3e-4, 1e-3), (1e-3, 1e-3), (3e-3, 0.0)]:
            reps = pool.map(mc_one, [(sw, sb, 31000000 + int(1e7 * sw) + int(1e6 * sb) + i) for i in range(NREP)])
            for cf in CF:
                for m in [1, 3]:
                    sel = [o for rep in reps for o in rep if o["cf"] == cf and o["m"] == m]
                    errs = np.array([o["fc_err"] for o in sel])
                    s_pred_fc = float(H.forecast_sd(TH, H.cov_theta(TH, LN, LD, GR, sw, sb), cf)[0])
                    row = dict(sigma_w=sw, sigma_b=sb, cf=cf, m=m, n_rep=len(sel),
                               fc_err_sd_empirical=float(errs.std()), fc_err_mean=float(errs.mean()),
                               fc_sd_delta_method=s_pred_fc, s_tot_plugin_median=float(np.median([o["s_tot"] for o in sel])),
                               tests={})
                    for key in sel[0]["rej"]:
                        k = sum(o["rej"][key] for o in sel)
                        pp, s_ = predicted_power(cf, m, sw, sb, key)
                        row["tests"][key] = dict(rate=k / len(sel), ci95=wilson(k, len(sel)), predicted=pp)
                    mc.append(row)
                    print(sw, sb, f"{cf:.0e}", m, f"err_sd={row['fc_err_sd_empirical']:.4f} dm={s_pred_fc:.4f}",
                          {k: (round(v["rate"], 2), round(v["predicted"], 2)) for k, v in row["tests"].items()},
                          f"{time.time()-t0:.0f}s", flush=True)
    res["monte_carlo"] = mc
    json.dump(res, open(f"{RES}/exp14_design_law.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for i, sw in enumerate([1e-4, 3e-4, 1e-3, 3e-3, 7.5e-3]):
        for sb, ls in [(0.0, "-"), (1e-3, "--")]:
            rs = [r for r in res["design_law"] if r["K"] == 12 and r["m"] == 1 and r["sigma_w"] == sw and r["sigma_b"] == sb]
            ax[0].plot([r["C_hi"] for r in rs], [r["C_H"] for r in rs], ls + "o", color=f"C{i}", ms=3,
                       label=f"$\\sigma_w$={sw:g}" + (", $\\sigma_b$=1e-3" if sb else ""))
    ax[0].plot([1e21, 1e25], [1e21, 1e25], "k:", lw=.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("top of sweep $C_{hi}$ (FLOP), 3-decade sweep, 240 runs")
    ax[0].set_ylabel("audit horizon $C_H$ (FLOP)"); ax[0].legend(fontsize=6); ax[0].axhline(4.6e26, color="r", lw=.7)
    ax[0].set_title("audit horizon vs sweep design (delta method)", fontsize=9)
    x = []; y = []
    for r in mc:
        for k, v in r["tests"].items():
            if k != "P1":
                x.append(v["predicted"]); y.append(v["rate"])
    ax[1].plot(x, y, "o", ms=3, alpha=.6); ax[1].plot([0, 1], [0, 1], "k:")
    ax[1].set_xlabel("closed-form predicted power"); ax[1].set_ylabel("Monte Carlo power (400 reps)")
    ax[1].set_title("validation: forecast-residual test, sweep 1e20-1e23", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{RES}/fig9_design_law.png", dpi=150)
    print("done", f"{time.time()-t0:.0f}s")
