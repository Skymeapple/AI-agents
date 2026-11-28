"""Exp 16 (revision 2): the audit ceiling C_dagger (Theorem 1) -- numerical checks, public-sweep ceilings,
nonlinear bootstrap certification bounds, and the design law for sd(ln E_hat).

Theorem 1 (paper Sec. 3.3).  With R(C) = G C^-gamma, total log-scale SD s of the forecast-residual comparison,
k = exp(z s) - 1 and z = z_{1-a} + z_power, the certifiable effective compute of a run at compute C is
    C_cert(C) = C / P_min(C) = (G / ((1+k) R(C) + k E))^(1/gamma)  <  C_dagger = (G / (k E))^(1/gamma).
Equivalently log10(C_dagger / C_hi) = (1/gamma) log10(R(C_hi) / (k E)) for any reference compute C_hi.

Parts
 (a) numerical check of the identity and the bound on the Chinchilla refit, several s;
 (b) public Chinchilla sweep: bootstrap sd(ln E_hat) under iid / compute-bin / size-group resampling, 240 and
     245 runs; C_dagger at 50% and 80% power; C_cert and P_min at 5e26 using the full bootstrap s(C);
     nonlinear (no Gaussian/delta approximation) 95% certification bound for a run landing exactly on the
     forecast, as a function of C, and its limit as C -> infinity;
 (c) design law: sd(ln E_hat) and reach log10(C_dagger/C_hi) for 3-decade iso-FLOP sweeps with n = 240 runs
     split as K budgets x M sizes, K in {6,12,24,48,120}, iid noise sigma_w and per-budget offset sigma_b
     (OLS sandwich and GLS / Cramer-Rao); the frontier run carries its own offset sigma_b and noise sigma_w.
"""
import json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm
from common import load_chinchilla, epoch_keep, RES, obj_base, OPT
import horizon as H

t0 = time.time()
P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
TH = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])


def G_of(th):
    E, R, _, _ = H.frontier_parts(th, 1.0)
    return float(R)  # R(C) = G C^-gamma  ->  G = R(1)


def zval(power):
    return norm.ppf(0.95) + norm.ppf(power)


def c_dagger(th, s, power=0.5):
    k = np.expm1(zval(power) * s)
    return (G_of(th) / (k * np.exp(th[2]))) ** (1.0 / H.gam(th))


def c_cert(th, C, s, power=0.5):
    k = np.expm1(zval(power) * s)
    E, R, _, _ = H.frontier_parts(th, C)
    return (G_of(th) / ((1 + k) * R + k * E)) ** (1.0 / H.gam(th))


INITS = [TH, TH + np.array([0.5, -0.5, 0.02, 0.02, -0.02]), TH + np.array([-0.5, 0.5, -0.02, -0.02, 0.02])]


def fit(lN, lD, lL):
    return min((minimize(obj_base, x0, args=(lN, lD, lL), method="BFGS", jac=True, options=OPT) for x0 in INITS),
               key=lambda r: r.fun).x


def boot(args):
    sets, seed, lN, lD, lL = args
    rng = np.random.default_rng(seed)
    ii = np.concatenate([sets[j] for j in rng.integers(0, len(sets), len(sets))])
    return fit(lN[ii], lD[ii], lL[ii])


def inv_Lstar(th, L):
    """Compute at which the frontier of th reaches loss L (inf if L <= E)."""
    E = np.exp(th[2])
    if L <= E:
        return np.inf
    return (G_of(th) / (L - E)) ** (1.0 / H.gam(th))


if __name__ == "__main__":
    out = {}
    gam = H.gam(TH); G = G_of(TH); E = float(np.exp(TH[2]))
    # (a) identity and bound
    CG = np.logspace(18, 45, 2701)
    chk = []
    for s in [1e-3, 3e-3, 1e-2, 2e-2]:
        num = CG / H.p_min(TH, CG, s)
        cl = c_cert(TH, CG, s)
        chk.append(dict(s=s, C_dagger=float(c_dagger(TH, s)), max_rel_err_identity=float(np.max(np.abs(num / cl - 1))),
                        sup_numeric=float(num.max()), monotone=bool(np.all(np.diff(num) > 0)),
                        C_cert_at_5e26=float(c_cert(TH, 5e26, s)), P_min_at_5e26=float(H.p_min(TH, 5e26, s)),
                        P_min_over_C_at_1e40_times_Cdag=float(H.p_min(TH, 1e40, s) / 1e40 * c_dagger(TH, s))))
    out["a_identity"] = dict(gamma=gam, G=G, E=E, rows=chk,
                             elasticity_dlnCdag_dlns=[float(np.log(c_dagger(TH, 2 * s) / c_dagger(TH, s)) / np.log(2))
                                                      for s in [1e-4, 1e-3, 1e-2]])
    print("(a)", json.dumps(out["a_identity"], indent=1), flush=True)

    # (b) public Chinchilla sweep
    N, D, C, L = load_chinchilla(); keep = epoch_keep(L)
    CGb = np.logspace(20, 40, 201)
    curves = {}
    out["b_chinchilla"] = {}
    with Pool(4) as pool:
        for tag, sel in [("240", keep), ("245", np.arange(len(L)))]:
            lN, lD, lL = np.log(N[sel]), np.log(D[sel]), np.log(L[sel])
            th = fit(lN, lD, lL)
            clC = np.unique(np.round(np.log10(C[sel]), 1), return_inverse=True)[1]
            clN = np.unique(np.round(np.log10(N[sel]) / 0.05).astype(int), return_inverse=True)[1]
            schemes = [("iid", np.arange(len(lL))), ("compute_bins", clC), ("size_groups", clN)]
            for name, cl in schemes:
                sets = [np.where(cl == g)[0] for g in np.unique(cl)]
                boots = np.array(pool.map(boot, [(sets, 16000 + i, lN, lD, lL) for i in range(300)]))
                lf = np.array([np.log(H.Lstar(b, CGb)) for b in boots])
                s_curve = lf.std(0)
                s_inf = float(boots[:, 2].std())  # sd of ln E_hat
                Lhat = H.Lstar(th, CGb)
                # nonlinear 95% certification bound: 5% quantile over replicates of the effective compute
                cert = []
                for i, c in enumerate(CGb):
                    ce = np.array([inv_Lstar(b, Lhat[i]) for b in boots])
                    cert.append(float(np.quantile(np.minimum(ce, 1e300), 0.05)))
                ce_inf = np.array([inv_Lstar(b, np.exp(th[2]) * (1 + 1e-12)) for b in boots])
                row = dict(n=int(len(lL)), E_hat=float(np.exp(th[2])), sd_lnE=s_inf, s_at_5e26=float(np.interp(np.log(5e26), np.log(CGb), s_curve)),
                           C_dagger_50=float(c_dagger(th, s_inf)), C_dagger_80=float(c_dagger(th, s_inf, 0.8)),
                           C_cert_5e26_50=float(5e26 / H.p_min(th, 5e26, np.interp(np.log(5e26), np.log(CGb), s_curve))),
                           P_min_5e26_50=float(H.p_min(th, 5e26, np.interp(np.log(5e26), np.log(CGb), s_curve))),
                           P_min_5e26_80=float(H.p_min(th, 5e26, np.interp(np.log(5e26), np.log(CGb), s_curve), power=0.8)),
                           nonlinear_cert95={f"{c:.0e}": cert[int(np.argmin(np.abs(np.log(CGb / c))))]
                                             for c in [1e22, 1e24, 1e25, 1e26, 5e26, 1e27, 1e28, 1e30, 1e35, 1e40]},
                           nonlinear_cert95_limit=float(np.quantile(np.minimum(ce_inf, 1e300), 0.05)),
                           frac_boot_E_below_Ehat=float(np.mean(np.exp(boots[:, 2]) < np.exp(th[2]))))
                out["b_chinchilla"][f"{tag}:{name}"] = row
                curves[f"{tag}:{name}"] = (th, s_curve, cert)
                print(tag, name, {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in row.items() if k != "nonlinear_cert95"},
                      {k: f"{v:.2e}" for k, v in row["nonlinear_cert95"].items()}, f"{time.time()-t0:.0f}s", flush=True)

    # (c) design law
    rows = []
    for Chi in [1e21, 1e22, 1e23, 1e24, 1e25]:
        for K in [6, 12, 24, 48, 120]:
            M = 240 // K
            lN, lD, lC, g = H.iso_design(Chi / 1e3, Chi, K, M)
            for sw in [1e-4, 3e-4, 1e-3, 3e-3]:
                for sb in [0.0, 3e-4, 1e-3, 3e-3]:
                    for est in ["ols", "gls"]:
                        if est == "gls" and sb == 0.0:
                            continue
                        cov = H.cov_theta(TH, lN, lD, g, sw, sb, estimator=est)
                        sdE = float(np.sqrt(cov[2, 2]))
                        s_tot = float(np.sqrt(sdE ** 2 + sb ** 2 + sw ** 2))  # frontier run: own offset + noise, m=1
                        Cd = float(c_dagger(TH, s_tot))
                        E_, R_, _, _ = H.frontier_parts(TH, Chi)
                        rows.append(dict(C_hi=Chi, K=K, M=M, sigma_w=sw, sigma_b=sb, est=est, sd_lnE=sdE, s_inf=s_tot,
                                         C_dagger=Cd, reach=float(np.log10(Cd / Chi)),
                                         reach_formula=float(np.log10(R_ / (np.expm1(zval(0.5) * s_tot) * E_)) / gam)))
    out["c_design_law"] = rows
    # scaling of sd(ln E) with sigma_b and K (OLS, sigma_w = 1e-4, C_hi = 1e23)
    sel = [r for r in rows if r["est"] == "ols" and r["sigma_w"] == 1e-4 and r["sigma_b"] > 0 and r["C_hi"] == 1e23]
    X = np.c_[np.ones(len(sel)), np.log([r["sigma_b"] for r in sel]), np.log([r["K"] for r in sel])]
    coef = np.linalg.lstsq(X, np.log([r["sd_lnE"] for r in sel]), rcond=None)[0]
    out["c_scaling_fit"] = dict(intercept=float(coef[0]), exp_sigma_b=float(coef[1]), exp_K=float(coef[2]))
    print("design scaling", out["c_scaling_fit"])
    for r in rows:
        if r["C_hi"] == 1e23 and r["sigma_w"] in (3e-4,) and r["est"] == "ols":
            print(f"C_hi=1e23 K={r['K']} sw={r['sigma_w']} sb={r['sigma_b']} sdE={r['sd_lnE']:.4f} Cdag={r['C_dagger']:.2e} reach={r['reach']:.2f} ({r['reach_formula']:.2f})")
    json.dump(out, open(f"{RES}/exp16_audit_ceiling.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s, c in zip([1e-3, 3e-3, 1e-2], ["C0", "C1", "C2"]):
        ax[0].loglog(CG, CG / H.p_min(TH, CG, s), color=c, label=f"s={s:g}")
        ax[0].axhline(c_dagger(TH, s), color=c, ls=":", lw=.8)
    ax[0].loglog(CG, CG, "k--", lw=.6, label="C_cert = C")
    ax[0].set_xlim(1e20, 1e40); ax[0].set_ylim(1e20, 1e36)
    ax[0].set_xlabel("actual compute of the run C"); ax[0].set_ylabel("certifiable effective compute C/P_min")
    ax[0].set_title("Theorem 1: certifiable compute saturates at C_dagger", fontsize=9); ax[0].legend(fontsize=7)
    for k_, sty in [("240:iid", "-"), ("240:compute_bins", "-"), ("240:size_groups", "-"), ("245:compute_bins", ":")]:
        th, s_curve, cert = curves[k_]
        ax[1].loglog(CGb, np.minimum(cert, 1e60), sty, label=k_)
    ax[1].loglog(CGb, CGb, "k--", lw=.6); ax[1].axvline(5e26, color="r", lw=.7)
    ax[1].set_xlim(1e20, 1e40); ax[1].set_ylim(1e20, 1e30)
    ax[1].set_xlabel("compute of a run landing exactly on the forecast"); ax[1].set_ylabel("95% lower bound on effective compute")
    ax[1].set_title("public Chinchilla law: nonlinear bootstrap certification", fontsize=9); ax[1].legend(fontsize=7)
    for i, sb in enumerate([0.0, 3e-4, 1e-3, 3e-3]):
        rs = [r for r in rows if r["C_hi"] == 1e23 and r["sigma_w"] == 3e-4 and r["sigma_b"] == sb and r["est"] == "ols"]
        ax[2].plot([r["K"] for r in rs], [r["reach"] for r in rs], "-o", color=f"C{i}", label=f"sigma_b={sb:g}")
    ax[2].set_xscale("log"); ax[2].set_xlabel("budgets K (n = 240 runs fixed)"); ax[2].set_ylabel("reach log10(C_dagger / C_hi)")
    ax[2].set_title("design law: sweep 1e20-1e23, sigma_w = 3e-4", fontsize=9); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{RES}/fig10_audit_ceiling.png", dpi=150)
    print("done", f"{time.time()-t0:.0f}s")
