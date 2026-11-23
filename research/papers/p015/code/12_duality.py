"""Replicate-length duality: distribution-free estimate of the task-level hazard shape,
an overidentification check, out-of-window (short -> long) forecasting test, and likelihood-ratio
bounds on day/week/month-long success.

Outputs
  results/duality_npmle_k.csv          per-agent NPMLE-frailty k with 95% profile CI (full data and variants)
  results/duality_moment_k.csv         model-free moment-matching k_j for j = 2, 3, 4 (overidentification)
  results/duality_forecast.csv         per agent x cut x method: predicted vs observed long-task success
  results/duality_bounds.csv           LR bounds on S(T*) for T* = 8 h, 40 h, 160 h (full data)
  results/duality_summary.txt          summary tables
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from common import *
import npmle

d = load_runs()
AGENTS = sorted(d.alias.unique())
TSTARS = [480.0, 2400.0, 9600.0]
CUTS = [60.0, 120.0, 240.0]


def tt_arrays(tt):
    return tt.n.values.astype(float), tt.s.values.astype(float), tt["T"].values.astype(float)


def kgrid_from(f, step=0.1):
    lo = f["k_lo"] if np.isfinite(f["k_lo"]) else f["k"]
    hi = f["k_hi"] if np.isfinite(f["k_hi"]) else f["k"]
    ks = np.unique(np.round(np.concatenate([np.arange(lo, hi + 1e-9, step), [lo, hi, f["k"]]]), 3))
    return ks


# ---------------------------------------------------------------- moment-matching (model-free in G)
def ustat(s, n, j):
    from scipy.special import comb
    return np.where(n >= j, comb(s, j) / np.maximum(comb(n, j), 1), np.nan)


def loclin(x, y, x0, h):
    m = np.isfinite(y)
    x, y = x[m], y[m]
    w = np.exp(-0.5 * ((x0[:, None] - x[None, :]) / h) ** 2)
    dx = x[None, :] - x0[:, None]
    s0, s1, s2 = w.sum(1), (w * dx).sum(1), (w * dx ** 2).sum(1)
    t0, t1 = (w * y).sum(1), (w * dx * y).sum(1)
    return (s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2)


def moment_k(tt, j, h=0.7, lo=4.0, hi=960.0):
    """Choose k so that M_j(T) = S(j^{1/k} T) on a grid of T, both sides smoothed in log T.
    Uses tasks >= 4 min (the sub-4-min tasks are near ceiling for most agents)."""
    n, s, T = tt_arrays(tt)
    x = np.log(T)
    p1 = s / n
    Mj = ustat(s, n, j)
    best = (np.inf, np.nan)
    for k in np.arange(0.4, 3.01, 0.02):
        r = j ** (1 / k)
        g = np.log(np.geomspace(lo, hi / r, 12))
        if g[-1] <= g[0]:
            continue
        lhs = loclin(x, Mj, g, h)
        rhs = loclin(x, p1, g + np.log(r), h)
        ok = (np.isfinite(lhs) & np.isfinite(rhs))
        sse = np.mean((lhs[ok] - rhs[ok]) ** 2)
        if sse < best[0]:
            best = (sse, k)
    return best[1]


# ---------------------------------------------------------------- per-agent work
def work(alias):
    da = d[d.alias == alias]
    tt = task_table(da)
    n, s, T = tt_arrays(tt)
    out = dict(alias=alias, npmle=[], moment=[], forecast=[], bounds=[])
    # (A) full-data NPMLE and variants
    variants = {"all tasks": np.ones(len(tt), bool),
                "baseline-timed only": (tt.hsrc == "baseline").values}
    allg = d.groupby("task_id").y.agg(["size", "sum"])
    never = set(allg.index[allg["sum"] == 0])
    always = set(allg.index[allg["sum"] == allg["size"]])
    variants["drop never-solved*"] = ~tt.task_id.isin(never).values
    variants["drop never- & always-solved*"] = ~tt.task_id.isin(never | always).values
    variants["drop RE-Bench"] = (tt.source != "RE-Bench").values
    full = None
    for vn, m in variants.items():
        f = npmle.fit(n[m], s[m], T[m])
        if vn == "all tasks":
            full = f
        out["npmle"].append(dict(alias=alias, variant=vn, ntask=int(m.sum()), k=f["k"], k_lo=f["k_lo"], k_hi=f["k_hi"],
                                 lo_edge=f["k_lo_at_edge"], hi_edge=f["k_hi_at_edge"], ll=f["ll"],
                                 S8h=float(npmle.S_pooled(f["pi"], f["k"], 480)[0]),
                                 S40h=float(npmle.S_pooled(f["pi"], f["k"], 2400)[0])))
    # (B) moment-matching k_j
    for j in (2, 3, 4):
        out["moment"].append(dict(alias=alias, j=j, k_moment=moment_k(tt, j)))
    # (D) bounds at long T* with full data
    ks = kgrid_from(full)
    B = npmle.lr_bounds(n, s, T, TSTARS, full["ll"], ks)
    for b, Ts in enumerate(TSTARS):
        pt = float(npmle.S_pooled(full["pi"], full["k"], Ts)[0])
        vals, w = npmle.p_draws(full["pi"], full["k"], Ts)
        out["bounds"].append(dict(alias=alias, Tstar_h=Ts / 60, point=pt, lo=np.nanmin(B[:, b, 0]),
                                  hi=np.nanmax(B[:, b, 1]), frac_p_ge_08=float(w[vals >= 0.8].sum()),
                                  frac_p_le_005=float(w[vals <= 0.05].sum())))
    # (C) out-of-window forecasts
    w_run = (1.0 / da.groupby("task_id").y.transform("size")).values
    Trun, yrun = da["T"].values, da["y"].values.astype(float)
    for C in CUTS:
        tr = T < C
        for band, te in {"[C,4C)": (T >= C) & (T < 4 * C), ">=C": T >= C}.items():
            if te.sum() < 3:
                continue
            obs = float(np.mean(s[te] / n[te]))
            rec = dict(alias=alias, cut_min=C, band=band, n_target_tasks=int(te.sum()), obs=obs)
            trr = Trun < C
            for nm in ["log-logistic (METR)", "Weibull"]:
                th, _ = fit_family(nm, Trun[trr], yrun[trr], w_run[trr])
                rec[nm] = float(np.mean(FAMILIES[nm][0](th, T[te])))
            pcm = fit_cm_mixture(Trun[trr], yrun[trr], w_run[trr])
            rec["CM mixture"] = float(np.mean(S_cm(pcm, T[te])))
            g = fit_glmm(tt[tr], link="cloglog", nested=False)
            mu, kk, lst = g.x[:3]
            from numpy.polynomial.hermite_e import hermegauss
            xg, wg = hermegauss(40)
            wg = wg / wg.sum()
            Sg = np.exp(-np.exp(np.clip(mu + kk * np.log(T[te] / T_REF)[:, None] + np.exp(lst) * xg[None, :], -30, 8))) @ wg
            rec["Gaussian frailty"] = float(np.mean(Sg))
            rec["k_gauss_window"] = float(kk)
            fw = npmle.fit(n[tr], s[tr], T[tr])
            rec["NPMLE frailty"] = float(np.mean(npmle.S_pooled(fw["pi"], fw["k"], T[te])))
            rec["k_npmle_window"] = fw["k"]
            f1 = npmle.em_weights(npmle.loglik_matrix(n[tr], s[tr], T[tr], 1.0))[0]
            rec["NPMLE frailty, k=1"] = float(np.mean(npmle.S_pooled(f1, 1.0, T[te])))
            # LR bounds for the target mean (linear functional of G): use per-task average of kernels
            ksw = kgrid_from(fw, step=0.1)
            lo_v, hi_v = np.inf, -np.inf
            import cvxpy as cp
            for k in ksw:
                LL = npmle.loglik_matrix(n[tr], s[tr], T[tr], k)
                mx = LL.max(1, keepdims=True)
                L = np.exp(LL - mx)
                keep = L.max(0) > 1e-12
                av = np.mean(np.exp(-np.exp(np.clip(k * np.log(T[te] / T_REF)[:, None] + npmle.U_GRID[None, :],
                                                    -40, 8))), 0)[keep]
                pi = cp.Variable(int(keep.sum()), nonneg=True)
                cons = [cp.sum(pi) == 1, cp.sum(cp.log(L[:, keep] @ pi)) >= fw["ll"] - 1.92 - float(mx.sum())]
                for sense in (cp.Minimize, cp.Maximize):
                    pr = cp.Problem(sense(av @ pi), cons)
                    try:
                        pr.solve(solver=cp.CLARABEL)
                        if pr.status in ("optimal", "optimal_inaccurate"):
                            lo_v, hi_v = min(lo_v, pr.value), max(hi_v, pr.value)
                    except Exception:
                        pass
            rec["NPMLE bound lo"], rec["NPMLE bound hi"] = lo_v, hi_v
            out["forecast"].append(rec)
    print(alias, "done", flush=True)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with Pool(nproc) as p:
        res = p.map(work, AGENTS)
    K = pd.DataFrame([r for o in res for r in o["npmle"]])
    M = pd.DataFrame([r for o in res for r in o["moment"]])
    F = pd.DataFrame([r for o in res for r in o["forecast"]])
    B = pd.DataFrame([r for o in res for r in o["bounds"]])
    K.to_csv(f"{RES}/duality_npmle_k.csv", index=False)
    M.to_csv(f"{RES}/duality_moment_k.csv", index=False)
    F.to_csv(f"{RES}/duality_forecast.csv", index=False)
    B.to_csv(f"{RES}/duality_bounds.csv", index=False)
    pd.set_option("display.width", 250)
    L = []
    L.append("(A) NPMLE-frailty k (G unrestricted), medians over 20 agents; * = outcome-selected filter")
    L.append(K.groupby("variant").agg(k_median=("k", "median"), k_q25=("k", lambda x: x.quantile(.25)),
                                      k_q75=("k", lambda x: x.quantile(.75)), n_k_gt1=("k", lambda x: int((x > 1).sum())),
                                      n_ci_above1=("k_lo", lambda x: int((x > 1).sum())),
                                      n_ci_below1=("k_hi", lambda x: int((x < 1).sum())),
                                      S40h_median=("S40h", "median")).round(3).to_string())
    L.append("\nPer-agent (all tasks):")
    L.append(K[K.variant == "all tasks"][["alias", "k", "k_lo", "k_hi", "S8h", "S40h"]].round(3).to_string(index=False))
    L.append("\nPer-agent (drop never-solved*):")
    L.append(K[K.variant == "drop never-solved*"][["alias", "k", "k_lo", "k_hi"]].round(3).to_string(index=False))
    L.append("\n(B) Model-free moment-matching k_j: M_j(T) = S(j^(1/k) T)")
    Mw = M.pivot(index="alias", columns="j", values="k_moment")
    L.append(Mw.round(2).to_string())
    L.append(f"median k_2, k_3, k_4: {Mw.median().round(3).to_dict()}; "
             f"median within-agent range max-min: {float((Mw.max(1) - Mw.min(1)).median()):.3f}")
    L.append("\n(C) Out-of-window forecasts of mean success on longer tasks (equal task weights)")
    meth = ["log-logistic (METR)", "Weibull", "CM mixture", "Gaussian frailty", "NPMLE frailty", "NPMLE frailty, k=1"]
    for (C, band), g in F.groupby(["cut_min", "band"]):
        err = g[meth].sub(g.obs, axis=0)
        cover = ((g.obs >= g["NPMLE bound lo"] - 1e-9) & (g.obs <= g["NPMLE bound hi"] + 1e-9)).mean()
        L.append(f"train T < {C:.0f} min, target {band}: agents={len(g)}, mean obs={g.obs.mean():.3f}, "
                 f"median target tasks={g.n_target_tasks.median():.0f}")
        L.append("   mean signed error: " + ", ".join(f"{m}: {err[m].mean():+.3f}" for m in meth))
        L.append("   mean abs error:    " + ", ".join(f"{m}: {err[m].abs().mean():.3f}" for m in meth))
        L.append(f"   observed inside NPMLE LR bounds: {cover:.2f}; mean bound width "
                 f"{(g['NPMLE bound hi'] - g['NPMLE bound lo']).mean():.3f}; median window k (NPMLE) "
                 f"{g.k_npmle_window.median():.2f}, (Gauss) {g.k_gauss_window.median():.2f}")
    L.append("\n(D) LR bounds (95%, union over k in its profile CI) on success on a random task of length T*")
    L.append(B.round(3).to_string(index=False))
    open(f"{RES}/duality_summary.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
