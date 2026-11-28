"""Analysis of results/raw/*.npz -> results/*.json, results/fig*.png"""
import os, glob, json
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(R, "raw")
THR = 0.9
TMIN = 20_000  # ignore checkpoints before this many tokens (initial transient)
summary = {}


def load(name):
    z = np.load(os.path.join(RAW, name + ".npz"))
    return {k: z[k] for k in z.files}


def fit_logistic(x, y):
    """P(y=1) = sigmoid((x - x0)/s). Returns x0, s, mean log-loss."""
    def nll(th):
        x0, ls = th
        q = 1 / (1 + np.exp(-(x - x0) / np.exp(ls)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    best = None
    for x0 in np.linspace(x.min(), x.max(), 15):
        r = minimize(nll, [x0, np.log(0.5)], method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1])), float(best.fun)


def checkpoint_table(d):
    rows = []
    for j, T in enumerate(d["T"]):
        rec = (d["maxcos"][j] >= THR).astype(float)
        rows.append((int(T), d["p"] * T, rec, d["p"]))
    return rows


# ---------------- E1: occurrence-count collapse ----------------
E1 = {}
alphas = [1.2, 1.5, 1.8]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ai, a in enumerate(alphas):
    res = {"per_ck": [], "pooled": {}}
    xs_m, xs_p, ys, Tl = [], [], [], []
    for seed in (0, 1):
        f = f"E1_a{a}_s{seed}"
        if not os.path.exists(os.path.join(RAW, f + ".npz")):
            continue
        d = load(f)
        for (T, m, rec, p) in checkpoint_table(d):
            if T < TMIN:
                continue
            xs_m.append(np.log10(np.maximum(m, 1e-3))); xs_p.append(np.log10(p)); ys.append(rec)
            Tl.append(np.full(len(rec), T))
            if seed == 0:
                # binned curves
                bins = np.linspace(-1, 4, 26)
                lm = np.log10(np.maximum(m, 1e-3)); lp = np.log10(p)
                for ax, lx, bb in ((axes[0, ai], lm, bins), (axes[1, ai], lp, np.linspace(-5, 0, 26))):
                    idx = np.digitize(lx, bb)
                    cx = [lx[idx == b].mean() for b in range(1, len(bb)) if (idx == b).sum() >= 3]
                    cy = [rec[idx == b].mean() for b in range(1, len(bb)) if (idx == b).sum() >= 3]
                    ax.plot(cx, cy, "-", color=plt.cm.viridis(np.log(T / TMIN) / np.log(3.1e6 / TMIN)), lw=1.3)
            if (rec.mean() < 1) and (rec.mean() > 0):
                x0, s, _ = fit_logistic(np.log10(np.maximum(m, 1e-3)), rec)
                res["per_ck"].append(dict(seed=seed, T=T, log10_mstar=x0, width=s, frac_rec=float(rec.mean())))
    if not ys:
        continue
    xm, xp, y, Tc = map(np.concatenate, (xs_m, xs_p, ys, Tl))
    x0m, sm, llm = fit_logistic(xm, y)
    x0p, sp_, llp = fit_logistic(xp, y)
    res["pooled"] = dict(log10_mstar=x0m, mstar=10 ** x0m, width_decades=sm, logloss_vs_m=llm,
                         log10_pstar=x0p, logloss_vs_p=llp, n_points=int(len(y)))
    pc = [r["log10_mstar"] for r in res["per_ck"]]
    Ts = [r["T"] for r in res["per_ck"]]
    if len(pc) > 2:
        res["mstar_vs_T_slope"] = float(np.polyfit(np.log10(Ts), pc, 1)[0])
        res["mstar_range_log10"] = [float(min(pc)), float(max(pc))]
        res["T_range"] = [int(min(Ts)), int(max(Ts))]
    E1[a] = res
    axes[0, ai].set_title(f"alpha={a}: recovery vs expected occurrences m=pT")
    axes[0, ai].set_xlabel("log10 m (expected occurrences seen)"); axes[0, ai].set_ylabel(f"P(max cos >= {THR})")
    axes[0, ai].axvline(x0m, ls="--", c="r", lw=0.8)
    axes[1, ai].set_title(f"alpha={a}: same data vs feature frequency p")
    axes[1, ai].set_xlabel("log10 p (per-token activation prob.)"); axes[1, ai].set_ylabel("P(recovered)")
sm_ = plt.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.LogNorm(TMIN, 3.1e6))
fig.colorbar(sm_, ax=axes, label="training tokens T", shrink=0.6)
fig.savefig(os.path.join(R, "fig1_collapse.png"), dpi=130, bbox_inches="tight")
summary["E1"] = E1

# ---------------- E3: tokens vs steps, lr ----------------
E3 = {}
for f in sorted(glob.glob(os.path.join(RAW, "E3_*.npz"))) + [os.path.join(RAW, "E1_a1.5_s0.npz")]:
    name = os.path.basename(f)[:-4]
    if not os.path.exists(f):
        continue
    d = load(name)
    cfg = json.loads(str(d["cfg"]))
    xs, ys, steps_x = [], [], []
    for (T, m, rec, p) in checkpoint_table(d):
        if T < TMIN:
            continue
        xs.append(np.log10(np.maximum(m, 1e-3))); ys.append(rec)
        steps_x.append(np.log10(np.maximum(m / cfg.get("batch", 256), 1e-5)))
    x, y, xs_steps = map(np.concatenate, (xs, ys, steps_x))
    x0, s, ll = fit_logistic(x, y)
    E3[name] = dict(batch=cfg.get("batch", 256), lr=cfg.get("lr", 1e-3), log10_mstar_tokens=x0, mstar_tokens=10 ** x0,
                    width=s, mstar_in_steps=10 ** x0 / cfg.get("batch", 256),
                    final_frac_rec=float((d["maxcos"][-1] >= THR).mean()), final_fvu=float(d["fvu"][-1]))
summary["E3"] = E3

# ---------------- E2: width-limited regime ----------------
E2 = {}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ai, a in enumerate(alphas):
    E2[a] = {}
    for M in (64, 128, 256, 512, 1024):
        f = f"E2_a{a}_M{M}"
        if not os.path.exists(os.path.join(RAW, f + ".npz")):
            continue
        d = load(f)
        rec = (d["maxcos"][-1] >= THR).astype(float)
        rank = np.arange(1, len(rec) + 1)
        p = d["p"]; T = int(d["T"][-1])
        x0r, sr, _ = fit_logistic(np.log10(rank), 1 - rec) if 0 < rec.mean() < 1 else (np.nan, np.nan, np.nan)
        # frequency at the 50% point
        pstar = float(np.interp(x0r, np.log10(rank), p)) if np.isfinite(x0r) else np.nan
        # mass-weighted coverage: fraction of feature activations that belong to recovered features
        cov = float((p * rec).sum() / p.sum())
        E2[a][M] = dict(n_recovered=int(rec.sum()), rank50=float(10 ** x0r) if np.isfinite(x0r) else None,
                        rank50_over_M=float(10 ** x0r / M) if np.isfinite(x0r) else None,
                        pstar=pstar, pstar_times_M=pstar * M if np.isfinite(pstar) else None,
                        mstar_at_pstar=pstar * T if np.isfinite(pstar) else None,
                        mass_coverage=cov, fvu=float(d["fvu"][-1]), T=T,
                        dead=int(d["dead"][-1]))
        w = 32
        sm_rec = np.convolve(rec, np.ones(w) / w, mode="valid")
        axes[ai].plot(rank[w // 2: w // 2 + len(sm_rec)], sm_rec, label=f"M={M}")
    axes[ai].set_xscale("log"); axes[ai].set_title(f"alpha={a}: recovery vs frequency rank (T=3.07M)")
    axes[ai].set_xlabel("feature frequency rank"); axes[ai].set_ylabel("P(recovered), running mean")
    axes[ai].legend(fontsize=7)
fig.savefig(os.path.join(R, "fig2_width.png"), dpi=130, bbox_inches="tight")
summary["E2"] = E2

# ---------------- Prediction check: unrecovered activation mass vs T ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pred = {}
for ai, a in enumerate(alphas):
    f = f"E1_a{a}_s0"
    if a not in E1 or not os.path.exists(os.path.join(RAW, f + ".npz")):
        continue
    d = load(f)
    ms = E1[a]["pooled"]["mstar"]
    obs, prd, Ts, fv = [], [], [], []
    for j, T in enumerate(d["T"]):
        rec = d["maxcos"][j] >= THR
        p = d["p"]
        obs.append((p * ~rec).sum() / p.sum())
        prd.append((p * (p * T < ms)).sum() / p.sum())
        Ts.append(T); fv.append(d["fvu"][j])
    Ts = np.array(Ts); obs = np.array(obs); prd = np.array(prd); fv = np.array(fv)
    c = plt.cm.tab10(ai)
    axes[0].loglog(Ts, obs, "o", c=c, ms=4, label=f"observed a={a}")
    axes[0].loglog(Ts, np.maximum(prd, 1e-6), "-", c=c, label=f"threshold model a={a}")
    axes[1].loglog(Ts, fv, "o-", c=c, ms=3, label=f"a={a}")
    sel = (Ts >= TMIN) & (obs > 0)
    slope = float(np.polyfit(np.log(Ts[sel]), np.log(obs[sel]), 1)[0]) if sel.sum() > 3 else None
    # theoretical slope for an infinite Zipf tail: -(a-1)/a
    pred[a] = dict(observed_unrec_mass_slope=slope, infinite_zipf_slope=-(a - 1) / a,
                   corr_log_obs_pred=float(np.corrcoef(np.log(obs[sel] + 1e-6), np.log(prd[sel] + 1e-6))[0, 1]) if sel.sum() > 3 else None,
                   mean_abs_err=float(np.abs(obs[sel] - prd[sel]).mean()) if sel.sum() else None,
                   fvu_final=float(fv[-1]))
axes[0].set_xlabel("training tokens T"); axes[0].set_ylabel("unrecovered activation mass")
axes[0].legend(fontsize=7); axes[0].set_title("threshold model vs observed")
axes[1].set_xlabel("training tokens T"); axes[1].set_ylabel("FVU"); axes[1].legend(fontsize=7)
axes[1].set_title("reconstruction FVU")
fig.savefig(os.path.join(R, "fig3_mass.png"), dpi=130, bbox_inches="tight")
summary["prediction_check"] = pred

json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
