"""Analysis of ../results/raw/*.npz  ->  ../results/summary.json and figures.
Usage: python3 analyze.py   (deterministic; bootstrap seeds fixed)"""
import os, glob, json
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results"); RAW = os.environ.get("RAW", os.path.join(R, "raw"))
THR = float(os.environ.get("THR", 0.9))
TMIN = int(os.environ.get("TMIN", 100_000))
SUF = os.environ.get("SUF", "")
rng = np.random.default_rng(0)
S = {"THR": THR, "TMIN": TMIN}


def load(name):
    z = np.load(os.path.join(RAW, name + ".npz"))
    d = {k: z[k] for k in z.files}; d["cfg"] = json.loads(str(d["cfg"])); return d


def sig(z): return 1 / (1 + np.exp(-z))


def nll_beta(th, lp, lT, y):
    """P(rec) = q * sigmoid((lp + b*lT - c)/s);  b=1: occurrence-count threshold; b=0: frequency threshold."""
    b, c, ls, lq = th
    P = sig(lq) * sig((lp + b * lT - c) / np.exp(ls))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean()


def fit_beta(lp, lT, y, fix_b=None):
    best = None
    for b0 in ([fix_b] if fix_b is not None else [0.0, 0.5, 1.0, 1.5]):
        for c0 in np.linspace(-1, 4, 6):
            f = (lambda th: nll_beta([fix_b, *th], lp, lT, y)) if fix_b is not None else (lambda th: nll_beta(th, lp, lT, y))
            x0 = [c0, np.log(0.3), 3.0] if fix_b is not None else [b0, c0, np.log(0.3), 3.0]
            r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
    th = [fix_b, *best.x] if fix_b is not None else list(best.x)
    return dict(beta=float(th[0]), log10_mstar=float(th[1]), width=float(np.exp(th[2])), ceiling=float(sig(th[3])),
                nll=float(best.fun), n=int(len(y)))


def stack(d, tmin=TMIN, tmax=np.inf, xcount=None):
    lp, lT, y = [], [], []
    for j, T in enumerate(d["T"]):
        if T < tmin or T > tmax: continue
        lp.append(np.log10(d["p"] if xcount is None else xcount)); lT.append(np.full(len(d["p"]), np.log10(T)))
        y.append((d["maxcos"][j] >= THR).astype(float))
    return map(np.concatenate, (lp, lT, y))


def boot_beta(lp, lT, y, nf, B=40):
    """bootstrap over features (resample feature indices, keep all their checkpoints)."""
    nck = len(y) // nf; bs = []
    for _ in range(B):
        idx = rng.integers(0, nf, nf)
        sel = (np.arange(nck)[:, None] * nf + idx[None, :]).ravel()
        bs.append(fit_beta(lp[sel], lT[sel], y[sel])["beta"])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------------------------------------------------------- E1: collapse test
E1 = {}
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
for ai, a in enumerate((1.2, 1.5, 1.8)):
    res = {}
    for s in (0, 1):
        n = f"E1_a{a}_s{s}"
        if not os.path.exists(os.path.join(RAW, n + ".npz")): continue
        d = load(n); lp, lT, y = stack(d)
        free = fit_beta(lp, lT, y)
        m1 = fit_beta(lp, lT, y, fix_b=1.0); m0 = fit_beta(lp, lT, y, fix_b=0.0)
        ci = boot_beta(lp, lT, y, len(d["p"])) if s == 0 else None
        res[f"seed{s}"] = dict(free=free, beta_CI95=ci, occ_model=m1, freq_model=m0,
                               final_rec=float((d["maxcos"][-1] >= THR).mean()), final_fvu=float(d["fvu"][-1]),
                               final_dead=int(d["dead"][-1]),
                               mstar_occ=10 ** m1["log10_mstar"])
        if s == 0:
            cm = plt.cm.viridis
            for j, T in enumerate(d["T"]):
                if T < TMIN: continue
                rec = (d["maxcos"][j] >= THR).astype(float); col = cm(np.log(T / TMIN) / np.log(d["T"][-1] / TMIN))
                for ax, lx, bb in ((axes[0, ai], np.log10(d["p"] * T), np.linspace(0, 6, 25)),
                                   (axes[1, ai], np.log10(d["p"]), np.linspace(-5, 0, 25))):
                    idx = np.digitize(lx, bb)
                    cx = [lx[idx == b].mean() for b in range(1, len(bb)) if (idx == b).sum() >= 4]
                    cy = [rec[idx == b].mean() for b in range(1, len(bb)) if (idx == b).sum() >= 4]
                    ax.plot(cx, cy, "-", color=col, lw=1.3)
            axes[0, ai].axvline(m1["log10_mstar"], ls="--", c="r", lw=0.8)
    # cross-seed prediction: fit on seed0, predict seed1 recovered fraction & unrecovered activation mass at every ckpt
    if "seed0" in res and "seed1" in res:
        th = res["seed0"]["free"]; d1 = load(f"E1_a{a}_s1"); rows = []
        for j, T in enumerate(d1["T"]):
            if T < TMIN: continue
            P = th["ceiling"] * sig((np.log10(d1["p"]) + th["beta"] * np.log10(T) - th["log10_mstar"]) / th["width"])
            rec = d1["maxcos"][j] >= THR; p = d1["p"]
            rows.append(dict(T=int(T), obs_frac=float(rec.mean()), pred_frac=float(P.mean()),
                             obs_unrec_mass=float((p * ~rec).sum() / p.sum()), pred_unrec_mass=float((p * (1 - P)).sum() / p.sum())))
        res["cross_seed_prediction"] = rows
        res["cross_seed_MAE_frac"] = float(np.mean([abs(r["obs_frac"] - r["pred_frac"]) for r in rows]))
        res["cross_seed_MAE_mass"] = float(np.mean([abs(r["obs_unrec_mass"] - r["pred_unrec_mass"]) for r in rows]))
    E1[str(a)] = res
    axes[0, ai].set_title(f"α={a}: vs expected occurrences m = pT", fontsize=10)
    axes[0, ai].set_xlabel("log10 m"); axes[0, ai].set_ylabel(f"P(max cos ≥ {THR})")
    axes[1, ai].set_title(f"α={a}: same data vs frequency p", fontsize=10)
    axes[1, ai].set_xlabel("log10 p"); axes[1, ai].set_ylabel(f"P(max cos ≥ {THR})")
fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.LogNorm(TMIN, 3e6)), ax=axes, label="training tokens T", shrink=0.6)
fig.savefig(os.path.join(R, f"fig1_collapse{SUF}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
S["E1"] = E1

# ---------------------------------------------------------------- E3: what sets m*?
E3 = {}
names = ["E1_a1.5_s0"] + sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(RAW, "E3_*.npz")) + glob.glob(os.path.join(RAW, "E5_*.npz")))
for n in names:
    d = load(n); c = d["cfg"]; lp, lT, y = stack(d)
    free = fit_beta(lp, lT, y); m1 = fit_beta(lp, lT, y, fix_b=1.0)
    E3[n] = dict(opt=c.get("opt", "adam"), beta2=c.get("beta2", 0.999), beta_CI95=boot_beta(lp, lT, y, len(d["p"]), B=20), lr=c["lr"], batch=c["batch"], D=c["D"], L0=c["L0"], K=c["K"], beta=free["beta"],
                 log10_mstar_tokens=m1["log10_mstar"], mstar_tokens=10 ** m1["log10_mstar"],
                 mstar_steps_equiv=10 ** m1["log10_mstar"] / c["batch"],
                 width=m1["width"], ceiling=m1["ceiling"], final_rec=float((d["maxcos"][-1] >= THR).mean()),
                 final_fvu=float(d["fvu"][-1]), superposition_ratio=c["N"] / c["D"])
S["E3"] = E3

# ---------------------------------------------------------------- E2: width-limited
E2 = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ai, a in enumerate((1.2, 1.5)):
    E2[str(a)] = {}
    Ms = [64, 128, 256, 512] + ([768] if os.path.exists(os.path.join(RAW, f"E1_a{a}_s0.npz")) else [])
    for M in Ms:
        n = f"E2_a{a}_M{M}" if M != 768 else f"E1_a{a}_s0"
        if not os.path.exists(os.path.join(RAW, n + ".npz")): continue
        d = load(n); rec = (d["maxcos"][-1] >= THR).astype(float); p = d["p"]; T = int(d["T"][-1])
        rank = np.arange(1, len(p) + 1)
        # recovered features among those that pass the data threshold (m >= 10 * m*_base)
        E2[str(a)][M] = dict(n_rec=int(rec.sum()), n_rec_over_M=float(rec.sum() / M),
                             mass_cov=float((p * rec).sum() / p.sum()), fvu=float(d["fvu"][-1]),
                             median_rank_recovered=float(np.median(rank[rec > 0])) if rec.sum() else None,
                             rank90_recovered=float(np.percentile(rank[rec > 0], 90)) if rec.sum() else None)
        w = 24; sm = np.convolve(rec, np.ones(w) / w, mode="valid")
        axes[ai].plot(rank[w // 2: w // 2 + len(sm)], sm, label=f"M={M}")
    axes[ai].set_xscale("log"); axes[ai].set_xlabel("feature frequency rank"); axes[ai].set_ylabel("P(recovered), running mean")
    axes[ai].set_title(f"α={a}, T=3M: recovery vs rank, by width M", fontsize=10); axes[ai].legend(fontsize=7)
fig.savefig(os.path.join(R, f"fig2_width{SUF}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
S["E2"] = E2

# ---------------------------------------------------------------- E4: replay buffer
E4 = {}
base = load("E1_a1.5_s0")
Ys, LE, LU = [], [], []
for n in ["E1_a1.5_s0"] + sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(RAW, "E4_*.npz"))):
    d = load(n); U = d["cfg"]["buffer"]
    for j, T in enumerate(d["T"]):
        if T < TMIN: continue
        y = (d["maxcos"][j] >= THR).astype(float)
        uniq = d["uniq_counts"] * 1.0 if U else d["p"] * T
        uniq_seen = np.minimum(uniq, d["p"] * T) if U else uniq
        Ys.append(y); LE.append(np.log10(np.maximum(d["p"] * T, 1e-3))); LU.append(np.log10(np.maximum(uniq_seen, 0.5)))
    E4[n] = dict(buffer=U, final_rec=float((d["maxcos"][-1] >= THR).mean()), final_fvu=float(d["fvu"][-1]),
                 epochs=float(d["T"][-1] / U) if U else 1.0,
                 rec_at_ck=[float((d["maxcos"][j] >= THR).mean()) for j in range(len(d["T"]))], T=d["T"].tolist())
y, le, lu = map(np.concatenate, (Ys, LE, LU)); z = np.zeros_like(le)
fe = fit_beta(le, z, y, fix_b=0.0); fu = fit_beta(lu, z, y, fix_b=0.0)
S["E4"] = dict(runs=E4, fit_on_exposures=fe, fit_on_unique_occurrences=fu,
               delta_nll_unique_minus_exposure=fu["nll"] - fe["nll"])
fig, ax = plt.subplots(figsize=(5.5, 4))
for n, v in E4.items():
    ax.semilogx(v["T"], v["rec_at_ck"], "o-", ms=3, label=(f"buffer {v['buffer']:,} unique tokens" if v["buffer"] else "fresh data (no replay)"))
ax.set_xlabel("total training tokens (incl. repeats)"); ax.set_ylabel(f"fraction recovered (cos ≥ {THR})")
ax.set_title("α=1.5: fresh data vs replayed buffer", fontsize=10); ax.legend(fontsize=7)
fig.savefig(os.path.join(R, f"fig3_replay{SUF}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

json.dump(S, open(os.path.join(R, f"summary{SUF}.json"), "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "E1" else {a: {s: (x["free"] if isinstance(x, dict) and "free" in x else x) for s, x in r.items() if s != "cross_seed_prediction"} for a, r in v.items()}) for k, v in S.items()}, indent=1, default=float)[:6000])
