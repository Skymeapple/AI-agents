"""Counterfactual fixed-n evaluations on the WalkSAT test bed.
Ground truth: 10,000 runs per instance (pass@k for k <= 2000 via the unbiased estimator) and the
exact satisfiability label (pass@infinity = fraction satisfiable, since every satisfiable instance
had at least one success in 10,000 runs; checked below).
Outputs results/walksat_summary.csv, results/fig_walksat.png"""
import sys, os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ttc import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")
KS = [32, 100, 300, 1000, 2000]
NS = [8, 16, 32, 64]
SUBSETS = 10


def truth_passk(runs, k):
    N = runs.shape[1]
    return unbiased_passk(runs.sum(1), N, k)


def main():
    rng = np.random.default_rng(2026)
    rows, curves = [], {}
    for f in sorted(glob.glob(os.path.join(RES, "walksat_*.npz"))):
        d = np.load(f); runs = d["runs"]; sat = d["sat"]
        name = os.path.basename(f)[8:-4]
        M, N = runs.shape
        phat = runs.mean(1)
        check = dict(cfg=name, M=int(M), sat_frac=float(sat.mean()),
                     sat_with_zero_successes=int(np.sum(sat & (phat == 0))),
                     unsat_with_successes=int(np.sum(~sat & (phat > 0))),
                     min_phat_sat=float(phat[sat].min()) if sat.any() else None)
        print(check, flush=True)
        json.dump(check, open(os.path.join(RES, f"walksat_check_{name}.json"), "w"), indent=1)
        truth = {f"pass@{k}": truth_passk(runs, k) for k in KS}
        truth["pass@inf"] = float(sat.mean())
        gver = g_verifier_ceiling(0.9, 0.01)
        truth["ceiling(t=.9,q=.01)"] = float(np.mean(gver(phat)))
        targets = {f"pass@{k}": g_passk(k) for k in KS}
        targets["ceiling(t=.9,q=.01)"] = gver
        kk = np.unique(np.round(np.logspace(0, np.log10(3000), 40)).astype(int))
        curves[name] = dict(k=kk, truth=[truth_passk(runs, k) for k in kk], sat=float(sat.mean()), n={})
        for n in NS:
            # oracle identified interval: count law of the benchmark's own p-hat distribution
            cl = binom.pmf(np.arange(n + 1)[None, :], n, phat[:, None]).mean(0)
            orc = {t: identified_interval_robust(g, n, cl, m=2000, slack=1e-6) for t, g in targets.items()}
            perm = rng.permutation(N)
            for s in range(SUBSETS):
                idx = perm[s * n:(s + 1) * n]
                X = runs[:, idx].sum(1).astype(int)
                a, b = betabinom_fit(X, n)
                for t, g in targets.items():
                    k = int(t.split("@")[1]) if t.startswith("pass@") else None
                    blo, bhi = box_interval(g, n, X, alpha=0.05, m=1000)
                    row = dict(cfg=name, n=n, subset=s, target=t, truth=truth[t],
                               oracle_lo=orc[t][0], oracle_hi=orc[t][1], box_lo=blo, box_hi=bhi)
                    if k is not None:
                        row["bb"] = float(betabinom_passk(a, b, k))
                        row["pl"] = float(powerlaw_passk(X, n, [k])[0])
                    rows.append(row)
                # pass@inf: parametric extrapolations vs truth; nonparametric lower bound = box lower bound at k=2000
                rows.append(dict(cfg=name, n=n, subset=s, target="pass@inf", truth=truth["pass@inf"],
                                 bb=1.0 if a > 0 else np.nan, pl=float(powerlaw_passk(X, n, [1e12])[0]),
                                 box_lo=box_interval(g_passk(2000), n, X, m=1000)[0], box_hi=1.0,
                                 oracle_lo=orc["pass@2000"][0], oracle_hi=1.0))
                if s == 0:
                    curves[name]["n"][n] = dict(
                        bb=[float(betabinom_passk(a, b, k)) for k in kk],
                        pl=list(powerlaw_passk(X, n, kk)),
                        box=[box_interval(g_passk(int(k)), n, X, m=600) for k in kk[::3]], kbox=kk[::3])
            print(name, n, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "walksat_raw.csv"), index=False, float_format="%.5f")
    df["box_cover"] = ((df.box_lo - 1e-9 <= df.truth) & (df.truth <= df.box_hi + 1e-9)).astype(float)
    df["bb_err"] = df.bb - df.truth; df["pl_err"] = df.pl - df.truth
    s = df.groupby(["cfg", "n", "target"], sort=False).agg(
        truth=("truth", "first"), oracle_lo=("oracle_lo", "first"), oracle_hi=("oracle_hi", "first"),
        box_lo=("box_lo", "mean"), box_hi=("box_hi", "mean"), box_cover=("box_cover", "mean"),
        bb_err=("bb_err", "mean"), bb_err_absmax=("bb_err", lambda x: np.abs(x).max()),
        pl_err=("pl_err", "mean")).reset_index()
    s.to_csv(os.path.join(RES, "walksat_summary.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))
    # figure
    fig, axs = plt.subplots(1, len(curves), figsize=(5 * len(curves), 3.8))
    for ax, (name, c) in zip(np.atleast_1d(axs), curves.items()):
        ax.semilogx(c["k"], c["truth"], "k-", lw=2.5, label="truth (10,000 runs)")
        ax.axhline(c["sat"], color="k", ls=":", lw=1, label="pass@inf = SAT fraction")
        cn = c["n"][16]
        ax.semilogx(c["k"], cn["bb"], "C1--", label="beta-binomial fit (n=16)")
        ax.semilogx(c["k"], cn["pl"], "C2-.", label="power-law fit (n=16)")
        bx = np.array(cn["box"])
        ax.fill_between(cn["kbox"], bx[:, 0], bx[:, 1], color="C0", alpha=0.25, label="95% nonparametric box (n=16)")
        ax.axvline(16, color="gray", lw=0.8); ax.axvline(256, color="gray", lw=0.8, ls="--")
        ax.set_title(name); ax.set_xlabel("k"); ax.set_ylim(0, 1.02)
    np.atleast_1d(axs)[0].set_ylabel("pass@k"); np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_walksat.png"), dpi=150)


if __name__ == "__main__":
    main()
