"""LLM data: the public Large Language Monkeys release (Brown et al. 2024), MATH, Pythia 70M ... 12B,
128 problems x 10,000 samples each (revision r1).

Data: results/llm_monkeys_math_pythia_counts.csv = per-problem number of correct samples out of 10,000,
taken from the processed file data/processed_data/large_language_monkeys_original_pass_at_k.parquet
(rows with Scaling Parameter == 1) of the public code release of Schaeffer et al. (2025)
(github.com/RylanSchaeffer/KoyejoLab-Large-How-Do-Language-Monkey-Power-Get-Their-Power, commit cd078ae),
which that release built from the Hugging Face dataset ScalingIntelligence/monkey_business (subsets MATH_Pythia-*).

Protocol (the one pre-registered in the first version of the paper, Sec. 7):
  For n in {16,32,64,128}, form 10 disjoint counterfactual evaluations per model by drawing n of the 10,000
  samples of each problem without replacement (sequential hypergeometric draws; exactly equivalent to
  subsampling the stored samples).  Targets pass@k for k in {4n, n^2/4, n^2, 4n^2} (k <= 10,000) and k = 10,000.
  Truth = Chen et al. unbiased estimate from all 10,000 samples (benchmark-specific estimand).
  Methods: 95% box interval; oracle identified interval (sharp interval for the law of p_hat_i = c_i/10^4);
  beta-binomial, zero-inflated BB, 2-/3-component Beta mixtures, ZI 2-mixture, BIC choice; power law.
  P1: intervals reported.  P2: oracle width at k = n^2/4 <= 2 E_n(pass@(n^2/4)) (a theorem).
  P3: beta-binomial forecasts at k = 4n^2 lie above the truth on models with unsolved fraction > 5%.
Outputs: results/llm_raw.csv, results/llm_summary.csv, results/llm_prereg.json, results/fig_llm.png
"""
import sys, os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import g_passk, box_interval, identified_interval, bernstein_matrix, powerlaw_passk, best_approx_error
from exp_baselines import fit_all
from scipy.special import gammaln

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N = 10000
NS = [16, 32, 64, 128]
NSUB = 10


def unbiased(c, Ntot, k):
    c = np.asarray(c)
    lf = gammaln(Ntot - c + 1) - gammaln(Ntot - c - k + 1) - gammaln(Ntot + 1) + gammaln(Ntot - k + 1)
    return float(np.mean(1 - np.where(Ntot - c >= k, np.exp(lf), 0.0)))


def targets(n):
    ks = [4 * n, n * n // 4, n * n, 4 * n * n]
    return sorted(set([k for k in ks if k <= N] + [N]))


def subsets(c, n, seed):
    rng = np.random.default_rng(seed)
    rem_c, rem_N = c.copy(), N
    out = []
    for s in range(NSUB):
        x = rng.hypergeometric(rem_c, rem_N - rem_c, n)
        out.append(x); rem_c = rem_c - x; rem_N -= n
    return out


def job(args):
    model, c, n, mi = args
    rows = []
    ks = targets(n)
    for s, X in enumerate(subsets(c, n, 1000 * mi + n)):
        pl = powerlaw_passk(X, n, ks)
        par = fit_all(X, np.full(len(X), n), ks, seed=s)
        for j, k in enumerate(ks):
            lo, hi = box_interval(g_passk(k), n, X, alpha=0.05)
            r = dict(model=model, n=n, subset=s, k=k, box_lo=lo, box_hi=hi, powerlaw=float(pl[j]))
            for m, pr in par.items():
                r[m] = pr[k]
            r["BIC_sel"] = par["BIC"]["selected"]
            rows.append(r)
    return rows


def oracle_interval(g, n, law, ph, m=4000, slack=1e-8):
    """Identified interval for the count law of the plug-in law (1/M) sum_i delta_{p_hat_i}.
    The grid is augmented with the p_hat_i (so the plug-in law is feasible) and rows of the Bernstein
    matrix are rescaled.  Exact equality constraints are numerically unreliable (HiGHS reports
    infeasibility), so we allow an L1 slack of 1e-8 on the count law; because the problem is severely
    ill-conditioned for k >> n this slack can widen the interval noticeably (see paper, Sec. 6.3)."""
    from scipy.optimize import linprog
    from ttc import cheb_grid
    p = np.unique(np.concatenate([cheb_grid(m), ph]))
    B = bernstein_matrix(n, p).T; gv = g(p)
    sc = B.max(1); Bs = B / sc[:, None]; ls = law / sc; K = n + 1; mm = len(p)
    A_ub = np.vstack([np.hstack([Bs, -np.eye(K)]), np.hstack([-Bs, -np.eye(K)]),
                      np.hstack([np.zeros((1, mm)), sc[None, :]])])
    b_ub = np.concatenate([ls, -ls, [slack]])
    A_eq = np.hstack([np.ones((1, mm)), np.zeros((1, K))])
    out = []
    for sgn in (+1, -1):
        r = linprog(np.concatenate([sgn * gv, np.zeros(K)]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                    bounds=[(0, None)] * (mm + K), method="highs")
        out.append(sgn * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]


def main():
    t0 = time.time()
    d = pd.read_csv(os.path.join(RES, "llm_monkeys_math_pythia_counts.csv"))
    models = sorted(d.model.unique(), key=lambda m: float(m.split()[1].rstrip("MB")) * (1e-3 if m.endswith("M") else 1))
    C = {m: d[d.model == m].sort_values("problem_idx").n_correct.values.astype(int) for m in models}
    jobs = [(m, C[m], n, mi) for mi, m in enumerate(models) for n in NS]
    ck = os.path.join(RES, "llm_fits.pkl")
    if os.path.exists(ck):
        rows = pd.read_pickle(ck)
    else:
        with Pool(int(os.environ.get("NPROC", 4))) as pool:
            rows = [r for rr in pool.map(job, jobs, chunksize=1) for r in rr]
        pd.to_pickle(rows, ck)
    df = pd.DataFrame(rows)
    df["truth"] = [unbiased(C[m], N, k) for m, k in zip(df.model, df.k)]
    df.to_csv(os.path.join(RES, "llm_raw.csv"), index=False, float_format="%.5f")
    print("fits done", time.time() - t0, flush=True)
    # oracle intervals and population worst case
    orc = []
    for m in models:
        ph = C[m] / N
        for n in NS:
            law = bernstein_matrix(n, ph).mean(0)
            for k in targets(n):
                lo, hi = oracle_interval(g_passk(k), n, law, ph)
                plug = float(np.mean(1 - (1 - ph) ** k))
                orc.append(dict(model=m, n=n, k=k, oracle_lo=lo, oracle_hi=hi, plugin_truth=plug,
                                twoE_n=2 * best_approx_error(g_passk(k), n, m=1500)[1]))
    orc = pd.DataFrame(orc)
    df = df.merge(orc, on=["model", "n", "k"])
    df["box_cover"] = (df.box_lo - 1e-9 <= df.truth) & (df.truth <= df.box_hi + 1e-9)
    mods = ["BB", "ZIBB", "BMix2", "BMix3", "ZIBMix2", "BIC", "powerlaw"]
    agg = {"truth": ("truth", "first"), "unsolved_frac": ("model", lambda x: 0.0),
           "box_cover": ("box_cover", "mean"), "box_half": ("box_hi", "mean"),
           "oracle_lo": ("oracle_lo", "first"), "oracle_hi": ("oracle_hi", "first"),
           "plugin_truth": ("plugin_truth", "first"), "twoE_n": ("twoE_n", "first")}
    for mo in mods:
        df[f"err_{mo}"] = df[mo] - df.truth
        agg[f"bias_{mo}"] = (f"err_{mo}", "mean")
    df["box_halfw"] = (df.box_hi - df.box_lo) / 2
    agg["box_half"] = ("box_halfw", "mean")
    s = df.groupby(["model", "n", "k"], sort=False).agg(**agg).reset_index()
    s["unsolved_frac"] = [float(np.mean(C[m] == 0)) for m in s.model]
    s["oracle_width"] = s.oracle_hi - s.oracle_lo
    s.to_csv(os.path.join(RES, "llm_summary.csv"), index=False, float_format="%.4f")
    df.to_csv(os.path.join(RES, "llm_raw.csv"), index=False, float_format="%.5f")
    # pre-registered checks
    pre = {}
    p2 = s[s.k == s.n ** 2 // 4]
    pre["P2_cells"] = len(p2)
    pre["P2_oracle_width_le_2En_all"] = bool((p2.oracle_width <= p2.twoE_n + 1e-4).all())
    pre["P2_max_oracle_width"] = float(p2.oracle_width.max())
    pre["P2_table"] = p2[["model", "n", "k", "oracle_width", "twoE_n"]].round(5).to_dict(orient="records")
    p3 = df[(df.k == 4 * df.n ** 2)]
    p3 = p3.assign(unsolved=[float(np.mean(C[m] == 0)) for m in p3.model])
    p3 = p3[p3.unsolved > 0.05]
    pre["P3_cells"] = len(p3)
    pre["P3_frac_BB_above_truth"] = float((p3.BB > p3.truth).mean())
    pre["P3_by_model_n"] = p3.groupby(["model", "n"]).apply(lambda g: float((g.BB > g.truth).mean())).reset_index(name="frac_above").to_dict(orient="records")
    pre["P3_mean_bias_BB"] = float((p3.BB - p3.truth).mean())
    pre["box_cover_all"] = float(df.box_cover.mean()); pre["box_cells"] = len(df)
    pre["models_unsolved_frac_at_1e4"] = {m: float(np.mean(C[m] == 0)) for m in models}
    pre["BIC_selection_counts"] = df.drop_duplicates(["model", "n", "subset"]).BIC_sel.value_counts().to_dict()
    json.dump(pre, open(os.path.join(RES, "llm_prereg.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps({k: v for k, v in pre.items() if k not in ("P2_table",)}, indent=1, default=float))
    # figure: n = 16 forecasts vs truth for three models
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = [models[0], models[3], models[-1]]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axs, show):
        kk = np.unique(np.round(np.logspace(0, 4, 40)).astype(int))
        ax.semilogx(kk, [unbiased(C[m], N, k) for k in kk], "k-", lw=2, label="truth (10,000 samples)")
        e = df[(df.model == m) & (df.n == 16)]
        g = e.groupby("k").agg(lo=("box_lo", "mean"), hi=("box_hi", "mean"), BB=("BB", "mean"), ZIBB=("ZIBB", "mean"),
                               BIC=("BIC", "mean"), PL=("powerlaw", "mean"), olo=("oracle_lo", "first"), ohi=("oracle_hi", "first")).reset_index()
        ax.fill_between(g.k, g.lo, g.hi, color="C0", alpha=0.2, label="95% box interval (mean ends)")
        ax.plot(g.k, g.olo, "C0:", g.k, g.ohi, "C0:")
        ax.plot(g.k, g.BB, "o-", color="C1", label="beta-binomial"); ax.plot(g.k, g.ZIBB, "s-", color="C2", label="zero-infl. BB")
        ax.plot(g.k, g.BIC, "^-", color="C4", label="BIC-selected mixture"); ax.plot(g.k, g.PL, "d-", color="C3", label="power law")
        ax.axvline(16, color="gray", lw=0.5); ax.axvline(256, color="gray", lw=0.5, ls="--")
        ax.set_title(f"{m}, MATH, n = 16"); ax.set_xlabel("k")
    axs[0].set_ylabel("pass@k"); axs[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_llm.png"), dpi=150)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
