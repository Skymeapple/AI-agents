"""Revision r2: summaries and figures for the horizon law and the real-data analyses.
Inputs: results/horizon_law_bounds.csv, results/minimax_bounds.csv, results/block_law.csv, results/realdata_bon.csv,
        results/realdata_math.csv, results/theorem5_converse_horizon.csv
Outputs: results/horizon_law_summary.json, results/horizon_law_table.csv, results/realdata_summary.json,
         results/fig_horizon.png, results/fig_bon.png"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
EPS = 0.05
out = {}


def cross(s, v, delta, increasing=True):
    """log-interpolated crossing of v(s) = delta on the grid (assumes local monotonicity; reported with grid values)."""
    s, v = np.asarray(s, float), np.asarray(v, float)
    for i in range(len(s) - 1):
        if v[i] <= delta < v[i + 1]:
            return float(np.exp(np.log(s[i]) + (np.log(delta) - np.log(v[i])) * (np.log(s[i + 1]) - np.log(s[i]))
                                / (np.log(v[i + 1]) - np.log(v[i]))))
    return np.nan


from block import design  # noqa: E402


def block_halves(n, k, Ms, Lmax=14):
    des = [d for d in (design(int(n), int(k), L) for L in range(0, min(Lmax, int(n) - 1) + 1)) if d is not None]
    res = {}
    for M in Ms:
        hs = [d["beta"] + d["zrange"] * np.sqrt(np.log(2 / EPS) / (2 * M)) for d in des]
        i = int(np.argmin(hs)); res[M] = (hs[i], des[i]["L"])
    return res


# ----------------------------------------------------------------------------- update of the r1 bracket (330 cells)
mb = pd.read_csv(os.path.join(RES, "minimax_bounds.csv"))
blk2 = []
for (n, k), _ in mb.groupby(["n", "k"]):
    for M, (h, L) in block_halves(n, k, sorted(mb.M.unique())).items():
        blk2.append(dict(n=n, k=k, M=M, R_block=h, block_L=L))
mb = mb.merge(pd.DataFrame(blk2), on=["n", "k", "M"])
mb["R_up_best"] = np.minimum(mb.R_up, mb.R_block)
mb["R_lo_best"] = np.maximum(mb.R_lo, (1 - 2 * EPS) * mb.pop_halfwidth)
mb["ratio_best"] = mb.R_up_best / mb.R_lo_best
mb.to_csv(os.path.join(RES, "minimax_bounds_r2.csv"), index=False, float_format="%.6g")
out["r1_grid_ratio_best"] = [float(mb.ratio_best.min()), float(mb.ratio_best.median()), float(mb.ratio_best.max())]
out["r1_grid_block_beats_chernoff"] = int((mb.R_block < mb.R_up).sum())
out["r1_grid_strengthening_binds"] = int(((1 - 2 * EPS) * mb.pop_halfwidth > mb.R_lo).sum())
hz = []
for delta in [0.02, 0.05]:
    for (n, M), d in mb.groupby(["n", "M"]):
        d = d.sort_values("k")
        hz.append(dict(delta=delta, n=n, M=M, k_ach=cross(d.k, d.R_up_best, delta), k_imp=cross(d.k, d.R_lo_best, delta),
                       k_ach_grid=d[d.R_up_best <= delta].k.max(), k_imp_grid=d[d.R_lo_best > delta].k.min(),
                       k_pop=cross(d.k, np.maximum(d.pop_halfwidth, 1e-300), delta)))
pd.DataFrame(hz).to_csv(os.path.join(RES, "minimax_horizon_r2.csv"), index=False, float_format="%.4g")

# ----------------------------------------------------------------------------- horizon law
if os.path.exists(os.path.join(RES, "horizon_law_bounds.csv")):
    hl = pd.read_csv(os.path.join(RES, "horizon_law_bounds.csv"))
    bl = pd.read_csv(os.path.join(RES, "block_law.csv"))
    blk = []
    for (n, k), _ in hl.groupby(["n", "k"]):
        for M, (h, L) in block_halves(n, k, sorted(hl.M.unique())).items():
            blk.append(dict(n=n, k=k, M=M, R_block=h, block_L=L))
    hl = hl.merge(pd.DataFrame(blk), on=["n", "k", "M"])
    hl["R_up_best"] = np.minimum(hl.R_up, hl.R_block)
    hl["R_lo_best"] = np.maximum(hl.R_lo, (1 - 2 * EPS) * hl.pop_halfwidth)
    hl["ratio"] = hl.R_up_best / hl.R_lo_best
    hl.to_csv(os.path.join(RES, "horizon_law_bounds_merged.csv"), index=False, float_format="%.6g")
    out["n_cells"] = int(len(hl))
    out["ratio_up_lo"] = [float(hl.ratio.min()), float(hl.ratio.median()), float(hl.ratio.max())]
    out["block_beats_chernoff_frac"] = float(np.mean(hl.R_block < hl.R_up))
    out["block_over_chernoff"] = [float((hl.R_block / hl.R_up).min()), float((hl.R_block / hl.R_up).median()),
                                  float((hl.R_block / hl.R_up).max())]
    # scale collapse: spread across n (n >= 16) at fixed (s, M)
    col = []
    for (s, M), d in hl[hl.n >= 16].groupby(["s", "M"]):
        if len(d) >= 3:
            col.append(dict(s=s, M=M, lo_spread=d.R_lo_best.max() / d.R_lo_best.min(), up_spread=d.R_up_best.max() / d.R_up_best.min()))
    col = pd.DataFrame(col)
    out["collapse_n_ge_16_lo_spread_median_max"] = [float(col.lo_spread.median()), float(col.lo_spread.max())]
    out["collapse_n_ge_16_up_spread_median_max"] = [float(col.up_spread.median()), float(col.up_spread.max())]
    tab = []
    for delta in [0.02, 0.05, 0.1]:
        for (n, M), d in hl.groupby(["n", "M"]):
            d = d.sort_values("s")
            tab.append(dict(delta=delta, n=n, M=M,
                            s_ach_grid=d[d.R_up_best <= delta].s.max(), s_imp_grid=d[d.R_lo_best > delta].s.min(),
                            s_ach=cross(d.s, d.R_up_best, delta), s_imp=cross(d.s, d.R_lo_best, delta),
                            s_pop=cross(d.s, np.maximum(d.pop_halfwidth, 1e-300), delta)))
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(RES, "horizon_law_table.csv"), index=False, float_format="%.4g")
    # slope of impossible-beyond horizon in log10 M (n >= 32, M >= 1e3, delta = 0.02)
    fits = {}
    for delta in [0.02, 0.05]:
        for n in sorted(hl.n.unique()):
            d = tab[(tab.delta == delta) & (tab.n == n) & (tab.M >= 1e3)].dropna(subset=["s_imp"])
            if len(d) >= 3:
                b, a = np.polyfit(np.log10(d.M), d.s_imp, 1)
                fits[f"delta{delta}_n{n}"] = dict(slope_per_decade=float(b), intercept=float(a), n_pts=int(len(d)))
    out["s_imp_vs_log10M_fits"] = fits
    # block estimator (achievable) horizons up to M = 1e12
    bh = []
    for delta in [0.02, 0.05]:
        for (n, M), d in bl.groupby(["n", "M"]):
            ok = d[d.half <= delta]
            bh.append(dict(delta=delta, n=n, M=M, s_ach_block=ok.s.max() if len(ok) else np.nan))
    bh = pd.DataFrame(bh)
    bh.to_csv(os.path.join(RES, "block_horizon_table.csv"), index=False, float_format="%.4g")
    fitsb = {}
    for delta in [0.02, 0.05]:
        for n in sorted(bl.n.unique()):
            d = bh[(bh.delta == delta) & (bh.n == n) & (bh.M >= 1e4)].dropna()
            if len(d) >= 3:
                b, a = np.polyfit(np.log10(d.M), d.s_ach_block, 1)
                fitsb[f"delta{delta}_n{n}"] = dict(slope_per_decade=float(b), intercept=float(a))
    out["block_s_ach_vs_log10M_fits"] = fitsb

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {8: "#999999", 16: "#1b9e77", 32: "#d95f02", 64: "#7570b3", 128: "#e7298a"}
    for ax, M in zip(axs[:2], [1e4, 1e6]):
        for n, d in hl[hl.M == M].groupby("n"):
            d = d.sort_values("s")
            ax.plot(d.s, d.R_up_best, "-o", ms=3, color=cols.get(n, "k"), label=f"n={n} achieved")
            ax.plot(d.s, d.R_lo_best, "--", color=cols.get(n, "k"), lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(0.02, color="gray", lw=0.5)
        ax.set_xlabel("s = k / n"); ax.set_title(f"M = {M:.0e}: minimax 95% half-width vs k/n")
    axs[0].set_ylabel("solid: achieved (upper), dashed: lower bound"); axs[0].legend(fontsize=7)
    ax = axs[2]
    for n in sorted(hl.n.unique()):
        d = tab[(tab.delta == 0.02) & (tab.n == n)].sort_values("M")
        ax.plot(d.M, d.s_imp, "--o", ms=3, color=cols.get(n, "k"), label=f"n={n}: impossible beyond")
        ax.plot(d.M, d.s_ach, "-s", ms=3, color=cols.get(n, "k"))
    for n, mk in [(256, "^"), (1024, "v")]:
        d = bh[(bh.delta == 0.02) & (bh.n == n)].sort_values("M")
        ax.plot(d.M, d.s_ach_block, ":" + mk, color="k", ms=4, label=f"n={n}: block estimator achieves")
    ax.set_xscale("log"); ax.set_xlabel("number of tasks M"); ax.set_ylabel("horizon k*/n at half-width 0.02")
    ax.set_title("forecast horizon grows like log M"); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_horizon.png"), dpi=150)

# ----------------------------------------------------------------------------- real data
rd = {}
if os.path.exists(os.path.join(RES, "realdata_bon.csv")):
    b = pd.read_csv(os.path.join(RES, "realdata_bon.csv"))
    for c in ["box", "blk", "blkeb"]:
        b[c + "_cover"] = (b[c + "_lo"] - 1e-9 <= b.truth) & (b.truth <= b[c + "_hi"] + 1e-9)
    for mo in ["BB", "ZIBB", "BMix2", "BMix3", "BIC", "powerlaw"]:
        b["err_" + mo] = b[mo] - b.truth
    b.to_csv(os.path.join(RES, "realdata_bon_scored.csv"), index=False, float_format="%.5f")
    rd["bon_cells"] = int(len(b))
    rd["bon_cover"] = {c: int(b[c + "_cover"].sum()) for c in ["box", "blk", "blkeb"]}
    kk = b[b.k >= 1000]
    rd["bon_err_k_ge_1000"] = {mo: dict(mean_abs=float(kk["err_" + mo].abs().mean()), min=float(kk["err_" + mo].min()),
                                        max=float(kk["err_" + mo].max()), frac_neg=float((kk["err_" + mo] < 0).mean()))
                               for mo in ["BB", "ZIBB", "BMix2", "BIC", "powerlaw"]}
    bk = b[b.k == b.groupby("model").k.transform("max")]
    rd["bon_kmax_table"] = bk[["model", "n", "k", "truth", "frac_zero", "BB", "ZIBB", "BIC", "BIC_sel", "powerlaw",
                               "box_lo", "box_hi", "blk_lo", "blk_hi", "local_lb_half"]].round(3).to_dict("records")
    rd["bon_local_lb_by_s"] = b.groupby("s").local_lb_half.median().round(4).to_dict()
    # first-success design cost actually spent in the BoN runs, vs. the fixed-n alternative
    dd = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "bon_text_first_success.csv"))
    rd["bon_mean_attempts_per_prompt"] = dd.groupby("model").n_recorded.mean().round(0).to_dict()
    rd["bon_dkw_halfwidth_M159"] = float(np.sqrt(np.log(2 / EPS) / (2 * 159)))
    # figure: 4 models, n = 60
    ms = ["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro-001", "GraySwanAI-Llama-3-8B-Instruct-RR"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, m in zip(axs, ms):
        d = b[(b.model == m) & (b.n == 60) & (b.k != 960)].sort_values("k")
        ax.plot(d.k, d.truth, "k-o", ms=3, label="realised (first-success run)")
        ax.fill_between(d.k, d.box_lo, d.box_hi, color="tab:blue", alpha=0.2, label="box 95%")
        ax.plot(d.k, d.blk_lo, "b:", lw=1); ax.plot(d.k, d.blk_hi, "b:", lw=1, label="block 95%")
        for mo, c in [("BB", "tab:orange"), ("ZIBB", "tab:red"), ("powerlaw", "tab:green")]:
            ax.plot(d.k, d[mo], "--", color=c, label=mo)
        ax.set_xscale("log"); ax.axvline(60, color="gray", lw=0.5); ax.set_title(m.replace("-20240620", ""), fontsize=9)
        ax.set_xlabel("k (attempts)")
    axs[0].set_ylabel("attack success rate (pass@k)"); axs[0].legend(fontsize=6, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_bon.png"), dpi=150)
if os.path.exists(os.path.join(RES, "realdata_math.csv")):
    mth = pd.read_csv(os.path.join(RES, "realdata_math.csv"))
    for c in ["box", "blk", "blkeb"]:
        mth[c + "_cover"] = (mth[c + "_lo"] - 1e-9 <= mth.truth) & (mth.truth <= mth[c + "_hi"] + 1e-9)
    rd["math_cells"] = int(len(mth))
    rd["math_cover"] = {c: int(mth[c + "_cover"].sum()) for c in ["box", "blk", "blkeb"]}
    rd["math_local_lb_by_n_s"] = mth.groupby(["n", "s"]).local_lb_half.median().round(4).reset_index().to_dict("records")
    rd["math_blk_vs_box_half_median"] = float((mth.blk_half / mth.box_half).median())
    rd["math_box_over_local_lb_median"] = float((mth.box_half / mth.local_lb_half).median())
json.dump(dict(horizon=out, realdata=rd), open(os.path.join(RES, "r2_summary.json"), "w"), indent=1, default=str)
print(json.dumps(dict(horizon=out), indent=1, default=str)[:4000])
