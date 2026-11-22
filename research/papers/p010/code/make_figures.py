"""Figures + variance decomposition (definition vs. statistical uncertainty) from analysis_metr.py outputs.
Usage: python3 make_figures.py   (after running analysis_metr.py for versions 1-1 and 1-0 with lam=1)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_metr import trend, TARGETS

R = "/home/user/AI-agents/research/papers/p010/results"
L167 = TARGETS["1 work-month (167h)"]
out = {}


def load(tag):
    point = pd.read_csv(f"{R}/{tag}_point.csv")
    z = np.load(f"{R}/{tag}_boot.npz", allow_pickle=True)
    cols = list(z["cols"])
    boots = [pd.DataFrame(a, columns=cols[1:]) for a in z["arr"]]
    return point, boots


def arrivals(P, metrics, L=L167):
    out = {}
    for c in metrics:
        a, s = trend(P.t.values, P[c].values, True, 0.0)
        out[c] = 2023 + (L - a) / s if s > 0 else np.inf
    return out


for v in ["1-1", "1-0"]:
    tag = f"metr_{v}_lam1"
    try:
        point, boots = load(tag)
    except FileNotFoundError:
        continue
    metrics = [c for c in point.columns if "|" in c]
    A = pd.DataFrame([arrivals(B, metrics) for B in boots])
    n_inf = int(np.isinf(A.values).sum())
    Af = A.replace(np.inf, np.nan)
    within = float(np.nanmean(Af.var(0)))
    between = float(np.nanvar(Af.mean(0)))
    pt = arrivals(point, metrics)
    out[v] = dict(
        n_boot=len(boots), n_infinite=n_inf,
        within_def_var=within, between_def_var=between,
        share_between=between / (between + within),
        pooled50_bootstrap_90=[float(np.nanpercentile(Af["pooled|0.5"], 5)), float(np.nanpercentile(Af["pooled|0.5"], 95))],
        range_point_estimates=[float(min(pt.values())), float(max(pt.values()))],
        argmin=min(pt, key=pt.get), argmax=max(pt, key=pt.get),
        # restricted menu: the two most commonly reported definitions (pooled p50 vs pooled p80)
        p50_vs_p80_gap_years=float(pt["pooled|0.8"] - pt["pooled|0.5"]),
    )
    # share-between within q=0.5 only (aggregator choice alone) and pooled only (threshold choice alone)
    for name, sel in [("aggregator_only_q0.5", [c for c in metrics if c.endswith("|0.5")]),
                      ("threshold_only_pooled", [c for c in metrics if c.startswith("pooled|")])]:
        w = float(np.nanmean(Af[sel].var(0))); b = float(np.nanvar(Af[sel].mean(0)))
        out[v][name] = dict(within=w, between=b, share_between=b / (b + w))

    # ---- Figure: trajectories
    if v == "1-1":
        sel = [("pooled|0.5", "pooled, q=0.5 (METR-style)"), ("pmean+1|0.5", "arith. mean over families, q=0.5"),
               ("q10fam|0.5", "10th-pct family, q=0.5"), ("pooled|0.8", "pooled, q=0.8"), ("minfam|0.95", "worst family, q=0.95")]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        tt = np.linspace(0, 10, 200)
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
        for (c, lab), col in zip(sel, cols):
            ax.scatter(2023 + point.t, point[c], s=14, color=col, alpha=0.8)
            a, s = trend(point.t.values, point[c].values, True, 0.0)
            ax.plot(2023 + tt, a + s * tt, color=col, lw=1.4, label=f"{lab}: DT={12/s:.1f} mo")
        ax.axhline(L167, c="k", ls="--", lw=0.8); ax.text(2023.1, L167 + 0.3, "1 work-month (167 h)", fontsize=8)
        ax.axhline(np.log2(60), c="grey", ls=":", lw=0.8); ax.text(2023.1, np.log2(60) + 0.3, "1 hour", fontsize=8, color="grey")
        ax.set_xlim(2023, 2033); ax.set_ylim(-6, 16)
        ax.set_ylabel("log2 horizon (minutes)"); ax.set_xlabel("release date")
        ax.set_title("METR v1.1 runs: same data, five definitions of 'horizon'")
        ax.legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(f"{R}/fig_trajectories.png", dpi=160); plt.close()

        # ---- Figure: arrival forest plot
        summ = pd.read_csv(f"{R}/{tag}_summary.csv")
        k = "1 work-month (167h)"
        summ = summ.sort_values(k)
        fig, ax = plt.subplots(figsize=(6.4, 6))
        y = np.arange(len(summ))
        colq = {0.5: "#1f77b4", 0.8: "#ff7f0e", 0.95: "#2ca02c"}
        for yi, (_, r) in zip(y, summ.iterrows()):
            ax.plot([r[f"{k}_lo"], r[f"{k}_hi"]], [yi, yi], c=colq[r.q], lw=1.5)
            ax.plot(r[k], yi, "o", c=colq[r.q], ms=4)
        ax.set_yticks(y); ax.set_yticklabels(summ.metric, fontsize=7)
        ax.set_xlabel("extrapolated date of 167-h horizon (point, 90% bootstrap interval)")
        ax.set_title("Definition choice vs. statistical uncertainty (METR v1.1)")
        plt.tight_layout(); plt.savefig(f"{R}/fig_arrival_forest.png", dpi=160); plt.close()

        # ---- Figure: dispersion over time
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for j, (c, lab) in enumerate([("disp_sd_delta", "SD of family offsets (logit units)"),
                                      ("b_pooled", "logistic slope b (per doubling of task length)")]):
            bs = np.array([B[c].values for B in boots])
            lo, hi = np.percentile(bs, [5, 95], axis=0)
            ax[j].fill_between(2023 + point.t, lo, hi, alpha=0.25)
            ax[j].plot(2023 + point.t, point[c], "o", ms=3)
            A_ = np.column_stack([np.ones(len(point)), point.t]); co = np.linalg.lstsq(A_, point[c], rcond=None)[0]
            ax[j].plot(2023 + point.t, A_ @ co, "k-", lw=1)
            ax[j].set_title(lab, fontsize=9); ax[j].set_xlabel("release date")
        plt.tight_layout(); plt.savefig(f"{R}/fig_dispersion_slope.png", dpi=160); plt.close()

json.dump(out, open(f"{R}/variance_decomposition.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
