"""E10 (revision r1): robustness of the E3 aggregation conclusions across sampling
distributions of success profiles and across detection schemes (common vs heterogeneous h_k).
For each (prior, h-scheme) cell: Spearman rho of each statistic with ln R_ck, best single-threshold
balanced accuracy for R_ck > 1, and pairwise discordance."""
import json, os
import numpy as np
from scipy.stats import spearmanr, norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import R_checkpoint, stage_odds

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(10)
K = 4; n = 20000


def logitnormal_corr(n, mu, sd, rho):
    C = np.full((K, K), rho) + (1 - rho) * np.eye(K)
    z = rng.multivariate_normal(np.full(K, mu), sd ** 2 * C, size=n)
    return 1 / (1 + np.exp(-z))


priors = {
    "Beta(0.6,1.2) [orig]": lambda: rng.beta(0.6, 1.2, (n, K)),
    "Uniform(0,1)": lambda: rng.uniform(0, 1, (n, K)),
    "Beta(2,2)": lambda: rng.beta(2, 2, (n, K)),
    "Beta(0.3,0.3)": lambda: rng.beta(0.3, 0.3, (n, K)),
    "Beta(1,4)": lambda: rng.beta(1, 4, (n, K)),
    "logit-N(-1.5,1.5), corr 0.6": lambda: logitnormal_corr(n, -1.5, 1.5, 0.6),
}
hschemes = {
    "common h=0.05 [orig]": lambda p: np.full_like(p, 0.05),
    "common h=0.01": lambda p: np.full_like(p, 0.01),
    "common h=0.2": lambda p: np.full_like(p, 0.2),
    "heterog. h_k ~ logU[0.003,0.3]": lambda p: np.exp(rng.uniform(np.log(0.003), np.log(0.3), p.shape)),
    "heterog., harder stages louder": lambda p: np.clip(0.3 * (1 - p) ** 3 *
                                                          np.exp(rng.normal(0, 0.5, p.shape)), 1e-3, 1),
}


def bal_acc(s, sup):
    if sup.all() or (~sup).all():
        return float("nan")
    order = np.unique(np.quantile(s, np.linspace(0, 1, 401)))
    return float(max(0.5 * (((s >= t) & sup).sum() / sup.sum() + ((s < t) & ~sup).sum() / (~sup).sum())
                     for t in order))


ii = rng.integers(0, n, 100000); jj = rng.integers(0, n, 100000)
table = []
for pn, pf in priors.items():
    p = pf().clip(1e-4, 0.999)
    for hn, hf in hschemes.items():
        h = hf(p)
        R = R_checkpoint(p, h)
        lR = np.log(R)
        sup = R > 1
        pass10 = 1 - (1 - p) ** 10
        stats = {"mean pass@1": p.mean(1), "mean pass@10": pass10.mean(1), "min pass@1": p.min(1),
                 "geo-mean pass@1": np.exp(np.log(p).mean(1)),
                 "min stage odds (needs h)": stage_odds(p, h).min(1)}
        row = dict(prior=pn, h_scheme=hn, frac_supercritical=float(sup.mean()))
        for sn, s in stats.items():
            row[sn] = dict(spearman=float(spearmanr(s, lR).correlation), bal_acc=bal_acc(s, sup),
                           discordance=float(np.mean(np.sign(s[ii] - s[jj]) != np.sign(R[ii] - R[jj]))))
        table.append(row)
        print(f"{pn:28s} {hn:32s} sup={sup.mean():.3f} " +
              " ".join(f"{sn[:12]}:{row[sn]['spearman']:.2f}/{row[sn]['bal_acc']:.2f}" for sn in stats))

# summary across cells
summary = {}
for sn in ["mean pass@1", "mean pass@10", "min pass@1", "geo-mean pass@1", "min stage odds (needs h)"]:
    for grp, cond in [("common h", lambda r: r["h_scheme"].startswith("common")),
                      ("heterogeneous h", lambda r: r["h_scheme"].startswith("heterog"))]:
        rs = [r for r in table if cond(r)]
        sp = [r[sn]["spearman"] for r in rs]
        ba = [r[sn]["bal_acc"] for r in rs if not np.isnan(r[sn]["bal_acc"])]
        dc = [r[sn]["discordance"] for r in rs]
        summary[f"{sn} | {grp}"] = dict(spearman_min=float(np.min(sp)), spearman_median=float(np.median(sp)),
                                       bal_acc_min=float(np.min(ba)) if ba else None,
                                       bal_acc_median=float(np.median(ba)) if ba else None,
                                       discordance_median=float(np.median(dc)), discordance_max=float(np.max(dc)))
for k, v in summary.items():
    print(k, {a: round(b, 3) for a, b in v.items() if b is not None})
json.dump(dict(cells=table, summary=summary), open(f"{OUT}/e10_aggregation_robust.json", "w"), indent=1)

# figure: spearman heat map for mean pass@10 and min odds
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, sn in zip(axs, ["mean pass@1", "mean pass@10", "min stage odds (needs h)"]):
    M = np.array([[r[sn]["spearman"] for r in table if r["prior"] == pn] for pn in priors])
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(hschemes))); ax.set_xticklabels([h[:18] for h in hschemes], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(priors))); ax.set_yticklabels(list(priors), fontsize=6)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[a, b] < 0.6 else "k")
    ax.set_title(f"Spearman with ln R: {sn}", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e10_aggregation_robust.png", dpi=150)
