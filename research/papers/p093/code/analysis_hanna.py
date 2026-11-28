"""Real-data analysis 1: HANNA story-generation human evaluation
(Chhun et al., COLING 2022). 1,056 stories x 3 MTurk raters x 6 criteria,
11 story sources (10 systems + human-written).  Run: python3 analysis_hanna.py"""
import itertools, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *

OUT = "../results/"
d = pd.read_csv("../data/hanna_stories_annotations.csv")
CRIT = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]
d["Mean6"] = d[CRIT].mean(1)
CRIT_ALL = CRIT + ["Mean6"]
models = sorted(d.Model.unique())
mcode = {m: i for i, m in enumerate(models)}
K = len(models)
pairs = list(itertools.combinations(range(K), 2))
MINC = 3   # min annotations per rater x system cell to enter contrasts
rng = np.random.default_rng(0)

summary, pairrows, rankrows = [], [], []
for c in CRIT_ALL:
    df = pd.DataFrame(dict(item=d["Story ID"], sys=d.Model.map(mcode), rater=d["Worker ID"], y=d[c]))
    alpha = krippendorff_alpha(df, "interval")
    obs, null, p = persistence_test(df, pairs, n_perm=999, seed=1, min_per_cell=MINC)
    # conventional leaderboard: volume-weighted mean score; story-bootstrap CI
    item_means = df.groupby("item").agg(y=("y", "mean"), sys=("sys", "first"))
    lb = item_means.groupby("sys").y.mean()
    B = 2000
    boots = np.empty((B, K))
    groups = [item_means[item_means.sys == k].y.values for k in range(K)]
    for b in range(B):
        boots[b] = [g[rng.integers(0, len(g), len(g))].mean() for g in groups]
    # equal-rater-weighted leaderboard over eligible raters (>= MINC per cell in all systems)
    cell = df.groupby(["rater", "sys"]).y.agg(["mean", "count"]).unstack()
    elig = cell["count"].fillna(0).min(1) >= MINC
    rmeans = cell["mean"][elig]
    lb_eq = rmeans.mean(0)
    ktau = stats.kendalltau(lb.values, lb_eq.values).statistic
    # per-rater rankings agreement
    rr = rmeans.values
    kt_r = [stats.kendalltau(rr[a], rr[b]).statistic for a, b in itertools.combinations(range(len(rr)), 2)]
    n_sig = n_frag = n_flip_eq = 0
    taus2, tvs, Gs = [], [], []
    for k, l in pairs:
        diff = lb[k] - lb[l]
        bd = boots[:, k] - boots[:, l]
        lo, hi = np.quantile(bd, [0.025, 0.975])
        sig = (lo > 0) or (hi < 0)
        D, v, n, raters = rater_contrasts(df, k, l, MINC)
        mu, t2, Q, dfree = dersimonian_laird(D, v)
        Dt = eb_shrink(D, v, mu, t2)
        s = np.sign(diff)
        G_gauss = gamma_star_gaussian(mu / np.sqrt(t2)) if t2 > 0 else np.inf
        if np.sign(mu) != s:
            G_gauss = 1.0  # rater-equal mean already disagrees with leaderboard
        G_disc = gamma_star_discrete(s * Dt)
        tv = tv_radius(s * Dt)
        if np.sign(Dt.mean()) != s:
            tv = 0.0; G_disc = 1.0
        n_sig += sig
        frag = sig and (G_gauss < 2)
        n_frag += frag
        taus2.append(t2); tvs.append(tv); Gs.append(G_gauss)
        pairrows.append(dict(criterion=c, a=models[k], b=models[l], lb_diff=diff, ci_lo=lo, ci_hi=hi, sig=sig,
                             R=len(D), mu_RE=mu, tau=np.sqrt(t2), Q=Q, df=dfree,
                             kappa=(abs(mu) / np.sqrt(t2) if t2 > 0 else np.inf),
                             G_gauss=G_gauss, G_disc=G_disc, tv_radius=tv,
                             raters_opposed_raw=int(np.sum(s * D < 0)),
                             raters_opposed_eb=int(np.sum(s * Dt < 0))))
    G = np.array(Gs); tvs = np.array(tvs)
    summary.append(dict(criterion=c, kripp_alpha=alpha, persist_Qex=obs, null_mean=null.mean(),
                        null_q95=np.quantile(null, .95), perm_p=p, n_eligible_raters=int(elig.sum()),
                        kendall_volume_vs_equal=ktau, mean_kendall_between_raters=np.mean(kt_r),
                        min_kendall_between_raters=np.min(kt_r),
                        pairs=len(pairs), pairs_sig=n_sig, pairs_sig_and_G_lt2=n_frag,
                        median_tau=np.median(np.sqrt(taus2)),
                        pairs_tau_pos=int(np.sum(np.array(taus2) > 0)),
                        median_G=np.median(G), frac_G_lt2=np.mean(G < 2), frac_tv_lt_0p2=np.mean(tvs < 0.2)))
    rankrows.append(pd.DataFrame(dict(criterion=c, model=models, lb_volume=lb.values, lb_equal_rater=lb_eq.values,
                                      rank_volume=stats.rankdata(-lb.values), rank_equal=stats.rankdata(-lb_eq.values))))
    print(c, "alpha=%.3f p=%.4f sig=%d frag=%d" % (alpha, p, n_sig, n_frag), flush=True)

S = pd.DataFrame(summary); P = pd.DataFrame(pairrows); RK = pd.concat(rankrows)
S.to_csv(OUT + "hanna_summary.csv", index=False)
P.to_csv(OUT + "hanna_pairs.csv", index=False)
RK.to_csv(OUT + "hanna_rankings.csv", index=False)
print(S.to_string())

# per-rater leaderboards for Mean6 and Engagement (figure)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, c in zip(ax, ["Engagement", "Mean6"]):
    df = pd.DataFrame(dict(item=d["Story ID"], sys=d.Model.map(mcode), rater=d["Worker ID"], y=d[c]))
    cell = df.groupby(["rater", "sys"]).y.agg(["mean", "count"]).unstack()
    elig = cell["count"].fillna(0).min(1) >= MINC
    rm = cell["mean"][elig]
    order = np.argsort(-rm.mean(0).values)
    for i, (r, row) in enumerate(rm.iterrows()):
        a.plot(range(K), row.values[order] - row.values.mean(), "o-", alpha=.8, label=f"rater {i+1} (n={int(cell['count'].loc[r].sum())})")
    a.set_xticks(range(K)); a.set_xticklabels([models[o] for o in order], rotation=60, ha="right", fontsize=8)
    a.set_ylabel("rater-centred mean score"); a.set_title(f"HANNA {c}: per-rater system profiles")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_hanna_profiles.png", dpi=160)

# distribution of Gamma* vs significance
fig, a = plt.subplots(figsize=(5.5, 4))
Pm = P[P.criterion == "Mean6"]
gg = np.minimum(Pm.G_gauss.values, 100)
a.scatter(np.abs(Pm.lb_diff) / ((Pm.ci_hi - Pm.ci_lo) / 3.92), gg, c=np.where(Pm.sig, "#2c6fbb", "#bbbbbb"), s=18)
a.set_yscale("log"); a.axhline(2, ls=":", c="gray"); a.axvline(1.96, ls=":", c="gray")
a.set_xlabel("|leaderboard difference| / SE  (sampling evidence)")
a.set_ylabel(r"$\Gamma^*$ (population robustness, capped 100)")
a.set_title("HANNA Mean6: 55 system pairs")
fig.tight_layout(); fig.savefig(OUT + "fig_hanna_gamma_vs_z.png", dpi=160)
