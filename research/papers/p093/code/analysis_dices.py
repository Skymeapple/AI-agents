"""Real-data analysis 2: DICES-350 (Aroyo et al., NeurIPS D&B 2023).
350 adversarial conversations x 123 demographically diverse raters (fully
crossed).  'Systems' are conditions an evaluation report would rank:
(a) the four expert degree-of-harm strata, (b) the six most frequent harm types.
Score: Q_overall unsafe = 1, unsure = 0.5, safe = 0.  Run: python3 analysis_dices.py"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *

OUT = "../results/"
x = pd.read_csv("../data/dices/dices350.csv")
x["y"] = x.Q_overall.map({"Yes": 1.0, "Unsure": 0.5, "No": 0.0})
# fully crossed design: remove item effects (each item mean averages 123 raters,
# so a rater's own score contributes <1%); keeps condition (system) means.
x["y_raw"] = x["y"]
x["y"] = x.y_raw - x.groupby("item_id").y_raw.transform("mean") + x.y_raw.mean()
x["type"] = x.rater_race.str.split(",").str[0].str.split("/").str[0] + "|" + x.rater_gender + "|" + x.rater_age

alpha_nom = krippendorff_alpha(pd.DataFrame(dict(item=x.item_id, y=x.Q_overall)), "nominal")
alpha_int = krippendorff_alpha(pd.DataFrame(dict(item=x.item_id, y=x.y_raw)), "interval")

# rater main-effect (leniency) share, for context
rm = x.groupby("rater_id").y_raw.mean()
im = x.groupby("item_id").y_raw.mean()
res = x.y_raw - x.item_id.map(im) - x.rater_id.map(rm) + x.y_raw.mean()
var_tot = x.y_raw.var()
decomp = dict(total=var_tot, item=im.var(), rater=rm.var(), residual=res.var())

top_types = x.drop_duplicates("item_id").harm_type.value_counts().index[:6].tolist()
designs = {"degree_of_harm": ("degree_of_harm", ["Benign", "Debatable", "Moderate", "Extreme"]),
           "harm_type_top6": ("harm_type", top_types)}
rows, summ = [], []
rng = np.random.default_rng(0)
for name, (col, levels) in designs.items():
    sub = x[x[col].isin(levels)].copy()
    code = {l: i for i, l in enumerate(levels)}
    df = pd.DataFrame(dict(item=sub.item_id, sys=sub[col].map(code), rater=sub.rater_id, y=sub.y))
    cm = sub.groupby(col).y_raw.mean()  # restore condition means removed by centring
    df["y"] = df.y - sub[col].map(sub.groupby(col).y.mean()).values + sub[col].map(cm).values
    pairs = list(itertools.combinations(range(len(levels)), 2))
    obs, null, p = persistence_test(df, pairs, n_perm=999, seed=2)
    types = sub.drop_duplicates("rater_id").set_index("rater_id")["type"]
    for k, l in pairs:
        D, v, n, raters = rater_contrasts(df, k, l)
        mu, t2, Q, dfree = dersimonian_laird(D, v)
        Dt = eb_shrink(D, v, mu, t2)
        s = np.sign(mu)
        # demographic types as mixture components (weights = sample shares)
        tdf = pd.DataFrame(dict(t=types.loc[raters].values, D=Dt))
        tg = tdf.groupby("t").D.agg(["mean", "size"])
        tv_types = tv_radius(s * tg["mean"].values, tg["size"].values)
        G_types = gamma_star_discrete(s * tg["mean"].values, tg["size"].values)
        # share of persistent heterogeneity explained by demographic type
        between = np.sum(tg["size"] * (tg["mean"] - Dt.mean()) ** 2) / tg["size"].sum()
        tl = tdf.t.values; codes_t = pd.factorize(tl)[0]; nt = np.bincount(codes_t)
        def _btw(c):
            mt = np.bincount(c, weights=Dt) / nt
            return np.sum(nt * (mt - Dt.mean()) ** 2) / nt.sum()
        bnull = np.array([_btw(rng.permutation(codes_t)) for _ in range(999)])
        p_demo = (1 + np.sum(bnull >= between)) / 1000
        # one-way ANOVA on raw contrasts by type (is demography predictive at all?)
        F, pF = stats.f_oneway(*[g.D.values for _, g in pd.DataFrame(dict(t=types.loc[raters].values, D=D)).groupby("t")])
        rows.append(dict(design=name, a=levels[k], b=levels[l], R=len(D), mu_RE=mu, tau=np.sqrt(t2), Q=Q, df=dfree,
                         kappa=abs(mu) / np.sqrt(t2) if t2 > 0 else np.inf,
                         G_gauss=gamma_star_gaussian(mu / np.sqrt(t2)) if t2 > 0 else np.inf,
                         G_raters=gamma_star_discrete(s * Dt), tv_raters=tv_radius(s * Dt),
                         raters_opposed_eb=int(np.sum(s * Dt < 0)), raters_opposed_raw=int(np.sum(s * D < 0)),
                         tv_demotypes=tv_types, G_demotypes=G_types, n_types=len(tg),
                         share_tau2_demographic=between / t2 if t2 > 0 else np.nan,
                         share_tau2_demographic_null=bnull.mean() / t2 if t2 > 0 else np.nan,
                         p_demographic_perm=p_demo, anova_p=pF))
    summ.append(dict(design=name, levels="; ".join(levels), persist_Qex=obs, null_mean=null.mean(),
                     null_q95=np.quantile(null, .95), perm_p=p))
R = pd.DataFrame(rows); S = pd.DataFrame(summ)
S["alpha_nominal"] = alpha_nom; S["alpha_interval"] = alpha_int
R.to_csv(OUT + "dices_pairs.csv", index=False); S.to_csv(OUT + "dices_summary.csv", index=False)
pd.Series(decomp).to_csv(OUT + "dices_variance_decomp.csv")
pd.set_option("display.width", 250)
print("alpha nominal %.3f interval %.3f" % (alpha_nom, alpha_int))
print(decomp)
print(S.to_string()); print(R.to_string())

# figure: distribution of EB rater contrasts for each harm-type pair vs 0
fig, ax = plt.subplots(figsize=(8, 4))
sub = x[x.harm_type.isin(top_types)]
code = {l: i for i, l in enumerate(top_types)}
df = pd.DataFrame(dict(item=sub.item_id, sys=sub.harm_type.map(code), rater=sub.rater_id, y=sub.y))
df["y"] = df.y - sub.harm_type.map(sub.groupby("harm_type").y.mean()).values + sub.harm_type.map(sub.groupby("harm_type").y_raw.mean()).values
labs = []
for j, (k, l) in enumerate(itertools.combinations(range(len(top_types)), 2)):
    D, v, n, raters = rater_contrasts(df, k, l)
    mu, t2, Q, dfree = dersimonian_laird(D, v)
    Dt = eb_shrink(D, v, mu, t2)
    ax.scatter(np.full(len(Dt), j) + rng.normal(0, .08, len(Dt)), Dt, s=4, alpha=.5, c="#2c6fbb")
    ax.plot([j - .3, j + .3], [mu, mu], c="k")
    labs.append(f"{top_types[k][:8]}-{top_types[l][:8]}")
ax.axhline(0, c="#d1495b", ls=":")
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=70, ha="right", fontsize=7)
ax.set_ylabel("EB-shrunk rater contrast\n(unsafe-rate difference)")
ax.set_title("DICES-350: persistent rater-level contrasts between harm types")
fig.tight_layout(); fig.savefig(OUT + "fig_dices_contrasts.png", dpi=160)
