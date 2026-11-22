"""Variance decomposition of headline estimates, sector-ranking robustness, and
first-order-stochastic-dominance (FOSD) test of Proposition 1."""
import os, pickle, itertools, json
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
from lib import R, MEASURES, WEIGHTS, eps_star

H = pd.read_csv(os.path.join(R, "multiverse_headline.csv"))
SC = pd.read_csv(os.path.join(R, "multiverse_sector.csv"), dtype={"sector": str})
out = {}


# ------------------------------------------------ 1. ANOVA (main-effect eta^2) decomposition
def eta2(df, y, factors):
    tot = ((df[y] - df[y].mean()) ** 2).sum()
    res = {}
    for f in factors:
        gm = df.groupby(f)[y].transform("mean")
        res[f] = ((gm - df[y].mean()) ** 2).sum() / tot
    res["interactions+residual"] = 1 - sum(res.values())
    return res


H["construct"] = H["measure"].str.split("_").str[1]       # alpha/beta/gamma/auto
H["rater"] = H["measure"].str.split("_").str[0]           # gpt/hum
dec = {}
for y in ["H_net", "H_gross", "H_half"]:
    dec[y] = eta2(H, y, ["measure", "weight", "rule"])
    # finer: rater vs construct (both are coarsenings of 'measure')
    dec[y + "_rater"] = eta2(H, y, ["rater"])["rater"]
    dec[y + "_construct"] = eta2(H, y, ["construct"])["construct"]
ces = H[H["family"].str.startswith("ces")].copy()
p = ces["rule"].str.extract(r"ces_(\w+)_s([\d.]+)_e([\d.]+)_t([\d.]+)")
ces["mode"], ces["sigma"], ces["eps"], ces["theta"] = p[0], p[1], p[2], p[3]
dec["ces_only_H_net"] = eta2(ces, "H_net", ["measure", "weight", "mode", "sigma", "eps", "theta"])
# within a single fixed exposure measure (the paper's preferred: human beta, importance weights)
sub = H[(H.measure == "hum_beta") & (H.weight == "im")]
dec["fixed_measure_range_H_net"] = [float(sub.H_net.min()), float(sub.H_net.max())]
out["anova"] = dec
# headline summaries
summ = H.groupby("family")["H_net"].agg(["min", "median", "max"])
out["headline_by_family"] = summ.round(4).to_dict()
out["headline_all"] = dict(min=float(H.H_net.min()), q05=float(H.H_net.quantile(.05)),
                           median=float(H.H_net.median()), q95=float(H.H_net.quantile(.95)),
                           max=float(H.H_net.max()), share_negative=float((H.H_net < 0).mean()))
out["preferred_measure_by_rule"] = sub.set_index("rule")[["H_net", "H_gross", "H_half"]].round(4).loc[
    ["linear", "threshold_0.3", "threshold_0.5", "threshold_0.7", "oring_indep", "oring_frechet_lo",
     "ces_automate_s0.5_e0.5_t0.9", "ces_automate_s0.5_e1.5_t0.9", "ces_augment_s0.5_e0.5_t0.9",
     "ces_augment_s0.5_e1.5_t0.9", "ces_labelled_s0.5_e1.0_t0.5", "ces_labelled_s2.0_e0.5_t0.9"]].to_dict("index")

# ------------------------------------------------ 2. sector ranking robustness
SC["key"] = SC["measure"] + "|" + SC["weight"] + "|" + SC["rule"]
piv = SC.pivot_table(index="key", columns="sector", values="D")
# drop degenerate specifications in which every sector has the same displacement (e.g. all-zero)
degenerate = piv.std(axis=1) < 1e-9
out["n_degenerate_sector_specs"] = int(degenerate.sum())
out["degenerate_specs"] = list(piv.index[degenerate])
piv_all = piv.copy()
piv = piv[~degenerate]
ranks = piv.rank(axis=1, ascending=False)            # 1 = most displaced
meta = SC.drop_duplicates("key").set_index("key")[["measure", "weight", "rule", "family"]].loc[piv.index]
# Kendall's W across all specs
m, n = ranks.shape
Rj = ranks.sum(axis=0)
W = 12 * ((Rj - Rj.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))
out["kendall_W_all_specs"] = float(W)
for fam in meta.family.unique():
    rk = ranks[meta.family == fam]; mm = len(rk); Rj = rk.sum(axis=0)
    out.setdefault("kendall_W_by_family", {})[fam] = float(12 * ((Rj - Rj.mean()) ** 2).sum() / (mm ** 2 * (n ** 3 - n)))
# tau vs linear rule for the same measure/weight
taus = []
for key in piv.index:
    mm_, ww_, rr_ = key.split("|")
    base = piv_all.loc[f"{mm_}|{ww_}|linear"]
    t = kendalltau(base.values, piv.loc[key].values).statistic
    taus.append(dict(measure=mm_, weight=ww_, rule=rr_, family=meta.loc[key, "family"], tau=t))
taus = pd.DataFrame(taus)
taus.to_csv(os.path.join(R, "sector_tau_vs_linear.csv"), index=False)
out["share_specs_tau_negative_by_family"] = taus.groupby("family")["tau"].apply(lambda x: float((x < 0).mean())).to_dict()
out["tau_vs_linear_by_family"] = taus.groupby("family")["tau"].agg(["min", "median", "mean", "max"]).round(3).to_dict("index")
rr = ranks.agg(["min", "median", "max"]).T
titles = pd.read_csv(os.path.join(R, "emp_sector.csv"), dtype={"NAICS": str}).drop_duplicates("NAICS").set_index("NAICS")["NAICS_TITLE"]
rr["title"] = rr.index.map(titles)
rr["top3_share"] = (ranks <= 3).mean()
rr["bottom3_share"] = (ranks >= n - 2).mean()
# rank under increasing-only rules (exclude specs where d decreases in s at the typical point)
rr.sort_values("median").to_csv(os.path.join(R, "sector_rank_ranges.csv"))
out["sector_rank_ranges"] = rr.sort_values("median").round(3).to_dict("index")

# ------------------------------------------------ 3. FOSD test (Proposition 1)
stats = pickle.load(open(os.path.join(R, "occ_stats.pkl"), "rb"))
S = pd.read_csv(os.path.join(R, "emp_sector.csv"), dtype={"soc": str, "NAICS": str})
S = S[S["emp"].notna()]
grid = np.linspace(0.005, 0.995, 199)
fosd_rows = []
for (mname, wname), st in stats.items():
    s = st["s"]
    Sx = S[S["soc"].isin(s.index)].assign(s=lambda x: x["soc"].map(s))
    cdf = {}
    for k, x in Sx.groupby("NAICS"):
        ww = x["emp"].values / x["emp"].sum()
        cdf[k] = np.array([ww[x["s"].values <= t].sum() for t in grid])   # F_k(t)
    secs = sorted(cdf)
    # share of exposure: mean
    mean = {k: np.average(x["s"], weights=x["emp"]) for k, x in Sx.groupby("NAICS")}
    n_pairs = n_fosd = n_inv_thr = n_inv_rules = 0
    # orderings under s-monotone increasing rules actually in the multiverse
    keys_inc = [k for k in piv.index if k.startswith(f"{mname}|{wname}|") and
                (k.endswith("linear") or "threshold" in k)]
    keys_ces = [k for k in piv.index if k.startswith(f"{mname}|{wname}|ces_automate") or
                k.startswith(f"{mname}|{wname}|ces_augment")]
    keys_other = [k for k in piv.index if k.startswith(f"{mname}|{wname}|") and
                  ("oring" in k or "labelled" in k)]
    n_rev_other = n_rev_ces = 0
    for a_, b_ in itertools.combinations(secs, 2):
        n_pairs += 1
        Fa, Fb = cdf[a_], cdf[b_]
        dom = np.all(Fa <= Fb + 1e-12) or np.all(Fb <= Fa + 1e-12)
        n_fosd += dom
        # dense-threshold invariance: sign of (share above t) difference constant over grid
        diff = (1 - Fa) - (1 - Fb)
        nz = diff[np.abs(diff) > 1e-9]
        inv_thr = len(nz) == 0 or np.all(nz > 0) or np.all(nz < 0)
        n_inv_thr += inv_thr
        sign_lin = np.sign(piv.loc[f"{mname}|{wname}|linear", a_] - piv.loc[f"{mname}|{wname}|linear", b_])
        sg = [np.sign(piv.loc[k, a_] - piv.loc[k, b_]) for k in keys_inc]
        n_inv_rules += all(x == sign_lin for x in sg)
        sc = [np.sign(piv.loc[k, a_] - piv.loc[k, b_]) for k in keys_ces]
        # CES rules may be decreasing in s; count pairs whose ordering is not +/- the linear one
        n_rev_ces += any(x not in (sign_lin, -sign_lin) for x in sc) or (any(x == -sign_lin for x in sc))
        so = [np.sign(piv.loc[k, a_] - piv.loc[k, b_]) for k in keys_other]
        n_rev_other += any(x == -sign_lin for x in so)
    fosd_rows.append(dict(measure=mname, weight=wname, pairs=n_pairs, fosd_share=n_fosd / n_pairs,
                          dense_threshold_invariant_share=n_inv_thr / n_pairs,
                          invariant_linear_threshold_rules_share=n_inv_rules / n_pairs,
                          reversed_by_oring_or_labelled_share=n_rev_other / n_pairs,
                          reversed_by_some_ces_share=n_rev_ces / n_pairs))
F = pd.DataFrame(fosd_rows)
F.to_csv(os.path.join(R, "fosd_pairs.csv"), index=False)
out["fosd_summary"] = F.drop(columns=["measure", "weight", "pairs"]).describe().loc[["mean", "min", "max"]].round(3).to_dict()
out["fosd_preferred"] = F[(F.measure == "hum_beta") & (F.weight == "im")].round(3).to_dict("records")

# ------------------------------------------------ 4. Proposition 2 numerics: employment-neutral elasticity
tab = []
for sig in [0.5, 1.0, 2.0]:
    for th in [0.5, 0.9, 0.99]:
        for s in [0.1, 0.3, 0.5]:
            tab.append(dict(sigma=sig, theta=th, s=s, eps_star=float(eps_star(s, sig, th))))
pd.DataFrame(tab).to_csv(os.path.join(R, "eps_star_table.csv"), index=False)
out["eps_star_examples"] = tab

# ------------------------------------------------ 5. preferred-measure sector table under selected rules
sel = ["linear", "threshold_0.5", "oring_indep", "ces_automate_s0.5_e0.5_t0.9", "ces_automate_s0.5_e1.5_t0.9",
       "ces_labelled_s0.5_e1.0_t0.9", "ces_augment_s0.5_e1.5_t0.9"]
T = pd.DataFrame({r: piv_all.loc[f"hum_beta|im|{r}"] for r in sel})
T.insert(0, "title", T.index.map(titles))
T = T.sort_values("linear", ascending=False)
for r in sel:
    T[r + "_rank"] = T[r].rank(ascending=False)
T.to_csv(os.path.join(R, "sector_table_preferred.csv"))
print(T.round(3).to_string())

with open(os.path.join(R, "analysis_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps(out["anova"], indent=1, default=float))
print(json.dumps(out["headline_all"], indent=1))
print(json.dumps(out["kendall_W_by_family"], indent=1), out["kendall_W_all_specs"])
print(json.dumps(out["tau_vs_linear_by_family"], indent=1)); print(out["share_specs_tau_negative_by_family"], out["n_degenerate_sector_specs"])
print(json.dumps(out["fosd_summary"], indent=1))
print(json.dumps(out["fosd_preferred"], indent=1))
print(rr.sort_values("median")[["title", "min", "median", "max", "top3_share", "bottom3_share"]].to_string())
