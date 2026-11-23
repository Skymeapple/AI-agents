"""Round-2 analyses built on ext_bootstrap.py outputs.

 (1) Variance decomposition of 167-h arrival dates into between-definition and within-definition parts, for several
     definition MENUS and several sets of NON-DEFINITIONAL uncertainty (task resampling; + human-time error;
     + leave-one-system-out; + release-date jitter; + trend window; + trend estimator), with a bootstrap CI on the
     definitional share.
 (2) Paired-bootstrap tests that aggregators change growth RATES (slope differences vs the pooled / geometric
     definition, same replicate).
 (3) Dispersion trends (all families / unsaturated families / lower semideviation) under ONE bootstrap
     (the task-hierarchical one), replacing the mixed task-vs-model bootstrap comparison of round 1.
 (4) Family-specific growth rates and a permutation test of rate heterogeneity.
 (5) Task-support diagnostics: how often the horizon that determines a definition lies outside the task lengths
     of the family it comes from.
Usage: python3 decompose.py --version 1-1
Output: results/decomp_{version}.json (+ CSV tables)
"""
import argparse, gzip, json, pickle
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from core import load_runs, RESULTS
from analysis_metr import QS, AGGS, TARGETS, frontier_mask

L167 = TARGETS["1 work-month (167h)"]
CAP = 2045.0
LOGIT = {q: np.log(q / (1 - q)) for q in QS}
DEFS = [f"{n}|{q}" for q in QS for n, _ in AGGS]


def load(tag):
    with gzip.open(f"{RESULTS}/{tag}.pkl.gz", "rb") as f:
        return pickle.load(f)["reps"]


def family_l2(s, q):
    return (LOGIT[q] - s["a"] - s["delta"].astype(float)) / s["b"]


def metrics(rep):
    """systems x definitions matrix of log2 horizons, and release times."""
    rows, t = [], []
    for s in sorted(rep, key=lambda s: s["t"]):
        r = {}
        for q in QS:
            r[f"pooled|{q}"] = (LOGIT[q] - s["pa"]) / s["pb"]
            l2 = family_l2(s, q)
            for n, p in AGGS[1:]:
                if isinstance(p, str):
                    r[f"{n}|{q}"] = np.percentile(l2, int(p[1:]))
                elif p == 0:
                    r[f"{n}|{q}"] = l2.mean()
                elif np.isinf(p):
                    r[f"{n}|{q}"] = l2.min()
                else:
                    z = p * l2 * np.log(2); m = z.max()
                    r[f"{n}|{q}"] = (m + np.log(np.mean(np.exp(z - m)))) / (p * np.log(2))
        rows.append(r); t.append(s["t"])
    return pd.DataFrame(rows)[DEFS], np.array(t)


def fit_trend(t, v, tmin=0.0, form="ols"):
    mask = np.isfinite(v) & (t >= tmin)
    fm = np.zeros(len(t), bool)
    fm[mask] = frontier_mask(t[mask], v[mask])
    tt, vv = t[fm], v[fm]
    if len(tt) < 3:
        return np.nan, np.nan
    if form == "theilsen":
        s, a, *_ = theilslopes(vv, tt)
    else:
        a, s = np.linalg.lstsq(np.column_stack([np.ones(len(tt)), tt]), vv, rcond=None)[0]
    return a, s


def arrival(a, s):
    if not np.isfinite(s) or s <= 0:
        return CAP
    return min(2023 + (L167 - a) / s, CAP)


def perturbations(n_sys, rng, which):
    p = dict(drop=None, jitter=np.zeros(n_sys), tmin=0.0, form="ols")
    if "loso" in which:
        p["drop"] = int(rng.integers(0, n_sys))
    if "dates" in which:
        p["jitter"] = rng.uniform(-45, 45, n_sys) / 365.25
    if "window" in which:
        p["tmin"] = float(rng.choice([0.0, 1.0]))
    if "form" in which:
        p["form"] = str(rng.choice(["ols", "theilsen"]))
    return p


def arrivals_and_slopes(M, t, pert):
    keep = np.ones(len(t), bool)
    if pert["drop"] is not None:
        keep[pert["drop"]] = False
    tt = (t + pert["jitter"])[keep]
    A, S = {}, {}
    for d in DEFS:
        a, s = fit_trend(tt, M[d].values[keep], pert["tmin"], pert["form"])
        A[d], S[d] = arrival(a, s), s
    return A, S


def share_stats(A):
    """A: B x D array of arrival dates. Returns share point estimate and bootstrap CI (see paper §4)."""
    W = float(np.mean(A.var(0)))
    Bt = float(A.mean(0).var())
    c = W - float(A.mean(1).var())
    Vb = A.var(1) - c                       # per-replicate (bias-corrected) between-definition variance
    sh = np.clip(Vb, 0, None) / (np.clip(Vb, 0, None) + W)
    # robust analogue (arrival dates are ~1/slope and heavy-tailed): squared MAD-scale instead of variance
    med = np.median(A, 0)
    Wr = float(np.mean((1.4826 * np.median(np.abs(A - med), 0)) ** 2))
    Br = float(med.var())
    return dict(within=W, between=Bt, share=Bt / (Bt + W), share_robust=Br / (Br + Wr),
                share_lo=float(np.percentile(sh, 5)), share_hi=float(np.percentile(sh, 95)),
                n_defs=A.shape[1], n_capped=int((A >= CAP).sum()))


MENUS = {
    "all24": DEFS,
    "defensible18 (drop p=+1, min-family)": [d for d in DEFS if not d.startswith(("pmean+1", "minfam"))],
    "defensible_q<=0.8 (12)": [d for d in DEFS if not d.startswith(("pmean+1", "minfam")) and not d.endswith("0.95")],
    "recommended8 (pooled,geo,q25,q10 x q=.5,.8)": [f"{a}|{q}" for q in (0.5, 0.8) for a in ("pooled", "geo(p=0)", "q25fam", "q10fam")],
    "aggregator_only q=.5 (defensible 6)": [d for d in DEFS if d.endswith("|0.5") and not d.startswith(("pmean+1", "minfam"))],
    "threshold_only pooled q=.5,.8": ["pooled|0.5", "pooled|0.8"],
    "threshold_only pooled q=.5,.8,.95": ["pooled|0.5", "pooled|0.8", "pooled|0.95"],
}
CONFIGS = [("task", "task", ()), ("task+time", "time", ()), ("task+loso", "task", ("loso",)),
           ("task+dates", "task", ("dates",)), ("task+window", "task", ("window",)), ("task+form", "task", ("form",)),
           ("all (time+loso+dates+window+form)", "time", ("loso", "dates", "window", "form"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="1-1"); ap.add_argument("--lam", default="1")
    a = ap.parse_args()
    v = a.version
    reps = {"task": load(f"ext_{v}_task_lam{a.lam}"), "time": load(f"ext_{v}_time_lam{a.lam}")}
    point = load(f"ext_{v}_point_lam{a.lam}")[0]
    MT = {k: [metrics(r) for r in R] for k, R in reps.items()}
    Mp, tp = metrics(point)
    out = dict(B_task=len(reps["task"]), B_time=len(reps["time"]))

    # point estimates
    Ap, Sp = arrivals_and_slopes(Mp, tp, perturbations(len(tp), None, ()))
    out["point_arrival"] = Ap; out["point_slope"] = Sp
    # window / form point variants
    for nm, pert in [("window2024", dict(drop=None, jitter=np.zeros(len(tp)), tmin=1.0, form="ols")),
                     ("theilsen", dict(drop=None, jitter=np.zeros(len(tp)), tmin=0.0, form="theilsen"))]:
        A_, S_ = arrivals_and_slopes(Mp, tp, pert)
        out[f"point_arrival_{nm}"] = A_
    # leave-one-system-out point spread for the headline definition
    loso = []
    for i in range(len(tp)):
        A_, _ = arrivals_and_slopes(Mp, tp, dict(drop=i, jitter=np.zeros(len(tp)), tmin=0.0, form="ols"))
        loso.append(A_)
    out["loso_point_range"] = {d: [float(min(x[d] for x in loso)), float(max(x[d] for x in loso))] for d in DEFS}

    # (1) decomposition
    dec, arrs, slopes = {}, {}, {}
    for cname, src, which in CONFIGS:
        rng = np.random.default_rng(7)
        A, S = [], []
        for M, t in MT[src]:
            A_, S_ = arrivals_and_slopes(M, t, perturbations(len(t), rng, which))
            A.append(A_); S.append(S_)
        A, S = pd.DataFrame(A)[DEFS], pd.DataFrame(S)[DEFS]
        arrs[cname], slopes[cname] = A, S
        dec[cname] = {mn: share_stats(A[ds].values) for mn, ds in MENUS.items()}
        dec[cname]["_pooled50_90CI"] = [float(np.percentile(A["pooled|0.5"], 5)), float(np.percentile(A["pooled|0.5"], 95))]
    out["decomposition"] = dec
    pt = pd.Series(Ap)
    out["point_range_by_menu"] = {mn: [float(pt[ds].min()), float(pt[ds].max())] for mn, ds in MENUS.items()}

    # (2) paired tests of slope (doublings / year) differences
    tests = {}
    for cname in ["task", "all (time+loso+dates+window+form)"]:
        S = slopes[cname]; T = {}
        for q in QS:
            for ref in ["pooled", "geo(p=0)"]:
                for agg, _ in AGGS:
                    if agg == ref:
                        continue
                    dlt = S[f"{agg}|{q}"] - S[f"{ref}|{q}"]
                    dpt = Sp[f"{agg}|{q}"] - Sp[f"{ref}|{q}"]
                    T[f"{agg} - {ref} |{q}"] = dict(point=float(dpt), lo=float(np.nanpercentile(dlt, 5)),
                                                   hi=float(np.nanpercentile(dlt, 95)), frac_lt0=float(np.nanmean(dlt < 0)))
        tests[cname] = T
    out["paired_slope_tests"] = tests

    # (3) dispersion trends, task-hierarchical bootstrap for every statistic
    def disp(rep):
        rows = []
        for s in rep:
            d = s["delta"].astype(float); r = s["rate"].astype(float)
            inner = d[(r > 0) & (r < 1)]; med = np.median(d); low = d[d < med]
            rows.append((s["t"], d.std(), inner.std() if len(inner) > 1 else np.nan,
                         np.sqrt(np.mean((low - med) ** 2)), np.sum(r == 1), np.sum(r == 0)))
        X = np.array(rows)
        sl = []
        for j in range(1, X.shape[1]):
            m = np.isfinite(X[:, j])
            sl.append(np.polyfit(X[m, 0], X[m, j], 1)[0])
        return sl
    names = ["sd_all", "sd_unsaturated", "lower_semidev", "n_saturated", "n_floored"]
    p0 = disp(point); bs = np.array([disp(r) for r in reps["task"]])
    out["dispersion_trends_task_bootstrap"] = {n: dict(slope=float(p0[i]), lo=float(np.nanpercentile(bs[:, i], 5)),
                                                       hi=float(np.nanpercentile(bs[:, i], 95)),
                                                       frac_pos=float(np.nanmean(bs[:, i] > 0))) for i, n in enumerate(names)}

    # (4) family-specific growth rates
    def fam_rates(rep, perm=None, rng=None):
        recs = []
        for s in rep:
            o = -s["delta"].astype(float) / s["b"]       # log2-horizon offset of family vs system baseline
            for k, val in zip(s["key"], o):
                recs.append((s["alias"], s["t"], k, val))
        D = pd.DataFrame(recs, columns=["m", "t", "k", "o"])
        if perm is not None:
            ms = D.drop_duplicates("m")[["m", "t"]]
            mp = dict(zip(ms.m, rng.permutation(ms.t.values)))
            D["t"] = D.m.map(mp)
        g = {}
        for k, h in D.groupby("k"):
            if h.t.nunique() >= 5:
                g[k] = np.polyfit(h.t, h.o, 1)[0]
        g = pd.Series(g)
        t5 = np.sort(D.t.unique())[4]
        early = D[D.t <= t5].groupby("k").o.mean().reindex(g.index)
        # slopes from LATER systems only, so that early level and later rate share no data (no regression-to-mean)
        gl = {}
        for k, h in D[D.t > t5].groupby("k"):
            if h.t.nunique() >= 5:
                gl[k] = np.polyfit(h.t, h.o, 1)[0]
        g.attrs["late"] = pd.Series(gl)
        return g, early
    g0, early0 = fam_rates(point)
    rng = np.random.default_rng(3)
    null = np.array([fam_rates(point, True, rng)[0].std() for _ in range(1000)])
    bsd = np.array([fam_rates(r)[0].std() for r in reps["task"][:300]])
    common = float(np.polyfit(tp, Mp["pooled|0.5"].values, 1)[0])
    out["family_rates"] = dict(
        n_families=int(len(g0)), sd_family_rate=float(g0.std()), perm_null_sd_mean=float(null.mean()),
        perm_null_sd_95=float(np.percentile(null, 95)), perm_p=float(np.mean(null >= g0.std())),
        task_boot_sd_90=[float(np.percentile(bsd, 5)), float(np.percentile(bsd, 95))],
        corr_early_level_vs_rate=float(np.corrcoef(early0.dropna(), g0[early0.notna()])[0, 1]),
        corr_early_level_vs_later_rate=float(pd.concat([early0, g0.attrs["late"]], axis=1).dropna().corr().iloc[0, 1]),
        common_rate_all_systems_log2_per_yr=common,
        quantiles_relative_rate=[float(x) for x in np.percentile(g0, [10, 25, 50, 75, 90])])
    g0.to_csv(f"{RESULTS}/family_rates_{v}.csv", header=["rel_rate_log2_per_yr"])
    # Prop. 2 plug-in: instantaneous growth rate of A_p implied by current family levels and family rates,
    # rate_p - rate_0 = sum_f pi_f(p) g_f - mean_f g_f, pi_f(p) ~ 2^{p * l2_f}; averaged over frontier systems (q=0.5).
    pred = {}
    fr50 = frontier_mask(tp, Mp["pooled|0.5"].values)
    for p in (-1.0, -2.0, -4.0):
        vals = []
        for i, s in enumerate(sorted(point, key=lambda s: s["t"])):
            if not fr50[i]:
                continue
            l2 = family_l2(s, 0.5); gg = g0.reindex(s["key"]).values; ok = np.isfinite(gg)
            z = p * l2[ok] * np.log(2); w = np.exp(z - z.max()); w /= w.sum()
            vals.append(float(np.sum(w * gg[ok]) - gg[ok].mean()))
        pred[str(p)] = float(np.mean(vals))
    out["family_rates"]["prop2_plugin_rate_diff_vs_geo_log2_per_yr"] = pred
    out["family_rates"]["observed_slope_diff_pmean-1_vs_geo_q0.5"] = float(Sp["pmean-1|0.5"] - Sp["geo(p=0)|0.5"])

    # (5) support diagnostics (point fit)
    df = load_runs(v); df = df[df.release >= pd.Timestamp("2023-01-01")]
    rngf = df.groupby("task_family").x.agg(["min", "max"])
    smin, smax = df.x.min(), df.x.max()
    sup = {}
    sys_sorted = sorted(point, key=lambda s: s["t"])
    for q in QS:
        for n, p in AGGS:
            d = f"{n}|{q}"
            fr = frontier_mask(tp, Mp[d].values)
            below, above, cens = [], [], []
            TOL = 2.0   # doublings of tolerance around a family's task lengths (families have median 2 tasks)
            for i, s in enumerate(sys_sorted):
                if n == "pooled":
                    L = Mp[d].values[i]
                    b_, a_, c_ = L < smin - TOL, L > smax + TOL, 0.0
                else:
                    l2 = family_l2(s, q); fmn = rngf.loc[s["fam"], "min"].values; fmx = rngf.loc[s["fam"], "max"].values
                    r = s["rate"].astype(float)
                    if isinstance(p, str) or np.isinf(p):
                        tgt = Mp[d].values[i]
                        j = int(np.argmin(np.abs(l2 - tgt)))   # family whose horizon determines the aggregate
                        b_, a_ = l2[j] < fmn[j] - TOL, l2[j] > fmx[j] + TOL
                        c_ = float(r[j] in (0.0, 1.0))
                    else:
                        b_, a_ = np.mean(l2 < fmn - TOL), np.mean(l2 > fmx + TOL)
                        c_ = float(np.mean((r == 0) | (r == 1)))
                below.append(float(b_)); above.append(float(a_)); cens.append(c_)
            below, above, cens = np.array(below), np.array(above), np.array(cens)
            sup[d] = dict(frontier_frac_below=float(below[fr].mean()), frontier_frac_above=float(above[fr].mean()),
                          frontier_frac_censored_family=float(cens[fr].mean()),
                          latest_below=float(below[-1]), latest_above=float(above[-1]), latest_censored=float(cens[-1]))
    out["support"] = sup
    insup = [d for d in DEFS if sup[d]["frontier_frac_below"] + sup[d]["frontier_frac_above"] <= 0.5]
    out["support_menu"] = insup
    for cname in arrs:
        out["decomposition"][cname]["support_menu (<=50% of frontier points >2 doublings outside family task range)"] = \
            share_stats(arrs[cname][insup].values)
    out["point_range_by_menu"]["support_menu"] = [float(pt[insup].min()), float(pt[insup].max())]

    json.dump(out, open(f"{RESULTS}/decomp_{v}.json", "w"), indent=1, default=float)
    # compact table of the decomposition
    rows = []
    for cname, D in out["decomposition"].items():
        for mn, s in D.items():
            if mn.startswith("_"):
                continue
            rows.append(dict(config=cname, menu=mn, **s))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/decomp_{v}_table.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
