"""p075: Substitution in the open-weight frontier.

Measures (i) the open-closed lag, (ii) counterfactual open frontiers when a developer,
a country's developers, or a release is removed, (iii) the superadditivity
("coordination gap") of restraint, (iv) Shapley attribution of open-frontier access,
(v) the effect of a staleness-based release rule applied by one jurisdiction or all.

Run:  python3 code/analysis.py      (writes results/*.json, results/*.csv)
"""
import json
import sys
import numpy as np
import pandas as pd
from common import (load, Grid, frontier, first_reach, hgap, setback, window_mask,
                    RES, SNAP_MAIN, SNAP_OLD, SEED)

DAYS_PER_MONTH = 30.4375
N_MC = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WIN = ("2025-01-01", None)          # main analysis window (end = data end)
WINDOWS = {"2024+": ("2024-01-01", None), "2025+": ("2025-01-01", None),
           "2026+": ("2026-01-01", None)}
THRESH = [145.0, 150.0, 155.0, 158.0]   # ECI levels for availability-delay metrics
DELTAS = [0, 3, 6, 9, 12]               # release-rule staleness requirements (months)


def mo(x):
    return x / DAYS_PER_MONTH


def summarize(arr, q=(5, 50, 95)):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": None, "lo90": None, "hi90": None}
    p = np.percentile(arr, q)
    return {"median": float(p[1]), "lo90": float(p[0]), "hi90": float(p[2]), "mean": float(arr.mean())}


class Setup:
    def __init__(self, d, open_mask=None):
        self.d = d
        self.grid = Grid(d["date"].max())
        self.idx = self.grid.idx(d["date"])
        self.open = d["open"].to_numpy() if open_mask is None else open_mask
        self.closed = (~d["open"]).to_numpy()
        self.dev = d["dev"].fillna("unknown").to_numpy()
        self.cc = d["cc"].to_numpy()
        self.open_devs = sorted(set(self.dev[self.open]))

    def scores(self, rng=None):
        s = self.d["eci"].to_numpy().copy()
        if rng is not None:
            s = s + rng.standard_normal(len(s)) * self.d["sd"].to_numpy()
        return s


def lag_metrics(S, s, win=WIN):
    g = S.grid
    O = frontier(g, S.idx, s, S.open)
    A = frontier(g, S.idx, s, np.ones(len(s), bool))
    C = frontier(g, S.idx, s, S.closed)
    w = window_mask(g, *win)
    L = hgap(g, O, A)
    Lc = hgap(g, O, C)
    return {"lag_all_mo": mo(np.nanmean(L[w])), "lag_closed_mo": mo(np.nanmean(Lc[w])),
            "gap_eci": float(np.mean((A - O)[w])), "O": O, "A": A, "C": C, "L": L}


def removal_metrics(S, s, remove, O, C, win=WIN):
    """Counterfactual open frontier without releases in `remove` (bool mask)."""
    g = S.grid
    Om = frontier(g, S.idx, s, S.open & ~remove)
    w = window_mask(g, *win)
    stale = setback(g, Om, O)              # frontier setback relative to actual open frontier
    lagc = hgap(g, Om, C)                  # counterfactual lag behind closed frontier
    T0 = first_reach(O, np.array(THRESH))
    T1 = first_reach(Om, np.array(THRESH))
    n = g.n
    delays = {}
    for c, a, b in zip(THRESH, T0, T1):
        if a >= n:
            delays[c] = np.nan            # never reached in reality
        elif b >= n:
            delays[c] = -mo(n - a)        # censored: negative flag encodes lower bound
        else:
            delays[c] = mo(b - a)
    return {"stale_mo": mo(np.nanmean(stale[w])),
            "gap_eci": float(np.nanmean(np.where(np.isfinite(Om), O - Om, np.nan)[w])),
            "lag_closed_cf_mo": mo(np.nanmean(lagc[w])), "delays": delays, "Om": Om}


def value_eci_months(S, s, members, base, w):
    if len(members) == 0:
        return 0.0
    m = S.open & np.isin(S.dev, list(members))
    F = frontier(S.grid, S.idx, s, m)
    return float(np.sum(np.clip(F[w] - base, 0, None)) / DAYS_PER_MONTH)


def shapley(S, s, w, n_perm, rng, value):
    devs = S.open_devs
    phi = {k: 0.0 for k in devs}
    for _ in range(n_perm):
        perm = rng.permutation(devs)
        prev, members = 0.0, []
        for k in perm:
            members.append(k)
            v = value(members)
            phi[k] += v - prev
            prev = v
    return {k: v / n_perm for k, v in phi.items()}


def release_rule(S, s, delta_mo, juris_mask, C, win=WIN):
    """Releases by covered developers are delayed until the closed frontier has held
    their capability for delta months. Returns lag of resulting open frontier."""
    g = S.grid
    idx = S.idx.copy()
    keep = S.open.copy()
    cov = S.open & juris_mask
    Tc = first_reach(C, s[cov])
    new = np.maximum(idx[cov], Tc + int(round(delta_mo * DAYS_PER_MONTH)))
    ok = new < g.n
    tmp_idx = idx[cov]
    tmp_idx[ok] = new[ok]
    idx[cov] = tmp_idx
    k2 = keep.copy()
    k2[np.where(cov)[0][~ok]] = False
    arr = np.full(g.n, -np.inf)
    np.maximum.at(arr, idx[k2], s[k2])
    Oc = np.maximum.accumulate(arr)
    w = window_mask(g, *win)
    O = frontier(g, S.idx, s, S.open)
    return {"lag_closed_mo": mo(np.nanmean(hgap(g, Oc, C)[w])),
            "stale_vs_actual_mo": mo(np.nanmean(setback(g, Oc, O)[w])),
            "n_delayed": int(np.sum(new > S.idx[cov]))}


def coalitions(S):
    cc = S.cc
    return {"CN developers": cc == "CN", "US developers": cc == "US",
            "non-CN developers": (cc != "CN"), "FR developers": cc == "FR"}


def main():
    rng = np.random.default_rng(SEED)
    d = load()
    S = Setup(d)
    out = {"n_models": int(len(d)), "n_open": int(S.open.sum()), "n_closed": int(S.closed.sum()),
           "data_end": str(d["date"].max().date()), "open_devs": S.open_devs,
           "n_open_devs": len(S.open_devs), "N_MC": N_MC, "window": WIN[0]}
    s0 = S.scores()
    base = lag_metrics(S, s0)
    O0, C0, A0 = base["O"], base["C"], base["A"]

    # ---------- (1) lag, point + MC ----------
    lag_mc = {k: {"lag_all_mo": [], "lag_closed_mo": [], "gap_eci": []} for k in WINDOWS}
    co = coalitions(S)
    devs = S.open_devs
    rem_mc = {k: {"stale_mo": [], "gap_eci": [], "lag_closed_cf_mo": [],
                  **{f"delay_{c}": [] for c in THRESH}} for k in list(co) + devs}
    holder_mc = {}
    sumuni_mc = {k: [] for k in co}
    rule_mc = {}
    monthly = []
    for b in range(N_MC):
        s = S.scores(rng)
        for wk, win in WINDOWS.items():
            lm = lag_metrics(S, s, win)
            for key in ("lag_all_mo", "lag_closed_mo", "gap_eci"):
                lag_mc[wk][key].append(lm[key])
        lm = lag_metrics(S, s, WIN)
        O, C = lm["O"], lm["C"]
        # monthly mean lag for trend
        ser = pd.Series(lm["L"], index=S.grid.days)
        monthly.append(ser.resample("MS").mean().to_numpy() / DAYS_PER_MONTH)
        # who holds the open frontier (country), window share of days
        w = window_mask(S.grid, *WIN)
        holder = holder_of(S, s)
        for c in set(holder[w]):
            holder_mc.setdefault(c, []).append(float(np.mean(holder[w] == c)))
        for c in set(S.cc[S.open]) - set(holder[w]):
            holder_mc.setdefault(c, []).append(0.0)
        dev_stale = {}
        for k in list(co) + devs:
            rm = co[k] if k in co else (S.dev == k)
            r = removal_metrics(S, s, rm, O, C)
            for key in ("stale_mo", "gap_eci", "lag_closed_cf_mo"):
                rem_mc[k][key].append(r[key])
            for c in THRESH:
                rem_mc[k][f"delay_{c}"].append(r["delays"][c])
            if k in devs:
                dev_stale[k] = r["stale_mo"]
        for k, m in co.items():
            members = set(S.dev[S.open & m])
            sumuni_mc[k].append(sum(dev_stale[x] for x in members if x in dev_stale))
        if b < max(200, N_MC // 5):
            for J, jm in {"all": np.ones(len(s), bool), "US only": S.cc == "US",
                          "CN only": S.cc == "CN", "non-CN": S.cc != "CN"}.items():
                for dl in DELTAS:
                    r = release_rule(S, s, dl, jm, C)
                    rule_mc.setdefault(f"{J}|{dl}", {"lag_closed_mo": [], "stale_vs_actual_mo": []})
                    for key in ("lag_closed_mo", "stale_vs_actual_mo"):
                        rule_mc[f"{J}|{dl}"][key].append(r[key])

    out["lag"] = {wk: {k: summarize(v) for k, v in dd.items()} for wk, dd in lag_mc.items()}
    out["lag_point"] = {wk: {k: lag_metrics(S, s0, win)[k] for k in ("lag_all_mo", "lag_closed_mo", "gap_eci")}
                        for wk, win in WINDOWS.items()}
    out["open_frontier_holder_share"] = {k: summarize(v) for k, v in holder_mc.items()}

    def delay_summary(v):
        v = np.asarray(v, float)
        cens = np.mean(v < 0)
        return {**summarize(np.abs(v)), "censored_frac": float(cens),
                "never_reached_frac": float(np.mean(np.isnan(v)))}

    rem_tab = []
    for k, dd in rem_mc.items():
        row = {"removed": k, "kind": "coalition" if k in co else "developer",
               "n_open_releases": int(np.sum(S.open & (co[k] if k in co else S.dev == k)))}
        for key in ("stale_mo", "gap_eci", "lag_closed_cf_mo"):
            sm = summarize(dd[key])
            row[key] = sm["median"]; row[key + "_lo"] = sm["lo90"]; row[key + "_hi"] = sm["hi90"]
        for c in THRESH:
            ds = delay_summary(dd[f"delay_{c}"])
            row[f"delay_{c}"] = ds["median"]; row[f"delay_{c}_cens"] = ds["censored_frac"]
        rem_tab.append(row)
    rem_df = pd.DataFrame(rem_tab).sort_values("stale_mo", ascending=False)
    rem_df.to_csv(f"{RES}/removal_counterfactuals.csv", index=False)
    out["removal"] = rem_df.set_index("removed")[["stale_mo", "stale_mo_lo", "stale_mo_hi", "gap_eci",
                                                  "lag_closed_cf_mo", "lag_closed_cf_mo_lo", "lag_closed_cf_mo_hi"]
                                                 + [f"delay_{c}" for c in THRESH]
                                                 + [f"delay_{c}_cens" for c in THRESH]].to_dict("index")

    # coordination gap: joint effect vs sum of unilateral effects
    cg = {}
    for k in co:
        joint = np.asarray(rem_mc[k]["stale_mo"]); uni = np.asarray(sumuni_mc[k])
        ratio = uni / np.where(joint > 0, joint, np.nan)
        cg[k] = {"joint_stale_mo": summarize(joint), "sum_unilateral_stale_mo": summarize(uni),
                 "sum_over_joint": summarize(ratio),
                 "frac_draws_superadditive": float(np.mean(joint >= uni - 1e-9))}
    out["coordination_gap"] = cg

    # release rule
    out["release_rule"] = {k: {kk: summarize(vv) for kk, vv in v.items()} for k, v in rule_mc.items()}

    # monthly trend in lag
    M = np.array(monthly)
    months = pd.date_range(S.grid.days[0], S.grid.days[-1], freq="MS")
    mt = pd.DataFrame({"month": months, "lag_median": np.nanmedian(M, 0),
                       "lag_lo": np.nanpercentile(M, 5, 0), "lag_hi": np.nanpercentile(M, 95, 0)})
    mt.to_csv(f"{RES}/monthly_lag.csv", index=False)
    sel = months >= pd.Timestamp("2024-01-01")
    x = (months[sel] - pd.Timestamp("2024-01-01")).days / 365.25
    slopes = [np.polyfit(x, row[sel], 1)[0] for row in M if np.all(np.isfinite(row[sel]))]
    out["lag_trend_mo_per_year_2024on"] = summarize(slopes)

    # ---------- (2) Shapley on point estimates ----------
    w = window_mask(S.grid, *WIN)
    b0 = float(O0[w][0])
    out["shapley_baseline_eci"] = b0
    vN = value_eci_months(S, s0, devs, b0, w)
    sh = shapley(S, s0, w, 2000, np.random.default_rng(SEED + 1),
                 lambda m: value_eci_months(S, s0, m, b0, w))
    thr = 155.0

    def v_thr(m):
        if not m:
            return 0.0
        F = frontier(S.grid, S.idx, s0, S.open & np.isin(S.dev, list(m)))
        return float(np.sum(F[w] >= thr) / DAYS_PER_MONTH)
    sh2 = shapley(S, s0, w, 2000, np.random.default_rng(SEED + 2), v_thr)
    vN2 = v_thr(devs)
    rows = []
    for k in devs:
        rest = [x for x in devs if x != k]
        rows.append({"developer": k, "country": S.cc[S.dev == k][0],
                     "shapley_eci_months": sh[k], "marginal_eci_months": vN - value_eci_months(S, s0, rest, b0, w),
                     "shapley_months_above_155": sh2[k], "marginal_months_above_155": vN2 - v_thr(rest)})
    shdf = pd.DataFrame(rows).sort_values("shapley_eci_months", ascending=False)
    shdf.to_csv(f"{RES}/shapley.csv", index=False)
    out["shapley_total_eci_months"] = vN
    out["shapley_sum_marginals_eci_months"] = float(shdf["marginal_eci_months"].sum())
    out["shapley_total_months_above_155"] = vN2
    out["shapley_sum_marginals_months_above_155"] = float(shdf["marginal_months_above_155"].sum())
    out["shapley_by_country"] = shdf.groupby("country")[["shapley_eci_months", "marginal_eci_months",
                                                          "shapley_months_above_155", "marginal_months_above_155"]].sum().to_dict("index")

    # ---------- (3) release-level: open records ----------
    out["open_records"] = open_records(S, s0, O0)

    # ---------- (4) robustness ----------
    rob = {}
    rob["point_main"] = {k: base[k] for k in ("lag_all_mo", "lag_closed_mo", "gap_eci")}
    d_old = load(SNAP_OLD); So = Setup(d_old)
    so = So.scores(); bo = lag_metrics(So, so)
    rob["vintage_2026-11-11"] = {k: bo[k] for k in ("lag_all_mo", "lag_closed_mo", "gap_eci")}
    rob["vintage_2026-11-11_CN_removal_stale_mo"] = removal_metrics(So, so, So.cc == "CN", bo["O"], bo["C"])["stale_mo"]
    unr = (d["Model accessibility"] == "Open weights (unrestricted)").to_numpy()
    Su = Setup(d, open_mask=unr)
    bu = lag_metrics(Su, s0)
    rob["unrestricted_only"] = {k: bu[k] for k in ("lag_all_mo", "lag_closed_mo", "gap_eci")}
    rob["unrestricted_only_CN_removal_stale_mo"] = removal_metrics(Su, s0, Su.cc == "CN", bu["O"], bu["C"])["stale_mo"]
    # CI read as 95% instead of 90% (narrower noise)
    d95 = load(ci_level=0.95); S95 = Setup(d95); r95 = np.random.default_rng(SEED + 3)
    l95 = [lag_metrics(S95, S95.scores(r95))["lag_all_mo"] for _ in range(200)]
    rob["ci_as_95pct_lag_all_mo"] = summarize(l95)
    # noise-free point removal for each coalition
    rob["point_removal_stale_mo"] = {k: removal_metrics(S, s0, m, O0, C0)["stale_mo"] for k, m in co.items()}
    # coalition removal and coordination gap across calendar windows (point estimates)
    wins = {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
            "2026": ("2026-01-01", None), "2024+": ("2024-01-01", None)}
    bywin = {}
    for wn, wv in wins.items():
        row = {}
        for k, m in co.items():
            joint = removal_metrics(S, s0, m, O0, C0, wv)["stale_mo"]
            uni = sum(removal_metrics(S, s0, S.dev == x, O0, C0, wv)["stale_mo"]
                      for x in set(S.dev[S.open & m]))
            row[k] = {"joint_setback_mo": joint, "sum_unilateral_mo": uni}
        hold = holder_of(S, s0)[window_mask(S.grid, *wv)]
        row["holder_share"] = {c: float(np.mean(hold == c)) for c in sorted(set(hold))}
        row["lag_mo"] = lag_metrics(S, s0, wv)["lag_all_mo"]
        bywin[wn] = row
    rob["by_window"] = bywin
    out["robustness"] = rob

    # frontier series for figures (point estimates)
    ser = pd.DataFrame({"day": S.grid.days, "open": O0, "closed": C0, "all": A0})
    for k, m in co.items():
        ser[f"open_minus_{k.split()[0]}"] = removal_metrics(S, s0, m, O0, C0)["Om"]
    ser.to_csv(f"{RES}/frontier_series.csv", index=False)

    with open(f"{RES}/results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("lag", "coordination_gap", "lag_trend_mo_per_year_2024on",
                                          "open_frontier_holder_share", "robustness")}, indent=1, default=float))
    print(rem_df.head(12).to_string())
    print(shdf.head(10).to_string())


def holder_of(S, s):
    """Country of the developer holding the open frontier on each day."""
    g = S.grid
    order = np.where(S.open)[0]
    order = order[np.argsort(S.idx[order], kind="stable")]
    holder = np.array(["none"] * g.n, dtype=object)
    best, who, j = -np.inf, "none", 0
    for t in range(g.n):
        while j < len(order) and S.idx[order[j]] <= t:
            i = order[j]
            if s[i] > best:
                best, who = s[i], S.cc[i]
            j += 1
        holder[t] = who
    return holder


def open_records(S, s, O):
    """Open releases that set a new open-frontier record; advancement and replacement time."""
    g = S.grid
    order = np.where(S.open)[0]
    order = order[np.argsort(S.idx[order], kind="stable")]
    rows, best = [], -np.inf
    for i in order:
        if s[i] > best:
            prev = best
            best = s[i]
            if S.d["date"].iloc[i] < pd.Timestamp("2024-01-01"):
                continue
            # leave-one-developer-out replacement time
            Om = frontier(g, S.idx, s, S.open & (S.dev != S.dev[i]))
            T = first_reach(Om, np.array([s[i]]))[0]
            cens = T >= g.n
            # leave-one-release-out (developer may still release later)
            m = S.open.copy(); m[i] = False
            Or = frontier(g, S.idx, s, m)
            Tr = first_reach(Or, np.array([s[i]]))[0]
            rows.append({"model": S.d["Model"].iloc[i], "developer": S.dev[i], "country": S.cc[i],
                         "date": str(S.d["date"].iloc[i].date()), "eci": float(s[i]),
                         "advance_eci": float(s[i] - prev) if np.isfinite(prev) else None,
                         "replace_other_devs_mo": mo((min(T, g.n) - S.idx[i])), "replace_other_devs_censored": bool(cens),
                         "replace_any_mo": mo((min(Tr, g.n) - S.idx[i])), "replace_any_censored": bool(Tr >= g.n),
                         "replacer_country": None if cens else S.cc[S.open & (S.dev != S.dev[i])][
                             np.argmax((S.idx[S.open & (S.dev != S.dev[i])] <= T) & (s[S.open & (S.dev != S.dev[i])] >= s[i]))]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/open_records.csv", index=False)
    return {"n_records_2024on": int(len(df)),
            "median_replace_other_devs_mo_uncensored": float(df.loc[~df.replace_other_devs_censored, "replace_other_devs_mo"].median()),
            "frac_censored": float(df.replace_other_devs_censored.mean()),
            "frac_replaced_within_3mo": float(np.mean((~df.replace_other_devs_censored) & (df.replace_other_devs_mo <= 3))),
            "records_by_country": df.country.value_counts().to_dict()}


if __name__ == "__main__":
    main()
