"""Definition-sensitivity analysis of METR time-horizon run-level data.

Usage:  python3 analysis_metr.py --version 1-1 --B 200 --lam 1.0
Outputs (results/):
  metr_{v}_lam{lam}_point.csv      per-model point estimates of every metric
  metr_{v}_lam{lam}_boot.npz       bootstrap replicates
  metr_{v}_lam{lam}_summary.csv    doubling times + forecast dates per definition (median, 5%, 95%)
  metr_{v}_lam{lam}_slopes.json    trend in logistic slope b and in cross-family dispersion
"""
import argparse, json, time
import numpy as np
import pandas as pd
from core import load_runs, fit_model, log2_horizon, power_mean_log2

OUT = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results")
QS = [0.5, 0.8, 0.95]
AGGS = [("pooled", None), ("pmean+1", 1.0), ("geo(p=0)", 0.0), ("pmean-1", -1.0),
        ("q50fam", "q50"), ("q25fam", "q25"), ("q10fam", "q10"), ("minfam", -np.inf)]
T0 = pd.Timestamp("2023-01-01")
TARGETS = {"1 work-week (40h)": np.log2(40 * 60), "1 work-month (167h)": np.log2(167 * 60)}


def years(ts):
    return (ts - T0).days / 365.25


def metrics_for_dataset(df, lam):
    """Return DataFrame (models x metrics) of log2-horizon metrics, plus b and dispersion."""
    rows = []
    for m, g in df.groupby("alias"):
        pooled = fit_model(g)
        fam = fit_model(g, lam=lam)
        r = {"alias": m, "t": years(g.release.iloc[0]), "b_pooled": pooled["b"], "b_fam": fam["b"]}
        for q in QS:
            r[f"pooled|{q}"] = log2_horizon(pooled["a"], pooled["b"], q)
            l2 = log2_horizon(fam["a"], fam["b"], q, fam["delta"])
            for name, p in AGGS[1:]:
                if isinstance(p, str):
                    r[f"{name}|{q}"] = np.percentile(l2, int(p[1:]))
                else:
                    r[f"{name}|{q}"] = power_mean_log2(l2, p)
            if q == 0.5:
                r["disp_sd_log2"] = np.std(l2)
                r["disp_sd_delta"] = np.std(fam["delta"])
                r["cover_1h"] = np.mean(l2 >= np.log2(60))
                r["cover_8h"] = np.mean(l2 >= np.log2(480))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def frontier_mask(t, v):
    """Running-max frontier in release order (models released the same day considered jointly)."""
    order = np.lexsort((-v, t))
    best = -np.inf
    keep = np.zeros(len(t), bool)
    for i in order:
        if v[i] > best:
            keep[i] = True
            best = v[i]
    return keep


def trend(t, v, frontier=True, tmin=None):
    mask = np.isfinite(v)
    if tmin is not None:
        mask &= t >= tmin
    if frontier:
        fm = np.zeros(len(t), bool)
        fm[mask] = frontier_mask(t[mask], v[mask])
        mask = fm
    A = np.column_stack([np.ones(mask.sum()), t[mask]])
    coef, *_ = np.linalg.lstsq(A, v[mask], rcond=None)
    return coef  # intercept, slope (log2 units per year)


def build_index(df):
    fam_tasks = {}
    for (f, tsk), idx in df.groupby(["task_family", "task_id"]).indices.items():
        fam_tasks.setdefault(f, {})[tsk] = idx
    return fam_tasks


def boot_sample(df, fam_tasks, rng):
    fams = list(fam_tasks)
    pick = rng.choice(len(fams), size=len(fams), replace=True)
    idx_parts, keys = [], []
    for j, fi in enumerate(pick):
        tasks = list(fam_tasks[fams[fi]].values())
        tp = rng.choice(len(tasks), size=len(tasks), replace=True)
        for k in tp:
            idx_parts.append(tasks[k])
            keys.append(np.full(len(tasks[k]), f"{fams[fi]}#{j}"))
    idx = np.concatenate(idx_parts)
    out = df.iloc[idx].copy()
    out["fam_key"] = np.concatenate(keys)
    return out


def summarise(point, boots, tmin):
    metric_cols = [c for c in point.columns if "|" in c]
    rows = []
    for c in metric_cols:
        def stats(P):
            a, s = trend(P.t.values, P[c].values, True, tmin)
            out = {"dt_months": 12 / s if s > 0 else np.inf}
            for name, L in TARGETS.items():
                out[name] = 2023 + (L - a) / s if s > 0 else np.inf
            out["level_latest"] = P[c].values[-1]
            return out
        p0 = stats(point)
        bs = pd.DataFrame([stats(B) for B in boots])
        row = {"metric": c, "agg": c.split("|")[0], "q": float(c.split("|")[1])}
        for k in p0:
            row[f"{k}"] = p0[k]
            row[f"{k}_lo"] = np.nanpercentile(bs[k], 5)
            row[f"{k}_med"] = np.nanpercentile(bs[k], 50)
            row[f"{k}_hi"] = np.nanpercentile(bs[k], 95)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1-1")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tmin", type=float, default=0.0, help="years since 2023-01-01 for trend window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = f"metr_{a.version}_lam{a.lam:g}"
    df = load_runs(a.version)
    df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
    df["fam_key"] = df.task_family
    t0 = time.time()
    point = metrics_for_dataset(df, a.lam)
    point.to_csv(f"{OUT}/{tag}_point.csv", index=False)
    print("point done", time.time() - t0, flush=True)
    rng = np.random.default_rng(a.seed)
    fam_tasks = build_index(df)
    boots = []
    for bi in range(a.B):
        boots.append(metrics_for_dataset(boot_sample(df, fam_tasks, rng), a.lam))
        if bi % 20 == 0:
            print("boot", bi, round(time.time() - t0, 1), flush=True)
    cols = list(point.columns)
    np.savez_compressed(f"{OUT}/{tag}_boot.npz", cols=np.array(cols, dtype=object),
                        arr=np.stack([B[cols[1:]].values.astype(float) for B in boots]),
                        alias=point.alias.values)
    summ = summarise(point, boots, a.tmin)
    summ.to_csv(f"{OUT}/{tag}_summary.csv", index=False)

    # trends in slope b and dispersion (all models, OLS on time)
    def slope_all(P, col):
        A = np.column_stack([np.ones(len(P)), P.t.values])
        return np.linalg.lstsq(A, P[col].values, rcond=None)[0][1]
    res = {}
    for col in ["b_pooled", "disp_sd_log2", "disp_sd_delta", "b_fam"]:
        est = slope_all(point, col)
        bs = np.array([slope_all(B, col) for B in boots])
        res[col] = dict(slope_per_year=est, lo=np.percentile(bs, 5), hi=np.percentile(bs, 95),
                        frac_boot_positive=float(np.mean(bs > 0)))
    # p80/p50 doubling-time ratio
    r = []
    for B in [point] + boots:
        s50 = trend(B.t.values, B["pooled|0.5"].values, True, a.tmin)[1]
        s80 = trend(B.t.values, B["pooled|0.8"].values, True, a.tmin)[1]
        r.append(s50 / s80)
    res["dt80_over_dt50"] = dict(est=r[0], lo=np.percentile(r[1:], 5), hi=np.percentile(r[1:], 95))
    json.dump(res, open(f"{OUT}/{tag}_slopes.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summ[["metric", "dt_months", "dt_months_lo", "dt_months_hi",
                    "1 work-month (167h)", "1 work-month (167h)_lo", "1 work-month (167h)_hi"]].round(2))
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()
