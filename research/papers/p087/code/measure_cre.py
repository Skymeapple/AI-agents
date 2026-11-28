"""
Part A: Contemporaneous Relative Efficiency (CRE) of non-NVIDIA AI accelerators.

For every non-NVIDIA data-centre accelerator j released at date t_j and every metric M,
    CRE_j(M) = M_j / max{ M_k : k is an NVIDIA data-centre chip, t_k <= t_j }        (frontier "F0")
A robustness variant uses the NVIDIA frontier at t_j + 12 months ("F+12": the NVIDIA chip
a buyer could alternatively get during the ASIC's first deployment year).
Lead/lag in years:  L_j = log10(CRE_j) / g_M,  where g_M (OOM/yr) is the growth rate of the
NVIDIA record-setting frontier for metric M, estimated by OLS on record-setters.

Input: Epoch AI "Machine Learning Hardware" CSV (CC-BY), snapshot last-modified 2026-11-19.
Outputs: results/cre_table.csv, results/cre_summary.json, results/fig_cre_*.png
"""
import csv, json, math, os, sys
from datetime import date
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "epoch_ml_hardware_2026-11-19.csv")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(20261122)

CONSUMER = ("geforce", "quadro", "titan", "rtx ", "rtx a", "radeon rx", "rtx pro", "l2 ", "l20", "a10", "a40", "l40", "l4", "t4", "p4", "tesla p4")


def fnum(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None


def todate(s):
    try:
        y, m, d = s[:10].split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def yfrac(d):
    return d.year + (d.timetuple().tm_yday - 0.5) / 365.25


def is_consumer(name):
    n = name.lower() + " "
    # keep flagship datacentre parts explicitly
    for keep in ("a100", "h100", "h200", "h800", "a800", "b200", "b300", "gb200", "gb300", "v100", "p100", "vr200", "gh100", "gh200", "h20", "k80", "m40", "m60"):
        if keep in n:
            return False
    return any(c in n for c in CONSUMER)


GROUP = {
    "Google": "hyperscaler_asic", "Amazon AWS": "hyperscaler_asic", "Meta": "hyperscaler_asic",
    "Microsoft": "hyperscaler_asic", "OpenAI": "hyperscaler_asic", "Tesla": "hyperscaler_asic",
    "Alibaba": "china_domestic",
    "AMD": "merchant_gpu", "Intel": "merchant_gpu",
    "Huawei": "china_domestic", "Cambricon": "china_domestic", "Kunlunxin Baidu": "china_domestic",
    "MetaX": "china_domestic", "Iluvatar CoreX": "china_domestic", "Biren": "china_domestic",
    "Moore Threads (Tencent)": "china_domestic", "Hygon": "china_domestic",
}

METRICS = {
    "bf16_flops": lambda r: fnum(r["Tensor-FP16/BF16 performance (FLOP/s)"]),
    "bf16_flops_per_w": lambda r: (fnum(r["Tensor-FP16/BF16 performance (FLOP/s)"]) / fnum(r["TDP (W)"]))
    if fnum(r["Tensor-FP16/BF16 performance (FLOP/s)"]) and fnum(r["TDP (W)"]) else None,
    "mem_bw": lambda r: fnum(r["Memory bandwidth (byte/s)"]),
    "mem_cap": lambda r: fnum(r["Memory (bytes)"]),
    "mem_bw_per_w": lambda r: (fnum(r["Memory bandwidth (byte/s)"]) / fnum(r["TDP (W)"]))
    if fnum(r["Memory bandwidth (byte/s)"]) and fnum(r["TDP (W)"]) else None,
    "fp8_flops": lambda r: fnum(r["FP8 performance (FLOP/s)"]),
}


def load():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    out = []
    for r in rows:
        d = todate(r["Release date"])
        if d is None or d.year < 2015:
            continue
        man = r["Manufacturer"].strip()
        name = r["Hardware name"].strip()
        rec = {"name": name, "man": man, "date": d, "t": yfrac(d)}
        for k, f in METRICS.items():
            rec[k] = f(r)
        rec["price"] = fnum(r["Release price (USD)"])
        rec["tdp"] = fnum(r["TDP (W)"])
        if man == "NVIDIA":
            if is_consumer(name):
                continue
            rec["group"] = "nvidia"
        elif man in GROUP:
            rec["group"] = GROUP[man]
        else:
            continue
        out.append(rec)
    return out


def nvidia_frontier(recs, metric):
    pts = sorted([(r["t"], r[metric], r["name"]) for r in recs if r["group"] == "nvidia" and r[metric]])
    return pts


def frontier_at(pts, t):
    vals = [(v, n) for (tt, v, n) in pts if tt <= t + 1e-9]
    if not vals:
        return None, None
    return max(vals)


def growth_rate(pts):
    """OLS slope (OOM/yr) of log10 value on time over NVIDIA record-setters (running max)."""
    best = -1
    rs = []
    for t, v, n in pts:
        if v > best:
            best = v
            rs.append((t, math.log10(v)))
    rs = np.array(rs)
    if len(rs) < 3:
        return None, None, rs
    A = np.vstack([rs[:, 0], np.ones(len(rs))]).T
    coef, *_ = np.linalg.lstsq(A, rs[:, 1], rcond=None)
    # bootstrap CI for slope
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(rs), len(rs))
        if len(set(rs[idx, 0])) < 2:
            continue
        c, *_ = np.linalg.lstsq(A[idx], rs[idx, 1], rcond=None)
        bs.append(c[0])
    return coef[0], (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), rs


def main():
    recs = load()
    table = []
    summary = {"n_records_used": len(recs), "n_nvidia_dc": sum(r["group"] == "nvidia" for r in recs)}
    growth = {}
    for m in METRICS:
        pts = nvidia_frontier(recs, m)
        g, ci, rs = growth_rate(pts)
        growth[m] = {"g_oom_per_yr": g, "g_ci95": ci, "doubling_yrs": (math.log10(2) / g) if g else None,
                     "n_record_setters": int(len(rs))}
        for r in recs:
            if r["group"] == "nvidia" or not r[m]:
                continue
            f0, n0 = frontier_at(pts, r["t"])
            f12, n12 = frontier_at(pts, r["t"] + 1.0)
            if f0 is None:
                continue
            cre0 = r[m] / f0
            cre12 = r[m] / f12
            table.append({"metric": m, "name": r["name"], "man": r["man"], "group": r["group"],
                          "date": r["date"].isoformat(), "t": round(r["t"], 3), "value": r[m],
                          "nv_frontier_F0": f0, "nv_frontier_chip_F0": n0, "CRE_F0": cre0,
                          "nv_frontier_F12": f12, "nv_frontier_chip_F12": n12, "CRE_F12": cre12,
                          "lead_years_F0": math.log10(cre0) / g if g else None,
                          "lead_years_F12": math.log10(cre12) / g if g else None})
    summary["nvidia_frontier_growth"] = growth

    with open(os.path.join(RES, "cre_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    def boot_median(x, B=5000):
        x = np.asarray(x)
        if len(x) == 0:
            return None
        bs = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
        return [float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    groups = ["hyperscaler_asic", "merchant_gpu", "china_domestic"]
    periods = [("2015-2019", 0, 2020), ("2020-2022", 2020, 2023), ("2023-2026", 2023, 2100)]
    stats = {}
    for m in METRICS:
        stats[m] = {}
        for gname in groups:
            rows = [x for x in table if x["metric"] == m and x["group"] == gname]
            if not rows:
                continue
            c0 = [x["CRE_F0"] for x in rows]
            c12 = [x["CRE_F12"] for x in rows]
            entry = {"n": len(rows),
                     "median_CRE_F0_ci": boot_median(c0),
                     "median_CRE_F12_ci": boot_median(c12),
                     "share_CRE_F0_ge1": float(np.mean(np.array(c0) >= 1)),
                     "share_CRE_F12_ge1": float(np.mean(np.array(c12) >= 1)),
                     "median_lead_years_F0": float(np.median([x["lead_years_F0"] for x in rows])),
                     "median_lead_years_F12": float(np.median([x["lead_years_F12"] for x in rows]))}
            per = {}
            for pn, a, b in periods:
                pr = [x["CRE_F0"] for x in rows if a <= x["t"] < b]
                if pr:
                    per[pn] = {"n": len(pr), "median_CRE_F0": float(np.median(pr))}
            entry["by_period"] = per
            # trend of log10 CRE on time
            if len(rows) >= 4:
                t = np.array([x["t"] for x in rows])
                y = np.log10(c0)
                A = np.vstack([t, np.ones(len(t))]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                bs = []
                for _ in range(3000):
                    idx = rng.integers(0, len(t), len(t))
                    if len(set(t[idx])) < 2:
                        continue
                    c, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
                    bs.append(c[0])
                entry["trend_log10CRE_per_yr"] = [float(coef[0]), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            stats[m][gname] = entry
    summary["group_stats"] = stats

    # hyperscaler ASIC: list for paper
    summary["hyperscaler_bf16"] = [
        {k: x[k] for k in ("name", "date", "CRE_F0", "nv_frontier_chip_F0", "CRE_F12", "lead_years_F0")}
        for x in table if x["metric"] == "bf16_flops" and x["group"] == "hyperscaler_asic"]
    summary["hyperscaler_bf16_per_w"] = [
        {k: x[k] for k in ("name", "date", "CRE_F0", "nv_frontier_chip_F0", "CRE_F12", "lead_years_F0")}
        for x in table if x["metric"] == "bf16_flops_per_w" and x["group"] == "hyperscaler_asic"]
    summary["hyperscaler_mem_bw"] = [
        {k: x[k] for k in ("name", "date", "CRE_F0", "nv_frontier_chip_F0", "CRE_F12")}
        for x in table if x["metric"] == "mem_bw" and x["group"] == "hyperscaler_asic"]

    # price-performance where both prices exist (sparse; reported descriptively only)
    pp = []
    for r in recs:
        if r["price"] and r["bf16_flops"] and r["t"] >= 2017:
            pp.append({"name": r["name"], "group": r["group"], "date": r["date"].isoformat(),
                       "price": r["price"], "bf16_flops_per_usd": r["bf16_flops"] / r["price"]})
    summary["price_performance_records"] = pp

    json.dump(summary, open(os.path.join(RES, "cre_summary.json"), "w"), indent=1, default=str)

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"hyperscaler_asic": "#1b7837", "merchant_gpu": "#b2182b", "china_domestic": "#2166ac"}
    labels = {"hyperscaler_asic": "Hyperscaler/in-house ASIC", "merchant_gpu": "Merchant GPU (AMD, Intel)",
              "china_domestic": "Chinese domestic"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, m, ttl in zip(axes, ["bf16_flops", "bf16_flops_per_w", "mem_bw"],
                          ["Dense BF16 FLOP/s", "BF16 FLOP/s per W (TDP)", "Memory bandwidth"]):
        for gname in groups:
            rows = [x for x in table if x["metric"] == m and x["group"] == gname]
            ax.scatter([x["t"] for x in rows], [x["CRE_F0"] for x in rows], s=28, c=colors[gname],
                       label=labels[gname], alpha=0.85, edgecolor="white", linewidth=0.5)
            if gname == "hyperscaler_asic":
                for x in rows:
                    short = x["name"].replace("Google ", "").replace("Amazon ", "").replace("Microsoft ", "").replace("Meta ", "")
                    ax.annotate(short[:16], (x["t"], x["CRE_F0"]), fontsize=6.5, xytext=(3, 2),
                                textcoords="offset points", color="#1b7837")
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(ttl + " relative to NVIDIA frontier at release", fontsize=9.5)
        ax.set_xlabel("Release date")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CRE (accelerator / best prior NVIDIA DC chip)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_cre_scatter.png"), dpi=170)

    # frontier growth figure
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for gname in ["nvidia"] + groups:
        rs = [r for r in recs if r["group"] == gname and r["bf16_flops"]]
        ax.scatter([r["t"] for r in rs], [r["bf16_flops"] for r in rs], s=20,
                   c="#555555" if gname == "nvidia" else colors[gname],
                   label="NVIDIA data-centre" if gname == "nvidia" else labels[gname], alpha=0.8)
    pts = nvidia_frontier(recs, "bf16_flops")
    tt = np.linspace(2017, 2026.9, 200)
    ax.step(tt, [frontier_at(pts, t)[0] or np.nan for t in tt], where="post", color="k", lw=1, label="NVIDIA frontier (F0)")
    ax.set_yscale("log")
    ax.set_ylabel("Dense BF16 FLOP/s per chip")
    ax.set_xlabel("Release date")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_frontier_bf16.png"), dpi=170)

    print(json.dumps({"growth": growth}, indent=1, default=str))
    for m in ["bf16_flops", "bf16_flops_per_w", "mem_bw", "mem_cap", "fp8_flops"]:
        print("==", m)
        for gname, e in stats[m].items():
            print(gname, json.dumps({k: e[k] for k in e if k != "by_period"}, default=str), e["by_period"])
    for x in summary["hyperscaler_bf16"]:
        print("BF16", x)
    for x in summary["hyperscaler_bf16_per_w"]:
        print("BF16/W", x)


if __name__ == "__main__":
    main()
