"""Secondary analyses from results/results.json: pairwise disagreement counts,
release-date trends (log-linear doubling times), convexity of pass->DVR map,
sensitivity summary.  Run after analysis.py.  Writes results/extra.json"""
import json, os, itertools
import numpy as np
from scipy.stats import linregress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results/results.json")))
main, sweep, sens, retry = d["main"], d["sweep"], d["sens"], d["retry"]
models = sorted(main)
out = {}

# 1. pairwise disagreement between pass_eq ranking and DVR rankings
def disagree(key_a, src_a, key_b, src_b):
    n = 0; tot = 0; lst = []
    for a, b in itertools.combinations(models, 2):
        da = src_a[a][key_a] - src_a[b][key_a]
        db = src_b[a][key_b] - src_b[b][key_b]
        if abs(db) < 1e-4:   # treat DVR ties (both ~0) as no information
            continue
        tot += 1
        if np.sign(da) != np.sign(db):
            n += 1; lst.append((a, b))
    return n, tot, lst

for rho in ["0.0", "0.2", "0.4"]:
    for k in ["dv_oracle", "dv_informed"]:
        n, tot, lst = disagree("pass_eq", main, k, sweep[rho])
        out[f"disagree_passeq_vs_{k}_rho{rho}"] = dict(n=n, of=tot, pairs=lst)
        n2, tot2, _ = disagree("h50", main, k, sweep[rho])
        out[f"disagree_h50_vs_{k}_rho{rho}"] = dict(n=n2, of=tot2)

# 2. trends over release date
def years(s):
    return (np.datetime64(s) - np.datetime64("2023-01-01")).astype(float) / 365.25

dm = [m for m in models if main[m]["release"]]
t = np.array([years(main[m]["release"]) for m in dm])
def fit(vals, mask=None):
    v = np.array(vals); msk = (v > 0) if mask is None else mask
    lr = linregress(t[msk], np.log2(v[msk]))
    return dict(slope_log2_per_year=lr.slope, doubling_months=12 / lr.slope, r2=lr.rvalue ** 2, n=int(msk.sum()))
out["trend_h50"] = fit([main[m]["h50"] for m in dm])
dvi = np.array([main[m]["dv_informed"] for m in dm])
out["trend_dv_informed_pos"] = fit(dvi, dvi > 0.005)
dvo = np.array([main[m]["dv_oracle"] for m in dm])
out["trend_dv_oracle"] = fit(dvo, dvo > 0)
pe = np.array([main[m]["pass_eq"] for m in dm])
lr = linregress(t, pe); out["trend_pass_eq_linear_per_year"] = dict(slope=lr.slope, r2=lr.rvalue ** 2)

# 3. convexity: DVR gained per pass-rate point in low vs high bands
pe_all = np.array([main[m]["pass_eq"] for m in models]); dv_all = np.array([main[m]["dv_informed"] for m in models])
lo = pe_all < 0.65; hi = pe_all >= 0.70
out["band_low"] = dict(pass_range=[float(pe_all[lo].min()), float(pe_all[lo].max())],
                       dv_range=[float(dv_all[lo].min()), float(dv_all[lo].max())],
                       slope=float(linregress(pe_all[lo], dv_all[lo]).slope))
out["band_high"] = dict(pass_range=[float(pe_all[hi].min()), float(pe_all[hi].max())],
                        dv_range=[float(dv_all[hi].min()), float(dv_all[hi].max())],
                        slope=float(linregress(pe_all[hi], dv_all[hi]).slope))

# 4. information value: oracle - informed and family vs length policies at base
out["info_gap"] = {m: dict(oracle=main[m]["dv_oracle"], family=main[m]["dv_family"], length=main[m]["dv_informed"],
                           blind=main[m]["dv_all"]) for m in models}

# 5. sensitivity summary
def sub(**kw):
    return [s for s in sens if all(abs(s[k] - v) < 1e-9 for k, v in kw.items())]
out["sens_rows"] = sens
out["sens_best_inf_range"] = [min(s["best_inf"] for s in sens), max(s["best_inf"] for s in sens)]
out["sens_tau_range"] = [min(s["tau_passeq"] for s in sens), max(s["tau_passeq"] for s in sens)]
out["sens_top_counts"] = {m: sum(1 for s in sens if s["top_inf"] == m) for m in set(s["top_inf"] for s in sens)}
out["sens_npos_range"] = [min(s["n_pos"] for s in sens), max(s["n_pos"] for s in sens)]

json.dump(out, open(os.path.join(ROOT, "results/extra.json"), "w"), indent=1, default=float)
for k, v in out.items():
    if k in ("sens_rows", "info_gap"):
        continue
    print(k, v if not isinstance(v, dict) or "pairs" not in v else {kk: vv for kk, vv in v.items()})
print("\nsensitivity (r0, s, delta, price, rho): best_inf best_or top tau npos")
for s in sens:
    print(s["r0"], s["s"], s["delta"], s["price"], s["rho"], round(s["best_inf"], 3), round(s["best_or"], 3), s["top_inf"], round(s["tau_passeq"], 2), s["n_pos"])
