"""Aggregates CPU replicates and tests the four model predictions T1-T4.
Outputs results/cpu_summary.json and results/fig_cpu_*.png"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(__file__), "..", "results")
reps = sorted(glob.glob(os.path.join(RES, "cpu_runs_rep*.json")))
cals = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "cpu_calibration_rep*.json")))]
runs = [json.load(open(f)) for f in reps]
key = lambda r: (r["exp"], r["n_kv"], r["ffn"], r["L"], r["b"], r["tau"])
agg = {}
for R in runs:
    for r in R:
        agg.setdefault(key(r), []).append(r)
rows = []
for k, lst in agg.items():
    t = np.array([x["t_per_token"] for x in lst])
    d = dict(lst[0]); d["t_per_token_med"] = float(np.median(t)); d["t_per_token_all"] = t.tolist()
    d["pred_per_token_med"] = float(np.median([x["pred_per_token"] for x in lst]))
    rows.append(d)
S = dict(n_replicates=len(runs), calibration=dict(
    {k: [float(np.median([c[k] for c in cals])), float(np.min([c[k] for c in cals])), float(np.max([c[k] for c in cals]))] for k in cals[0]}))

def sel(**kw):
    return sorted([r for r in rows if all(r[k] == v for k, v in kw.items())], key=lambda r: (r["L"], r["b"], r["tau"]))

# T1: saturation. For each n_kv and L, per-token time at largest b vs predicted asymptote L*kappa/B_kv
T1 = []
for nkv in [16, 4, 1]:
    for L in [256, 512, 1024, 2048, 4096]:
        rs = [r for r in sel(exp="T1T2", n_kv=nkv) if r["L"] == L]
        if not rs: continue
        rb = max(rs, key=lambda r: r["b"]); r1 = min(rs, key=lambda r: r["b"])
        T1.append(dict(n_kv=nkv, L=L, b_max=rb["b"], t_tok_b1_ms=r1["t_per_token_med"] * 1e3,
                       t_tok_bmax_ms=rb["t_per_token_med"] * 1e3, kappa=rb["kappa"],
                       kv_only_ms=L * rb["kappa"] / S["calibration"]["B_attn_kernel"][0] * 1e3))
S["T1_saturation"] = T1
# slope of per-token time at max batch vs L  (should ~ kappa / B_kv)
slopes = {}
for nkv in [16, 4, 1]:
    pts = [(t["L"], t["t_tok_bmax_ms"]) for t in T1 if t["n_kv"] == nkv]
    x, y = np.array(pts).T
    A = np.vstack([x, np.ones_like(x)]).T
    (s, c), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ np.array([s, c])) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    kappa = [t["kappa"] for t in T1 if t["n_kv"] == nkv][0]
    slopes[nkv] = dict(slope_us_per_ctx_token=s * 1e3, intercept_ms=c, r2=r2, kappa=kappa,
                       predicted_slope_us=kappa / S["calibration"]["B_attn_kernel"][0] * 1e6)
S["T1_T2_slopes"] = slopes
S["T2_slope_ratios"] = dict(r16_4=slopes[16]["slope_us_per_ctx_token"] / slopes[4]["slope_us_per_ctx_token"],
                            r4_1=slopes[4]["slope_us_per_ctx_token"] / slopes[1]["slope_us_per_ctx_token"],
                            r16_1=slopes[16]["slope_us_per_ctx_token"] / slopes[1]["slope_us_per_ctx_token"],
                            kappa_ratios=[4, 4, 16])
# prediction error of the fully independent roofline over all T1T2 points
pe = [(r["pred_per_token_med"], r["t_per_token_med"], r["b"], r["n_kv"]) for r in rows if r["exp"] == "T1T2"]
ratios = np.array([m / p for p, m, b, n in pe])
S["roofline_measured_over_predicted"] = dict(median=float(np.median(ratios)), p10=float(np.percentile(ratios, 10)),
                                             p90=float(np.percentile(ratios, 90)),
                                             median_nkv16=float(np.median([m / p for p, m, b, n in pe if n == 16])),
                                             median_nkv1=float(np.median([m / p for p, m, b, n in pe if n == 1])))
# T3
T3 = []
for L in [256, 4096]:
    for b in [1, 8]:
        rs = [r for r in rows if r["exp"] == "T3" and r["L"] == L and r["b"] == b]
        rs.sort(key=lambda r: r["ffn"])
        T3.append(dict(L=L, b=b, ffn=[r["ffn"] for r in rs], W_MB=[r["W_bytes"] / 1e6 for r in rs],
                       t_tok_ms=[r["t_per_token_med"] * 1e3 for r in rs],
                       t_tok_all=[[v * 1e3 for v in r["t_per_token_all"]] for r in rs],
                       elasticity_lnT_lnW=float(np.polyfit(np.log([r["W_bytes"] for r in rs]),
                                                           np.log([r["t_per_token_med"] for r in rs]), 1)[0])))
S["T3_weight_insensitivity"] = T3
# T4
T4 = []
for L in [512, 4096]:
    rs = sorted([r for r in rows if r["exp"] == "T4" and r["L"] == L], key=lambda r: r["tau"])
    t1 = rs[0]["t_per_token_med"]
    T4.append(dict(L=L, tau=[r["tau"] for r in rs], t_tok_ms=[r["t_per_token_med"] * 1e3 for r in rs],
                   speedup=[t1 / r["t_per_token_med"] for r in rs],
                   pred_speedup=[rs[0]["pred_per_token_med"] / r["pred_per_token_med"] for r in rs]))
S["T4_multitoken"] = T4
json.dump(S, open(os.path.join(RES, "cpu_summary.json"), "w"), indent=1)

# Figures
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
cols = {256: "#9ecae1", 1024: "#3182bd", 4096: "#08306b"}
ax = axes[0]
for L in [1024, 4096]:
    rs = [r for r in sel(exp="T1T2", n_kv=16) if r["L"] == L]
    b = [r["b"] for r in rs]; t = [r["t_per_token_med"] * 1e3 for r in rs]
    ax.loglog(b, t, "o-", color=cols[L], label=f"measured, L={L}")
    ax.axhline(L * rs[0]["kappa"] / S["calibration"]["B_attn_kernel"][0] * 1e3, ls=":", color=cols[L])
ax.set_xlabel("batch size b"); ax.set_ylabel("ms per generated token"); ax.set_title("T1: batching saturates at L*kappa/B (dotted)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axes[1]
mk = {16: "o", 4: "s", 1: "^"}
for nkv in [16, 4, 1]:
    pts = [(t["L"], t["t_tok_bmax_ms"]) for t in T1 if t["n_kv"] == nkv]
    x, y = np.array(pts).T
    ax.plot(x, y, mk[nkv] + "-", label=f"n_kv={nkv} (kappa={slopes[nkv]['kappa']//1024} KiB)")
ax.set_xlabel("context L"); ax.set_ylabel("ms/token at largest batch"); ax.set_title("T2: floor slope tracks KV bytes", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[2]
for d in T4:
    ax.plot(d["tau"], d["speedup"], "o-", label=f"measured L={d['L']}")
    ax.plot(d["tau"], d["pred_speedup"], "x--", label=f"roofline L={d['L']}")
ax.plot([1, 16], [1, 16], color="grey", lw=0.6)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("positions verified per pass tau"); ax.set_ylabel("per-token speedup (all accepted)")
ax.set_title("T4: multi-token verification divides KV term", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_cpu_tests.png"), dpi=150); plt.close()
print(json.dumps({k: v for k, v in S.items()}, indent=1)[:9000])

# ---- Post-hoc two-term per-context-token traffic model: slope = (kappa + sigma)/B_eff ----
# sigma = per-context-token bytes of materialised per-head scores (independent of n_kv),
# which fused GPU attention kernels avoid.  Two parameters fitted to the n_kv=16 and n_kv=1
# slopes; n_kv=4 slope and all per-point times are then out-of-sample checks.
s16, s1, s4 = (slopes[n]["slope_us_per_ctx_token"] * 1e-6 for n in (16, 1, 4))
B_eff = (65536 - 4096) / (s16 - s1)
sigma = s1 * B_eff - 4096
pred_s4 = (16384 + sigma) / B_eff
# semi-empirical per-point prediction: measured weight-stage time t_w(b) (1-token cache run)
tw = {}
for r in rows:
    if r["exp"] == "W":
        tw[(r["n_kv"], r["b"])] = r["t_per_token_med"] * r["b"]   # step time
errs = []
for r in rows:
    if r["exp"] != "T1T2": continue
    step_pred = tw[(r["n_kv"], r["b"])] + r["b"] * r["L"] * (r["kappa"] + sigma) / B_eff
    errs.append((r["t_per_token_med"] * r["b"]) / step_pred)
errs = np.array(errs)
S["posthoc_two_term"] = dict(B_eff_GBps=B_eff / 1e9, sigma_bytes_per_ctx_token=sigma,
                             B_attn_kernel_calibrated_GBps=S["calibration"]["B_attn_kernel"][0] / 1e9,
                             nkv4_slope_pred_us=pred_s4 * 1e6, nkv4_slope_meas_us=s4 * 1e6,
                             semi_empirical_measured_over_pred=dict(median=float(np.median(errs)),
                                 p10=float(np.percentile(errs, 10)), p90=float(np.percentile(errs, 90)),
                                 mean_abs_log_err=float(np.mean(np.abs(np.log(errs)))), n=int(len(errs))))
S["weight_stage_step_ms"] = {f"nkv{n}_b{b}": v * 1e3 for (n, b), v in sorted(tw.items())}
json.dump(S, open(os.path.join(RES, "cpu_summary.json"), "w"), indent=1)
print(json.dumps(S["posthoc_two_term"], indent=1)); print(json.dumps(S["weight_stage_step_ms"], indent=1))
