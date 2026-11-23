"""Round-2 statistics. Reads results/runs/*.json, results/runs_r2/*.json, results/r2_models_{old,new}.json.
Writes results/r2_stats.json and figures fig5_runlevel.png, fig6_gating.png."""
import os, glob, json, math
import numpy as np
from scipy.stats import spearmanr, pearsonr, beta as Beta
import cotlab as C
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = 23
out = {}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else Beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else Beta.ppf(1 - a / 2, k + 1, n - k)
    return [k, n, round(k / n, 3), round(float(lo), 3), round(float(hi), 3)]


def quad(acc_clean, acc_trunc, follow, th=0.5):
    return ("N+" if acc_clean - acc_trunc > th else "N-") + ("S+" if follow > th else "S-")


def mquad(r, th=0.5):
    """Mechanistic quadrant: N_mech+ if blocking ANS->CoT-position (true CoT present) costs > th accuracy;
    S_mech+ if the CoT route alone (strict internal cut) gives > th accuracy."""
    return ("N+" if r["acc_clean"] - r["acc_cot_cut"] > th else "N-") + ("S+" if r["acc_internal_cut_strict"] > th else "S-")


def eq2(h):
    return C.additive_route_fit(h, P)["pred_trunc"]


# ------------------------------------------------------------------ A. round-1 runs: corrections
r1 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "runs", "*.json")))]
r1f = [r for r in r1 if r["regime"] == "free"]
A = {}
A["never_reliant_runs"] = {r["tag"]: [round(h["follow"], 3) for h in r["history"][:4]] + [round(r["final"]["follow"], 3)]
                           for r in r1f if max(h["follow"] for h in r["history"]) < 0.5}
A["max_follow_never_reliant"] = {t: max(h["follow"] for h in r["history"]) for r in r1f for t in [r["tag"]]
                                 if max(h["follow"] for h in r["history"]) < 0.5}
# Eq. 2 vs trivial predictor P(ignore) on graded checkpoints
def eq2_vs_trivial(recs, cond):
    xs, bs, ys, tags = [], [], [], []
    for r in recs:
        if not cond(r): continue
        for h in r["history"]:
            if h["other"] >= 0.02:
                xs.append(eq2(h)); bs.append(h["ignore"]); ys.append(h["acc_trunc"]); tags.append(r["tag"])
    xs, bs, ys = map(np.array, (xs, bs, ys))
    if len(xs) < 3: return {"n": int(len(xs))}
    return {"n": int(len(xs)), "n_runs": len(set(tags)),
            "eq2_r": float(pearsonr(xs, ys)[0]), "eq2_mae": float(np.abs(xs - ys).mean()),
            "trivial_r": float(pearsonr(bs, ys)[0]), "trivial_mae": float(np.abs(bs - ys).mean())}
reg_c = lambda r: r["eta"] == 0 and r["delta"] == 0.3
A["eq2_vs_trivial_excl_c"] = eq2_vs_trivial(r1f, lambda r: not reg_c(r))
A["eq2_vs_trivial_c"] = eq2_vs_trivial(r1f, reg_c)
A["eq2_vs_trivial_all"] = eq2_vs_trivial(r1f, lambda r: True)
# cluster (run-level) bootstrap for the checkpoint-level Spearman drop-vs-follow
rng = np.random.default_rng(0)
per_run = []
for r in r1f:
    g = [h for h in r["history"] if h["acc_clean"] > 0.9]
    if g: per_run.append((np.array([h["acc_clean"] - h["acc_trunc"] for h in g]), np.array([h["follow"] for h in g])))
bs = []
for _ in range(2000):
    idx = rng.integers(0, len(per_run), len(per_run))
    d = np.concatenate([per_run[i][0] for i in idx]); f = np.concatenate([per_run[i][1] for i in idx])
    bs.append(spearmanr(d, f)[0])
d = np.concatenate([x[0] for x in per_run]); f = np.concatenate([x[1] for x in per_run])
A["ckpt_spearman_drop_follow"] = {"rho": float(spearmanr(d, f)[0]), "cluster_bootstrap_95ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_runs": len(per_run), "n_ckpt": int(len(d))}
# N-S+ checkpoints by run in regime (c)
A["regc_NminusSplus_ckpts_by_run"] = {r["tag"]: int(sum(quad(h["acc_clean"], h["acc_trunc"], h["follow"]) == "N-S+" for h in r["history"] if h["acc_clean"] > 0.9)) for r in r1f if reg_c(r)}
# truncation verdict of the regime-(c) long runs at the originally planned 4000 steps
A["regc_long_quadrant_at_4000"] = {r["tag"]: [quad(h["acc_clean"], h["acc_trunc"], h["follow"]) for h in r["history"] if h["step"] == 4000][0] for r in r1f if reg_c(r) and r["steps"] == 8000}
# threshold sweep of the Table-2 asymmetry (round-1 checkpoints)
sw = {}
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    row = {}
    for key, cond in [("c", reg_c), ("eta>0", lambda r: r["eta"] > 0)]:
        g = [h for r in r1f if cond(r) for h in r["history"] if h["acc_clean"] > 0.9]
        q = [quad(h["acc_clean"], h["acc_trunc"], h["follow"], th) for h in g]
        row[key] = {"N-S+": round(q.count("N-S+") / len(q), 3), "N+S-": round(q.count("N+S-") / len(q), 3)}
    sw[str(th)] = row
A["threshold_sweep"] = sw
out["round1"] = A

# ------------------------------------------------------------------ B. per-model mechanistic rows
def load(fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else []
old = load("r2_models_old.json"); new = load("r2_models_new.json")
for r in old + new:
    r["bq"] = quad(r["acc_clean"], r["acc_trunc"], r["follow"]); r["mq"] = mquad(r)
B = {}
oldf = [r for r in old if r["regime"] == "free"]
# probe ranges correction (M2) come from final_table.json
ft = json.load(open(os.path.join(R, "final_table.json")))
fr = [r for r in ft if r["regime"] == "free"]
B["probe_L2_true_s_followers"] = sorted([(r["tag"], round(r["L2_true_s"], 3)) for r in fr if r["follow"] > 0.5], key=lambda x: x[1])
B["probe_L2_true_s_ignorers"] = sorted([(r["tag"], round(r["L2_true_s"], 3)) for r in fr if r["follow"] <= 0.5], key=lambda x: x[1])
out["probes_r1"] = B

def summarize_models(rows, label):
    S = {}
    if not rows: return S
    f = np.array([r["follow"] for r in rows]); t = np.array([r["acc_trunc"] for r in rows])
    cc = np.array([r["acc_cot_cut"] for r in rows]); ic = np.array([r["acc_internal_cut_strict"] for r in rows])
    ac = np.array([r["acc_clean"] for r in rows])
    # full 2x2 tests-vs-references (W5), Spearman and Kendall-free agreement of binary verdicts
    def sp(a, b):
        v = spearmanr(a, b)[0]; return None if np.isnan(v) else round(float(v), 3)
    S["n"] = len(rows)
    S["spearman"] = {"follow_vs_strictcut": sp(f, ic), "follow_vs_cotcut": sp(f, cc),
                     "trunc_vs_cotcut": sp(t, cc), "trunc_vs_strictcut": sp(t, ic)}
    S["mae"] = {"follow_vs_strictcut": round(float(np.abs(f - ic).mean()), 3), "trunc_vs_cotcut": round(float(np.abs(t - cc).mean()), 3)}
    bN = np.array([r["bq"][:2] == "N+" for r in rows]); bS = np.array([r["bq"][2:] == "S+" for r in rows])
    mN = np.array([r["mq"][:2] == "N+" for r in rows]); mS = np.array([r["mq"][2:] == "S+" for r in rows])
    S["verdict_agreement"] = {"trunc_vs_mechN": round(float((bN == mN).mean()), 3), "corrupt_vs_mechS": round(float((bS == mS).mean()), 3),
                              "trunc_vs_mechS": round(float((bN == mS).mean()), 3), "corrupt_vs_mechN": round(float((bS == mN).mean()), 3)}
    S["mismatch_trunc_mechN"] = [(r["tag"], r["bq"], r["mq"]) for r in rows if (r["bq"][:2] == "N+") != (r["mq"][:2] == "N+")]
    S["mismatch_corrupt_mechS"] = [(r["tag"], r["bq"], r["mq"]) for r in rows if (r["bq"][2:] == "S+") != (r["mq"][2:] == "S+")]
    # bimodality: fraction of models with follow in (0.1, 0.9)
    S["frac_follow_intermediate"] = round(float(((f > 0.1) & (f < 0.9)).mean()), 3)
    return S
out["mech_r1_final_free"] = summarize_models(oldf, "r1")
out["mech_r1_controls"] = [{k: r[k] for k in ["tag", "acc_trunc", "follow", "acc_cot_cut", "acc_internal_cut_strict", "trunc_SEP", "trunc_meanemb", "trunc_zeroemb"]} | {"D_IIA_p013_corrupt": r["D_corrupt"]["p013"]["s_aligned"], "D_IIA_p3_corrupt": r["D_corrupt"]["p3"]["s_aligned"]} for r in old if r["regime"] != "free"]

# ------------------------------------------------------------------ C. round-2 pre-specified sweep
r2 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "runs_r2", "*.json")))]
regs = [(e, d) for (e, d) in [(0.0, 0.3), (0.2, 0.0)] if any(r['eta'] == e and r['delta'] == d for r in r2)]  # amended design (sweep_r2_spec.json)
Csw = {"n_runs": len(r2)}
quads = ["N+S+", "N-S+", "N+S-", "N-S-"]
runlevel = {}
for (e, dl) in regs:
    for step in (4000, 8000, 16000):
        rows = [r for r in new if r["eta"] == e and r["delta"] == dl and r["step"] == step]
        if not rows: continue
        k = f"e{e:g}_d{dl:g}_step{step}"
        n = len(rows)
        runlevel[k] = {"n": n, "behavioural": {q: cp(sum(r["bq"] == q for r in rows), n) for q in quads},
                       "mechanistic": {q: cp(sum(r["mq"] == q for r in rows), n) for q in quads},
                       "behav_disagree": cp(sum(r["bq"] in ("N-S+", "N+S-") for r in rows), n),
                       "follow": [round(r["follow"], 3) for r in sorted(rows, key=lambda r: r["seed"])],
                       "trunc": [round(r["acc_trunc"], 3) for r in sorted(rows, key=lambda r: r["seed"])],
                       "clean": [round(r["acc_clean"], 3) for r in sorted(rows, key=lambda r: r["seed"])],
                       "cotcut": [round(r["acc_cot_cut"], 3) for r in sorted(rows, key=lambda r: r["seed"])],
                       "strictcut": [round(r["acc_internal_cut_strict"], 3) for r in sorted(rows, key=lambda r: r["seed"])],
                       "bq": [r["bq"] for r in sorted(rows, key=lambda r: r["seed"])],
                       "mq": [r["mq"] for r in sorted(rows, key=lambda r: r["seed"])]}
Csw["runlevel"] = runlevel
# per-run checkpoint-level disagreement (each run contributes one number)
pr = {}
for (e, dl) in regs:
    rs = [r for r in r2 if r["eta"] == e and r["delta"] == dl]
    vals = {"NmSp": [], "NpSm": []}
    for r in rs:
        g = [h for h in r["history"] if h["acc_clean"] > 0.9 and h["step"] <= 8000]
        if not g: continue
        q = [quad(h["acc_clean"], h["acc_trunc"], h["follow"]) for h in g]
        vals["NmSp"].append(q.count("N-S+") / len(q)); vals["NpSm"].append(q.count("N+S-") / len(q))
    def mb(v):
        if not v: return None
        v = np.array(v); b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        return [round(float(v.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3), int(len(v)), int((v > 0).sum())]
    pr[f"e{e:g}_d{dl:g}"] = {"NmSp_mean_ci_nruns_nruns_with_any": mb(vals["NmSp"]), "NpSm_mean_ci_nruns_nruns_with_any": mb(vals["NpSm"])}
Csw["per_run_ckpt_disagreement_steps_le_8000"] = pr
# reliance first? per run: first checkpoint follow and whether ever reliant
first = {}
for r in r2:
    fl = [h["follow"] for h in r["history"]]
    first[r["tag"]] = {"max_follow": round(max(fl), 3), "follow_at_250": round(fl[0], 3),
                       "ever_reliant": bool(max(fl) >= 0.5)}
Csw["ever_reliant_by_regime"] = {f"e{e:g}_d{dl:g}": cp(sum(first[r["tag"]]["ever_reliant"] for r in r2 if r["eta"] == e and r["delta"] == dl), sum(1 for r in r2 if r["eta"] == e and r["delta"] == dl)) for (e, dl) in regs}
Csw["never_reliant_runs"] = [t for t, v in first.items() if not v["ever_reliant"]]
# Eq. 2 vs trivial on the new sweep, and run-level detection of disagreement by the Eq.2 residual
Csw["eq2_vs_trivial_excl_c"] = eq2_vs_trivial(r2, lambda r: not reg_c(r))
Csw["eq2_vs_trivial_c"] = eq2_vs_trivial(r2, reg_c)
# model-level soft residual as a detector of behavioural N/S disagreement (final models at 8000 steps)
det = {}
for src, rows in [("r2_step8000", [r for r in new if r["step"] == 8000]), ("r1_final", oldf)]:
    if not rows: continue
    res = np.array([abs(r["soft_trunc"] - r["soft_pred_trunc"]) for r in rows])
    dis = np.array([r["bq"] in ("N-S+", "N+S-") for r in rows])
    flag = res > 0.2
    tp = int((flag & dis).sum()); fp = int((flag & ~dis).sum()); fn = int((~flag & dis).sum())
    det[src] = {"n": len(rows), "n_disagree": int(dis.sum()), "n_flag": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
                "precision": None if tp + fp == 0 else round(tp / (tp + fp), 3), "recall": None if tp + fn == 0 else round(tp / (tp + fn), 3)}
Csw["soft_residual_detector"] = det
# mechanistic summaries on the new sweep (8000-step models; all regimes)
Csw["mech_step8000"] = summarize_models([r for r in new if r["step"] == 8000], "r2")
# removal formats: per regime, mean over runs at 8000 (and 16000 for c)
rf = {}
for (e, dl) in regs:
    for step in (8000, 16000):
        rows = [r for r in new if r["eta"] == e and r["delta"] == dl and r["step"] == step]
        if not rows: continue
        rf[f"e{e:g}_d{dl:g}_step{step}"] = {k: [round(r[k], 3) for r in sorted(rows, key=lambda r: r["seed"])] for k in ["trunc_MASK", "trunc_SEP", "trunc_meanemb", "trunc_zeroemb"]}
Csw["removal_formats"] = rf
# N+S- models: symmetric mechanistic check
Csw["NplusSminus_models"] = [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ["tag", "acc_clean", "acc_trunc", "follow", "ignore", "acc_cot_cut", "acc_internal_cut_strict", "trunc_SEP", "trunc_meanemb", "trunc_zeroemb", "mq"]} for r in new + oldf if r["bq"] == "N+S-"]
# presence gating: models with trunc > 0.7 and CoT-cut < 0.1
def gate_summary(r):
    g = r["gating"]; curve = g["interp"]
    ig = [c[2] for c in curve]
    lam50 = next((c[0] for c in curve if c[2] >= 0.5), None)
    return {"tag": r["tag"], "trunc": round(r["acc_trunc"], 3), "cotcut": round(r["acc_cot_cut"], 3), "follow": round(r["follow"], 3),
            "ignore_at_lambda": [round(x, 2) for x in ig], "follow_at_lambda": [round(c[1], 2) for c in curve], "lambda_ignore_ge_0.5": lam50,
            "patch_mask_into_corrupt_L1_ignore": round(g["patch_mask_into_corrupt_L1_ignore"], 3),
            "D_mask_IIA_p013": round(r["D_mask"]["p013"]["s_aligned"], 3)}
pg = [r for r in new + oldf if r["acc_trunc"] > 0.7 and r["acc_cot_cut"] < 0.1]
Csw["presence_gated"] = [gate_summary(r) for r in pg]
# stability 8000 -> 16000 in regime c
st = {}
for s in sorted(set(r["seed"] for r in new if reg_c(r))):
    st[s] = [next((r["bq"] for r in new if reg_c(r) and r["seed"] == s and r["step"] == k), None) for k in (4000, 8000, 16000)]
Csw["regc_quadrant_trajectory_4k_8k_16k"] = st
# D (interchange IIA at the c-free site set {0,1,3}, layer 1) vs behavioural/mechanistic measures
allm = [r for r in new if r["step"] == 8000] + oldf
def dval(r):
    return r["D_corrupt"]["p013"]["s_aligned"] if r["D_corrupt"]["base_acc"] > 0.9 else None
dv = [(dval(r), r) for r in allm if dval(r) is not None]
Csw["D_IIA_ignorers"] = {"n_models_with_internal_route_answering": len(dv),
                         "IIA_p013": [round(x, 3) for x, _ in dv],
                         "IIA_p3": [round(r["D_corrupt"]["p3"]["s_aligned"], 3) for _, r in dv],
                         "IIA_p3_probe22": [round(r["D_corrupt"]["p3_s22"]["s_aligned"], 3) for _, r in dv],
                         "IIA_p3_rand22": [round(r["D_corrupt"]["p3_rand22"]["s_aligned"], 3) for _, r in dv],
                         "trunc": [round(r["acc_trunc"], 3) for _, r in dv], "follow": [round(r["follow"], 3) for _, r in dv],
                         "spearman_IIA_vs_trunc": float(spearmanr([x for x, _ in dv], [r["acc_trunc"] for _, r in dv])[0]) if len(dv) > 3 else None,
                         "spearman_IIA_vs_ignore": float(spearmanr([x for x, _ in dv], [r["ignore"] for _, r in dv])[0]) if len(dv) > 3 else None}
out["round2"] = Csw
json.dump(out, open(os.path.join(R, "r2_stats.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------ figures
if runlevel:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    labs, keys = [], []
    for (e, dl) in regs:
        for step in (4000, 8000, 16000):
            k = f"e{e:g}_d{dl:g}_step{step}"
            if k in runlevel: labs.append(f"η={e:g},δ={dl:g}\n{step//1000}k"); keys.append(k)
    for ax, q, col in [(axes[0], "N-S+", "#ff7f0e"), (axes[1], "N+S-", "#1f77b4")]:
        x = np.arange(len(keys))
        m = [runlevel[k]["behavioural"][q][2] for k in keys]
        lo = [runlevel[k]["behavioural"][q][2] - runlevel[k]["behavioural"][q][3] for k in keys]
        hi = [runlevel[k]["behavioural"][q][4] - runlevel[k]["behavioural"][q][2] for k in keys]
        ax.bar(x, m, color=col, yerr=[lo, hi], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6); ax.set_ylim(0, 1)
        ax.set_title(f"run-level fraction of {q} models (10 seeds; 4 at 16k; 95% CP CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig5_runlevel.png"), dpi=160); plt.close(fig)
if pg:
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for r in pg:
        c = r["gating"]["interp"]
        ax.plot([x[0] for x in c], [x[2] for x in c], "-", color="#2ca02c", alpha=0.6)
        ax.plot([x[0] for x in c], [x[1] for x in c], "--", color="#1f77b4", alpha=0.6)
    ax.set_xlabel("λ (CoT-position embedding: (1-λ)E[s'] + λE[MASK])"); ax.set_ylabel("rate")
    ax.set_title("presence-gated models: solid=ignore (correct), dashed=follow", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig6_gating.png"), dpi=160); plt.close(fig)
print(json.dumps(out, indent=1, default=str)[:20000])
