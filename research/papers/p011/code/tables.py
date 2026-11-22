"""Supplementary statistics used in the paper's Results section (reads results/final_table.json and runs/*.json).
Writes results/supp_stats.json and results/supp_table.md."""
import os, json, glob
import numpy as np
from scipy.stats import spearmanr
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows = json.load(open(os.path.join(R, "final_table.json")))
fr = [r for r in rows if r["regime"] == "free"]
out = {}
def quad(r):
    N = r["acc_clean"] - r["acc_trunc"]; S = r["follow"]
    return ("N+" if N > 0.5 else "N-") + ("S+" if S > 0.5 else "S-")
for r in rows: r["quadrant"] = quad(r)
out["final_free_n"] = len(fr)
out["final_free_quadrants"] = {q: sum(r["quadrant"] == q for r in fr) for q in ["N+S+", "N-S+", "N+S-", "N-S-"]}
out["final_free_verdict_disagree"] = sum(r["quadrant"] in ("N-S+", "N+S-") for r in fr)
# Prop-3 regimes (free models)
def grp(cond):
    g = [r for r in fr if cond(r)]
    return {"n": len(g), "follow_mean": float(np.mean([r["follow"] for r in g])),
            "follow_values": [round(r["follow"], 3) for r in g],
            "trunc_values": [round(r["acc_trunc"], 3) for r in g],
            "quadrants": [r["quadrant"] for r in g]}
out["regime_eta_ge_0.1"] = grp(lambda r: r["eta"] >= 0.1)
out["regime_eta_0.05"] = grp(lambda r: r["eta"] == 0.05)
out["regime_eta0_delta0.3"] = grp(lambda r: r["eta"] == 0 and r["delta"] == 0.3)
out["regime_eta0_delta0"] = grp(lambda r: r["eta"] == 0 and r["delta"] == 0)
# mechanistic agreement (Spearman) on final free models
f = np.array([r["follow"] for r in fr]); ic = np.array([r["acc_internal_cut_strict"] for r in fr])
t = np.array([r["acc_trunc"] for r in fr]); cc = np.array([r["acc_cot_cut"] for r in fr])
p2 = np.array([r["L2_true_s"] for r in fr])
out["spearman_follow_vs_strict_internal_cut"] = list(map(float, spearmanr(f, ic)))
out["spearman_trunc_vs_cot_cut"] = list(map(float, spearmanr(t, cc)))
out["spearman_follow_vs_probe_true_s_L2"] = list(map(float, spearmanr(f, p2)))
out["spearman_trunc_vs_probe_true_s_L2"] = list(map(float, spearmanr(t, p2)))
# presence-gated models: truncation acc high but CoT-route cut acc at chance
pg = [r for r in fr if r["acc_trunc"] > 0.7 and r["acc_cot_cut"] < 0.1]
out["presence_gated_tags"] = [r["tag"] for r in pg]
# cross-test (soft) residuals
res = np.array([r["soft_trunc"] - r["soft_pred_trunc"] for r in fr])
out["soft_residual_abs_mean"] = float(np.abs(res).mean())
out["soft_residual_large_tags"] = [r["tag"] for r, e in zip(fr, res) if abs(e) > 0.2]
out["soft_residual_large_quadrants"] = [r["quadrant"] for r, e in zip(fr, res) if abs(e) > 0.2]
out["soft_residual_abs_mean_excluding_large"] = float(np.abs(res[np.abs(res) <= 0.2]).mean())
# checkpoint-level disagreement, free models only
pts = []
for fn in glob.glob(os.path.join(R, "runs", "*.json")):
    rec = json.load(open(fn))
    if rec["regime"] != "free": continue
    pts += [h for h in rec["history"] if h["acc_clean"] > 0.9]
tv = np.array([(h["acc_clean"] - h["acc_trunc"]) > 0.5 for h in pts]); cv = np.array([h["follow"] > 0.5 for h in pts])
out["ckpt_free_n"] = len(pts)
out["ckpt_free_disagree_frac"] = float((tv != cv).mean())
out["ckpt_free_NminusSplus_frac"] = float(((~tv) & cv).mean())
out["ckpt_free_NplusSminus_frac"] = float((tv & ~cv).mean())
# continuous: correlation between truncation drop and follow over checkpoints
d = np.array([h["acc_clean"] - h["acc_trunc"] for h in pts]); fl = np.array([h["follow"] for h in pts])
out["ckpt_free_spearman_drop_vs_follow"] = list(map(float, spearmanr(d, fl)))
json.dump(out, open(os.path.join(R, "supp_stats.json"), "w"), indent=1)
keys = ["tag", "quadrant", "acc_trunc", "follow", "ignore", "other", "acc_cot_cut", "acc_internal_cut_strict", "L2_true_s", "soft_trunc", "soft_pred_trunc"]
with open(os.path.join(R, "supp_table.md"), "w") as fh:
    fh.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
    for r in rows:
        fh.write("| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |\n")
print(json.dumps(out, indent=1))

# checkpoint-level disagreement split by training regime (free models)
byreg = {}
for key, cond in [("eta0_delta0.3", lambda r: r["eta"] == 0 and r["delta"] == 0.3),
                  ("eta0_delta0", lambda r: r["eta"] == 0 and r["delta"] == 0), ("eta>0", lambda r: r["eta"] > 0)]:
    P_ = []
    for fn in glob.glob(os.path.join(R, "runs", "*.json")):
        rec = json.load(open(fn))
        if rec["regime"] == "free" and cond(rec):
            P_ += [h for h in rec["history"] if h["acc_clean"] > 0.9]
    a_ = np.array([(h["acc_clean"] - h["acc_trunc"]) > 0.5 for h in P_]); b_ = np.array([h["follow"] > 0.5 for h in P_])
    byreg[key] = dict(n=len(P_), disagree=float((a_ != b_).mean()), NminusSplus=float(((~a_) & b_).mean()),
                      NplusSminus=float((a_ & ~b_).mean()))
json.dump(byreg, open(os.path.join(R, "ckpt_disagree_by_regime.json"), "w"), indent=1)
print(byreg)

# cross-test identity (Eq. 2) on graded checkpoints, split by regime
import cotlab as C
split = {}
for key, cond in [("eta0_delta0.3", lambda r: r["eta"] == 0 and r["delta"] == 0.3), ("other_free", lambda r: not (r["eta"] == 0 and r["delta"] == 0.3))]:
    xs, ys = [], []
    for fn in glob.glob(os.path.join(R, "runs", "*.json")):
        rec = json.load(open(fn))
        if rec["regime"] != "free" or not cond(rec): continue
        for h in rec["history"]:
            if h["other"] >= 0.02:
                xs.append(C.additive_route_fit(h, 23)["pred_trunc"]); ys.append(h["acc_trunc"])
    xs, ys = np.array(xs), np.array(ys)
    split[key] = dict(n=len(xs), pearson_r=float(np.corrcoef(xs, ys)[0, 1]), mae=float(np.abs(xs - ys).mean()),
                      frac_resid_gt_03=float((np.abs(ys - xs) > 0.3).mean()))
json.dump(split, open(os.path.join(R, "crosstest_by_regime.json"), "w"), indent=1)
print(split)
