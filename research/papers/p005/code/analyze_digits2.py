"""Summarise Experiment 6 (results/digits2.json): per-seed means and 95% t-intervals, paired
score differences, and a leave-one-seed-out logistic-regression test of whether hull features
add to the logit margin for error detection.  -> results/digits2_summary.json"""
import os, json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(__file__), "..", "results")
D = json.load(open(os.path.join(RES, "digits2.json")))
runs, perts = D["runs"], D["perts"]
SC = ["msp", "logit_margin", "energy", "mahalanobis", "knn1", "knn10", "hull_dist", "hull_contrast", "cert_slack", "certvis_slack"]


def ci(v):
    v = np.array([x for x in v if x is not None], float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return {"n": 1, "mean": float(v[0])}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "lo": float(v.mean() - h), "hi": float(v.mean() + h)}


def zfeat(r, model, pert, names):
    """features z-scored with the statistics of the same seed's *unshifted* test set (unlabelled)."""
    ref = r["perts"]["none"]["models"][model]["scores"]
    cur = r["perts"][pert]["models"][model]["scores"]
    cols = []
    for n in names:
        a, b = np.array(ref[n]), np.array(cur[n])
        cols.append((b - a.mean()) / (a.std() + 1e-12))
    return np.stack(cols, 1)


out = {}
for model in ["mlp", "cnn"]:
    for pert in perts:
        blks = [r["perts"][pert]["models"][model] for r in runs]
        s = {k: ci([b[k] for b in blks]) for k in ["acc", "I_pen", "C_pred", "Cvis_pred", "acc_cert", "acc_uncert", "acc_certvis", "acc_uncertvis"]}
        s["I0"] = ci([r["perts"][pert]["I0"] for r in runs])
        s["n_err_total"] = int(sum(b["n_err"] for b in blks))
        s["n_cert_total"] = int(sum(b["n_cert"] for b in blks))
        s["auroc"] = {k: ci([b["auroc"][k] for b in blks]) for k in SC}
        for k in ["hull_contrast", "certvis_slack", "knn10"]:
            d = [b["auroc"][k] - b["auroc"]["logit_margin"] for b in blks if b["auroc"][k] is not None]
            s[f"diff_{k}_minus_logit_margin"] = ci(d)
            s[f"diff_{k}_minus_logit_margin_p"] = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else None
        # leave-one-seed-out logistic regression
        lo = {}
        for fs_name, fs in {"margin": ["logit_margin"], "margin+hull_contrast": ["logit_margin", "hull_contrast"],
                            "margin+certvis": ["logit_margin", "certvis_slack"], "margin+knn10": ["logit_margin", "knn10"]}.items():
            aus = []
            for i in range(len(runs)):
                Xtr = np.vstack([zfeat(runs[j], model, pert, fs) for j in range(len(runs)) if j != i])
                ytr = np.concatenate([runs[j]["perts"][pert]["models"][model]["err"] for j in range(len(runs)) if j != i])
                Xte = zfeat(runs[i], model, pert, fs); yte = np.array(runs[i]["perts"][pert]["models"][model]["err"])
                if 0 < ytr.sum() < len(ytr) and 0 < yte.sum() < len(yte):
                    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                    aus.append(float(roc_auc_score(yte, clf.decision_function(Xte))))
            lo[fs_name] = {"per_seed": aus, **(ci(aus) or {})}
        base = np.array(lo["margin"]["per_seed"])
        for k in ["margin+hull_contrast", "margin+certvis", "margin+knn10"]:
            d = np.array(lo[k]["per_seed"]) - base
            lo[k]["gain_over_margin"] = ci(d)
            lo[k]["gain_p"] = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else None
        s["loso_lr"] = lo
        out[f"{model}/{pert}"] = s
out["padding"] = {a: {k: ci([r["padding"]["by_scale"][a][k] for r in runs]) for k in ["acc", "max_abs_logit_change", "pad_norm_ratio", "C_pred_padded", "Cvis_pred_padded"]}
                  for a in runs[0]["padding"]["by_scale"]}
out["padding"]["orig"] = {"C_pred": ci([r["padding"]["C_pred_orig"] for r in runs]), "Cvis_pred": ci([r["padding"]["Cvis_pred_orig"] for r in runs])}
json.dump(out, open(os.path.join(RES, "digits2_summary.json"), "w"), indent=1)

f = lambda c: "--" if c is None else (f"{c['mean']:.3f}" + (f"±{c['sd']:.3f}" if "sd" in c else ""))
for key, s in out.items():
    if key == "padding":
        continue
    print(f"== {key}: acc {f(s['acc'])} I0 {f(s['I0'])} Ipen {f(s['I_pen'])} C {f(s['C_pred'])} Cvis {f(s['Cvis_pred'])} "
          f"acc_cert {f(s['acc_cert'])} acc_unc {f(s['acc_uncert'])} acc_certvis {f(s['acc_certvis'])} acc_uncvis {f(s['acc_uncertvis'])} errs {s['n_err_total']}")
    print("   AUROC:", {k: f(v) for k, v in s["auroc"].items()})
    print("   diff HC-LM", f(s["diff_hull_contrast_minus_logit_margin"]), "p", s["diff_hull_contrast_minus_logit_margin_p"],
          "| LOSO:", {k: (f(v), f(v.get("gain_over_margin")) if "gain_over_margin" in v else "", v.get("gain_p")) for k, v in s["loso_lr"].items()})
print(json.dumps(out["padding"], indent=0)[:1500])
