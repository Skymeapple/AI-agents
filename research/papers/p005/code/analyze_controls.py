"""Analyse Experiments 4-5 (controls_*.json): crossing steps / lags of the certificate versus
margin, loss and neural-collapse statistics; best-fitting fixed margin threshold; per-point
AUROC of margin for certification.  Usage: python3 analyze_controls.py -> results/controls_summary.json"""
import os, json, glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAUS = np.round(np.arange(0.0, 40.01, 0.25), 2)


def cross(steps, v, thr):
    """sustained crossing: first checkpoint from which the statistic stays >= thr until the end of training."""
    v = np.asarray(v, float)
    if not (v[-1] >= thr):
        return None
    below = np.where(~(v >= thr))[0]
    return int(steps[below[-1] + 1]) if len(below) else int(steps[0])


def half_decline(steps, v):
    v = np.log(np.maximum(np.asarray(v, float), 1e-12))
    ip = int(np.argmax(v)); target = (v[ip] + v[-1]) / 2   # half of the log-decline from the peak to the end
    idx = np.where(v[ip:] <= target)[0]
    return int(steps[ip + idx[0]]) if len(idx) else None


def analyse(r):
    H = r["history"]
    steps = np.array([h["step"] for h in H])
    g = lambda k: np.array([h[k] if h.get(k) is not None else np.nan for h in H], float)
    out = {"arch": r["arch"], "frac": r["frac"], "seed": r["seed"], "wd": r["wd"], "steps": r["steps"]}
    fin = H[-1]
    out["final"] = {k: fin[k] for k in ["test_acc_full", "C", "C_vis", "M_q10", "M_q50", "M_abs2", "M_abs5", "M_abs10",
                                         "test_loss", "NC1_train", "NC1_test", "weight_norm", "train_margin_median", "test_margin_median"]}
    for thr in (0.5, 0.9):
        for k in ["test_acc_full", "C", "C_vis", "M_q10", "M_q50", "M_abs2", "M_abs5", "M_abs10"]:
            out[f"step_{k}_ge{thr}"] = cross(steps, g(k), thr)
    tl = g("test_loss")
    out["step_test_loss_le0.1"] = (int(steps[np.where(tl <= 0.1)[0][0]]) if (tl <= 0.1).any() else None)
    for k in ["NC1_test", "NC1_train", "test_loss"]:
        out[f"step_{k}_half_decline"] = half_decline(steps, g(k))
    # per-point checkpoints
    pp = [(h["step"], h["pp"]) for h in H if "pp" in h]
    ps = np.array([s for s, _ in pp])
    C_pp = np.array([np.mean(p["cert"]) for _, p in pp])
    Cv_pp = np.array([np.mean(p["cert_vis"]) for _, p in pp])
    Mgrid = np.array([[np.mean(np.array(p["margin"]) > t) for t in TAUS] for _, p in pp])   # (ckpt, tau)
    out["fit"] = {}
    for name, Cs in (("C", C_pp), ("C_vis", Cv_pp)):
        rmse = np.sqrt(((Mgrid - Cs[:, None]) ** 2).mean(0))
        j = int(np.argmin(rmse)); tau = float(TAUS[j])
        out["fit"][name] = {"tau_star": tau, "rmse": float(rmse[j]),
                            "step_Mtaustar_ge0.5": cross(ps, Mgrid[:, j], 0.5), "step_ge0.5_pp": cross(ps, Cs, 0.5)}
        # per-point AUROC (margin predicting certification among correct points) at the checkpoint where Cs is closest to .5
        k = int(np.argmin(np.abs(Cs - 0.5)))
        p = pp[k][1]
        corr = np.array(p["correct"]) == 1; c = np.array(p[("cert" if name == "C" else "cert_vis")])[corr]
        m = np.array(p["margin"])[corr]
        au = float(roc_auc_score(c, m)) if 0 < c.sum() < len(c) else None
        # agreement of the best single-threshold rule with the certificate at that checkpoint
        acc_rule = float(np.max([np.mean((m > t) == (c == 1)) for t in TAUS])) if len(c) else None
        out["fit"][name].update({"auroc_ckpt_step": int(pp[k][0]), "auroc_margin_predicts": au,
                                 "best_threshold_agreement": acc_rule, "C_at_ckpt": float(Cs[k])})
    out["series"] = {"step": steps.tolist(), **{k: g(k).tolist() for k in
                     ["test_acc_full", "C", "C_vis", "M_q10", "M_q50", "M_abs2", "M_abs5", "M_abs10", "test_loss", "NC1_test", "NC1_train",
                      "weight_norm", "test_margin_median", "train_margin_median"]}}
    return out


def ci(v):
    v = np.array([x for x in v if x is not None], float)
    if len(v) < 2:
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95": [float(v.mean() - h), float(v.mean() + h)],
            "min": float(v.min()), "max": float(v.max())}


if __name__ == "__main__":
    runs = [analyse(json.load(open(f))) for f in sorted(glob.glob(os.path.join(RES, "controls_*_s*_wd*.json")))]
    groups = {}
    for r in runs:
        groups.setdefault(f"{r['arch']}_f{r['frac']}_wd{r['wd']}", []).append(r)
    summ = {}
    for gname, rs in groups.items():
        s = {"n_runs": len(rs), "seeds": [r["seed"] for r in rs]}
        base = [r["step_test_acc_full_ge0.5"] for r in rs]
        s["n_grokked"] = int(sum(b is not None for b in base))
        for k in ["C", "C_vis", "M_q10", "M_q50", "M_abs2", "M_abs5", "M_abs10"]:
            lags = [(r[f"step_{k}_ge0.5"] - r["step_test_acc_full_ge0.5"]) if (r[f"step_{k}_ge0.5"] is not None and r["step_test_acc_full_ge0.5"] is not None) else None for r in rs]
            s[f"lag50_{k}"] = ci(lags); s[f"lag50_{k}_per_seed"] = lags
            s[f"never_{k}_ge0.5"] = int(sum(r[f"step_{k}_ge0.5"] is None for r in rs if r["step_test_acc_full_ge0.5"] is not None))
        for k in ["NC1_test_half_decline", "NC1_train_half_decline", "test_loss_half_decline"]:
            lags = [(r[f"step_{k}"] - r["step_test_acc_full_ge0.5"]) if (r[f"step_{k}"] is not None and r["step_test_acc_full_ge0.5"] is not None) else None for r in rs]
            s[f"lag_{k}"] = ci(lags); s[f"lag_{k}_per_seed"] = lags
        s["lag_test_loss_le0.1"] = ci([(r["step_test_loss_le0.1"] - r["step_test_acc_full_ge0.5"]) if (r["step_test_loss_le0.1"] is not None and r["step_test_acc_full_ge0.5"] is not None) else None for r in rs])
        s["step_test_acc_ge0.5"] = base
        for k in ["test_acc_full", "C", "C_vis", "M_q50", "M_abs5", "NC1_test", "test_loss", "test_margin_median", "train_margin_median"]:
            s[f"final_{k}"] = ci([r["final"][k] for r in rs])
        for name in ("C", "C_vis"):
            s[f"fit_{name}_tau_star"] = [r["fit"][name]["tau_star"] for r in rs]
            s[f"fit_{name}_rmse"] = ci([r["fit"][name]["rmse"] for r in rs])
            s[f"fit_{name}_auroc"] = ci([r["fit"][name]["auroc_margin_predicts"] for r in rs])
            s[f"fit_{name}_agree"] = ci([r["fit"][name]["best_threshold_agreement"] for r in rs])
        summ[gname] = s
    json.dump({"groups": summ, "runs": runs}, open(os.path.join(RES, "controls_summary.json"), "w"), indent=1)
    for g, s in summ.items():
        print("==", g, "n", s["n_runs"], "grokked", s["n_grokked"])
        for k in ["C", "C_vis", "M_q10", "M_q50", "M_abs2", "M_abs5", "M_abs10"]:
            print(f"  lag50 {k:8s}", s[f"lag50_{k}_per_seed"], "never:", s[f"never_{k}_ge0.5"])
        for k in ["NC1_test_half_decline", "NC1_train_half_decline", "test_loss_half_decline"]:
            print(f"  lag {k}", s[f"lag_{k}_per_seed"])
        print("  final acc", round(s["final_test_acc_full"]["mean"], 3), "C", s["final_C"], "\n  Cvis", s["final_C_vis"])
        print("  tau*", s["fit_C_tau_star"], "rmse", s["fit_C_rmse"].get("mean"), "auroc", s["fit_C_auroc"].get("mean"), "agree", s["fit_C_agree"].get("mean"))
        print("  tau*vis", s["fit_C_vis_tau_star"], "rmse", s["fit_C_vis_rmse"].get("mean"), "auroc", s["fit_C_vis_auroc"].get("mean"))


def global_tau():
    """One margin threshold for all runs of an architecture: does it reproduce the certificate's crossing step?"""
    res = {}
    files = sorted(glob.glob(os.path.join(RES, "controls_*_s*_wd*.json")))
    R = [json.load(open(f)) for f in files]
    for arch in sorted(set(r["arch"] for r in R)):
        rs = [r for r in R if r["arch"] == arch]
        data = []
        for r in rs:
            pp = [(h["step"], h["pp"]) for h in r["history"] if "pp" in h]
            ps = np.array([s for s, _ in pp])
            C = np.array([np.mean(p["cert"]) for _, p in pp]); Cv = np.array([np.mean(p["cert_vis"]) for _, p in pp])
            A = np.array([np.mean(p["correct"]) for _, p in pp])
            M = np.array([[np.mean(np.array(p["margin"]) > t) for t in TAUS] for _, p in pp])
            data.append((r, ps, C, Cv, A, M))
        out = {}
        for name, ci_ in (("C", 2), ("C_vis", 3)):
            sse = sum(((d[5] - d[ci_][:, None]) ** 2).sum(0) for d in data)
            n = sum(len(d[1]) for d in data)
            j = int(np.argmin(sse)); tau = float(TAUS[j])
            per = []
            for d in data:
                r, ps = d[0], d[1]
                sa, sc, sm = cross(ps, d[4], 0.5), cross(ps, d[ci_], 0.5), cross(ps, d[5][:, j], 0.5)
                per.append({"frac": r["frac"], "wd": r["wd"], "seed": r["seed"], "step_acc50": sa, "step_cert50": sc, "step_Mtau50": sm,
                            "lag_cert": (sc - sa) if (sc is not None and sa is not None) else None,
                            "lag_Mtau": (sm - sa) if (sm is not None and sa is not None) else None})
            lc = np.array([p["lag_cert"] for p in per if p["lag_cert"] is not None and p["lag_Mtau"] is not None], float)
            lm = np.array([p["lag_Mtau"] for p in per if p["lag_cert"] is not None and p["lag_Mtau"] is not None], float)
            out[name] = {"tau_global": tau, "rmse_global": float(np.sqrt(sse[j] / n)), "per_run": per,
                         "n_pairs": int(len(lc)),
                         "pearson_lag_cert_vs_lag_Mtau": float(np.corrcoef(lc, lm)[0, 1]) if len(lc) > 2 else None,
                         "mean_abs_diff_steps": float(np.mean(np.abs(lc - lm))) if len(lc) else None,
                         "mean_diff_cert_minus_Mtau": float(np.mean(lc - lm)) if len(lc) else None}
        res[arch] = out
    return res


if __name__ == "__main__":
    G = global_tau()
    S = json.load(open(os.path.join(RES, "controls_summary.json")))
    S["global_tau"] = G
    json.dump(S, open(os.path.join(RES, "controls_summary.json"), "w"), indent=1)
    for a, o in G.items():
        for n, v in o.items():
            print(a, n, "tau", v["tau_global"], "rmse", round(v["rmse_global"], 3), "r", v["pearson_lag_cert_vs_lag_Mtau"],
                  "MAD", v["mean_abs_diff_steps"], "meandiff", v["mean_diff_cert_minus_Mtau"], "n", v["n_pairs"])
            print("   ", [(p["frac"], p["wd"], p["seed"], p["lag_cert"], p["lag_Mtau"]) for p in v["per_run"]])
