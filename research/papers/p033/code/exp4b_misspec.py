"""Experiment 4b (added in revision): estimating the ROC tail R(alpha0) of an
evaluation-awareness discriminator under (a) correct and (b) misspecified models,
and the test-set size needed to hit a stated precision target.

Ground truths for the discriminator score s (E = evaluation items, D = deployment items):
  binormal      E ~ N(0,1), D ~ N(a,1), AUC 0.83                         (control)
  chat_agentic  50/50 mixture of a 'chat' subpopulation (within-group AUC 0.74) and an
                'agentic' subpopulation (within-group AUC 0.92) whose scores are
                shifted by +s; these within-group AUCs are illustrative values inside
                the ranges reported by Needham et al. (arXiv:2505.23836, v2+:
                chat 0.71-0.77, agentic 0.88-0.95). s is set so pooled AUC = 0.83.
  obvious_cue   D = (1-eps) N(d,1) + eps N(5,1): a 5% subset of deployment items carry an
                unmistakable 'real' cue. E ~ N(0,1). d set so AUC = 0.83.
  eval_lookalike E = (1-eps) N(0,1) + eps N(d,1): 2% of evaluation items are
                indistinguishable from deployment. D ~ N(d,1). d set so AUC = 0.83.
The truth is the ROC of the score itself at alpha0 = 1e-3 and 1e-4 (computed exactly).
Estimators: empirical ROC; equal-variance (EV) binormal fit from empirical AUC;
unequal-variance (UV) probit fit on FPR in [0.02, 0.98]; tail-weighted UV fit on
FPR in [min(5/n, 0.01), 0.10].
Precision targets: P(estimate within a factor 2 of truth) >= 0.9, and within a factor 1.25.
Run: python3 exp4b_misspec.py   (~6 min, 1 core)
"""
import json, os
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS
from theory import binormal, auc_to_a

A0 = 1e-3
A0S = [1e-3, 1e-4]
AUC = 0.83
NS = [450, 1000, 2000, 5000, 10000, 20000, 50000]


class Mix:
    def __init__(self, comps):  # list of (w, mean, sd)
        self.c = comps

    def sf(self, t):
        return sum(w * norm.sf((t - m) / s) for w, m, s in self.c)

    def sample(self, n, rng):
        w = np.array([c[0] for c in self.c])
        k = rng.choice(len(self.c), size=n, p=w / w.sum())
        m = np.array([c[1] for c in self.c])[k]
        s = np.array([c[2] for c in self.c])[k]
        return rng.normal(m, s)


def auc_of(E, D):
    t = np.linspace(-12, 18, 60001)
    # AUC = P(sD > sE) = int f_E(t) P_D(s > t) dt
    fE = -np.gradient(E.sf(t), t)
    return float(np.trapezoid(fE * D.sf(t), t))


def roc_at(E, D, a0):
    tau = brentq(lambda t: E.sf(t) - a0, -20, 30)
    return float(D.sf(tau))


def build_truths():
    T = {}
    a = auc_to_a(AUC)
    T["binormal"] = (Mix([(1, 0, 1)]), Mix([(1, a, 1)]))
    dc, da = auc_to_a(0.74), auc_to_a(0.92)

    def ca(s):
        return Mix([(0.5, 0, 1), (0.5, s, 1)]), Mix([(0.5, dc, 1), (0.5, s + da, 1)])
    s = brentq(lambda s: auc_of(*ca(s)) - AUC, -0.5, 0.5)
    T["chat_agentic"] = ca(s)

    def oc(d):
        return Mix([(1, 0, 1)]), Mix([(0.95, d, 1), (0.05, 5.0, 1)])
    d = brentq(lambda d: auc_of(*oc(d)) - AUC, -2, 3)
    T["obvious_cue"] = oc(d)

    def el(d):
        return Mix([(0.98, 0, 1), (0.02, d, 1)]), Mix([(1, d, 1)])
    d2 = brentq(lambda d: auc_of(*el(d)) - AUC, 0, 4)
    T["eval_lookalike"] = el(d2)
    params = dict(chat_agentic_shift=float(s), obvious_cue_d=float(d), eval_lookalike_d=float(d2))
    return T, params


def emp_auc(se, sd):
    allv = np.concatenate([se, sd])
    ranks = allv.argsort().argsort() + 1
    return (ranks[len(se):].sum() - len(sd) * (len(sd) + 1) / 2) / (len(se) * len(sd))


def emp_R(se, sd, a):
    k = int(np.floor(a * len(se)))
    thr = np.sort(se)[::-1][min(k, len(se) - 1)]
    return float(np.mean(sd > thr))


def probit_fit(se, sd, lo, hi, m=25):
    fq = np.geomspace(lo, hi, m)
    thr = np.quantile(se, 1 - fq)
    fpr = np.array([np.mean(se > t) for t in thr])
    tpr = np.array([np.mean(sd > t) for t in thr])
    ok = (tpr > 0.5 / len(sd)) & (tpr < 1 - 0.5 / len(sd)) & (fpr > 0) & (fpr < 1)
    if ok.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(norm.ppf(fpr[ok]), norm.ppf(tpr[ok]), 1)
    return a, b


def estimates(se, sd, a0):
    n = len(se)
    out = {"emp": emp_R(se, sd, a0)}
    au = emp_auc(se, sd)
    out["ev"] = float(binormal(a0, auc_to_a(min(au, 0.9999)), 1.0))
    a, b = probit_fit(se, sd, 0.02, 0.98)
    out["uv"] = float(binormal(a0, a, b))
    a, b = probit_fit(se, sd, min(5.0 / n, 0.01), 0.10)
    out["uv_tail"] = float(binormal(a0, a, b)) if np.isfinite(a) else np.nan
    return out


def main():
    rng = np.random.default_rng(4242)
    T, params = build_truths()
    res = {"params": params, "truth": {}, "auc": {}, "sims": {}}
    for name, (E, D) in T.items():
        res["truth"][name] = roc_at(E, D, A0)
        res["truth_1e-4"] = res.get("truth_1e-4", {})
        res["truth_1e-4"][name] = roc_at(E, D, 1e-4)
        res["auc"][name] = auc_of(E, D)
    # binormal slope variants for the precision target (correct specification)
    for b in [0.5, 1.5]:
        a = auc_to_a(AUC, b)
        T[f"binormal_b{b}"] = (Mix([(1, 0, 1)]), Mix([(1, a / b, 1 / b)]))
        res["truth"][f"binormal_b{b}"] = float(binormal(A0, a, b))
        res["truth_1e-4"][f"binormal_b{b}"] = float(binormal(1e-4, a, b))
        res["auc"][f"binormal_b{b}"] = AUC
    print(json.dumps(res["truth"], indent=1), json.dumps(res["auc"], indent=1), params)
    for name, (E, D) in T.items():
        for n in NS:
            reps = 400 if n <= 5000 else (150 if n <= 20000 else 60)
            est = {(a0, k): [] for a0 in A0S for k in ["emp", "ev", "uv", "uv_tail"]}
            for _ in range(reps):
                se, sd = E.sample(n, rng), D.sample(n, rng)
                for a0 in A0S:
                    for k, v in estimates(se, sd, a0).items():
                        est[(a0, k)].append(v)
            for a0 in A0S:
                truth = res["truth"][name] if a0 == A0 else res["truth_1e-4"][name]
                row = {"truth": truth, "reps": reps}
                for k in ["emp", "ev", "uv", "uv_tail"]:
                    v = np.array(est[(a0, k)]); v = v[np.isfinite(v)]
                    ratio = v / truth
                    row[k] = dict(median=float(np.median(v)), q05=float(np.quantile(v, 0.05)),
                                  q95=float(np.quantile(v, 0.95)),
                                  median_ratio=float(np.median(ratio)),
                                  p_within_2x=float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
                                  p_within_1p25x=float(np.mean((ratio >= 0.8) & (ratio <= 1.25))))
                key = f"{name}_n{n}" if a0 == A0 else f"{name}_n{n}_a1e-4"
                res["sims"][key] = row
                print(key, {k: (round(row[k]["median_ratio"], 3), round(row[k]["p_within_2x"], 2),
                                round(row[k]["p_within_1p25x"], 2)) for k in ["emp", "ev", "uv", "uv_tail"]}, flush=True)
    # smallest n meeting the target, per truth and estimator
    target = {}
    for suf in ["", "_a1e-4"]:
        for crit in ["p_within_2x", "p_within_1p25x"]:
            for name in T:
                for k in ["emp", "ev", "uv", "uv_tail"]:
                    ok = [n for n in NS if res["sims"][f"{name}_n{n}{suf}"][k][crit] >= 0.9]
                    target[f"{name}_{k}_{crit}{suf or '_a1e-3'}"] = ok[0] if ok else None
    res["n_for_target_within2x_p90"] = target
    print(json.dumps(target, indent=1))
    json.dump(res, open(os.path.join(RESULTS, "exp4b_misspec.json"), "w"), indent=1)

    names = ["binormal", "chat_agentic", "obvious_cue", "eval_lookalike"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    cols = {"emp": "#1f77b4", "ev": "#2ca02c", "uv": "#d62728", "uv_tail": "#9467bd"}
    labs = {"emp": "empirical", "ev": "eq.-var. fit", "uv": "uneq.-var. fit", "uv_tail": "tail probit fit"}
    for i, name in enumerate(names):
        for j, k in enumerate(cols):
            rows = [res["sims"][f"{name}_n{n}"][k] for n in NS]
            xs = np.array(NS) * (1 + 0.07 * (j - 1.5))
            med = np.array([r["median"] for r in rows])
            ax[i].errorbar(xs, med, yerr=[med - [r["q05"] for r in rows], [r["q95"] for r in rows] - med],
                           fmt="o-", ms=3, capsize=2, color=cols[k], label=labs[k])
        ax[i].axhline(res["truth"][name], color="k", ls="--", lw=1, label="truth")
        ax[i].set_xscale("log"); ax[i].set_yscale("log")
        ax[i].set_title(f"{name} (AUC {res['auc'][name]:.2f})", fontsize=9)
        ax[i].set_xlabel("items per class n")
    ax[0].set_ylabel("estimate of R(1e-3) (median, 90% band)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig4b_misspec.png"), dpi=150)


if __name__ == "__main__":
    main()
