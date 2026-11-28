"""Figures and derived numbers for the revision (reads results/*.json; also computes the
label-efficiency ratio of Remark 5 from closed forms + Exp 3b RMSE). Run after exp_revision.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Params, closed_form, gold_peak

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = json.load(open(os.path.join(OUT, "revision_results.json")))
T = json.load(open(os.path.join(OUT, "theory_results.json")))

# ---- Fig 5: regret vs mean cumulative gold labels, W, q = 0.5 ----
d = R["R1_W"]["q=0.5"]
MS = [5, 10, 20, 40, 80]
series = [("cov-lev", "cov test, leverage audit"), ("cov-lev-win3+rollback", "cov test, leverage, window-3 + rollback"),
          ("cov-unif", "cov test, uniform audit"), ("goldtrend", "gold-mean trend (w=5)"),
          ("goldtrend-reg", "gold-mean trend, regression est."), ("goldtrend-oracle-tuned", "gold-mean trend, window oracle-tuned"),
          ("proxypoint", "proxy-point search (kmin=5)"), ("proxypoint-oracle-tuned", "proxy-point search, kmin oracle-tuned"),
          ("cov-lev-anytime(a=0.1)", "cov test, anytime-valid (alpha=0.1)")]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for key, lab in series:
    xs = [d[f"{key}|m={m}"]["labels_mean"] for m in MS]; ys = [d[f"{key}|m={m}"]["regret"] for m in MS]
    es = [d[f"{key}|m={m}"]["se"] for m in MS]
    ls = "-" if key.startswith("cov") else "--"
    ax.errorbar(xs, ys, yerr=es, marker="o", ls=ls, ms=3.5, capsize=2, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("mean cumulative gold labels until stop"); ax.set_ylabel("gold regret (peak gain 0.427)")
ax.set_title("Exploit-mixture W, q = 0.5: regret vs label budget (200 reps, +-1 SE)")
ax.legend(fontsize=6.5, ncol=2); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_budget_frontier.png"), dpi=150); plt.close()

# ---- Fig 6: W vs W' trade-off (m = 20) ----
rules = [("cov-lev|m=20", "cov, z=0"), ("cov-lev-win3+rollback|m=20", "cov, win3+rb"), ("cov-lev-anytime(a=0.1)|m=20", "cov, anytime"),
         ("gap-alarm(tuned q=0, out-of-sample)", "gap alarm"), ("hack-rate-alarm (perfect clf, tuned q=0)", "hack-rate")]
twin_key = {"gap-alarm(tuned q=0, out-of-sample)": "gap-alarm(tuned q=0)",
            "hack-rate-alarm (perfect clf, tuned q=0)": "hack-rate-alarm (exploit-type clf, tuned q=0)"}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for j, q in enumerate(["q=0.5", "q=1.0"]):
    for k, lab in rules:
        xW = R["R1_W"][q][k]["regret"]; xT = R["R3_twin"][q][twin_key.get(k, k)]["regret"]
        ax[j].scatter(xW, xT, s=30); ax[j].annotate(lab, (xW, xT), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[j].set_xscale("symlog", linthresh=0.01); ax[j].set_yscale("symlog", linthresh=0.01)
    ax[j].set_xlabel("regret in W (exploits bad)"); ax[j].set_ylabel("regret in W' (shared exploits good)")
    ax[j].set_title(f"{q}: same rule, both worlds (m=20)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_twin_tradeoff.png"), dpi=150); plt.close()

# ---- Remark 5: variance ratio, OLS slope of audited gold means vs averaged covariance ----
P = Params(q=0.5); lam_s = gold_peak(P)[0]; cf = closed_form(P, lam_s)
pi = float(cf["pi"]); mean_g0 = P.sg2 * lam_s
Eg = (1 - pi) * mean_g0 - pi * P.delta
Eg2 = (1 - pi) * (P.sg2 + mean_g0 ** 2) + pi * P.delta ** 2
var_g = Eg2 - Eg ** 2
rm = T["exp3b_rmse_m20"][f"q=0.5|lam={lam_s:.3f}|COV-lev"]["rmse"]
VC = rm ** 2 * 20  # per-label variance of the HH term (leverage design) at the peak
step = 0.05
ratio = {w: float(12 * var_g / (VC * step ** 2 * (w ** 2 - 1))) for w in [3, 5, 8, 12, 16]}
derived = dict(var_g_on_policy_at_peak=float(var_g), per_label_var_HH_leverage_at_peak=float(VC),
               variance_ratio_slope_over_cov_by_window=ratio, step=step)
json.dump(derived, open(os.path.join(OUT, "derived_label_efficiency.json"), "w"), indent=1)
print(derived)
