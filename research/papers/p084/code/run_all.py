"""Run every experiment in the paper. Usage:  python3 run_all.py  (about 1-3 min on 4 CPUs)
Outputs: ../results/results.json, ../results/*.csv, ../results/fig*.png
"""
import json
import os
import copy
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import config, default_sites
from model import (simulate, risk_metrics, slack_tolerance, output_under_capacity_loss,
                   regional_cutoff_output, required_outside_capacity)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
N = 100000
SEED = 12345
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "figure.dpi": 150, "axes.edgecolor": "#52514e"})
OUT = {}


def cfg_with(p_geo=0.01, mode="priority", **kw):
    c = config()
    c["hazards"]["tw_geo"]["rate"] = p_geo
    c["model"]["allocation"] = mode
    for k, v in kw.items():
        sec, key = k.split("__", 1)
        if sec == "model":
            c["model"][key] = v
        elif sec == "ai":
            c["ai_share"][key] = v
        else:  # hazard__param
            c["hazards"][sec][key] = v
    return c


def set_cowos_tw_share(c, x):
    for s in c["sites"]:
        if s["inp"] == "cowos":
            s["share"] = x / 6 if s["region"] == "TW" else 1 - x
    return c


def set_logic_us_share(c, x):
    """TSMC-process logic outside Taiwan = x; Taiwan TSMC = 0.93 - x."""
    for s in c["sites"]:
        if s["inp"] == "logic" and s["supplier"] == "TSMC":
            s["share"] = x if s["region"] == "US" else (0.93 - x) / 6
    return c


def m(r):
    return risk_metrics(r["loss"])


# ---------------------------------------------------------------- E1 headline sweep
print("E1 headline sweep")
E1 = []
for mode in ("priority", "proportional"):
    for p in (0.0, 0.005, 0.01, 0.02, 0.05):
        r = simulate(cfg_with(p, mode), N, SEED)
        mm = m(r)
        mm.update(mode=mode, p_geo=p,
                  mean_nongeo=float(r["loss"][~r["geo"]].mean()),
                  mean_given_geo=float(r["loss"][r["geo"]].mean()) if r["geo"].any() else None,
                  n_geo_years=int(r["geo"].sum()),
                  share_of_mean_from_geo_years=float(r["loss"][r["geo"]].sum() / max(r["loss"].sum(), 1e-12)))
        E1.append(mm)
        print(f"  {mode:12s} p={p:.3f} mean={mm['mean']:.4f} CVaR99={mm['CVaR0.99']:.3f} "
              f"P>10%={mm['p_gt_10pct']:.4f} geo-share={mm['share_of_mean_from_geo_years']:.2f}")
OUT["E1"] = E1

# exceedance curves figure
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, mode in zip(axes, ("priority", "proportional")):
    for k, p in enumerate((0.0, 0.01, 0.05)):
        r = simulate(cfg_with(p, mode), N, SEED)
        x = np.sort(r["loss"])[::-1]
        ex = np.arange(1, len(x) + 1) / len(x)
        ax.step(x * 100, ex, where="post", color=C[k], label=f"p_TW-geo = {p:.0%}/yr")
    ax.set_yscale("log"); ax.set_xlim(0, 80); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("annual accelerator output lost (%)")
    ax.set_title(f"{mode} allocation", fontsize=9)
axes[0].set_ylabel("P(annual loss > x)")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_exceedance.png")); plt.close(fig)

# ---------------------------------------------------------------- E2 hazard decomposition
print("E2 hazard decomposition")
HZ = ["local", "tw_quake", "kr_regional", "hbm_yield", "tw_geo"]
E2 = []
for mode in ("priority", "proportional"):
    full = simulate(cfg_with(0.01, mode), N, SEED)
    fm = m(full)
    for h in HZ:
        alone = m(simulate(cfg_with(0.01, mode), N, SEED, hazards_on=[h]))
        wo = m(simulate(cfg_with(0.01, mode), N, SEED, hazards_on=[x for x in HZ if x != h]))
        E2.append(dict(mode=mode, hazard=h, alone_mean=alone["mean"], alone_CVaR99=alone["CVaR0.99"],
                       alone_p10=alone["p_gt_10pct"],
                       drop_mean_if_removed=fm["mean"] - wo["mean"], drop_CVaR99_if_removed=fm["CVaR0.99"] - wo["CVaR0.99"],
                       full_mean=fm["mean"], full_CVaR99=fm["CVaR0.99"]))
        print(f"  {mode:12s} {h:12s} alone mean={alone['mean']:.5f} CVaR99={alone['CVaR0.99']:.3f}")
OUT["E2"] = E2

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
labels = {"local": "site incidents", "tw_quake": "Taiwan quake", "kr_regional": "Korea regional",
          "hbm_yield": "HBM yield/qual.", "tw_geo": "Taiwan cut-off (1%/yr)"}
for ax, metric, ttl in zip(axes, ("alone_mean", "alone_CVaR99"), ("expected annual loss (%)", "CVaR$_{99}$ of annual loss (%)")):
    for k, mode in enumerate(("priority", "proportional")):
        vals = [next(e[metric] for e in E2 if e["mode"] == mode and e["hazard"] == h) * 100 for h in HZ]
        ax.barh(np.arange(len(HZ)) + (k - 0.5) * 0.38, vals, height=0.36, color=C[k], label=mode)
    ax.set_yticks(np.arange(len(HZ))); ax.set_yticklabels([labels[h] for h in HZ])
    ax.set_xscale("symlog", linthresh=0.01); ax.set_title(ttl, fontsize=9); ax.grid(axis="y", visible=False)
axes[1].set_yticklabels([])
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Each hazard class simulated alone", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_hazard_decomposition.png")); plt.close(fig)

# ---------------------------------------------------------------- E3 input hardening (risk-reduction worth)
print("E3 input hardening")
E3 = []
for mode in ("priority", "proportional"):
    base = m(simulate(cfg_with(0.01, mode), N, SEED))
    for imm in ((), ("logic",), ("cowos",), ("hbm",), ("logic", "cowos"), ("cowos", "hbm"), ("logic", "hbm"), ("logic", "cowos", "hbm")):
        mm = m(simulate(cfg_with(0.01, mode), N, SEED, immune_inputs=imm))
        E3.append(dict(mode=mode, immune="+".join(imm) or "none", mean=mm["mean"], CVaR99=mm["CVaR0.99"],
                       mean_reduction_pct=100 * (1 - mm["mean"] / base["mean"]),
                       CVaR99_reduction_pct=100 * (1 - mm["CVaR0.99"] / base["CVaR0.99"])))
        print(f"  {mode:12s} immune={E3[-1]['immune']:18s} mean={mm['mean']:.5f} (-{E3[-1]['mean_reduction_pct']:.1f}%) "
              f"CVaR99={mm['CVaR0.99']:.3f} (-{E3[-1]['CVaR99_reduction_pct']:.1f}%)")
OUT["E3"] = E3

# ---------------------------------------------------------------- E4 regional cut-off analytics
print("E4 regional cut-off analytics")
s = config()["ai_share"]
cases = {
    "2025-26 baseline (short run)": {"logic": 0.03, "cowos": 0.05, "hbm": 0.81},
    "after porting to non-TSMC foundries": {"logic": 0.10, "cowos": 0.05, "hbm": 0.81},
    "US logic x3 only (0.09), no CoWoS move": {"logic": 0.09, "cowos": 0.05, "hbm": 0.81},
    "CoWoS outside TW 30%, logic unchanged": {"logic": 0.03, "cowos": 0.30, "hbm": 0.81},
    "logic 0.09 + CoWoS 30% outside": {"logic": 0.09, "cowos": 0.30, "hbm": 0.81},
    "logic 0.09 + CoWoS 50% outside": {"logic": 0.09, "cowos": 0.50, "hbm": 0.81},
}
E4 = {"ai_share": s, "cases": [], "korea_cutoff": None}
for name, o in cases.items():
    for mode in ("priority", "proportional"):
        y, per = regional_cutoff_output(o, s, mode)
        E4["cases"].append(dict(case=name, mode=mode, output=y, per_input=per, binding=min(per, key=per.get)))
        print(f"  {name:42s} {mode:12s} output={y:.3f} binding={min(per, key=per.get)}")
# Korea full cut-off: logic outside KR (TSMC TW+US)=0.93, cowos 1.0, hbm outside KR 0.19
for mode in ("priority", "proportional"):
    y, per = regional_cutoff_output({"logic": 0.93, "cowos": 1.0, "hbm": 0.19}, s, mode)
    E4.setdefault("korea", []).append(dict(mode=mode, output=y, per_input=per))
    print(f"  Korea cut-off {mode}: output={y:.3f}")
E4["required_outside_share_y50"] = required_outside_capacity(0.5, s)
E4["required_outside_share_y80"] = required_outside_capacity(0.8, s)
# physical CoWoS translation using TrendForce Jun-2026 industry capacity range 170-200k wpm (TSMC 120-140k + OSAT 50-60k)
E4["cowos_wpm_needed_y50"] = [0.5 * 0.90 * x for x in (170000, 200000)]
E4["cowos_wpm_needed_y80"] = [0.8 * 0.90 * x for x in (170000, 200000)]
OUT["E4"] = E4

gl = np.linspace(0, 0.20, 81); gc = np.linspace(0, 0.6, 121)
Z = np.array([[regional_cutoff_output({"logic": a, "cowos": b, "hbm": 0.81}, s, "priority")[0] for a in gl] for b in gc])
fig, ax = plt.subplots(figsize=(4.4, 3.4))
cs = ax.contourf(gl * 100, gc * 100, Z * 100, levels=np.arange(0, 101, 10), cmap="Blues")
ax.contour(gl * 100, gc * 100, Z * 100, levels=[25, 50, 75], colors="#0b0b0b", linewidths=0.8)
ax.plot([3], [5], marker="o", color=C[1], ms=8, mec="white", mew=1.5)
ax.annotate("2025-26", (3, 5), xytext=(5, 9), fontsize=8, color="#0b0b0b")
ax.set_ylim(0, 60); ax.set_xlim(0, 20)
ax.set_xlabel("TSMC-process logic capacity outside Taiwan (% of advanced logic)")
ax.set_ylabel("CoWoS-class capacity outside Taiwan (%)")
cb = fig.colorbar(cs, ax=ax); cb.set_label("AI output during Taiwan cut-off (% of demand)")
ax.set_title("Weakest-link resilience (priority allocation)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_weakest_link.png")); plt.close(fig)

# ---------------------------------------------------------------- E5 buffers vs heavy tails
print("E5 buffers")
E5 = []
Bs = [0, 1, 2, 4, 8, 13, 26]
for B in Bs:
    for label, hz, p in (("routine (no TW cut-off)", None, 0.0), ("TW cut-off only", ["tw_geo"], 0.01)):
        r = simulate(cfg_with(p, "proportional", model__buffer_weeks=B), N, SEED, hazards_on=hz)
        mm = m(r)
        E5.append(dict(B=B, regime=label, mean=mm["mean"], CVaR99=mm["CVaR0.99"]))
OUT["E5"] = E5


def lognorm_excess_frac(b, med, sig):
    """Closed form E[(D-b)^+]/E[D], D ~ LogNormal(ln med, sig)."""
    if b <= 0:
        return 1.0
    mu = np.log(med)
    d1 = (mu + sig ** 2 - np.log(b)) / sig
    d2 = d1 - sig
    ED = np.exp(mu + sig ** 2 / 2)
    return float((ED * norm.cdf(d1) - b * norm.cdf(d2)) / ED)


hz = config()["hazards"]
OUT["E5_analytic"] = {
    "local": [lognorm_excess_frac(b, hz["local"]["dur_median"], hz["local"]["dur_sigma"]) for b in Bs],
    "tw_quake": [lognorm_excess_frac(b, hz["tw_quake"]["dur_median"], hz["tw_quake"]["dur_sigma"]) for b in Bs],
    "tw_geo": [lognorm_excess_frac(b, hz["tw_geo"]["dur_median"], hz["tw_geo"]["dur_sigma"]) for b in Bs],
    "B": Bs}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for k, reg in enumerate(("routine (no TW cut-off)", "TW cut-off only")):
    v = [e["mean"] for e in E5 if e["regime"] == reg]
    axes[0].plot(Bs, np.array(v) / v[0] * 100, marker="o", color=C[k], label=reg)
axes[0].set_xlabel("inventory buffer (weeks of demand)"); axes[0].set_ylabel("expected loss, % of no-buffer value")
axes[0].set_title("Simulation (proportional allocation)", fontsize=9); axes[0].legend(frameon=False, fontsize=8)
for k, h in enumerate(("tw_quake", "local", "tw_geo")):
    axes[1].plot(Bs, np.array(OUT["E5_analytic"][h]) * 100, marker="o", color=C[k + 2],
                 label={"tw_quake": "quake (median 1 wk)", "local": "site incident (10 wk)", "tw_geo": "cut-off (26 wk)"}[h])
axes[1].set_xlabel("buffer b (weeks)"); axes[1].set_ylabel("E[(D-b)+]/E[D] (%)")
axes[1].set_title("Prop. 3 closed form, full outage", fontsize=9); axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_buffers.png")); plt.close(fig)

# ---------------------------------------------------------------- E6 sensitivity (one-at-a-time, common random numbers)
print("E6 sensitivity")
SENS = [
    ("local site-incident rate", "local__rate_per_site", 0.025, 0.10),
    ("local outage median (wk)", "local__dur_median", 5.0, 20.0),
    ("Taiwan quake rate", "tw_quake__rate", 0.04, 0.15),
    ("Taiwan quake median (wk)", "tw_quake__dur_median", 0.5, 4.0),
    ("Korea regional rate", "kr_regional__rate", 0.003, 0.03),
    ("HBM yield-shock rate", "hbm_yield__rate_per_supplier", 0.075, 0.30),
    ("TW cut-off probability", "tw_geo__rate", 0.005, 0.02),
    ("TW cut-off median (wk)", "tw_geo__dur_median", 13.0, 52.0),
    ("buffer (wk)", "model__buffer_weeks", 1.0, 8.0),
    ("AI share of CoWoS", "ai__cowos", 0.80, 0.95),
    ("AI share of HBM", "ai__hbm", 0.80, 0.95),
]
E6 = []
for mode in ("priority", "proportional"):
    base = m(simulate(cfg_with(0.01, mode), N, SEED))
    for name, key, lo, hi in SENS:
        row = dict(mode=mode, param=name, lo=lo, hi=hi, base_mean=base["mean"], base_CVaR99=base["CVaR0.99"])
        for tag, v in (("lo", lo), ("hi", hi)):
            kw = {key: v}
            if key == "tw_geo__rate":
                c = cfg_with(v, mode)
            else:
                c = cfg_with(0.01, mode, **kw)
            mm = m(simulate(c, N, SEED))
            row[f"mean_{tag}"] = mm["mean"]; row[f"CVaR99_{tag}"] = mm["CVaR0.99"]
        # CoWoS Taiwan share separately below
        E6.append(row)
    row = dict(mode=mode, param="CoWoS share outside Taiwan", lo=0.0, hi=0.15, base_mean=base["mean"], base_CVaR99=base["CVaR0.99"])
    for tag, x in (("lo", 1.0), ("hi", 0.85)):
        mm = m(simulate(set_cowos_tw_share(cfg_with(0.01, mode), x), N, SEED))
        row[f"mean_{tag}"] = mm["mean"]; row[f"CVaR99_{tag}"] = mm["CVaR0.99"]
    E6.append(row)
OUT["E6"] = E6
for e in E6:
    print(f"  {e['mode']:12s} {e['param']:28s} mean {e['mean_lo']:.5f}..{e['mean_hi']:.5f}  CVaR99 {e['CVaR99_lo']:.3f}..{e['CVaR99_hi']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
for ax, mode in zip(axes, ("priority", "proportional")):
    rows = [e for e in E6 if e["mode"] == mode]
    rows.sort(key=lambda e: abs(e["mean_hi"] - e["mean_lo"]))
    b = rows[0]["base_mean"] * 100
    for i, e in enumerate(rows):
        lo, hi = e["mean_lo"] * 100, e["mean_hi"] * 100
        ax.barh(i, hi - b, left=b, color=C[0], height=0.6)
        ax.barh(i, lo - b, left=b, color=C[1], height=0.6)
    ax.axvline(b, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([e["param"] for e in rows])
    ax.set_title(f"{mode}: expected annual loss (%)", fontsize=9); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=C[0], label="high value"), Patch(color=C[1], label="low value")],
               frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_tornado.png")); plt.close(fig)

# ---------------------------------------------------------------- E7 slack asymmetry (falsifiable prediction)
print("E7 slack asymmetry")
E7 = {"tolerance": {k: slack_tolerance(v) for k, v in s.items()}}
lf = np.linspace(0, 1, 101)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for k, inp in enumerate(("logic", "cowos", "hbm")):
    ax.plot(lf * 100, output_under_capacity_loss(lf, s[inp]) * 100, color=C[k], label=f"{inp} (AI share {s[inp]:.0%}), priority")
ax.plot(lf * 100, output_under_capacity_loss(lf, 1, "proportional") * 100, color="#52514e", ls="--", label="any input, proportional")
ax.set_xlabel("capacity of one input lost (%)"); ax.set_ylabel("AI accelerator output (% of demand)")
ax.legend(frameon=False, fontsize=7); ax.set_title("Prop. 1: slack asymmetry", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig6_slack.png")); plt.close(fig)
# scenario check by simulation-free evaluation: 20% one-input loss for 8 weeks
E7["loss20pct_8wk_output_during"] = {inp: float(output_under_capacity_loss(0.2, s[inp])) for inp in s}
OUT["E7"] = E7

with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
# CSVs
import csv
for key in ("E1", "E2", "E3", "E5", "E6"):
    rows = OUT[key]
    with open(os.path.join(RES, f"{key}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
print("done")
