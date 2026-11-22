"""Make figures and summary tables from ../results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = json.load(open(os.path.join(RES, "results.json")))
NAMES = ["WinMLP", "SplitMLP", "GRU"]
COL = {"A": "#222222", "WinMLP": "#2a7ab9", "SplitMLP": "#d9822b", "GRU": "#3a9d5d"}
LAB = {"A": "Reference A (integrator)", "WinMLP": "Window mimic (shared)",
       "SplitMLP": "Window mimic (split heads)", "GRU": "Recurrent mimic (GRU)"}
lags = R["A"]["persistence"]["lags"]
out = []

def ms(v):
    v = np.array(v, float); return np.nanmean(v), np.nanstd(v)

# ---- Table 1: indicator summary
out.append("| Indicator | A | " + " | ".join(NAMES) + " |")
out.append("|---|---|" + "---|" * len(NAMES))
def row(label, fA, f):
    cells = [label, "%.3f" % fA if fA is not None else "-"]
    for n in NAMES:
        m, s = ms([f(e) for e in R[n]]); cells.append("%.3f ± %.3f" % (m, s))
    out.append("| " + " | ".join(cells) + " |")
row("In-dist. per-step KL eps", 0.0, lambda e: e["fidelity"]["eps"])
row("Prop.1 TV bound sqrt(L eps/2), L=12", 0.0, lambda e: e["fidelity"]["tv_bound"])
row("|gap| P(choose noxious) under D", 0.0, lambda e: e["fidelity"]["T_noxious_gap"])
row("|gap| P(noxious | post-shock) under D", 0.0, lambda e: e["fidelity"]["T_postshock_gap"])
row("Report corr. with A's report", 1.0, lambda e: e["fidelity"]["rep_corr"])
for c in ["none", "one", "two"]:
    row("Trade-off indifference premium (%s shock)" % c, R["A"]["tradeoff"][c]["indiff"],
        lambda e, c=c: e["tradeoff"][c]["indiff"])
i30 = lags.index(30); i50 = lags.index(50); i2 = lags.index(2)
for li, L in [(i2, 2), (lags.index(11), 11), (lags.index(15), 15), (i30, 30), (i50, 50)]:
    row("Avoidance response at lag %d" % L, R["A"]["persistence"]["d_avoid"][li],
        lambda e, li=li: e["persistence"]["d_avoid"][li])
row("Report response at lag 30", R["A"]["persistence"]["d_report"][i30],
    lambda e: e["persistence"]["d_report"][i30])
row("Linear-probe R^2 for report", R["A"]["steering"]["probe_r2"], lambda e: e["steering"]["probe_r2"])
row("Coupling rho (probe direction)", R["A"]["steering"]["probe"]["rho"], lambda e: e["steering"]["probe"]["rho"])
row("Coupling rho (report-gradient direction)", R["A"]["steering"]["gradient"]["rho"],
    lambda e: e["steering"]["gradient"]["rho"])
out.append("")
out.append("Analytic A: " + json.dumps(R["analytic_A"]))

# ---- Table 2: suppression
out.append("\n| Mimic | metric | before | after |\n|---|---|---|---|")
for n, lst in R["suppression"].items():
    for k in ["report_state_corr", "report_sd", "d_report_lag2", "d_avoid_lag2", "action_kl", "state_probe_r2"]:
        b = ms([x["before"][k] for x in lst]); a = ms([x["after"][k] for x in lst])
        out.append("| %s | %s | %.3f ± %.3f | %.3f ± %.3f |" % (n, k, b[0], b[1], a[0], a[1]))

# ---- Table 3: coverage
out.append("\n| Mimic | T_train | eps on T=48 episodes | report MSE T=48 | avoid lag 30 | avoid lag 50 |\n|---|---|---|---|---|---|")
for n, d in R["coverage"].items():
    for T, lst in d.items():
        e = ms([x["eps_T48"] for x in lst]); rm = ms([x["rep_mse_T48"] for x in lst])
        a30 = ms([x["persistence"]["d_avoid"][i30] for x in lst]); a50 = ms([x["persistence"]["d_avoid"][i50] for x in lst])
        out.append("| %s | %s | %.4f ± %.4f | %.4f ± %.4f | %.3f ± %.3f | %.3f ± %.3f |" % (n, T, *e, *rm, *a30, *a50))
out.append("A avoid lag30 %.3f lag50 %.3f" % (R["A"]["persistence"]["d_avoid"][i30], R["A"]["persistence"]["d_avoid"][i50]))
open(os.path.join(RES, "tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---- Figure 1: trade-off curves (in-distribution)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
g = np.array(R["A"]["tradeoff"]["grid"])
for ax, c in zip(axs, ["none", "one", "two"]):
    ax.plot(g, R["A"]["tradeoff"][c]["curve"], color=COL["A"], lw=2.5, label=LAB["A"])
    for n in NAMES:
        C = np.array([e["tradeoff"][c]["curve"] for e in R[n]])
        ax.plot(g, C.mean(0), color=COL[n], lw=1.3, ls="--", label=LAB[n])
    ax.set_title({"none": "no recent shock", "one": "1 shock (lag 2)", "two": "2 shocks (lags 2,3)"}[c])
    ax.set_xlabel("reward premium for noxious option"); ax.grid(alpha=.3)
axs[0].set_ylabel("P(accept noxious option)"); axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_tradeoff.png"), dpi=150)

# ---- Figure 2: persistence
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, key, yl in [(axs[0], "d_report", "report response (shock - no shock)"),
                    (axs[1], "d_avoid", "avoidance response (shock - no shock)")]:
    ax.plot(lags, R["A"]["persistence"][key], color=COL["A"], lw=2.5, marker="o", label=LAB["A"])
    for n in NAMES:
        C = np.array([e["persistence"][key] for e in R[n]])
        ax.errorbar(lags, C.mean(0), yerr=C.std(0), color=COL[n], lw=1.3, ls="--", marker=".", capsize=2, label=LAB[n])
    ax.axvspan(0, 11, color="grey", alpha=.12)
    ax.text(0.5, ax.get_ylim()[1] * 0.92 if key == "d_avoid" else 0.47, "training horizon", fontsize=7)
    ax.set_xlabel("lag since shock (steps)"); ax.set_ylabel(yl); ax.grid(alpha=.3)
axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_persistence.png"), dpi=150)

# ---- Figure 3: coupling rho
fig, ax = plt.subplots(figsize=(6, 3.4))
xs = np.arange(4); w = 0.35
for j, dn in enumerate(["probe", "gradient"]):
    vals = [R["A"]["steering"][dn]["rho"]] + [ms([e["steering"][dn]["rho"] for e in R[n]])[0] for n in NAMES]
    errs = [0] + [ms([e["steering"][dn]["rho"] for e in R[n]])[1] for n in NAMES]
    ax.bar(xs + (j - .5) * w, vals, w, yerr=errs, capsize=3, label="%s direction" % dn,
           color=["#7a7a7a", "#b0b0b0"][j], edgecolor="k")
ax.axhline(1, color="k", lw=.8, ls=":"); ax.axhline(0, color="k", lw=.8)
ax.set_xticks(xs); ax.set_xticklabels(["A", "Window\n(shared)", "Window\n(split)", "GRU"])
ax.set_ylabel(r"coupling $\rho$ (steered / natural slope)"); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_coupling.png"), dpi=150)

# ---- Figure 4: coverage
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(lags, R["A"]["persistence"]["d_avoid"], color="k", lw=2.5, marker="o", label="A")
cm = {"12": "#a6dba0", "24": "#5aae61", "48": "#1b7837"}
for T, lst in R["coverage"]["GRU"].items():
    C = np.array([x["persistence"]["d_avoid"] for x in lst])
    ax.errorbar(lags, C.mean(0), yerr=C.std(0), color=cm[T], ls="--", marker=".", capsize=2, label="GRU, train horizon %s" % T)
ax.set_xlabel("lag since shock"); ax.set_ylabel("avoidance response"); ax.grid(alpha=.3); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_coverage.png"), dpi=150)
print("figures written")
