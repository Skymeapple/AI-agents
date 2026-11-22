"""Build tables, figures and numerical checks of Propositions 1-3 from results/*.jsonl."""
import json, os, sys, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
import model
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 2, "figure.dpi": 150})
NAMES = {"abilene_tx": "Abilene, TX", "loudoun_va": "Loudoun, VA", "phoenix_az": "Phoenix, AZ"}
def load(e):
    p = os.path.join(RES, f"{e}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
P = model.params(); out = {}

# ---------- E1 ----------
E1 = [r for r in load("E1") if r.get("status") == 0]
idx = {(r["site"], r["mix"], r["phi"], r["v"]): r for r in E1}
PHIS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]; VS = [250, 500, 1000, 2000, 10000]
lines = ["| Site | Mix | v ($/MWh) | " + " | ".join(f"CFE {int(p*100)}%" for p in PHIS) + " |", "|" + "---|" * (3 + len(PHIS))]
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for v in [500, 1000, 10000]:
            cells = []
            for p in PHIS:
                r = idx.get((s, mix, p, v))
                cells.append("--" if r is None else f"{r['lcoe_served']:.0f} ({100*r['curtail_frac']:.1f}%)")
            lines.append(f"| {NAMES[s]} | {'with firm' if mix=='all' else 'no firm'} | {v} | " + " | ".join(cells) + " |")
open(os.path.join(RES, "table_E1_cost.md"), "w").write("Supply cost per served MWh ($/MWh) and (curtailed share of compute load)\n\n" + "\n".join(lines) + "\n")

# capacity table at v=1000
lines = ["| Site | Mix | CFE | PV | Wind | Bat MW | Bat MWh | Gas CT | Gas CC | Firm | spill % | curtail h |", "|" + "---|" * 12]
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for p in [0.0, 0.9, 0.99, 1.0]:
            r = idx.get((s, mix, p, 1000))
            if r is None: continue
            c = r["caps"]
            lines.append(f"| {NAMES[s]} | {mix} | {p:.2f} | " + " | ".join(f"{c[k]:.2f}" for k in ["pv", "wind", "bat_p", "bat_e", "ct", "cc", "firm"]) + f" | {100*r['spill_frac']:.0f} | {r['curtail_hours']} |")
open(os.path.join(RES, "table_E1_caps.md"), "w").write("Optimal capacities per MW of flat compute load, v = 1000 $/MWh\n\n" + "\n".join(lines) + "\n")

# Figure 1: cost vs CFE target
if E1:
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, s in zip(axs, NAMES):
        for j, (mix, v, ls) in enumerate([("nofirm", 10000, "--"), ("nofirm", 1000, "-"), ("all", 10000, "--"), ("all", 1000, "-")]):
            xs = [p for p in PHIS if (s, mix, p, v) in idx]
            ys = [idx[(s, mix, p, v)]["lcoe_served"] for p in xs]
            ax.plot([100 * x for x in xs], ys, ls, color=C[0] if mix == "nofirm" else C[1], marker="o", ms=4,
                    label=f"{'no firm' if mix=='nofirm' else 'with SMR option'}, {'inflexible' if v==10000 else 'v=$1000/MWh'}")
        ax.set_title(NAMES[s]); ax.set_xlabel("Carbon-free share of served energy (%)")
    axs[0].set_ylabel("Supply cost per served MWh ($)"); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_cost_vs_cfe.png")); plt.close(fig)

    # Figure 2: curtailment curves U*(v)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, s in zip(axs, NAMES):
        k = 0
        for mix in ["nofirm", "all"]:
            for p in [0.95, 0.99, 1.0]:
                xs = [v for v in VS if (s, mix, p, v) in idx]
                ax.plot(xs, [100 * idx[(s, mix, p, v)]["curtail_frac"] for v in xs], "-" if mix == "nofirm" else "--",
                        marker="o", ms=4, color=C[[0, 2, 6][[0.95, 0.99, 1.0].index(p)]],
                        label=f"CFE {int(p*100)}%, {'no firm' if mix=='nofirm' else 'SMR option'}")
        ax.set_xscale("log"); ax.set_title(NAMES[s]); ax.set_xlabel("Value of lost compute v ($/MWh)")
        ax.axvspan(403, 968, color="#999999", alpha=0.12, lw=0)
    axs[0].set_ylabel("Optimal curtailed share of load (%)"); axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_curtailment_vs_v.png")); plt.close(fig)

# ---------- Proposition checks ----------
chk = collections.Counter(); viol = []
for r in E1:
    # Prop 1: price ceiling lambda_t <= v + (1-phi) mu
    ceil = r["v"] + (1 - r["phi"]) * r["cfe_dual"]
    chk["p1_n"] += 1; chk["p1_ok"] += r["max_price"] <= ceil * (1 + 1e-4) + 1e-3
    # Prop 3 duration bound, firm option available
    if r["mix"] == "all":
        b = P["firm"] / (P["firm_avail"] * (r["v"] - P["firm_var"]))
        chk["p3_n"] += 1; ok = r["curtail_hours"] <= b + 2  # +2 h for dt=2 step rounding
        chk["p3_ok"] += ok
        if not ok: viol.append((r["site"], r["phi"], r["v"], r["curtail_hours"], b))
    if r["phi"] == 0.0:
        b = P["ct"] / (r["v"] - P["ct_var"])
        chk["p3ct_n"] += 1; chk["p3ct_ok"] += r["curtail_hours"] <= b + 2
# Prop 2: monotone U*(v), concavity sandwich U(v2)dv <= Z(v2)-Z(v1) <= U(v1)dv  (Z = total cost incl. curtailment)
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for p in PHIS:
            rs = [idx[(s, mix, p, v)] for v in VS if (s, mix, p, v) in idx]
            for a, b in zip(rs, rs[1:]):
                dv = b["v"] - a["v"]; dz = b["total_cost"] - a["total_cost"]
                chk["p2_n"] += 1
                Ua, Ub = a["curtail_frac"] * 4380 * 2, b["curtail_frac"] * 4380 * 2  # MWh/yr
                tol = 1e-3 * a["total_cost"]
                chk["p2_mono_ok"] += Ub <= Ua + 1e-6 * 8760 + 1
                chk["p2_sand_ok"] += (Ub * dv - tol <= dz <= Ua * dv + tol)
out["prop_checks"] = dict(chk); out["p3_violations"] = viol[:10]
out["p3_bound_hours"] = {v: P["firm"] / (P["firm_avail"] * (v - P["firm_var"])) for v in VS}
out["p3ct_bound_hours"] = {v: P["ct"] / (v - P["ct_var"]) for v in VS}

# key numbers
def g(s, mix, p, v, k="lcoe_served"):
    r = idx.get((s, mix, p, v)); return None if r is None else r[k]
kn = {}
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for p in PHIS:
            for v in VS:
                r = idx.get((s, mix, p, v))
                if r: kn[f"{s}|{mix}|{p}|{v}"] = dict(lcoe=round(r["lcoe_served"], 1), curtail_pct=round(100 * r["curtail_frac"], 2),
                                                      curtail_h=r["curtail_hours"], total_per_load_mwh=round(r["total_cost"] / 8760, 1),
                                                      firm=round(r["caps"]["firm"], 3), pv=round(r["caps"]["pv"], 2), wind=round(r["caps"]["wind"], 2),
                                                      bat_e=round(r["caps"]["bat_e"], 2), co2=round(r["co2_t_per_mwh"], 4))
out["E1"] = kn

# flexibility dividend: inflexible (v=10000) vs v=1000 -> % reduction of supply cost per served MWh and of total cost
fd = {}
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for p in PHIS:
            a, b = idx.get((s, mix, p, 10000)), idx.get((s, mix, p, 1000))
            if a and b: fd[f"{s}|{mix}|{p}"] = dict(total_saving_pct=round(100 * (1 - b["total_cost"] / a["total_cost"]), 2),
                                                    curtail_pct=round(100 * b["curtail_frac"], 2))
out["flex_dividend_v1000"] = fd

# ---------- E2 firm capex ----------
E2 = [r for r in load("E2") if r.get("status") == 0]
if E2:
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
    for j, (s, phi, v) in enumerate([(s, phi, v) for s in ["abilene_tx", "loudoun_va"] for phi in [0.9, 1.0] for v in [500, 2000]]):
        rs = sorted([r for r in E2 if r["site"] == s and r["phi"] == phi and r["v"] == v], key=lambda r: r["firm_capex"])
        if not rs: continue
        x = [r["firm_capex"] for r in rs]
        lab = f"{NAMES[s]}, CFE {int(phi*100)}%, v={v}"
        st = "-" if v == 2000 else ":"
        axs[0].plot(x, [r["caps"]["firm"] * P["firm_avail"] for r in rs], st, marker="o", ms=3, color=C[j % 8], label=lab)
        axs[1].plot(x, [r["total_cost"] / 8760 for r in rs], st, marker="o", ms=3, color=C[j % 8], label=lab)
    axs[0].set_xlabel("Firm clean (SMR) overnight-incl. capex ($/kW)"); axs[0].set_ylabel("Firm output share of load (avail.-adj.)")
    axs[1].set_xlabel("Firm clean (SMR) capex ($/kW)"); axs[1].set_ylabel("Total cost per MWh of potential load ($)")
    axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_firm_capex.png")); plt.close(fig)
    out["E2"] = [dict(site=r["site"], phi=r["phi"], v=r["v"], capex=r["firm_capex"], firm=round(r["caps"]["firm"], 3),
                      lcoe=round(r["lcoe_served"], 1), curtail_pct=round(100 * r["curtail_frac"], 2)) for r in sorted(E2, key=lambda r: (r["site"], r["phi"], r["v"], r["firm_capex"]))]

# ---------- E5 sensitivity ----------
E5 = [r for r in load("E5") if r.get("status") == 0]
if E5:
    base = [r for r in E5 if r["sens"] == "base"][0]
    out["E5"] = {r["sens"]: dict(lcoe=round(r["lcoe_served"], 1), firm=round(r["caps"]["firm"], 3), pv=round(r["caps"]["pv"], 2),
                                 wind=round(r["caps"]["wind"], 2), bat_e=round(r["caps"]["bat_e"], 2), curtail_pct=round(100 * r["curtail_frac"], 3))
                 for r in E5}
# ---------- E6 exogenous curtailment cap with free curtailment ----------
E6 = [r for r in load("E6") if r.get("status") == 0]
if E6:
    out["E6"] = []
    for s in ["abilene_tx", "loudoun_va"]:
        for mix in ["all", "nofirm"]:
            rs = sorted([r for r in E6 if r["site"] == s and r["mix"] == mix], key=lambda r: r["max_curtail_frac"])
            if not rs: continue
            z0 = rs[0]["supply_cost"]
            for a, b in zip(rs, rs[1:]):
                dU = (b["curtail_frac"] - a["curtail_frac"]) * 8760
                out["E6"].append(dict(site=s, mix=mix, f=b["max_curtail_frac"], lcoe=round(b["lcoe_served"], 1),
                                      saving_vs_f0_pct=round(100 * (1 - b["supply_cost"] / z0), 2),
                                      marginal_value_per_mwh_curtailed=round((a["supply_cost"] - b["supply_cost"]) / dU, 0) if dU > 0 else None))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for j, (s, mix) in enumerate([(s, m) for s in ["abilene_tx", "loudoun_va"] for m in ["nofirm", "all"]]):
        rs = sorted([r for r in E6 if r["site"] == s and r["mix"] == mix], key=lambda r: r["max_curtail_frac"])
        if not rs: continue
        z0 = rs[0]["supply_cost"]
        ax.plot([100 * r["max_curtail_frac"] for r in rs], [100 * (1 - r["supply_cost"] / z0) for r in rs], marker="o", ms=3,
                color=C[j], label=f"{NAMES[s]}, {'no firm' if mix=='nofirm' else 'SMR option'}")
    ax.set_xlabel("Allowed annual curtailment (% of load energy)"); ax.set_ylabel("Supply-cost saving vs. 0% (%)")
    ax.set_title("100% CFE islanded campus", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_exogenous_flex.png")); plt.close(fig)
# ---------- E4 hourly validation ----------
E4 = [r for r in load("E4") if r.get("status") == 0]
if E4 and E1:
    out["E4"] = []
    for r in E4:
        r2 = idx.get((r["site"], r["mix"], r["phi"], r["v"]))
        if r2: out["E4"].append(dict(site=r["site"], mix=r["mix"], phi=r["phi"], v=r["v"], lcoe_1h=round(r["lcoe_served"], 2),
                                     lcoe_2h=round(r2["lcoe_served"], 2), rel_err_pct=round(100 * (r2["lcoe_served"] / r["lcoe_served"] - 1), 2),
                                     curtail_1h=round(100 * r["curtail_frac"], 3), curtail_2h=round(100 * r2["curtail_frac"], 3), secs_1h=round(r["secs"])))
# ---------- E3 ----------
p3 = os.path.join(RES, "E3.json")
if os.path.exists(p3):
    E3 = json.load(open(p3)); out["E3"] = E3
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
    for ax, s in zip(axs, ["abilene_tx", "loudoun_va"]):
        for j, phi in enumerate([0.95, 1.0]):
            ins = {d["design_year"]: 100 * d["curtail"] for d in E3["designs"] if d["site"] == s and d["phi"] == phi}
            oos = [e for e in E3["evals"] if e["site"] == s and e["phi"] == phi]
            ax.scatter([e["design_year"] + (j - 0.5) * 0.15 for e in oos], [100 * e["curtail"] for e in oos], s=14, color=C[j], alpha=0.7,
                       label=f"CFE {int(phi*100)}%: other weather years")
            ax.scatter([y + (j - 0.5) * 0.15 for y in ins], list(ins.values()), marker="D", s=26, color=C[j], edgecolor="k", lw=0.6,
                       label=f"CFE {int(phi*100)}%: design year")
        ax.set_title(NAMES[s] + ", no firm, v=$1000/MWh", fontsize=8.5); ax.set_xlabel("Design weather year"); ax.set_xticks([2010, 2011, 2012, 2013])
    axs[0].set_ylabel("Curtailed share of load (%)"); axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_weather_years.png")); plt.close(fig)


# ---------- Total cost per MWh of potential load (supply + lost-compute cost) ----------
lines = ["| Site | Mix | v | " + " | ".join(f"CFE {int(p*100)}%" for p in PHIS) + " |", "|" + "---|" * (3 + len(PHIS))]
for s in NAMES:
    for mix in ["all", "nofirm"]:
        for v in [250, 500, 1000, 2000, 10000]:
            cells = [("--" if (s, mix, p, v) not in idx else f"{idx[(s, mix, p, v)]['total_cost']/8760:.0f}") for p in PHIS]
            lines.append(f"| {NAMES[s]} | {'with firm' if mix=='all' else 'no firm'} | {v} | " + " | ".join(cells) + " |")
open(os.path.join(RES, "table_E1_total.md"), "w").write("Total cost (supply + v x curtailed MWh) per MWh of potential compute load ($/MWh)\n\n" + "\n".join(lines) + "\n")

# ---------- E7 + Prop-2 brackets on the VoLC at which renewable-only + flexibility matches the firm option ----------
E7 = [r for r in load("E7") if r.get("status") == 0]
grid = collections.defaultdict(dict)
for r in E1 + E7:
    if r["mix"] == "nofirm" and r["phi"] in (0.99, 1.0):
        grid[(r["site"], r["phi"])][r["v"]] = r
be = {}
for (s, phi), d in grid.items():
    firm = idx.get((s, "all", phi, 1000))
    if firm is None: continue
    zf = firm["total_cost"] / 8760
    vs = sorted(d); z = {v: d[v]["total_cost"] / 8760 for v in vs}; U = {v: d[v]["curtail_frac"] for v in vs}
    lo = hi = None
    # Z concave nondecreasing; tangent upper bound Z(w) <= Z(v)+U(v)(w-v) gives a certified lower end of the crossing,
    # chord lower bound between grid points gives a certified upper end.
    for a, b in zip(vs, vs[1:]):
        if z[a] <= zf < z[b]:
            lo = a + (zf - z[a]) / U[a] if U[a] > 0 else a
            hi = a + (zf - z[a]) * (b - a) / (z[b] - z[a])
    be[f"{s}|{phi}"] = dict(firm_total=round(zf, 1), grid={v: dict(total=round(z[v], 1), curtail_pct=round(100 * U[v], 2),
                         supply_per_served=round(d[v]["lcoe_served"], 1), hours=d[v]["curtail_hours"]) for v in vs},
                         crossing_bracket=None if lo is None else [round(lo, 0), round(hi, 0)],
                         below_grid=z[vs[0]] > zf)
out["breakeven_vs_firm"] = be
if grid:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
    for ax, phi in zip(axs, [0.99, 1.0]):
        for j, s in enumerate(NAMES):
            d = grid.get((s, phi), {}); vs = sorted(d)
            ax.plot(vs, [d[v]["total_cost"] / 8760 for v in vs], "-o", ms=3, color=C[j], label=f"{NAMES[s]}: no firm + flexibility")
            f = idx.get((s, "all", phi, 1000))
            if f: ax.axhline(f["total_cost"] / 8760, color=C[j], ls=":", lw=1.2)
        ax.axvspan(403, 968, color="#999999", alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("Value of lost compute v ($/MWh)"); ax.set_title(f"CFE {int(phi*100)}% (dotted: SMR-option optimum)", fontsize=8.5)
    axs[0].set_ylabel("Total cost per MWh of potential load ($)"); axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig7_breakeven.png")); plt.close(fig)

# ---------- SMR capex at which a firm-only campus costs the same as the optimal renewable-only + flexibility campus ----------
def firm_only_cost(capex):
    Pc = model.params(firm_capex=capex)
    return Pc["firm"] / Pc["firm_avail"] / 8760 + Pc["firm_var"]
from scipy.optimize import brentq
bc = {}
for s in NAMES:
    for v in VS:
        r = idx.get((s, "nofirm", 1.0, v))
        if r is None: continue
        z = r["total_cost"] / 8760
        bc[f"{s}|{v}"] = dict(nofirm_total=round(z, 1), firm_only_at_atb=round(firm_only_cost(9650.0), 1),
                              breakeven_capex=round(brentq(lambda c: firm_only_cost(c) - z, 100, 1e5), -1))
out["breakeven_firm_capex_cfe100"] = bc

# ---------- E8 lumpy firm availability ----------
E8 = [r for r in load("E8") if r.get("status") == 0]
if E8:
    out["E8"] = []
    for r in sorted(E8, key=lambda r: (r["site"], r["phi"], r["modules"], r["v"])):
        base = idx.get((r["site"], "all", r["phi"], r["v"]))
        c = r["caps"]
        out["E8"].append(dict(site=r["site"], phi=r["phi"], modules=r["modules"], v=r["v"], mean_avail=round(r["mean_avail"], 3),
                              min_avail=round(r["min_avail"], 3), total=round(r["total_cost"] / 8760, 1),
                              total_const_avail=None if base is None else round(base["total_cost"] / 8760, 1),
                              curtail_pct=round(100 * r["curtail_frac"], 2), curtail_h=r["curtail_hours"],
                              firm=round(c["firm"], 2), pv=round(c["pv"], 2), wind=round(c["wind"], 2), bat_e=round(c["bat_e"], 2),
                              ct=round(c["ct"], 2), cc=round(c["cc"], 2)))

json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(out["prop_checks"]), out.get("p3_violations"))
