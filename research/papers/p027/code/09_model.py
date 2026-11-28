"""Entry-migration model of revision diffusion across a breaking SDK boundary.

State (npm servers with a runtime dependency on an official SDK):
    L legacy (v1 SDK), M modern (v2 SDK).  Per week:
    M += b*E + h*L ;  L += (1-b)*E - h*L
E weekly entrants, b share of entrants born modern, h weekly migration hazard of legacy servers.
Continuous-time share dynamics with relative inflow g = E/N:  dl/dt = (1-b) g - (g+h) l,
so for constant g the legacy share converges to l* = (1-b) g / (g+h) at rate g+h.

Reads results/summary.json and results/cohort_weekly.csv (written by 08_analyze.py).
Writes results/model.json and results/fig4_model.png.
"""
import csv, json, math, os

import numpy as np

B = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(B, "results")
S = json.load(open(os.path.join(R, "summary.json")))
W = [r for r in csv.DictReader(open(os.path.join(R, "cohort_weekly.csv"))) if int(r["week_since_ga"]) >= 0]
E = np.array([int(r["entrants"]) for r in W], float)
Bm = np.array([int(r["born_modern"]) for r in W], float)
Mg = np.array([int(r["migrations"]) for r in W], float)
nE = np.array([int(r["ns_entrants"]) for r in W], float)
nB = np.array([int(r["ns_born_modern"]) for r in W], float)
L0, M0 = S["entrants"]["stock_at_ga"]["sdk1"], S["entrants"]["stock_at_ga"]["v2"]
WPY = 365.25 / 7
out = {"inputs": {"L0": L0, "M0": M0, "weeks_observed": len(W), "entrants_total": int(E.sum()),
                  "born_modern_total": int(Bm.sum()), "migrations_total": int(Mg.sum())}}


def stocks(upto):
    """Observed bookkeeping: legacy and modern stock at the start of each week 0..upto."""
    L, M = [L0], [M0]
    for w in range(upto):
        L.append(L[-1] + E[w] - Bm[w] - Mg[w])
        M.append(M[-1] + Bm[w] + Mg[w])
    return np.array(L), np.array(M)


def simulate(L, M, inflow, b, h, weeks):
    tr = []
    for w in range(weeks):
        e = inflow(w) if callable(inflow) else inflow[w]
        mig = h * L
        L, M = L + (1 - b) * e - mig, M + b * e + mig
        tr.append(M / (M + L))
    return L, M, tr


Lobs, Mobs = stocks(len(W))
out["observed_share_end"] = round(Mobs[-1] / (Mobs[-1] + Lobs[-1]), 4)
out["observed_share_by_week"] = [round(m / (m + l), 4) for l, m in zip(Lobs, Mobs)]

# ---- parameter estimates on the full window
h_w = Mg.sum() / Lobs[:-1].sum()
b_pkg = Bm.sum() / E.sum()
b_ns = nB.sum() / nE.sum()
out["fit_full"] = {"h_per_week": h_w, "h_per_year": round(h_w * WPY, 4), "b_package": round(b_pkg, 4),
                   "b_namespace_first": round(b_ns, 4),
                   "g_rel_inflow_per_year_at_end": round(np.median(E[-8:]) / (Lobs[-1] + Mobs[-1]) * WPY, 4),
                   "weekly_entrants_median_last8": float(np.median(E[-8:])), "weekly_entrants_mean_last8": float(E[-8:].mean()),
                   "weekly_entrants_median_all": float(np.median(E))}

# ---- backtest: fit on the first K weeks, predict the rest given realised inflow
bt = {}
for K in (4, 6, 8, 10):
    hK = Mg[:K].sum() / Lobs[:K].sum()
    bK = Bm[:K].sum() / E[:K].sum()
    _, _, tr = simulate(Lobs[K], Mobs[K], E[K:], bK, hK, len(W) - K)
    # naive baseline: share stays at its week-K value
    bt[K] = {"h_per_year": round(hK * WPY, 4), "b": round(bK, 4), "share_at_fit_end": round(Mobs[K] / (Mobs[K] + Lobs[K]), 4),
             "predicted_end": round(tr[-1], 4), "observed_end": out["observed_share_end"],
             "abs_err": round(abs(tr[-1] - out["observed_share_end"]), 4),
             "persistence_abs_err": round(abs(Mobs[K] / (Mobs[K] + Lobs[K]) - out["observed_share_end"]), 4)}
out["backtest"] = bt
# two-class variant: "bulk" entrants (namespace launching >= 5 servers in one week) have their own b
bE = np.array([int(r["bulk_entrants"]) for r in W], float)
bB = np.array([int(r["bulk_born_modern"]) for r in W], float)
oE, oB = E - bE, Bm - bB
out["fit_full"]["b_bulk"] = round(bB.sum() / bE.sum(), 4)
out["fit_full"]["b_non_bulk"] = round(oB.sum() / oE.sum(), 4)
out["fit_full"]["bulk_share_of_entrants"] = round(bE.sum() / E.sum(), 4)
bt2 = {}
for K in (4, 6, 8, 10):
    hK = Mg[:K].sum() / Lobs[:K].sum()
    b_o = oB[:K].sum() / oE[:K].sum()
    b_b = bB[:K].sum() / bE[:K].sum() if bE[:K].sum() > 0 else b_o
    L, M = Lobs[K], Mobs[K]
    for w in range(K, len(W)):
        mig = hK * L
        born = b_o * oE[w] + b_b * bE[w]
        L, M = L + E[w] - born - mig, M + born + mig
    bt2[K] = {"b_non_bulk": round(b_o, 4), "b_bulk": round(b_b, 4), "bulk_entrants_in_fit": int(bE[:K].sum()),
              "predicted_end": round(M / (M + L), 4), "abs_err": round(abs(M / (M + L) - out["observed_share_end"]), 4)}
out["backtest_two_class"] = bt2

# ---- closed form vs simulation (constant relative inflow g)
chk = []
for g_y, b, h_y in ((2.0, 0.08, 0.15), (0.5, 0.3, 0.15), (5.0, 0.16, 0.5)):
    g, h = g_y / WPY, h_y / WPY
    L, M = 1000.0, 0.0
    for w in range(6000):
        e = g * (L + M)
        mig = h * L
        L, M = L + (1 - b) * e - mig, M + b * e + mig
    chk.append({"g_per_year": g_y, "b": b, "h_per_year": h_y, "simulated_modern_share": round(M / (M + L), 4),
                "closed_form": round((h_y + b * g_y) / (g_y + h_y), 4)})
out["closed_form_check"] = chk
g_now = out["fit_full"]["g_rel_inflow_per_year_at_end"]
h_y = out["fit_full"]["h_per_year"]
out["steady_state_at_current_rates"] = {
    "b_package": round((h_y + b_pkg * g_now) / (g_now + h_y), 4),
    "b_namespace": round((h_y + b_ns * g_now) / (g_now + h_y), 4),
    "relaxation_time_years": round(1 / (g_now + h_y), 3),
    "d_mstar_d_b": round(g_now / (g_now + h_y), 4),
    "d_mstar_d_h_per_unit_year": round((1 - b_pkg) * g_now / (g_now + h_y) ** 2, 4)}

# ---- zero-inflow waiting times from the observed end state
l_end = 1 - out["observed_share_end"]
out["zero_inflow_years_to"] = {"50pct": round(math.log(l_end / 0.5) / h_y, 2), "90pct": round(math.log(l_end / 0.1) / h_y, 2),
                               "legacy_half_life_years": round(math.log(2) / h_y, 2)}

# ---- projections with parameter uncertainty
rng = np.random.default_rng(20261128)
N = 4000
h_draw = rng.gamma(Mg.sum(), 1 / Lobs[:-1].sum(), N)
b_draw = {"package": rng.beta(Bm.sum() + 1, E.sum() - Bm.sum() + 1, N), "namespace": rng.beta(nB.sum() + 1, nE.sum() - nB.sum() + 1, N)}
slope = np.polyfit(np.arange(len(E)), np.log(E), 1)[0]
out["fit_full"]["entrant_log_growth_per_week"] = round(float(slope), 4)
out["fit_full"]["entrant_doubling_weeks"] = round(math.log(2) / slope, 1) if slope > 0 else None
E_med = float(np.median(E[-8:]))
inflows = {"none": lambda w: 0.0, "constant": lambda w: E_med, "exponential": lambda w: E_med * math.exp(slope * (w + 1))}
horizons = {"2027-07-28": 52 - len(W), "2028-07-28": 104 - len(W), "2029-07-28": 156 - len(W)}


def project(bs, hs, inflow, weeks):
    res = []
    for b, h in zip(bs, hs):
        _, _, tr = simulate(Lobs[-1], Mobs[-1], inflow, b, h, weeks)
        res.append(tr[-1])
    q = np.quantile(res, [0.025, 0.5, 0.975])
    return {"median": round(float(q[1]), 4), "ci95": [round(float(q[0]), 4), round(float(q[2]), 4)]}


proj = {}
for bname, bs in b_draw.items():
    for iname, fn in inflows.items():
        for hname, wk in horizons.items():
            proj["b=%s|inflow=%s|%s" % (bname, iname, hname)] = project(bs, h_draw, fn, wk)
out["projection"] = proj
# interventions at constant inflow, evaluated 2027-07-28 and 2028-07-28
iv = {}
ones = np.ones(N)
for name, bs, hs in (("status_quo", b_draw["package"], h_draw), ("h_x2", b_draw["package"], 2 * h_draw),
                     ("h_x5", b_draw["package"], 5 * h_draw), ("b=0.58 (PyPI-like default)", 0.58 * ones, h_draw),
                     ("b=0.90", 0.9 * ones, h_draw), ("b=0.90 and h_x5", 0.9 * ones, 5 * h_draw)):
    iv[name] = {hn: project(bs, hs, inflows["constant"], wk) for hn, wk in horizons.items()}
out["interventions_constant_inflow"] = iv

# ---- sensitivity grid: modern share on 2027-07-28 at constant inflow
bg = [0.05, 0.16, 0.3, 0.58, 0.9]
hg = [0.07, 0.135, 0.27, 0.5, 1.0, 2.0]
grid = [[round(simulate(Lobs[-1], Mobs[-1], inflows["constant"], b, h / WPY, horizons["2027-07-28"])[2][-1], 3) for h in hg] for b in bg]
out["sensitivity_2027_07_28"] = {"b": bg, "h_per_year": hg, "modern_share": grid}
json.dump(out, open(os.path.join(R, "model.json"), "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
wk = np.arange(len(W) + 1)
axs[0].plot(wk, out["observed_share_by_week"], "k-", lw=2, label="observed (npm, runtime-declared)")
for K, c in ((4, "#4C72B0"), (8, "#C44E52")):
    hK = Mg[:K].sum() / Lobs[:K].sum(); bK = Bm[:K].sum() / E[:K].sum()
    _, _, tr = simulate(Lobs[K], Mobs[K], E[K:], bK, hK, len(W) - K)
    axs[0].plot(np.arange(K + 1, len(W) + 1), tr, "--", color=c, label="model fit on weeks 0-%d" % (K - 1))
H = 104 - len(W)
for lab, b, h, c in (("status quo", b_pkg, h_w, "k"), ("h x5", b_pkg, 5 * h_w, "#4C72B0"),
                     ("b = 0.58", 0.58, h_w, "#55A868"), ("b = 0.90", 0.9, h_w, "#C44E52")):
    _, _, tr = simulate(Lobs[-1], Mobs[-1], inflows["constant"], b, h, H)
    axs[1].plot(np.arange(len(W) + 1, 105), tr, color=c, label=lab)
axs[1].axvline(52, color="grey", ls=":"); axs[1].text(52.5, 0.02, "12 months", fontsize=8)
axs[0].set_xlabel("weeks since v2 SDK GA"); axs[0].set_ylabel("modern (2026-07-28-capable) share"); axs[0].legend(fontsize=8)
axs[1].set_xlabel("weeks since v2 SDK GA"); axs[1].set_ylabel("projected modern share (constant inflow)"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_model.png"), dpi=150)
print(json.dumps(out, indent=1))
