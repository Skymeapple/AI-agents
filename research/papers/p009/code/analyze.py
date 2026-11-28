"""Experiment D: combine (B) trained capacities and (C) measured recall demand.

1. Capacity N*(acc=0.5) of each trained mixer, compared with the random-code theory of Prop. 1
   (efficiency eta = N*_trained / N*_theory).
2. State needed by a fixed-state memory to hold the context-specific associations of real text,
   S_req(T) = d_k d_v with N*(d_k,d_v,V) = N_ctx(T)/eta, using the large-d_v rule N* ~= S/(2 ln V),
   versus the KV cache of ONE softmax attention layer, 2 d T floats.
3. Critical context length T*(S) beyond which a fixed state S (per layer, ideal or trained eta)
   cannot hold N_ctx(T).
Outputs: results/capacity_table.json, results/tradeoff.json, results/fig_trained.png, results/fig_tradeoff.png, results/fig_dvsweep.png
"""
import os, json, glob, numpy as np
from capacity_sim import theory_acc

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
V = 512


def nstar(Ns, accs):
    pts = sorted(zip(Ns, accs))
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 >= 0.5 > a1:
            f = (a0 - 0.5) / (a0 - a1)
            return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0))))
    return float("inf") if pts[-1][1] >= 0.5 else None


runs = []
for f in sorted(glob.glob(os.path.join(R, "train", "*.json"))):
    d = json.load(open(f)); ar = d["args"]
    Ns = [int(k) for k in d["acc"]]; acc = [d["acc"][k] for k in d["acc"]]
    th = [theory_acc(1.0, n - 1, ar["dk"], ar["dv"], V) for n in Ns]
    ns_tr = nstar(Ns, acc); ns_th = nstar(Ns, th)
    if ns_th is None or ns_th == float("inf"):  # extend grid for theory
        grid = np.unique(np.logspace(0.5, 4, 60).astype(int))
        ns_th = nstar(list(grid), [theory_acc(1.0, n - 1, ar["dk"], ar["dv"], V) for n in grid])
    runs.append({"arch": ar["arch"], "d": ar.get("d", 128), "dk": ar["dk"], "dv": ar["dv"], "S": ar["dk"] * ar["dv"], "acc": dict(zip(Ns, acc)),
                 "acc_theory_random_codes": dict(zip(Ns, th)), "Nstar_trained": ns_tr, "Nstar_theory": ns_th,
                 "eta": (ns_tr / ns_th) if (ns_tr not in (None, float("inf"))) else None,
                 "final_loss": d["final_loss"], "train_time_s": d["train_time_s"],
                 "gates_on_pair_tokens": d.get("gates_on_pair_tokens")})
json.dump(runs, open(os.path.join(R, "capacity_table.json"), "w"), indent=1)
print(f"{'arch':7s} {'d':>4s} {'dk':>3s} {'dv':>4s} {'S':>6s} {'N*tr':>7s} {'N*th':>7s} {'eta':>5s}  acc@N=4..256")
for r in sorted(runs, key=lambda r: (r["arch"], r["dv"], r["dk"])):
    nt = r["Nstar_trained"]; e = r["eta"]
    print(f"{r['arch']:7s} {r['d']:4d} {r['dk']:3d} {r['dv']:4d} {r['S']:6d} {str(round(nt,1)) if nt else '-':>7s} {r['Nstar_theory']:7.1f} "
          f"{(f'{e:.2f}' if e else '-'):>5s}  " + " ".join(f"{a:.2f}" for a in r["acc"].values()))

# ---- fit eta per arch (median over dk at dv=64) and S-scaling exponent of N*_trained
summary = {}
for arch in sorted({r["arch"] for r in runs}):
    rr = [r for r in runs if r["arch"] == arch and r["dv"] == 64 and r["d"] == 128 and r["eta"]]
    if len(rr) >= 2:
        X = np.log([r["S"] for r in rr]); Y = np.log([r["Nstar_trained"] for r in rr])
        slope = float(np.polyfit(X, Y, 1)[0])
    else:
        slope = None
    summary[arch] = {"eta_median_dv64": float(np.median([r["eta"] for r in rr])) if rr else None,
                     "eta_range_dv64": [float(min(r["eta"] for r in rr)), float(max(r["eta"] for r in rr))] if rr else None,
                     "Nstar_vs_dk_loglog_slope_dv64": slope}
    rv = [r for r in runs if r["arch"] == arch and r["dk"] == 16 and r["d"] == 128 and r["eta"]]
    if len(rv) >= 2:
        X = np.log([r["dv"] for r in rv]); Y = np.log([r["Nstar_trained"] for r in rv])
        summary[arch]["Nstar_vs_dv_loglog_slope_dk16"] = float(np.polyfit(X, Y, 1)[0])
print(json.dumps(summary, indent=1))

# ---- tradeoff with measured demand
dem = json.load(open(os.path.join(R, "demand.json")))
trade = {"summary_trained": summary, "corpora": {}}
etas = {"ideal additive (eta=1)": 1.0}
for arch in ["linear", "gla", "delta", "gdelta"]:
    if arch in summary and summary[arch]["eta_median_dv64"]:
        etas[f"trained {arch} (eta={summary[arch]['eta_median_dv64']:.2f})"] = summary[arch]["eta_median_dv64"]
for corpus in ["prose", "code"]:
    D = dem[corpus]; Vc = D["vocab"]; lnV = np.log(Vc)
    rows = D["rows"]; T = np.array([r["T"] for r in rows]); N = np.array([r["N_ctx"] for r in rows])
    fit = D["fits"]["N_ctx"]
    out = {"vocab": Vc, "rows": []}
    for t, n in zip(T, N):
        row = {"T": int(t), "N_ctx": float(n), "KV_floats_one_layer_d4096": float(2 * 4096 * t)}
        for name, e in etas.items():
            row[f"S_req[{name}]"] = float(2 * lnV * n / e)
        out["rows"].append(row)
    # critical context length for a fixed per-layer state S, for three write policies:
    #   every-token  : the memory receives one association per token (demand = T)
    #   dedup        : one write per distinct bigram (demand = N_bi(T))
    #   oracle       : only context-specific associations that will recur (demand = N_ctx(T))
    def inv(Ncap, key):
        Nk = np.array([r[key] for r in rows]) if key != "T" else T.astype(float)
        if Ncap <= Nk[0]:
            return float(T[0] * Ncap / Nk[0])
        if Ncap <= Nk[-1]:
            return float(np.exp(np.interp(np.log(Ncap), np.log(Nk), np.log(T))))
        b = np.polyfit(np.log(T[-3:]), np.log(Nk[-3:]), 1)[0]   # local exponent of last three points
        return float(T[-1] * (Ncap / Nk[-1]) ** (1 / b))
    out["Tstar"] = []
    for S in [2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22]:
        for name, e in etas.items():
            Ncap = e * S / (2 * lnV)
            out["Tstar"].append({"S": S, "eta_label": name, "eta": e, "Ncap": Ncap,
                                 "T*_every_token": inv(Ncap, "T"), "T*_dedup": inv(Ncap, "N_bi"),
                                 "T*_oracle": inv(Ncap, "N_ctx")})
    Tstar = lambda Ncap: inv(Ncap, "N_ctx")
    out["beta_local_last3"] = float(np.polyfit(np.log(T[-3:]), np.log(N[-3:]), 1)[0])
    trade["corpora"][corpus] = out
json.dump(trade, open(os.path.join(R, "tradeoff.json"), "w"), indent=1)
for c in trade["corpora"]:
    print(c, "beta_local", trade["corpora"][c]["beta_local_last3"])
    for r in trade["corpora"][c]["Tstar"]:
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    r = trade["corpora"][c]["rows"][-1]
    print("   T=32768:", {k: f"{v:.3g}" for k, v in r.items()})

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
archs = ["attn", "linear", "gla", "delta", "gdelta"]
fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for j, dk in enumerate([8, 16, 32, 64]):
    for i, arch in enumerate(archs):
        rr = [r for r in runs if r["arch"] == arch and r["dk"] == dk and r["dv"] == 64 and r["d"] == 128]
        if rr:
            a = rr[0]["acc"]; ax[j].plot(list(a.keys()), list(a.values()), "o-", color=f"C{i}", label=arch)
    th = [r for r in runs if r["dk"] == dk and r["dv"] == 64]
    if th:
        a = th[0]["acc_theory_random_codes"]; ax[j].plot(list(a.keys()), list(a.values()), "k--", label="theory (random codes)")
    ax[j].set_xscale("log"); ax[j].set_title(f"trained, d_k={dk}, d_v=64"); ax[j].set_xlabel("stored pairs N")
ax[0].set_ylabel("recall accuracy"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_trained.png"), dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, c in enumerate(["prose", "code"]):
    rows = trade["corpora"][c]["rows"]; T = [r["T"] for r in rows]
    ax[0].loglog(T, [r["KV_floats_one_layer_d4096"] for r in rows], "k-", lw=2, label="KV cache, 1 attn layer, d=4096" if i == 0 else None)
    for k, name in enumerate(etas):
        ax[0].loglog(T, [r[f"S_req[{name}]"] for r in rows], linestyle=["-", "--", ":", "-.", (0, (1, 4))][k % 5], color=f"C{i}",
                     label=f"{c}: {name}")
    ts = [r for r in trade["corpora"][c]["Tstar"] if r["eta"] == 1.0]
    for k, (key, mk) in enumerate([("T*_every_token", "k:"), ("T*_dedup", "s--"), ("T*_oracle", "o-")]):
        ax[1].loglog([r["S"] for r in ts], [r[key] for r in ts], mk if k else ":", color="k" if k == 0 else f"C{i}",
                     label=(f"{c}: {key}" if k else ("every-token write (both)" if i == 0 else None)), ms=4)
ax[0].set_xlabel("context length T"); ax[0].set_ylabel("floats of memory"); ax[0].legend(fontsize=6)
ax[0].set_title("(a) state needed to hold N_ctx(T) vs one KV cache")
ax[1].set_xlabel("fixed state size S per layer (floats)"); ax[1].set_ylabel("critical context length T*")
ax[1].set_title("(b) critical context length T*, eta=1"); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_tradeoff.png"), dpi=130)

# ---- value-dimension sweep at d_k=16 (trained vs theory)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
for i, arch in enumerate(["linear", "delta"]):
    for dd, mk in [(128, "o-"), (512, "s:")]:
        rv = sorted([r for r in runs if r["arch"] == arch and r["dk"] == 16 and r["d"] == dd and r["eta"]], key=lambda r: r["dv"])
        if rv:
            ax.loglog([r["dv"] for r in rv], [r["Nstar_trained"] for r in rv], mk, color=f"C{i+1}",
                      label=f"trained {arch}, embedding d={dd} (value half {dd//2})")
sim = json.load(open(os.path.join(R, "sim_capacity.json")))
simdv = sorted([c for c in sim["capacities"] if c["kind"] == "additive_dvsweep"], key=lambda c: c["dv"])
ax.loglog([c["dv"] for c in simdv], [c["Nstar_sim"] for c in simdv], "k^:", label="simulated ideal additive memory")
eta_sim = {}
for r in runs:
    m_ = [c for c in sim["capacities"] if c["kind"] in ("additive", "additive_dvsweep") and c["dk"] == r["dk"] and c["dv"] == r["dv"]]
    if m_ and r["Nstar_trained"] not in (None, float("inf")):
        eta_sim[f'{r["arch"]}_d{r["d"]}_dk{r["dk"]}_dv{r["dv"]}'] = r["Nstar_trained"] / float(np.mean([c["Nstar_sim"] for c in m_]))
print("eta relative to simulated ideal additive memory:", json.dumps({k: round(v, 3) for k, v in eta_sim.items()}, indent=0))
json.dump(eta_sim, open(os.path.join(R, "eta_vs_sim.json"), "w"), indent=1)

rv = sorted([r for r in runs if r["arch"] == "linear" and r["dk"] == 16], key=lambda r: r["dv"])
ax.loglog([r["dv"] for r in rv], [r["Nstar_theory"] for r in rv], "k--", label="Eq. (1), random codes")
ax.set_xlabel("value dimension d_v (d_k = 16)"); ax.set_ylabel("N* (50% recall)"); ax.legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_dvsweep.png"), dpi=130)
print("gates:", [(r["arch"], r["dk"], r["dv"], r["gates_on_pair_tokens"]) for r in runs if r["gates_on_pair_tokens"]])
